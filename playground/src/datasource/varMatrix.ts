// Reading `X` (and `layers/*`) out of an AnnData table — the zarr-facing half of expression
// ingestion. The pure indexing arithmetic is `src/datasource/sparseColumns.ts`; this module finds
// the arrays, works out the encoding, and fetches only what a selection needs.
//
// Why this exists at all: `src/spatial/gram.ts` takes per-cell **weights**, and a cell-type channel
// is just the one-hot case of one. So a gene's expression column drops straight into the same
// statistic — the N-way cell-type matrix and a gene–gene spatial co-expression matrix are the same
// computation with a different mark. See `docs/cell-stats.md` §12.
//
// **This is a selected-genes feature, not an all-genes one, and the cost is structural.** The Gram
// matrix is `O(G·P)` to splat and `O(G²·P)` to reduce, so tens of genes are comfortable and
// thousands are not. That matches ADR-0005's existing position that sparse gene columns are a
// *selection* mechanism and the full matrix is never densified. The UI enforces the selection; this
// module reports what a selection will cost before it is paid.

import type { SpatialData } from "@spatialdata/core";
import {
  type ColumnStats,
  columnStats,
  denseToColumns,
  type SparseEncoding,
  sparseColumnCost,
  sparseToColumns,
} from "../../../src/datasource/sparseColumns";
import { isLeaf, leafAt, readStrings1D, storeTree, symbolAttrs, type ZarrArrayLike } from "./cellTable";

/** What a table's expression matrix looks like, without having read any of it. */
export interface VarCatalog {
  /** Var (gene / marker / protein) names, in var order. */
  readonly names: string[];
  readonly nCells: number;
  readonly nVars: number;
  /** `"dense"`, or the sparse layout AnnData recorded in the group's `encoding-type`. */
  readonly encoding: "dense" | SparseEncoding;
  /** Selectable matrices: `"X"` plus any `layers/*`. A layer is often the one you want — `X` is
   *  frequently already scaled/centred, which a mass-normalised statistic should not be fed. */
  readonly matrices: string[];
  /** Populated instead of the rest when there is no readable `X`. */
  readonly error?: string;
}

export interface VarColumns {
  readonly names: string[];
  readonly nCells: number;
  /** Column-major, `values[g * nCells + i]` — the layout `channelsFromExpression` consumes. */
  readonly values: Float64Array;
  readonly stats: ColumnStats[];
  readonly matrix: string;
}

/** Read a whole 1-D zarr array into a typed array.
 *
 *  Deliberately not `readArray1D`'s `number[]`: a CSR `indices` array for a real store runs to
 *  millions of entries, where a boxed JS array costs several times the memory and is materially
 *  slower to walk. int64 (`BigInt64Array` chunks) is narrowed to `Number` here — var and cell
 *  indices are bounded by array lengths, so the 2^53 limit is not in play. */
async function readTyped1D(arr: ZarrArrayLike): Promise<Float64Array> {
  const n = arr.shape[0] ?? 0;
  const cn = arr.chunks[0] ?? (n || 1);
  const out = new Float64Array(n);
  for (let ci = 0; ci < Math.ceil(n / cn); ci++) {
    const chunk = await arr.getChunk([ci]);
    const base = ci * cn;
    const extent = Math.min(cn, n - base);
    const stride = chunk.stride[0] ?? 1;
    for (let i = 0; i < extent; i++) out[base + i] = Number(chunk.data[i * stride]!);
  }
  return out;
}

/** The `var` dataframe's index column holds the gene names. AnnData records WHICH column that is
 *  in the group's `_index` attribute rather than fixing the name, so read the attribute; falling
 *  back to the conventional `_index` only when the attrs are missing. */
async function readVarNames(varNode: unknown, nVars: number): Promise<string[]> {
  const attrs = symbolAttrs(varNode, (v) => "_index" in v || "column-order" in v);
  const indexName = typeof attrs?._index === "string" ? attrs._index : "_index";
  for (const name of [indexName, "_index", "index"]) {
    const node = varNode && typeof varNode === "object" ? (varNode as Record<string, unknown>)[name] : undefined;
    if (isLeaf(node)) {
      try {
        return await readStrings1D(await node.get());
      } catch {
        // fall through to the next candidate
      }
    }
  }
  // No names in the store is survivable — the statistic does not need them, only the UI does.
  return Array.from({ length: nVars }, (_, i) => `var${i}`);
}

interface MatrixHandle {
  readonly encoding: "dense" | SparseEncoding;
  readonly nCells: number;
  readonly nVars: number;
  readonly node: unknown;
}

/** Resolve `X` or `layers/<name>` to its encoding and shape without reading the data.
 *
 *  A dense matrix is a zarr array and carries its own shape. A sparse one is a GROUP of
 *  `data`/`indices`/`indptr` whose logical shape lives only in the group's attributes — `indptr`'s
 *  length pins the compressed dimension but says nothing about the other, so the `shape` attribute
 *  is not optional here. */
function describeMatrix(node: unknown, name: string): MatrixHandle {
  if (isLeaf(node)) {
    throw new Error(`varMatrix: ${name} is a dense array; its shape is read via describeDense`);
  }
  const attrs = symbolAttrs(node, (v) => "encoding-type" in v || "shape" in v);
  const enc = String(attrs?.["encoding-type"] ?? "");
  const shape = attrs?.shape;
  if (!Array.isArray(shape) || shape.length < 2) {
    throw new Error(`varMatrix: ${name} is a group with no usable 'shape' attribute (encoding-type: ${enc || "absent"})`);
  }
  const encoding: SparseEncoding = enc.startsWith("csc") ? "csc" : "csr";
  return { encoding, nCells: Number(shape[0]), nVars: Number(shape[1]), node };
}

/** Find `X` / `layers/<name>` on a table node and describe it, dense or sparse. */
async function resolveMatrix(tableNode: unknown, matrix: string): Promise<MatrixHandle> {
  const path = matrix === "X" ? ["X"] : ["layers", matrix];
  let node: unknown = tableNode;
  for (const key of path) {
    node = node && typeof node === "object" ? (node as Record<string, unknown>)[key] : undefined;
  }
  if (!node) throw new Error(`varMatrix: no ${path.join("/")} in this table`);
  if (isLeaf(node)) {
    const arr = await node.get();
    return { encoding: "dense", nCells: arr.shape[0] ?? 0, nVars: arr.shape[1] ?? 0, node };
  }
  return describeMatrix(node, path.join("/"));
}

/**
 * Catalogue a table's expression matrix: gene names, shape, encoding, and which matrices exist.
 *
 * Cheap by construction — it reads the `var` index (a few thousand strings) and some attributes,
 * never the matrix. Returns `{ error }` rather than throwing for a table with no `X`, because a
 * store where only some tables carry expression should still be usable.
 */
export async function listVars(sdata: SpatialData, tableName: string): Promise<VarCatalog> {
  const empty = { names: [], nCells: 0, nVars: 0, encoding: "dense" as const, matrices: [] };
  try {
    const tableNode = storeTree(sdata).tables?.[tableName];
    if (!tableNode) return { ...empty, error: `no table '${tableName}'` };

    const handle = await resolveMatrix(tableNode, "X");
    // UPSTREAM(anndata.js): `AnnData.varNames()` returns a lazy `zarr.Array`, not `string[]`, and
    // decoding a zarr string / categorical index is exactly the fiddly part — so we read the `var`
    // index off the tree ourselves (`readVarNames`). A `varNames({ decode: true }): string[]`, or a
    // `TableElement.getVarNames()` on the sd.js side mirroring `getObsColumnNames()`, would delete it.
    // (The earlier `await tableNode.varNames()` bound the lazy Array and silently fell back to
    // `var{i}` for every gene — the names never actually appeared.)
    const names = await readVarNames((tableNode as Record<string, unknown>).var, handle.nVars);
    // UPSTREAM(anndata.js): `AnnData.layers` is an `AxisArrays` with `get()`/`has()` but no
    // `keys()`, so "which layers exist?" — a plain catalogue question — can't be answered on the
    // high-level object. We enumerate the group's members off the tree instead.
    const layersNode = (tableNode as Record<string, unknown>).layers;
    const layers = layersNode && typeof layersNode === "object" ? Object.keys(layersNode as Record<string, unknown>) : [];
    return {
      names: names.length === handle.nVars ? names : Array.from({ length: handle.nVars }, (_, i) => names[i] ?? `var${i}`),
      nCells: handle.nCells,
      nVars: handle.nVars,
      encoding: handle.encoding,
      matrices: ["X", ...layers],
    };
  } catch (err) {
    return { ...empty, error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * What fraction of the stored matrix a selection has to read.
 *
 * Exposed so the UI can warn *before* starting rather than appearing to hang: CSC can slice out a
 * gene directly, but CSR scatters each gene's entries across every cell's row, so pulling one gene
 * costs a full scan — and CSR is what scanpy writes by default.
 */
export function selectionCost(cat: VarCatalog, nWanted: number): number {
  if (cat.encoding === "dense") return Math.min(1, nWanted / Math.max(cat.nVars, 1));
  return sparseColumnCost(cat.encoding, nWanted, cat.nVars);
}

/** Densify a dense `[nCells, nVars]` X's selected columns, fetching only the chunks that hold them.
 *  Wanted vars are grouped by column-chunk so a chunk covering several selected genes is read once
 *  rather than once per gene. */
async function readDenseColumns(arr: ZarrArrayLike, wanted: readonly number[]): Promise<Float64Array> {
  const rows = arr.shape[0] ?? 0;
  const cols = arr.shape[1] ?? 0;
  const cr = arr.chunks[0] ?? (rows || 1);
  const cc = arr.chunks[1] ?? (cols || 1);
  const out = new Float64Array(wanted.length * rows);

  const byChunk = new Map<number, Array<{ slot: number; v: number }>>();
  wanted.forEach((v, slot) => {
    if (v < 0 || v >= cols) return;
    const ic = Math.floor(v / cc);
    const list = byChunk.get(ic);
    if (list) list.push({ slot, v });
    else byChunk.set(ic, [{ slot, v }]);
  });

  for (const [ic, members] of byChunk) {
    for (let ir = 0; ir < Math.ceil(rows / cr); ir++) {
      const chunk = await arr.getChunk([ir, ic]);
      const sR = chunk.stride[0] ?? 1;
      const sC = chunk.stride[1] ?? 1;
      const rBase = ir * cr;
      const rExtent = Math.min(cr, rows - rBase);
      for (const { slot, v } of members) {
        const localC = v - ic * cc;
        const base = slot * rows;
        for (let i = 0; i < rExtent; i++) out[base + rBase + i] = Number(chunk.data[i * sR + localC * sC]!);
      }
    }
  }
  return out;
}

/**
 * Load the selected vars as column-major per-cell weights, ready for `channelsFromExpression`.
 *
 * `vars` are indices into `catalogue.names`. Order is preserved and repeats are honoured, so the
 * returned `names` line up with the columns one-for-one.
 */
export async function readVarColumns(
  sdata: SpatialData,
  opts: { table: string; vars: readonly number[]; matrix?: string; names?: readonly string[] },
): Promise<VarColumns> {
  const matrix = opts.matrix ?? "X";
  const tableNode = storeTree(sdata).tables?.[opts.table];
  if (!tableNode) throw new Error(`varMatrix: no table '${opts.table}'`);

  const handle = await resolveMatrix(tableNode, matrix);
  const wanted = [...opts.vars];
  let values: Float64Array;

  if (handle.encoding === "dense") {
    values = await readDenseColumns(await (handle.node as { get(): Promise<ZarrArrayLike> }).get(), wanted);
  } else {
    // Sparse: three 1-D arrays. For CSC only the wanted slices are needed in principle, but the
    // arrays are chunked along their own length rather than by var, so a partial read would still
    // fetch whole chunks — reading them whole keeps this simple and is what CSR requires anyway.
    //
    // UPSTREAM(anndata.js): `SparseArray.get()` takes `number | Slice | null` per axis — one column
    // or a *contiguous* range, never an arbitrary set of genes. So the high-level object can't
    // express "these 14 columns" in one read; per-gene calls would be N CSR scans. We read the raw
    // arrays and scatter every wanted column in a single pass (`sparseColumns.ts`). A
    // `getColumns(indices: number[])` that did one pass would honour what `selectionCost` promises
    // ("one CSR scan for N genes") instead of quietly defeating it.
    const [indptr, indices, data] = await Promise.all([
      readTyped1D(await leafAt(handle.node, ["indptr"]).get()),
      readTyped1D(await leafAt(handle.node, ["indices"]).get()),
      readTyped1D(await leafAt(handle.node, ["data"]).get()),
    ]);
    values = sparseToColumns({ encoding: handle.encoding, indptr, indices, data, nCells: handle.nCells, nVars: handle.nVars }, wanted);
  }

  const names = wanted.map((v, i) => opts.names?.[i] ?? `var${v}`);
  return { names, nCells: handle.nCells, values, stats: columnStats(values, handle.nCells, names), matrix };
}

/** Re-exported so the demo can densify an already-fetched block without importing two modules. */
export { denseToColumns };
