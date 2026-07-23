// B2b — the zarr-facing half of the cell-centroid ingestion (docs/stream-b-bridge-plan.md).
//
// Reads a SpatialData **regions table** (AnnData, zarr v3) and turns it into per-`cell_type_id`
// centroid clouds, each a graph points source built through the dep-free B2a converter
// (`centroidsToField`). Heavy zarr access + all transform/placement reasoning live here on the
// playground side (ADR-0015 ownership boundary); `src/datasource/centroidsToField.ts` receives only
// resolved values.
//
//   openExtraConsolidated(url) → tree.tables[name]   (zarrextra's own zarrita; consolidated
//                                                      metadata lives at the store root)
//     obsm/spatial   [N,2] float64 → per-cell (x,y) centroids
//     obs/cell_type_id  [N] int64  → per-cell integer type id
//
// ADR-0018: a points cloud is keyed per `(table, region, cell_type)` — we group by `cell_type_id`
// into SEPARATE clouds and NEVER merge them. The centroids are carried, not host-transformed: the
// placement rides along on each cloud and is applied on the GPU (splatDensity).

import { centroidsToField } from "../../../src/datasource";
import type { Affine3 } from "../../../src/datasource";
import type { Graph } from "../../../src/gpu/graph/graph";
import type { FieldProvenance, GpuField, ResolvedPlacement } from "../../../src/gpu/graph/handle";

/** Default target table on the Leap034 store. */
export const DEFAULT_CELL_TABLE = "Leap034_imc_cells";

/** A single cell type's centroid cloud, ready to add to a graph as a points source (via B2a). */
export interface CellTypeCloud {
  /** The `cell_type_id` this cloud was split on (integer). */
  readonly id: number;
  /** Number of cells of this type. */
  readonly n: number;
  /** This type's centroids, parallel arrays (for the scatter render; the GPU path packs them). */
  readonly xs: readonly number[];
  readonly ys: readonly number[];
  /** Build this cloud as a graph points source, carrying the shared placement + per-type
   *  provenance. Deferred so one read can seed many graphs / re-runs. */
  readonly source: (g: Graph) => GpuField;
}

export interface CellTable {
  /** Per-`cell_type_id` clouds, ascending by id. Never one merged cloud (ADR-0018). */
  readonly types: CellTypeCloud[];
  /** The shared array→world placement for every cloud (same table ⇒ same system). */
  readonly placement: ResolvedPlacement;
  /** The table's region + instance key (NGFF `region` / `instance_key`), for provenance/HUD. */
  readonly provenance: { region: string; instanceKey: string };
  /** Total cells read across all types. */
  readonly totalCells: number;
  /** The coordinate-system name the placement targets (mirrors `placement.system`, for the HUD). */
  readonly system: string;
  /** Human label for the HUD. */
  readonly label: string;
}

// ---- structural zarr shapes (zarrita is zarrextra's dep, not ours — no value import) ---------------

interface ZarrChunk {
  data: ArrayLike<number | bigint>;
  shape: number[];
  stride: number[];
}
interface ZarrArrayLike {
  readonly shape: number[];
  readonly chunks: number[];
  readonly dtype: string;
  getChunk(coords: number[]): Promise<ZarrChunk>;
}
interface LazyZarrArrayLike {
  get(): Promise<ZarrArrayLike>;
}

/** Descend a zarrextra ZarrTree node to a named lazy-array leaf by path, e.g. `["obsm","spatial"]`.
 *  A leaf is identified by exposing a `get()`; group nodes are plain objects. */
function leafAt(node: unknown, path: string[]): LazyZarrArrayLike {
  let cur: unknown = node;
  for (const key of path) {
    if (!cur || typeof cur !== "object") throw new Error(`cellTable: path .${path.join(".")} — '${key}' parent is not a group`);
    cur = (cur as Record<string, unknown>)[key];
  }
  if (!cur || typeof (cur as LazyZarrArrayLike).get !== "function") {
    throw new Error(`cellTable: no zarr array at .${path.join(".")}`);
  }
  return cur as LazyZarrArrayLike;
}

const IDENTITY_AFFINE: Affine3 = {
  origin: [0, 0, 0],
  axes: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

/** Read a full 1-D zarr array into a plain `number[]`, reassembling across its chunk grid.
 *  Handles int64 (`BigInt64Array` → `Number`) as well as float dtypes. */
async function readArray1D(arr: ZarrArrayLike): Promise<number[]> {
  const n = arr.shape[0] ?? 0;
  const cn = arr.chunks[0] ?? n || 1;
  const out = new Array<number>(n);
  const nChunks = Math.ceil(n / cn);
  for (let ci = 0; ci < nChunks; ci++) {
    const chunk = await arr.getChunk([ci]);
    const base = ci * cn;
    const extent = Math.min(cn, n - base);
    const stride = chunk.stride[0] ?? 1;
    for (let i = 0; i < extent; i++) out[base + i] = Number(chunk.data[i * stride]!);
  }
  return out;
}

/** Read a full `[N,2]` zarr array into parallel `xs`/`ys`, reassembling across the (possibly
 *  column-chunked) grid. The Leap034 store chunks this `[24875,1]`, so column 0 (x) and column 1
 *  (y) land in separate chunks — the generic grid walk below handles that. */
async function readCentroids2D(arr: ZarrArrayLike): Promise<{ xs: number[]; ys: number[] }> {
  const rows = arr.shape[0] ?? 0;
  const cols = arr.shape[1] ?? 0;
  if (cols < 2) throw new Error(`cellTable: obsm/spatial has ${cols} columns, expected >= 2 (x,y)`);
  const cr = arr.chunks[0] ?? rows || 1;
  const cc = arr.chunks[1] ?? cols || 1;
  const xs = new Array<number>(rows);
  const ys = new Array<number>(rows);
  const nR = Math.ceil(rows / cr);
  // Only the chunks covering columns 0 and 1 matter.
  const colChunks = new Set([Math.floor(0 / cc), Math.floor(1 / cc)]);
  for (let ir = 0; ir < nR; ir++) {
    const rBase = ir * cr;
    const rExtent = Math.min(cr, rows - rBase);
    for (const ic of colChunks) {
      const chunk = await arr.getChunk([ir, ic]);
      const sR = chunk.stride[0] ?? 1;
      const sC = chunk.stride[1] ?? 1;
      const cBase = ic * cc;
      for (const col of [0, 1]) {
        if (Math.floor(col / cc) !== ic) continue;
        const localC = col - cBase;
        const target = col === 0 ? xs : ys;
        for (let i = 0; i < rExtent; i++) target[rBase + i] = Number(chunk.data[i * sR + localC * sC]!);
      }
    }
  }
  return { xs, ys };
}

/** Read the group attributes (NGFF `region` / `instance_key`) off the table node's zarrextra attrs.
 *  Symbol-keyed in the tree; falls back to sensible Leap034 defaults if absent. */
function readTableAttrs(tableNode: unknown): { region: string; instanceKey: string } {
  const attrs = collectAttrs(tableNode);
  const region = typeof attrs?.region === "string" ? attrs.region : "";
  const instanceKey = typeof attrs?.instance_key === "string" ? attrs.instance_key : "";
  return { region, instanceKey };
}

/** zarrextra stashes a node's zarr.json attributes under a Symbol key; pull the first symbol-keyed
 *  object that looks like NGFF regions-table attrs. */
function collectAttrs(node: unknown): Record<string, unknown> | undefined {
  if (!node || typeof node !== "object") return undefined;
  for (const sym of Object.getOwnPropertySymbols(node)) {
    const v = (node as Record<symbol, unknown>)[sym];
    if (v && typeof v === "object" && ("region" in v || "instance_key" in v)) return v as Record<string, unknown>;
  }
  return undefined;
}

/**
 * Read a SpatialData regions table and group its centroids into per-`cell_type_id` clouds.
 *
 * Placement: the demo has NO image overlay, so the cloud is placed at **identity** in the store's
 * named coordinate system (`Leap034` here) — an honest "already in system S" rather than a
 * fabricated transform. The real array→world affine for the annotated shape (`shapes/<region>`'s
 * coordinateTransformations) is a straightforward future upgrade; wiring it here is deferred.
 * TODO(stream-B): compose the annotated shape's transform into `worldFromArray` once an image is
 * co-displayed; until then identity-in-system is correct for a standalone splat.
 *
 * ADR-0018: never merges the clouds — one points source per `cell_type_id`, each carrying the
 * shared placement + its own provenance `{ region, instanceKey, cellTypeId }`.
 */
export async function readCellTable(
  url: string,
  opts: { table?: string; system?: string } = {},
): Promise<CellTable> {
  const tableName = opts.table ?? DEFAULT_CELL_TABLE;
  const zx = await import("zarrextra");
  const opened = await zx.openExtraConsolidated(url);
  const { tree } = zx.unwrap(opened) as { tree: { tables?: Record<string, unknown> } };
  const tableNode = tree.tables?.[tableName];
  if (!tableNode) {
    const have = Object.keys(tree.tables ?? {}).join(", ") || "(none)";
    throw new Error(`cellTable: no table '${tableName}' in store; available: ${have}`);
  }

  const spatialArr = await leafAt(tableNode, ["obsm", "spatial"]).get();
  const idArr = await leafAt(tableNode, ["obs", "cell_type_id"]).get();

  const rows = spatialArr.shape[0] ?? 0;
  if ((idArr.shape[0] ?? 0) !== rows) {
    throw new Error(`cellTable: obsm/spatial rows (${rows}) != obs/cell_type_id length (${idArr.shape[0]})`);
  }

  const { xs, ys } = await readCentroids2D(spatialArr);
  const ids = await readArray1D(idArr);

  const attrs = readTableAttrs(tableNode);
  const region = attrs.region || opts.system || "";
  const instanceKey = attrs.instanceKey || "cell_id";
  const system = opts.system ?? "Leap034";

  // Identity in the named system (no image overlay in this demo — see the TODO above).
  const placement: ResolvedPlacement = { system, worldFromArray: IDENTITY_AFFINE };

  // Group by cell_type_id into separate clouds (ADR-0018 — never one merged cloud).
  const byId = new Map<number, { xs: number[]; ys: number[] }>();
  for (let i = 0; i < rows; i++) {
    const id = ids[i]!;
    let bucket = byId.get(id);
    if (!bucket) {
      bucket = { xs: [], ys: [] };
      byId.set(id, bucket);
    }
    bucket.xs.push(xs[i]!);
    bucket.ys.push(ys[i]!);
  }

  const types: CellTypeCloud[] = [...byId.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([id, cloud]) => {
      const provenance: FieldProvenance = { region, instanceKey, cellTypeId: id };
      return {
        id,
        n: cloud.xs.length,
        xs: cloud.xs,
        ys: cloud.ys,
        source: (g: Graph) => g.source(centroidsToField(cloud.xs, cloud.ys, { placement, provenance }), `cellType:${id}`),
      };
    });

  return {
    types,
    placement,
    provenance: { region, instanceKey },
    totalCells: rows,
    system,
    label: `${tableName} · ${rows} cells · ${types.length} types`,
  };
}

/** A dep-free synthetic 2-type cloud that flows through the identical path, so the demo works with
 *  no store reachable. Two gaussian blobs, distinct `cell_type_id`s, placed in a fixture system. */
export function syntheticCellTable(opts: { perType?: number; system?: string } = {}): CellTable {
  const perType = opts.perType ?? 800;
  const system = opts.system ?? "fixture";
  const placement: ResolvedPlacement = { system, worldFromArray: IDENTITY_AFFINE };
  // Deterministic PRNG so the fixture is stable.
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) / 2; // ~N(0, ~0.29)
  const blob = (cx: number, cy: number, spread: number) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < perType; i++) {
      xs.push(cx + gauss() * spread);
      ys.push(cy + gauss() * spread);
    }
    return { xs, ys };
  };
  const clouds: Array<{ id: number; blob: { xs: number[]; ys: number[] } }> = [
    { id: 1, blob: blob(30, 30, 12) },
    { id: 2, blob: blob(75, 70, 10) },
  ];
  const types: CellTypeCloud[] = clouds.map(({ id, blob: cloud }) => {
    const provenance: FieldProvenance = { region: "synthetic", instanceKey: "cell_id", cellTypeId: id };
    return {
      id,
      n: cloud.xs.length,
      xs: cloud.xs,
      ys: cloud.ys,
      source: (g: Graph) => g.source(centroidsToField(cloud.xs, cloud.ys, { placement, provenance }), `cellType:${id}`),
    };
  });
  return {
    types,
    placement,
    provenance: { region: "synthetic", instanceKey: "cell_id" },
    totalCells: perType * clouds.length,
    system,
    label: `synthetic · ${perType * clouds.length} cells · 2 types`,
  };
}
