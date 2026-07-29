// Reading `X` from and writing `obsm/*` into an AnnData zarr store, from Node.
//
// This is the repo's first WRITE path into a zarr store — everything before it reads.
// That asymmetry is why this module is more careful than its size suggests: a reader
// that misunderstands a store produces a wrong picture, whereas a writer that
// misunderstands one produces a *corrupt dataset that other tools will later fail on*,
// possibly long after the fact.
//
// Three things have to be right for scanpy / anndata / MDV to read back what we write:
//
//   • **`encoding-type` / `encoding-version` attributes.** AnnData does not infer types
//     from zarr structure; it dispatches on these attributes. A `[N, d]` float array in
//     `obsm` without `encoding-type: "array"` is not an `obsm` entry, it is a stray
//     zarr array that `read_zarr` will skip or choke on.
//   • **The `obsm` group itself** must exist and carry `encoding-type: "dict"`.
//   • **Row count must equal `n_obs`.** AnnData validates this on read; a mismatch is a
//     hard error there, so it is a hard error here, where the message can be useful.
//
// The sparse-matrix reading logic deliberately is NOT duplicated here — the indexing
// arithmetic lives in `src/datasource/sparseColumns.ts` and is shared with the browser
// path (`playground/src/datasource/varMatrix.ts`). This module is the Node-side zarr
// plumbing plus the write half.

import * as zarr from "zarrita";
import { type SparseEncoding, sparseToColumns } from "./sparseColumns";

/** Column-major `[nVars][nCells]` is what `sparseColumns` produces; UMAP wants
 *  row-major `[nCells][nVars]`. Named so call sites cannot confuse the two. */
export interface ExpressionMatrix {
  /** Row-major `values[cell * nVars + gene]`. */
  readonly values: Float32Array;
  readonly nCells: number;
  readonly nVars: number;
  readonly varNames: string[];
}

/** Open a local directory or an http(s) URL as a zarr store. Local stores are opened
 *  writable; an http(s) store is read-only and `writeObsm` will reject it. */
export async function openStore(location: string): Promise<zarr.Location<never>> {
  if (/^https?:\/\//.test(location)) {
    return zarr.root(new zarr.FetchStore(location)) as unknown as zarr.Location<never>;
  }
  // Node-only: dynamic so a browser bundle never pulls `node:fs` in.
  const mod = (await import("@zarrita/storage/fs")) as { default: new (path: string) => never };
  return zarr.root(new mod.default(location));
}

async function tryOpenArray(loc: zarr.Location<never>, path: string) {
  try {
    return await zarr.open(loc.resolve(path), { kind: "array" });
  } catch {
    return undefined;
  }
}

async function tryOpenGroup(loc: zarr.Location<never>, path: string) {
  try {
    return await zarr.open(loc.resolve(path), { kind: "group" });
  } catch {
    return undefined;
  }
}

/** Read a whole 1-D array as f64. int64 indices are narrowed via `Number` — they are
 *  bounded by array lengths, so the 2^53 limit is not in play. */
async function readWhole1D(arr: zarr.Array<zarr.DataType, never>): Promise<Float64Array> {
  const chunk = await zarr.get(arr as never);
  const src = chunk.data as ArrayLike<number | bigint>;
  const out = new Float64Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = Number(src[i]!);
  return out;
}

/** AnnData records WHICH column of `var` holds the names in the group's `_index`
 *  attribute rather than fixing the name. Mirrors `varMatrix.ts`'s `readVarNames`. */
async function readVarNames(loc: zarr.Location<never>, tablePath: string, nVars: number): Promise<string[]> {
  const varGroup = await tryOpenGroup(loc, `${tablePath}/var`);
  const indexName = typeof varGroup?.attrs?._index === "string" ? (varGroup.attrs._index as string) : "_index";
  for (const name of [indexName, "_index", "index"]) {
    const arr = await tryOpenArray(loc, `${tablePath}/var/${name}`);
    if (!arr) continue;
    try {
      const chunk = await zarr.get(arr as never);
      const data = chunk.data as ArrayLike<unknown>;
      if (data.length === nVars) return Array.from({ length: nVars }, (_, i) => String(data[i]));
    } catch {
      // fall through
    }
  }
  return Array.from({ length: nVars }, (_, i) => `var${i}`);
}

export interface ReadMatrixOptions {
  /** Path to the AnnData group inside the store, e.g. `"tables/table"` for SpatialData
   *  or `""` for a bare `.h5ad`-style zarr. */
  readonly tablePath?: string;
  /** `"X"` or a `layers/<name>`. */
  readonly matrix?: string;
  /** Restrict to these var indices (a gene subset). Order preserved. */
  readonly vars?: readonly number[];
}

/**
 * Read `X` (or a layer) as a dense row-major matrix.
 *
 * Densifies, which is only defensible because `vars` is expected to be a *selection* —
 * ADR-0005's position, and the same one `varMatrix.ts` takes. Reading all 20k genes of
 * a 500k-cell store into memory is not a supported mode and will exhaust the heap
 * before it exhausts your patience.
 */
export async function readExpressionMatrix(loc: zarr.Location<never>, opts: ReadMatrixOptions = {}): Promise<ExpressionMatrix> {
  const tablePath = opts.tablePath ?? "";
  const matrix = opts.matrix ?? "X";
  const rel = matrix === "X" ? "X" : `layers/${matrix}`;
  const path = tablePath ? `${tablePath}/${rel}` : rel;

  const dense = await tryOpenArray(loc, path);
  if (dense) {
    const nCells = dense.shape[0] ?? 0;
    const nVars = dense.shape[1] ?? 0;
    const varNames = await readVarNames(loc, tablePath, nVars);
    const wanted = opts.vars ? [...opts.vars] : undefined;
    const outVars = wanted?.length ?? nVars;
    const chunk = await zarr.get(dense as never);
    const src = chunk.data as ArrayLike<number>;
    const values = new Float32Array(nCells * outVars);
    for (let i = 0; i < nCells; i++) {
      for (let c = 0; c < outVars; c++) {
        values[i * outVars + c] = Number(src[i * nVars + (wanted ? wanted[c]! : c)]!);
      }
    }
    return { values, nCells, nVars: outVars, varNames: wanted ? wanted.map((v) => varNames[v] ?? `var${v}`) : varNames };
  }

  const group = await tryOpenGroup(loc, path);
  if (!group) throw new Error(`annDataObsm: no ${path} in this store (neither array nor group)`);
  const enc = String(group.attrs["encoding-type"] ?? "");
  const shape = group.attrs.shape;
  if (!Array.isArray(shape) || shape.length < 2) {
    throw new Error(`annDataObsm: ${path} is a group with no usable 'shape' attribute (encoding-type: ${enc || "absent"})`);
  }
  const nCells = Number(shape[0]);
  const nVars = Number(shape[1]);
  const encoding: SparseEncoding = enc.startsWith("csc") ? "csc" : "csr";

  const [indptrArr, indicesArr, dataArr] = await Promise.all([
    tryOpenArray(loc, `${path}/indptr`),
    tryOpenArray(loc, `${path}/indices`),
    tryOpenArray(loc, `${path}/data`),
  ]);
  if (!indptrArr || !indicesArr || !dataArr) throw new Error(`annDataObsm: ${path} is missing indptr/indices/data`);
  const [indptr, indices, data] = await Promise.all([readWhole1D(indptrArr), readWhole1D(indicesArr), readWhole1D(dataArr)]);

  const varNames = await readVarNames(loc, tablePath, nVars);
  const wanted = opts.vars ? [...opts.vars] : Array.from({ length: nVars }, (_, i) => i);
  // `sparseToColumns` yields COLUMN-major [gene][cell]; transpose into row-major.
  const cols = sparseToColumns({ encoding, indptr, indices, data, nCells, nVars }, wanted);
  const outVars = wanted.length;
  const values = new Float32Array(nCells * outVars);
  for (let g = 0; g < outVars; g++) {
    const base = g * nCells;
    for (let i = 0; i < nCells; i++) values[i * outVars + g] = cols[base + i]!;
  }
  return { values, nCells, nVars: outVars, varNames: wanted.map((v) => varNames[v] ?? `var${v}`) };
}

/** How many rows `obs` has, so a write can be validated against it. */
export async function readNObs(loc: zarr.Location<never>, tablePath = ""): Promise<number | undefined> {
  const base = tablePath ? `${tablePath}/obs` : "obs";
  const obs = await tryOpenGroup(loc, base);
  if (!obs) return undefined;
  const indexName = typeof obs.attrs?._index === "string" ? (obs.attrs._index as string) : "_index";
  for (const name of [indexName, "_index", "index"]) {
    const arr = await tryOpenArray(loc, `${base}/${name}`);
    if (arr) return arr.shape[0];
  }
  return undefined;
}

export interface WriteObsmOptions {
  readonly tablePath?: string;
  /** Key under `obsm`, e.g. `"X_umap"`. */
  readonly key: string;
  /** Overwrite an existing key. Without this an existing key is an error. */
  readonly force?: boolean;
}

/**
 * Write a row-major `[nObs, dim]` embedding into `obsm/<key>`.
 *
 * Refuses to clobber an existing key unless `force` is set — the destructive case is
 * the one worth a speed bump, since re-running with the wrong parameters would
 * otherwise silently replace a colleague's embedding.
 */
export async function writeObsm(
  loc: zarr.Location<never>,
  embedding: Float32Array,
  nObs: number,
  dim: number,
  opts: WriteObsmOptions,
): Promise<string> {
  if (embedding.length !== nObs * dim) {
    throw new Error(`writeObsm: embedding has ${embedding.length} values, expected ${nObs * dim}`);
  }
  const tablePath = opts.tablePath ?? "";
  const obsmPath = tablePath ? `${tablePath}/obsm` : "obsm";
  const target = `${obsmPath}/${opts.key}`;

  const declared = await readNObs(loc, tablePath);
  if (declared !== undefined && declared !== nObs) {
    throw new Error(`writeObsm: obs has ${declared} rows but the embedding has ${nObs}; refusing to write a mismatched obsm`);
  }

  const existing = await tryOpenArray(loc, target);
  if (existing && !opts.force) {
    throw new Error(`writeObsm: ${target} already exists — pass --force to overwrite`);
  }

  const store = (loc as unknown as { store: { set?: (key: string, value: Uint8Array) => Promise<void> } }).store;
  if (typeof store.set !== "function") {
    throw new Error("writeObsm: this store is read-only (an http(s) URL cannot be written to)");
  }

  const version = await detectZarrFormat(loc, tablePath);
  const rowsPerChunk = Math.min(nObs, 8192);

  if (version === 3) {
    // `obsm` must exist and be typed, or AnnData will not treat its members as obsm
    // entries. Creating it when absent is safe; when present we leave its attrs alone
    // rather than stomping on keys another writer set.
    if (!(await tryOpenGroup(loc, obsmPath))) {
      await zarr.create(loc.resolve(obsmPath), { attributes: { "encoding-type": "dict", "encoding-version": "0.1.0" } });
    }
    const arr = await zarr.create(loc.resolve(target), {
      shape: [nObs, dim],
      chunkShape: [rowsPerChunk, dim],
      dtype: "float32",
      attributes: { "encoding-type": "array", "encoding-version": "0.2.0" },
    });
    await zarr.set(arr as never, null, { data: embedding, shape: [nObs, dim], stride: [dim, 1] } as never);
    return target;
  }

  await writeV2Array(
    store as { set: (key: string, value: Uint8Array) => Promise<void> },
    obsmPath,
    target,
    embedding,
    nObs,
    dim,
    rowsPerChunk,
  );
  return target;
}

const enc = new TextEncoder();
const json = (v: unknown) => enc.encode(JSON.stringify(v, null, 1));

/**
 * Which zarr format the surrounding store uses.
 *
 * This matters more than it looks. `zarrita`'s `create` only ever emits **zarr v3**
 * (`zarr.json`), while the great majority of AnnData / SpatialData stores in the wild —
 * anything scanpy wrote before it gained v3 support — are **v2** (`.zarray` / `.zattrs`).
 * Dropping a v3 array into a v2 store produces a mixed-format hierarchy that
 * `anndata.read_zarr` does not expect and will not load, so the write would appear to
 * succeed and the result would be unreadable by the very tools it exists to feed.
 *
 * Detection is by probing for the marker files rather than trusting a flag, and the
 * default when nothing is found is **v2**, the conservative choice: v2 is readable by
 * every version of anndata, v3 only by recent ones.
 */
async function detectZarrFormat(loc: zarr.Location<never>, tablePath: string): Promise<2 | 3> {
  const store = loc as unknown as { store: { get(key: string): Promise<Uint8Array | undefined> } };
  const base = tablePath ? `/${tablePath}` : "";
  for (const [key, version] of [
    [`${base}/zarr.json`, 3],
    [`${base}/.zgroup`, 2],
    ["/zarr.json", 3],
    ["/.zgroup", 2],
  ] as const) {
    try {
      if (await store.store.get(key)) return version;
    } catch {
      // keep probing
    }
  }
  return 2;
}

/**
 * Write a `[nObs, dim]` float32 array as **zarr v2**, uncompressed.
 *
 * Hand-rolled because zarrita cannot emit v2. Uncompressed (`compressor: null`) is a
 * deliberate simplification and is fully valid v2: it costs disk on an array that is
 * `nObs × dim × 4` bytes — 8 MB for a million cells in 2-D — and it removes any
 * dependency on a compressor implementation agreeing byte-for-byte with what
 * zarr-python expects.
 *
 * Chunk keys use the v2 default `.` dimension separator, and chunks are **full-size**:
 * v2 requires a trailing partial chunk to be padded out to the declared chunk shape,
 * not truncated. Getting that wrong yields a file zarr-python reads as garbage at the
 * tail rather than an error, so the padding below is load-bearing.
 */
async function writeV2Array(
  store: { set: (key: string, value: Uint8Array) => Promise<void> },
  obsmPath: string,
  target: string,
  embedding: Float32Array,
  nObs: number,
  dim: number,
  rowsPerChunk: number,
): Promise<void> {
  await store.set(`/${obsmPath}/.zgroup`, json({ zarr_format: 2 }));
  await store.set(`/${obsmPath}/.zattrs`, json({ "encoding-type": "dict", "encoding-version": "0.1.0" }));

  await store.set(
    `/${target}/.zarray`,
    json({
      zarr_format: 2,
      shape: [nObs, dim],
      chunks: [rowsPerChunk, dim],
      dtype: "<f4",
      compressor: null,
      fill_value: 0,
      order: "C",
      filters: null,
    }),
  );
  await store.set(`/${target}/.zattrs`, json({ "encoding-type": "array", "encoding-version": "0.2.0" }));

  const nChunks = Math.ceil(nObs / rowsPerChunk);
  for (let ci = 0; ci < nChunks; ci++) {
    const startRow = ci * rowsPerChunk;
    const rows = Math.min(rowsPerChunk, nObs - startRow);
    // Always allocate the full chunk; the tail is zero-padded, as v2 requires.
    const chunk = new Float32Array(rowsPerChunk * dim);
    chunk.set(embedding.subarray(startRow * dim, (startRow + rows) * dim));
    await store.set(`/${target}/${ci}.0`, new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
  }
}
