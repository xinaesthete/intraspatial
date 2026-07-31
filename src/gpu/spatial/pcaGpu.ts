// PCA's two heavy steps on the device — the covariance `CᵀC` and the projection `CV`.
//
// `src/spatial/pca.ts` lists the host timings; the short version is that at 60k cells and
// 300 genes the host PCA is 5.4 seconds and at 100k it is 8.6, all of it **synchronous on
// the main thread**. The page is not slow during that, it is frozen. That is the reason
// this module exists: the asymptotics were never the problem (O(N·G²) with G in the
// hundreds is not much work for a GPU) — the problem is *where* the work runs.
//
// **Both kernels are tiled matmuls, and the tiling is what makes them worth writing.** The
// naive shape — one thread per covariance entry `(p, q)`, striding the rows — reads columns
// of a row-major matrix, so consecutive threads touch addresses `dim` floats apart and
// every memory transaction is thrown away. The 16x16 tile instead stages a block of rows in
// workgroup memory, where each thread's load is contiguous along the *gene* axis, and then
// does 16 multiply-adds per element loaded.
//
// Three things about this are deliberate and would be easy to get wrong:
//
//   • **Only the upper triangle of tiles runs.** The covariance is symmetric, so
//     `wid.y < wid.x` returns immediately and the diagonal tile masks its own lower half at
//     write time. Halving the work is the small part; the important part is that each
//     `(p, q)` is then owned by exactly one thread in exactly one workgroup, which is what
//     makes the accumulate a plain `+=` with no atomics and no chance of the two halves of
//     a symmetric matrix disagreeing in the last bit.
//   • **The data is uploaded per row-tile, not once.** A 100k x 300 f32 matrix is 120 MB,
//     against a `maxStorageBufferBindingSize` that is 128 MiB by default in Chrome — so the
//     obvious "upload once, dispatch many times" runs out of headroom at exactly the sizes
//     this module is for. Tiling the upload costs one extra pass over the bytes and removes
//     the ceiling. See `planTiles`.
//   • **f32 accumulation, in a two-level sum.** The host oracle is f64 and this is not; the
//     per-tile products go into a fresh accumulator that is added into the running total
//     once per 16 rows, so the longest dependent add chain is `n/16` rather than `n`. The
//     residual against the f64 host is measured in `pcaGpu.gpu.test.ts` rather than assumed
//     — and the numbers there are the justification for not needing Kahan on top.
//
// Raw WGSL rather than TGSL for the usual two reasons (ADR-0003): `var<workgroup>` shared
// memory with barriers, and integer division on `u32` — `lid.x / 32u` here — which in TGSL
// "use gpu" kernels silently lowers to float division.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
  type ColumnStats,
  checkPcaArgs,
  columnStats,
  normaliseCovariance,
  type PcaBasis,
  type PcaOptions,
  type PcaResult,
  pcaBasis,
  pcaComponentCount,
} from "../../spatial/pca";
import { getDevice } from "../device";

/** Output tile edge, and the row block staged in workgroup memory. 16x16 = 256 threads,
 *  the same shape as a textbook shared-memory matmul; 16 f32 is also one 64-byte burst per
 *  row, so the staging loads are contiguous rather than merely nearby. */
const TILE = 16;
/** Threads per projection workgroup: 32 component lanes x 2 rows. */
const PROJECT_WG = 64;
const PROJECT_LANES = 32;

const Params = d.struct({
  /** Rows in THIS tile — the data buffer holds exactly these, at local index. */
  rows: d.u32,
  dim: d.u32,
  nComp: d.u32,
  /** Global index of local row 0, needed only by the projection to address `scores`. */
  rowOffset: d.u32,
});

const covLayout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  data: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [rows, dim]
  mean: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [dim]
  scale: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [dim]
  cov: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // [dim, dim], accumulated
});

const projectLayout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  data: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [rows, dim]
  mean: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  scale: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  /** Components TRANSPOSED to `[dim, nComp]`, so the 32 lanes of a half-warp read 32
   *  consecutive floats instead of 32 addresses `dim` apart. */
  compT: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  scores: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // [n, nComp]
});

const COV_TEMPLATE = /* wgsl */ `
var<workgroup> As: array<f32, ${TILE * TILE}u>;
var<workgroup> Bs: array<f32, ${TILE * TILE}u>;

@compute @workgroup_size(${TILE}, ${TILE})
fn covariance(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  // Upper triangle of TILES only. Uniform across the workgroup, so returning here cannot
  // strand anyone at a barrier.
  if (wid.y < wid.x) { return; }
  let dim = params.dim;
  let rows = params.rows;
  let p0 = wid.x * ${TILE}u;
  let q0 = wid.y * ${TILE}u;
  let tx = lid.x;
  let ty = lid.y;

  var total: f32 = 0.0;
  var row: u32 = 0u;
  loop {
    if (row >= rows) { break; }
    // Stage rows [row, row+16) for both column blocks. Thread (tx, ty) fetches column
    // p0+tx (resp. q0+tx) of row row+ty: consecutive tx, consecutive addresses.
    let r = row + ty;
    let ca = p0 + tx;
    let cb = q0 + tx;
    var va: f32 = 0.0;
    var vb: f32 = 0.0;
    if (r < rows) {
      if (ca < dim) { va = (data[r * dim + ca] - mean[ca]) * scale[ca]; }
      if (cb < dim) { vb = (data[r * dim + cb] - mean[cb]) * scale[cb]; }
    }
    As[ty * ${TILE}u + tx] = va;
    Bs[ty * ${TILE}u + tx] = vb;
    workgroupBarrier();

    // Fresh accumulator per row block, folded into 'total' once — the two-level sum from
    // the module header. Out-of-range rows and columns staged exact zeros above, so they
    // contribute nothing and need no test here.
    var acc: f32 = 0.0;
    for (var rr: u32 = 0u; rr < ${TILE}u; rr = rr + 1u) {
      acc = acc + As[rr * ${TILE}u + ty] * Bs[rr * ${TILE}u + tx];
    }
    total = total + acc;
    workgroupBarrier();
    row = row + ${TILE}u;
  }

  // 'q >= p' masks the lower half of a diagonal tile. Every surviving (p, q) is written by
  // exactly one thread of exactly one workgroup, so this read-modify-write is race-free —
  // and across row tiles the dispatches are serialised, which is what lets the covariance
  // accumulate in place instead of being reduced afterwards.
  let p = p0 + ty;
  let q = q0 + tx;
  if (p < dim && q < dim && q >= p) {
    cov[p * dim + q] = cov[p * dim + q] + total;
  }
}
`;

const PROJECT_TEMPLATE = /* wgsl */ `
@compute @workgroup_size(${PROJECT_WG})
fn project(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let dim = params.dim;
  let nComp = params.nComp;
  // 64 threads = 2 rows x 32 component lanes. The 32 lanes of a row all read the same
  // 'data[...]' address, which is a broadcast rather than 32 transactions, and 32 adjacent
  // floats of 'compT'.
  let sub = lid.x / ${PROJECT_LANES}u;
  let lane = lid.x - sub * ${PROJECT_LANES}u;
  let li = wid.x * 2u + sub;
  if (li >= params.rows) { return; }

  var c = lane;
  loop {
    if (c >= nComp) { break; }
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < dim; t = t + 1u) {
      acc = acc + (data[li * dim + t] - mean[t]) * scale[t] * compT[t * nComp + c];
    }
    scores[(params.rowOffset + li) * nComp + c] = acc;
    c = c + ${PROJECT_LANES}u;
  }
}
`;

interface Pipes {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  covariance: GPUComputePipeline;
  project: GPUComputePipeline;
}
let pipeCache: Promise<Pipes> | undefined;

function getPipes(): Promise<Pipes> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const finish = (resolved: ReturnType<typeof tgpu.resolveWithContext>, entryPoint: string): GPUComputePipeline => {
      const module = device.createShaderModule({ code: resolved.code });
      const layout = device.createPipelineLayout({
        bindGroupLayouts: resolved.usedBindGroupLayouts.map((l) => root.unwrap(l)),
      });
      return device.createComputePipeline({ layout, compute: { module, entryPoint } });
    };
    return {
      device,
      root,
      covariance: finish(
        tgpu.resolveWithContext({ template: COV_TEMPLATE, externals: { ...covLayout.bound }, names: "strict" }),
        "covariance",
      ),
      project: finish(
        tgpu.resolveWithContext({ template: PROJECT_TEMPLATE, externals: { ...projectLayout.bound }, names: "strict" }),
        "project",
      ),
    };
  })();
  return pipeCache;
}

// Grow-only and never destroyed, the constraint every other module here works under:
// `.destroy()` segfaults Dawn-on-Node's teardown.
type Root = Pipes["root"];
// Written as helpers so the buffer types stay concrete: `ReturnType<Root["createBuffer"]>`
// widens to `TgpuBuffer<AnyData>`, which no bind group will accept.
const makeF32 = (root: Root, cap: number) => root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage");
const makeParams = (root: Root) => root.createBuffer(Params).$usage("uniform");
type F32Buf = ReturnType<typeof makeF32>;

const pool = new Map<string, { buf: F32Buf; cap: number }>();

function ensure(root: Root, key: string, floats: number): F32Buf {
  const got = pool.get(key);
  if (got && got.cap >= floats) return got.buf;
  const cap = Math.max(floats, (got?.cap ?? 0) * 2, 4);
  const buf = makeF32(root, cap);
  pool.set(key, { buf, cap });
  return buf;
}

let paramsBuf: ReturnType<typeof makeParams> | undefined;
function ensureParams(root: Root) {
  paramsBuf ??= makeParams(root);
  return paramsBuf;
}

/** Work per dispatch, in multiply-adds. Same watchdog reasoning as `knn.ts`: a dispatch
 *  past roughly two seconds is killed with NO error and the output keeps whatever was
 *  there, so the failure looks like a plausible answer. 2e9 is the figure the descent
 *  kernel uses and is comfortably inside the largest known-good dispatch here. */
const TARGET_MACS_PER_DISPATCH = 2_000_000_000;

/** Cap on the staged row tile, in floats. 64 MB — half of Chrome's default
 *  `maxStorageBufferBindingSize`, and the actual device limit is applied on top. */
const TILE_FLOAT_CAP = 16_000_000;

/**
 * Rows per dispatch, satisfying three separate ceilings at once.
 *
 * Getting this wrong is not a slowdown, it is a wrong answer that looks right — which is
 * why all three are computed rather than one being assumed generous:
 *   1. the GPU watchdog (`TARGET_MACS_PER_DISPATCH`),
 *   2. `maxStorageBufferBindingSize`, since the tile is uploaded not the whole matrix,
 *   3. `maxComputeWorkgroupsPerDimension` — the projection issues `rows / 2` workgroups,
 *      and exceeding that limit invalidates the command buffer silently. That exact bug
 *      cost a day in `umapLayoutGpu`; it is cheaper to clamp than to rediscover.
 */
export function planTiles(n: number, dim: number, macsPerRow: number, limits: { maxBufferFloats: number; maxWorkgroups: number }): number {
  const byWork = Math.floor(TARGET_MACS_PER_DISPATCH / Math.max(macsPerRow, 1));
  const byBuffer = Math.floor(limits.maxBufferFloats / Math.max(dim, 1));
  const byWorkgroups = limits.maxWorkgroups * 2;
  const rows = Math.min(n, byWork, byBuffer, byWorkgroups);
  // Whole row blocks, so the covariance staging never straddles a tile boundary, and at
  // least one block however tight the limits are.
  return Math.max(TILE, Math.floor(rows / TILE) * TILE);
}

export interface TileOptions {
  /** Rows per dispatch, overriding `planTiles`. A test seam, as in `knn.ts` — real data
   *  large enough to need several tiles is too slow to put in a unit test, and the
   *  accumulate-across-dispatches path would otherwise never run under test. */
  readonly rowsPerTile?: number;
}

function limitsFor(device: GPUDevice): { maxBufferFloats: number; maxWorkgroups: number } {
  return {
    maxBufferFloats: Math.min(Math.floor(device.limits.maxStorageBufferBindingSize / 4), TILE_FLOAT_CAP),
    maxWorkgroups: device.limits.maxComputeWorkgroupsPerDimension,
  };
}

/** Copy rows `[start, start + rows)` of a row-major matrix into a flat f32 tile.
 *
 *  The `<ArrayBuffer>` argument is load-bearing, not decoration: a bare `Float32Array`
 *  annotation widens to `Float32Array<ArrayBufferLike>`, which `queue.writeBuffer` rejects
 *  because it will not accept a possibly-shared buffer. Same wart as `gramMatrix.ts`. */
function sliceRows(
  data: ArrayLike<number>,
  dim: number,
  start: number,
  rows: number,
  into: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> {
  const from = start * dim;
  const count = rows * dim;
  if (data instanceof Float32Array) {
    into.set(data.subarray(from, from + count));
  } else {
    for (let t = 0; t < count; t++) into[t] = data[from + t]!;
  }
  return into;
}

/**
 * Step 2 on the device: the `dim x dim` covariance of the centred, scaled columns.
 *
 * Same result as `covarianceHost` — normalised by `n - 1` and symmetric — but accumulated
 * in f32, so it is close rather than equal. `pcaGpu.gpu.test.ts` measures how close.
 */
export async function covarianceGpu(
  data: ArrayLike<number>,
  n: number,
  dim: number,
  stats: ColumnStats,
  opts: TileOptions = {},
): Promise<Float64Array> {
  const { device, root, covariance } = await getPipes();

  const rowsPerTile = opts.rowsPerTile ?? planTiles(n, dim, (dim * dim) / 2, limitsFor(device));
  const dataBuf = ensure(root, "data", rowsPerTile * dim);
  const meanBuf = ensure(root, "mean", dim);
  const scaleBuf = ensure(root, "scale", dim);
  const covBuf = ensure(root, "cov", dim * dim);
  const params = ensureParams(root);
  device.queue.writeBuffer(root.unwrap(meanBuf), 0, Float32Array.from(stats.mean) as BufferSource);
  device.queue.writeBuffer(root.unwrap(scaleBuf), 0, Float32Array.from(stats.scale) as BufferSource);
  // The pool is grow-only and shared across calls, so the accumulator must be cleared
  // explicitly — a stale covariance from a previous call would otherwise be added to.
  device.queue.writeBuffer(root.unwrap(covBuf), 0, new Float32Array(dim * dim) as BufferSource);

  const bind = root.unwrap(root.createBindGroup(covLayout, { params, data: dataBuf, mean: meanBuf, scale: scaleBuf, cov: covBuf }));
  const tiles = Math.ceil(dim / TILE);
  const staging = new Float32Array(rowsPerTile * dim);

  for (let start = 0; start < n; start += rowsPerTile) {
    const rows = Math.min(rowsPerTile, n - start);
    device.queue.writeBuffer(root.unwrap(dataBuf), 0, sliceRows(data, dim, start, rows, staging), 0, rows * dim);
    params.write({ rows, dim, nComp: 0, rowOffset: start });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(covariance);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(tiles, tiles);
    pass.end();
    device.queue.submit([enc.finish()]);
    // One tile in flight at a time: `params` is a single uniform rewritten per tile, and
    // the accumulate into `cov` is only race-free because the dispatches are serialised.
    await device.queue.onSubmittedWorkDone();
  }

  const got = (await covBuf.read()) as ArrayLike<number>;
  const cov = new Float64Array(dim * dim);
  for (let p = 0; p < dim; p++) {
    for (let q = p; q < dim; q++) cov[p * dim + q] = got[p * dim + q]!;
  }
  return normaliseCovariance(cov, n, dim);
}

/** Step 4 on the device: project the cells onto `basis.components`. */
export async function projectScoresGpu(
  data: ArrayLike<number>,
  n: number,
  dim: number,
  stats: ColumnStats,
  basis: PcaBasis,
  opts: TileOptions = {},
): Promise<Float32Array> {
  const { device, root, project } = await getPipes();
  const nComp = basis.nComponents;

  const rowsPerTile = opts.rowsPerTile ?? planTiles(n, dim, dim * nComp, limitsFor(device));
  const dataBuf = ensure(root, "data", rowsPerTile * dim);
  const meanBuf = ensure(root, "mean", dim);
  const scaleBuf = ensure(root, "scale", dim);
  const compBuf = ensure(root, "compT", dim * nComp);
  const scoresBuf = ensure(root, "scores", n * nComp);
  const params = ensureParams(root);

  // Transpose the basis to `[dim, nComp]` — see the `compT` note on the layout.
  const compT = new Float32Array(dim * nComp);
  for (let c = 0; c < nComp; c++) {
    for (let t = 0; t < dim; t++) compT[t * nComp + c] = basis.components[c * dim + t]!;
  }
  device.queue.writeBuffer(root.unwrap(meanBuf), 0, Float32Array.from(stats.mean) as BufferSource);
  device.queue.writeBuffer(root.unwrap(scaleBuf), 0, Float32Array.from(stats.scale) as BufferSource);
  device.queue.writeBuffer(root.unwrap(compBuf), 0, compT as BufferSource);

  const bind = root.unwrap(
    root.createBindGroup(projectLayout, {
      params,
      data: dataBuf,
      mean: meanBuf,
      scale: scaleBuf,
      compT: compBuf,
      scores: scoresBuf,
    }),
  );
  const staging = new Float32Array(rowsPerTile * dim);

  for (let start = 0; start < n; start += rowsPerTile) {
    const rows = Math.min(rowsPerTile, n - start);
    device.queue.writeBuffer(root.unwrap(dataBuf), 0, sliceRows(data, dim, start, rows, staging), 0, rows * dim);
    params.write({ rows, dim, nComp, rowOffset: start });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(project);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(rows / 2));
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const got = (await scoresBuf.read()) as ArrayLike<number>;
  const scores = new Float32Array(n * nComp);
  for (let t = 0; t < n * nComp; t++) scores[t] = got[t]!;
  return scores;
}

/**
 * PCA with both heavy steps on the device.
 *
 * Same signature and same result shape as `pca` in `src/spatial/pca.ts`, which is its
 * oracle: column statistics and the eigendecomposition still run on the host in f64
 * (together a few percent of the cost, and f64 is load-bearing for the eigensolver),
 * while the two O(N·G·…) steps run as tiled matmuls.
 */
export async function pcaGpu(data: ArrayLike<number>, n: number, dim: number, opts: PcaOptions = {}): Promise<PcaResult> {
  const nComponents = pcaComponentCount(n, dim, opts);
  checkPcaArgs(n, nComponents);
  const stats = columnStats(data, n, dim, opts);
  const cov = await covarianceGpu(data, n, dim, stats);
  const basis = pcaBasis(cov, dim, nComponents);
  const scores = await projectScoresGpu(data, n, dim, stats, basis);
  return { scores, ...basis, mean: stats.mean };
}
