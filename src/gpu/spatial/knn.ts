// Exact k-nearest-neighbours in D dimensions — the search UMAP is built on, and the
// first spatial primitive here that is not hard-wired to 2-D points.
//
// Every existing kernel in this directory indexes `pts[2u*i]` / `pts[2u*i + 1u]`,
// because the discrete-cell front is about points on a slide. A UMAP over gene
// expression is a k-NN in a 50-to-2000-dimensional feature space, so `dim` becomes a
// uniform and the distance accumulates over a loop. Nothing else changes — which is
// why this sits alongside `kthNeighborDistance.ts` rather than replacing it. (That one
// stays: it returns only ρ_i, needs no index array, and the 2-D specialisation is
// meaningfully faster for the density-estimate use.)
//
// **Output shape is the load-bearing decision.** This returns a sparse `[n, k]` pair
// of index/distance arrays, NOT the dense n×n matrix that `fuzzyAdjacency` and `cknn`
// produce. Dense is correct for a persistence sweep over a few hundred points and
// impossible here: 100k cells is 10^10 f32, i.e. 40 GB. Every stage downstream of this
// module is sparse for the same reason.
//
// Brute force O(N²·D), one thread per query point, k smallest kept in a private array.
// That is exact and (per ADR-0016) the thing to get right before approximating. It is
// also genuinely adequate up to ~10-20k cells: at N=10k, D=50 the inner loop is 5·10⁹
// fused multiply-adds, seconds on a laptop GPU. Past that the O(N²) wall is real and
// the fix is an approximate index behind this same `KnnResult` interface — see
// `docs/umap-on-anndata.md` §3.
//
// WGSL template rather than `"use gpu"`, for the reason recorded in ADR-0003 and hit
// first by `kthNeighborDistance`: the per-thread k-smallest selection needs a local
// mutable array, which this TypeGPU version cannot express in TGSL.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { KnnResult } from "../../spatial/umapGraph";
import { getDevice } from "../device";

const WG = 64;
/** Compile-time bound on the private per-thread arrays. k is a uniform ≤ this. 32 is
 *  well past UMAP's useful range (n_neighbors is conventionally 5–50, and the graph
 *  stops changing much above ~30). */
const MAX_K = 32;

const Params = d.struct({ n: d.u32, dim: d.u32, k: d.u32, rowOffset: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  data: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [n, dim] row-major
  outIdx: { storage: (n: number) => d.arrayOf(d.u32, n), access: "mutable" }, // [n, k]
  outDist: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // [n, k]
});

const TEMPLATE = /* wgsl */ `
@compute @workgroup_size(${WG})
fn knn(@builtin(global_invocation_id) gid: vec3u) {
  // Query rows are processed in tiles; this dispatch owns [rowOffset, rowOffset + count).
  let i = gid.x + params.rowOffset;
  let n = params.n;
  if (i >= n) { return; }
  let dim = params.dim;
  let k = min(params.k, ${MAX_K}u);

  // Ascending by construction: bd[0] is the nearest kept so far, bd[k-1] the furthest.
  var bd: array<f32, ${MAX_K}u>;
  var bi: array<u32, ${MAX_K}u>;
  for (var t: u32 = 0u; t < k; t = t + 1u) { bd[t] = 3.4e38; bi[t] = 0u; }

  let ibase = i * dim;
  for (var j: u32 = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
    let jbase = j * dim;
    // Squared distance; the sqrt is paid once per kept neighbour, not per candidate.
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
      let delta = data[ibase + c] - data[jbase + c];
      acc = acc + delta * delta;
    }
    // Early reject against the current k-th best, still in squared space.
    let worst = bd[k - 1u];
    if (acc >= worst) { continue; }
    // Shift the tail down and drop the candidate into place.
    var p: u32 = k - 1u;
    loop {
      if (p == 0u) { break; }
      if (bd[p - 1u] <= acc) { break; }
      bd[p] = bd[p - 1u];
      bi[p] = bi[p - 1u];
      p = p - 1u;
    }
    bd[p] = acc;
    bi[p] = j;
  }

  let obase = i * k;
  for (var t: u32 = 0u; t < k; t = t + 1u) {
    outIdx[obase + t] = bi[t];
    outDist[obase + t] = sqrt(bd[t]);
  }
}
`;

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;
function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({
      template: TEMPLATE,
      externals: { ...layout.bound },
      names: "strict",
    });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "knn" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makePool(root: Root, values: number, pairs: number) {
  return {
    values,
    pairs,
    data: root.createBuffer(d.arrayOf(d.f32, values)).$usage("storage"),
    outIdx: root.createBuffer(d.arrayOf(d.u32, pairs)).$usage("storage"),
    outDist: root.createBuffer(d.arrayOf(d.f32, pairs)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, values: number, pairs: number) {
  if (pool && pool.values >= values && pool.pairs >= pairs) return pool;
  pool = makePool(root, Math.max(values, pool?.values ?? 0, 1), Math.max(pairs, pool?.pairs ?? 0, 1));
  return pool;
}

export const KNN_MAX_K = MAX_K;

/** Target work per dispatch, in (row x column x dimension) products.
 *
 *  Calibrated against measurement, not guessed: a single dispatch of 1.28e10 products
 *  (n=16000, dim=50) completed reliably, while 3.4e10 (n=26000) was killed. 4e9 sits ~3x
 *  below the largest known-good dispatch. Larger tiles mean fewer per-tile syncs and so
 *  more throughput; the cost of guessing too large is not a slowdown but a silently
 *  all-zero result, so the margin is deliberate. */
const TARGET_PAIRS_PER_DISPATCH = 4_000_000_000;

export interface KnnGpuOptions {
  /** Rows. */
  readonly n: number;
  /** Columns (feature dimension). */
  readonly dim: number;
  /** Neighbours per row, excluding self. 1..32, and < n. */
  readonly k: number;
}

/**
 * Exact k-NN over a row-major `[n, dim]` feature matrix.
 *
 * Returns the same `KnnResult` shape as `knnBruteForceCpu`, so the two are drop-in
 * substitutes and the GPU test can diff them directly. Rows come back sorted ascending
 * by distance with self excluded.
 */
export async function knnGpu(data: ArrayLike<number>, opts: KnnGpuOptions): Promise<KnnResult> {
  const { n, dim, k } = opts;
  if (data.length < n * dim) throw new Error(`knnGpu: data has ${data.length} values, need ${n * dim}`);
  if (k < 1 || k > MAX_K) throw new Error(`knnGpu: k must be in 1..${MAX_K}`);
  if (k >= n) throw new Error(`knnGpu: need k < n (k=${k}, n=${n})`);

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n * dim, n * k);

  const flat = data instanceof Float32Array && data.length === n * dim ? data : Float32Array.from({ length: n * dim }, (_, t) => data[t]!);
  device.queue.writeBuffer(root.unwrap(p.data), 0, flat as BufferSource);

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, data: p.data, outIdx: p.outIdx, outDist: p.outDist }));

  // Tiled over query rows, and this is a correctness fix rather than a tuning knob.
  // One dispatch covering every row is O(n^2 * dim) of work in a single command, and past
  // roughly two seconds the OS GPU watchdog kills it — Dawn then reports NO error, the
  // output buffer keeps its zeroes, and the caller gets a complete-looking result whose
  // every index is 0. Measured before this: n=26000 and n=30000 returned all-zero
  // neighbour lists (recall 0.000) while n=28000 happened to survive, so it presented as
  // an intermittent wrong answer rather than a failure. Bounding each dispatch to a tile
  // of rows keeps every command well inside the watchdog.
  // Each query row costs n*dim products, so the tile is TARGET / (n*dim) rows, rounded
  // up to a whole number of workgroups and clamped into [WG, n].
  const perRow = Math.max(n * dim, 1);
  const wantRows = Math.max(WG, Math.min(n, Math.floor(TARGET_PAIRS_PER_DISPATCH / perRow)));
  const rowsPerTile = Math.min(n, Math.ceil(wantRows / WG) * WG);
  for (let start = 0; start < n; start += rowsPerTile) {
    const count = Math.min(rowsPerTile, n - start);
    p.params.write({ n, dim, k, rowOffset: start });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(count / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    // Wait for each tile before queuing the next, so the queue never holds more than one
    // long command and `params` (one uniform, rewritten per tile) cannot be overwritten
    // before the dispatch that reads it has run.
    await device.queue.onSubmittedWorkDone();
  }

  const gotIdx = (await p.outIdx.read()) as ArrayLike<number>;
  const gotDist = (await p.outDist.read()) as ArrayLike<number>;
  const indices = new Uint32Array(n * k);
  const distances = new Float32Array(n * k);
  for (let t = 0; t < n * k; t++) {
    indices[t] = gotIdx[t]!;
    distances[t] = gotDist[t]!;
  }
  return { n, k, indices, distances };
}
