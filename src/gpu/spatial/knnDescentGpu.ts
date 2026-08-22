// NN-descent's local join on the GPU — the piece that decides whether this project can
// embed a whole tissue section rather than a subsample.
//
// The exact search is O(N²·D) and no device fixes that: `knnGpu` is fast, but at 200k
// cells it is still a minute and a half of quadratic work. NN-descent replaces it with a
// fixed number of O(N·c²·D) passes, where c is the candidate-list width (~21) rather than
// N. That is the asymptotic change; moving the pass to the GPU is what makes the constant
// small enough to matter.
//
// **Why this parallelises cleanly, which is not obvious for NN-descent.** The textbook
// local join updates BOTH endpoints of every pair it evaluates, which on a GPU means many
// threads writing the same point's neighbour list and needing locks or atomics. The host
// implementation here (`knnDescentCpu`) already does not do that: thread `i` walks its own
// candidates, then its candidates' candidates, and offers them **only to `i`'s own list**.
// Each thread therefore owns exactly one heap and touches no other. No atomics, no locks,
// no races — and the GPU result can be compared against the host's directly rather than
// only statistically, which is the opposite of the situation in `umapLayoutGpu`.
//
// The cost of that choice is that a discovered pair improves one side rather than two, so
// convergence takes marginally more passes than a both-ends variant would. That is a good
// trade at this width: passes are cheap and correctness arguments are not.
//
// **What stays on the host.** Initialisation (random-projection seeding) and the candidate
// build, both O(N·k) and neither a good fit for a device — the reverse-neighbour pass is a
// counting sort with reservoir sampling. They are also exactly the parts where being
// precisely right matters and being fast does not. The heap and candidate arrays are flat
// typed arrays for this reason: they upload verbatim.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { buildCandidates, type DescentOptions, finalise, initialiseHeap } from "../../spatial/knnDescent";
import type { KnnResult } from "../../spatial/umapGraph";
import { mulberry32 } from "../../spatial/umapLayout";
import { getDevice, writeView } from "../device";

const WG = 64;
/** Compile-time bound on the private heap, matching `knn.ts`. */
const MAX_K = 32;

const Params = d.struct({ n: d.u32, dim: d.u32, k: d.u32, width: d.u32, rowOffset: d.u32, _pad0: d.u32, _pad1: d.u32, _pad2: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  data: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // [n, dim]
  candidates: { storage: (n: number) => d.arrayOf(d.i32, n), access: "readonly" }, // [n, width]
  counts: { storage: (n: number) => d.arrayOf(d.i32, n), access: "readonly" }, // [n]
  heapIdx: { storage: (n: number) => d.arrayOf(d.i32, n), access: "mutable" }, // [n, k]
  heapDist: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // [n, k]
  changed: { storage: (n: number) => d.arrayOf(d.u32, n), access: "mutable" }, // [n]
});

// WGSL template rather than `"use gpu"`, same reason as `knn.ts` (ADR-0003): the k-smallest
// selection needs a local mutable array, which TGSL cannot express.
const TEMPLATE = /* wgsl */ `
@compute @workgroup_size(${WG})
fn localJoin(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + params.rowOffset;
  let n = params.n;
  if (i >= n) { return; }
  let dim = params.dim;
  let k = min(params.k, ${MAX_K}u);
  let width = params.width;

  // This thread's heap, in registers for the duration of the pass. Nothing else writes it.
  var hi: array<i32, ${MAX_K}u>;
  var hd: array<f32, ${MAX_K}u>;
  let hbase = i * k;
  for (var t: u32 = 0u; t < k; t = t + 1u) {
    hi[t] = heapIdx[hbase + t];
    hd[t] = heapDist[hbase + t];
  }
  var worst: f32 = hd[k - 1u];
  var changes: u32 = 0u;

  let ibase = i * dim;
  let cn = counts[i];
  for (var a: i32 = 0; a < cn; a = a + 1) {
    let p = candidates[i * width + u32(a)];
    if (p < 0) { continue; }
    // The local join: everything p knows about is a candidate for i.
    let pn = counts[u32(p)];
    for (var b: i32 = 0; b < pn; b = b + 1) {
      let q = candidates[u32(p) * width + u32(b)];
      if (q < 0 || u32(q) == i) { continue; }

      let qbase = u32(q) * dim;
      var acc: f32 = 0.0;
      for (var c: u32 = 0u; c < dim; c = c + 1u) {
        let delta = data[ibase + c] - data[qbase + c];
        acc = acc + delta * delta;
      }
      // Compare in SQUARED space; the root is taken once, on write-back.
      if (acc >= worst) { continue; }

      // Reject duplicates. Without this a point can fill its whole list with copies of one
      // good neighbour found along several paths, and the descent then stops improving
      // while still reporting changes.
      var dup = false;
      for (var t: u32 = 0u; t < k; t = t + 1u) {
        if (hi[t] == q) { dup = true; break; }
      }
      if (dup) { continue; }

      var s: u32 = k - 1u;
      loop {
        if (s == 0u) { break; }
        if (hd[s - 1u] <= acc) { break; }
        hd[s] = hd[s - 1u];
        hi[s] = hi[s - 1u];
        s = s - 1u;
      }
      hd[s] = acc;
      hi[s] = q;
      worst = hd[k - 1u];
      changes = changes + 1u;
    }
  }

  for (var t: u32 = 0u; t < k; t = t + 1u) {
    heapIdx[hbase + t] = hi[t];
    heapDist[hbase + t] = hd[t];
  }
  changed[i] = changes;
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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({ template: TEMPLATE, externals: { ...layout.bound }, names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "localJoin" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makeBuffers(root: Root, values: number, cands: number, rows: number, slots: number) {
  return {
    values,
    cands,
    rows,
    slots,
    data: root.createBuffer(d.arrayOf(d.f32, values)).$usage("storage"),
    candidates: root.createBuffer(d.arrayOf(d.i32, cands)).$usage("storage"),
    counts: root.createBuffer(d.arrayOf(d.i32, rows)).$usage("storage"),
    heapIdx: root.createBuffer(d.arrayOf(d.i32, slots)).$usage("storage"),
    heapDist: root.createBuffer(d.arrayOf(d.f32, slots)).$usage("storage"),
    changed: root.createBuffer(d.arrayOf(d.u32, rows)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}

/** Work per dispatch, in (row x candidate-pair x dimension) products.
 *
 *  Same watchdog reasoning as `knn.ts`: a dispatch that runs past roughly two seconds is
 *  killed with no error and the output silently keeps whatever was there. One pass is
 *  `n * width^2 * dim`, which stays under this at every size tried (2.2e9 at n=100k,
 *  width=21, dim=51), so the tiling below is usually a single dispatch — but it is not
 *  free to assume that, since `width` grows with k. Tiling is over ROWS, which is safe
 *  here in a way it was not for the exact search: threads are independent within a pass,
 *  so no state is carried between tiles. */
const TARGET_PRODUCTS_PER_DISPATCH = 2_000_000_000;

/** How often the early-exit test costs a device read. See the note in the pass loop —
 *  in a browser a buffer map dominates the pass it follows, so convergence is sampled
 *  rather than polled. */
const CHECK_EVERY = 4;

export interface DescentGpuOptions extends DescentOptions {
  /** Rows per dispatch, overriding the budget. A test seam, as in `knn.ts`. */
  readonly rowsPerTile?: number;
}

/**
 * Approximate k-NN by NN-descent, with the local join on the GPU.
 *
 * Same contract as `knnDescentCpu` and `knnGpu`: row-major `[n, dim]` in, `KnnResult` out
 * with self excluded and each row ascending — except approximate, so judge it by recall.
 */
export async function knnDescentGpu(data: ArrayLike<number>, n: number, dim: number, opts: DescentGpuOptions): Promise<KnnResult> {
  const { k } = opts;
  if (k >= n) throw new Error(`knnDescentGpu: need k < n (k=${k}, n=${n})`);
  if (k < 1 || k > MAX_K) throw new Error(`knnDescentGpu: k must be in 1..${MAX_K}`);
  const maxIters = opts.maxIters ?? 12;
  const tol = opts.tol ?? 0.001;
  const rnd = mulberry32((opts.seed ?? 42) ^ 0x5bf03635);

  // Seeding and the candidate build stay on the host — see the module header.
  const heap = initialiseHeap(data, n, dim, opts);
  const maxReverse = Math.max(4, Math.floor(k / 2));

  const { device, root, pipeline } = await getPipe();
  const flat = data instanceof Float32Array && data.length === n * dim ? data : Float32Array.from({ length: n * dim }, (_, t) => data[t]!);
  const width = k + maxReverse;
  const buf = makeBuffers(root, n * dim, n * width, n, n * k);
  const bind = root.unwrap(root.createBindGroup(layout, buf));
  writeView(device.queue, root.unwrap(buf.data), flat);

  // The heap lives on the device across passes; only the candidate lists go up each pass
  // and only the indices come back at the end. Squared distances on the device, rooted on
  // the way out — the host heap arrives with real distances, so square them going in.
  const squared = Float32Array.from(heap.distances, (v) => (Number.isFinite(v) ? v * v : 3.4e38));
  writeView(device.queue, root.unwrap(buf.heapIdx), heap.indices);
  writeView(device.queue, root.unwrap(buf.heapDist), squared);

  const rowsPerTile = Math.max(
    WG,
    Math.min(n, opts.rowsPerTile ?? Math.floor(TARGET_PRODUCTS_PER_DISPATCH / Math.max(width * width * dim, 1))),
  );

  const hostIdx = heap.indices;
  const hostDist = heap.distances;
  for (let iter = 0; iter < maxIters; iter++) {
    const built = buildCandidates({ n, k, indices: hostIdx, distances: hostDist }, maxReverse, rnd);
    writeView(device.queue, root.unwrap(buf.candidates), built.candidates);
    writeView(device.queue, root.unwrap(buf.counts), built.counts);

    for (let start = 0; start < n; start += rowsPerTile) {
      const count = Math.min(rowsPerTile, n - start);
      buf.params.write({ n, dim, k, width, rowOffset: start, _pad0: 0, _pad1: 0, _pad2: 0 });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / WG));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    // **Read back as little as possible, as rarely as possible.** Each `read()` is a
    // device-to-host map, and in a browser that is a round trip costing far more than the
    // kernel it follows — the first version read three buffers every pass and a 32k-cell
    // build took 11.3 s in Chrome against ~1 s in Node, where maps are nearly free. Two of
    // the three were pure waste:
    //
    //   • The DISTANCES are never used between passes. `buildCandidates` reads indices
    //     only, so they stay on the device until the end.
    //   • The CHANGE counts only decide early exit, so they are sampled every few passes
    //     rather than every one. The cost of noticing convergence a little late is at most
    //     `CHECK_EVERY - 1` extra passes; the cost of asking every time is a round trip
    //     per pass.
    const gotIdx = (await buf.heapIdx.read()) as ArrayLike<number>;
    for (let t = 0; t < n * k; t++) hostIdx[t] = gotIdx[t]!;

    if (iter % CHECK_EVERY === CHECK_EVERY - 1) {
      const changes = (await buf.changed.read()) as ArrayLike<number>;
      let total = 0;
      for (let i = 0; i < n; i++) total += changes[i]!;
      if (total <= tol * n * k * CHECK_EVERY) break;
    }
  }

  const gotDist = (await buf.heapDist.read()) as ArrayLike<number>;
  for (let t = 0; t < n * k; t++) hostDist[t] = Math.sqrt(gotDist[t]!);
  return finalise({ n, k, indices: hostIdx, distances: hostDist });
}
