// The QCM's permutation null on the GPU — the GPU twin of `src/spatial/quadratCorrelation.ts`.
//
// ## What is on the GPU, and what deliberately is not
//
// **Only the null.** The OBSERVED `r` and `pc` are still computed by the f64 CPU functions and
// returned unchanged, so the statistic this reports is bit-identical to the CPU path — the one that
// reproduces the published `quadratCounts` and `MH_PC` columns to 1e-9. What moves here is the
// reference distribution: 999 shuffles, each needing a K×Q count, a K×K correlation and a K×K
// inverse. Measured on COVID_SAMPLE_16_ROI_3 (K=49, Q=400, n=29,536, S=999), the CPU loop splits
//
//     rowCorrelation 56% · partialCorrelation 21% · Fisher-Yates 19% · counts 4%
//
// so there is no useful subset: moving the matmul alone caps the speedup at 2.3× by Amdahl. All
// four stages are here. Measured on that ROI: **85 ms against 1,614 ms under node (19×)**, and in the
// browser **668 ms against 42,364 ms at 9,999 shuffles (63×)**.
//
// The point of the speed is not the speed. A permutation p cannot fall below 1/(S+1), so
// Benjamini–Hochberg over m pairs can declare nothing until m/((S+1)·α) of them tie at that floor —
// 113 pairs at S=199, 23 at S=999. On this ROI exactly 23 clear it at 999, which means the discovery
// set sits ON the threshold and is unstable: two runs of the same null disagree on 6 of 1,128 pairs
// from Monte Carlo alone. Raising S is the only fix, and this is what makes raising it free.
//
// ## The shuffle, and why there is no Fisher-Yates
//
// Fisher-Yates is inherently sequential and needs a private n-element scratch per shuffle — 120 kB
// × 999. Instead each shuffle's permutation is a **format-preserving Feistel cipher** over
// [0, n): π(i) is computed independently per thread in O(1), with no scratch and no sort, so the
// scatter becomes one thread per (shuffle, cell). Six rounds over the two halves of the index,
// with cycle-walking to fold the power-of-two domain back onto [0, n) — the walk is what keeps it a
// bijection rather than merely a hash.
//
// This is a *pseudo-random* permutation where the CPU draws a uniform one, which is a real
// difference and is treated as one: `quadratCorrelationGpu.gpu.test.ts` checks that the null's mean
// and standard deviation agree with the CPU label-shuffle null within Monte Carlo error, which is
// the only property the effect size actually consumes.
//
// The counts are `atomic<u32>` rather than floats. Counts ARE integers, so the accumulation is
// exact and order-independent — the one place in this file where f32 would have been a silent
// approximation for no gain.
//
// ## The swap null here is BETTER than SpOOx's, which means it does not match it
//
// A swap chain is sequential, so the device's parallelism has to come from running many chains at
// once — 256 of them, each supplying every 256th draw. SpOOx runs exactly ONE chain, with only 500
// successful swaps between consecutive draws on a 19,600-entry table, so its 1000 nulls are a short
// slice of a single trajectory rather than a sample of the stationary distribution. That
// under-mixing does not merely add noise, it BIASES the null's spread downward and so inflates the
// effect sizes.
//
// Measured on COVID_SAMPLE_16_ROI_3, 1000 draws, median |Δ| against the published `MH_SES`, with
// each path's own run-to-run spread beside it:
//
//     CPU, one chain (SpOOx's shape) .. 0.144   self 0.145   1397 ms
//     GPU, 256 chains ................. 0.168   self 0.060    106 ms
//
// The CPU path lands ON the published column — its disagreement equals its own reproducibility, so
// the two are statistically the same answer. The GPU path is three times more precise and therefore
// visibly different, because it is estimating the quantity SpOOx was approximating rather than the
// approximation. So: **use the CPU swap path to reproduce the published numbers, and this one to get
// the number the published one was trying to be.** SpOOx's chain is cheap in absolute terms
// (~510,000 swaps for all 1000 draws), so parity never needs the GPU.
//
// ## Why the keep-set is recomputed per shuffle
//
// A type with no variance across quadrats cannot enter the inverse. A label shuffle preserves every
// type's abundance exactly, so an ABSENT type is absent in every shuffle — but a present type whose
// observed counts happen to be constant will generally not be constant once shuffled. The CPU
// recomputes the keep-set inside each shuffle, so the kernel does too, off the per-shuffle standard
// deviations. Getting this wrong is not hypothetical: the CPU version shipped with a keep test that
// scanned whole rows, so ONE degenerate type dropped every other type with it.
//
// ## Precision
//
// The correlation is f32 and the inverse of a 49×49 correlation matrix in f32 can lose several
// digits when it is near-singular. That is tolerable here and nowhere else: these values are only
// ever consumed as a mean and a standard deviation over ~1000 draws, and the observed statistic they
// are compared against is the f64 one. Singular shuffles are dropped rather than fudged, exactly as
// the CPU drops non-finite ones.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { LabelledCells } from "../../spatial/pcf";
import {
  benjaminiHochbergMatrix,
  partialCorrelation,
  type QuadratCorrelationParams,
  type QuadratCorrelationResult,
  quadratCounts,
  rowCorrelation,
  SWAP_BETWEEN,
  SWAP_BURN_IN,
} from "../../spatial/quadratCorrelation";
import { getDevice } from "../device";

/** Shuffles resident on the device at once. Bounds the count buffer (B·K·Q u32) and every dispatch,
 *  so no single command approaches the ~2 s the OS watchdog kills silently. */
const BATCH = 64;
/** The swap null uses a much bigger batch, because its parallelism has nowhere else to come from: a
 *  chain is sequential by construction (each move reads what the last wrote), so the only thing to
 *  run in parallel is other chains. One thread per chain at BATCH=64 would leave the device almost
 *  entirely idle. 256 × 49 × 400 × 4 B is 20 MB of counts, which is the price. */
const SWAP_BATCH = 256;
/** Swap moves per dispatch. A chain of millions in ONE command is exactly the shape the OS GPU
 *  watchdog kills at ~2 s, silently — see docs and `gpu-watchdog-silent-zeroes`. Chunking bounds each
 *  command; the chain state lives in the buffer, so splitting it costs nothing but a submit. */
const SWAP_CHUNK = 100_000;
const SCATTER_WG = 256;
const MOMENT_WG = 128;
const PAIR_WG = 64;
const TALLY_WG = 64;
/** The inverse runs in workgroup shared memory: MAX_TYPES² f32 = 12.5 kB at 56, inside the 16 kB
 *  floor every WebGPU device guarantees. Above it the GPU path declines and the caller falls back,
 *  rather than failing pipeline creation on some devices and not others. */
const MAX_TYPES = 56;
/** Written where a correlation is undefined. Real correlations live in [-1, 1] and partial ones in
 *  [-1, 1] too, so any sentinel outside that is unambiguous — and unlike NaN it survives a `<`
 *  comparison in WGSL without depending on how the driver treats unordered compares. */
const UNDEFINED = 1e30;

const SHADER = /* wgsl */ `
struct Uni {
  n: u32, K: u32, Q: u32, batch: u32,
  seedBase: u32, halfBits: u32, halfMask: u32, swapSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> cellQuadrat: array<u32>;
@group(0) @binding(2) var<storage, read> cellLabel: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;  // batch*K*Q
@group(0) @binding(4) var<storage, read_write> moments: array<f32>;         // batch*K*2 (mean, sd)
@group(0) @binding(5) var<storage, read_write> corr: array<f32>;            // batch*K*K
@group(0) @binding(6) var<storage, read_write> part: array<f32>;            // batch*K*K
@group(0) @binding(7) var<storage, read> obs: array<f32>;                   // 2*K*K (r then pc)
@group(0) @binding(8) var<storage, read_write> acc: array<f32>;             // K*K*6

const UNDEF: f32 = ${UNDEFINED};

// ---- the permutation -----------------------------------------------------------------------------

fn mix32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u); v = v * 0x7feb352du;
  v = v ^ (v >> 15u); v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

// One pass of a balanced Feistel network on 2*halfBits bits. Invertible for ANY round function,
// which is what makes the composition a permutation rather than a hash.
fn feistel(idx: u32, key: u32) -> u32 {
  var l = idx & U.halfMask;
  var r = (idx >> U.halfBits) & U.halfMask;
  for (var round = 0u; round < 6u; round = round + 1u) {
    let f = mix32(r ^ key ^ (round * 0x9e3779b9u)) & U.halfMask;
    let nl = r;
    r = l ^ f;
    l = nl;
  }
  return (r << U.halfBits) | l;
}

// Cycle-walking: the Feistel is a bijection on [0, 2^(2*halfBits)), which over-covers [0, n).
// Re-encrypting until the image lands in range maps each out-of-range point along its own cycle,
// and because cycles are closed the result is a bijection ON [0, n) — not merely a rejection
// sample. The domain is under 4n, so the expected number of laps is below 4.
fn permute(i: u32, key: u32) -> u32 {
  var x = i;
  for (var guard = 0u; guard < 64u; guard = guard + 1u) {
    x = feistel(x, key);
    if (x < U.n) { return x; }
  }
  return i;
}

fn keyFor(s: u32) -> u32 { return mix32(U.seedBase + s * 0x9e3779b9u + 1u); }

// ---- 1. scatter: one thread per (shuffle, cell) ---------------------------------------------------

@compute @workgroup_size(${SCATTER_WG})
fn clearCounts(@builtin(global_invocation_id) gid: vec3u) {
  let total = U.batch * U.K * U.Q;
  if (gid.x >= total) { return; }
  atomicStore(&counts[gid.x], 0u);
}

@compute @workgroup_size(${SCATTER_WG})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let s = gid.y;
  // cell i keeps its quadrat and takes the label of cell π(i) — exactly the CPU's label shuffle,
  // read from the other end.
  let lab = cellLabel[permute(i, keyFor(s))];
  atomicAdd(&counts[(s * U.K + lab) * U.Q + cellQuadrat[i]], 1u);
}

// ---- 1b. the swap null: seed each slot from the observed table, then walk a chain ------------------

@compute @workgroup_size(${SCATTER_WG})
fn seedChains(@builtin(global_invocation_id) gid: vec3u) {
  let per = U.K * U.Q;
  let total = U.batch * per;
  if (gid.x >= total) { return; }
  // The observed table rides in the tail of 'obs' rather than a binding of its own: WebGPU
  // guarantees only 8 storage buffers per stage and bindings 1-8 already use all of them, so a
  // ninth makes the LAYOUT invalid — every dispatch then silently does nothing and the only
  // results that survive are the ones the CPU computed. Counts are small integers, so f32 holds
  // them exactly.
  atomicStore(&counts[gid.x], u32(obs[2u * U.K * U.K + (gid.x % per)]));
}

fn nextRand(state: ptr<function, u32>) -> u32 {
  *state = *state * 1664525u + 1013904223u;
  return mix32(*state);
}

// One thread per chain, transcribing SpOOx's 'changeSomeElements': adaptive diagonal, step drawn
// from {1, …, minDiag}, and — the part that changes the arithmetic — 'swapSteps' counts SUCCESSFUL
// moves, not attempts. On a sparse table most attempts land on a 2×2 block with a zero on both
// diagonals and do nothing.
//
// A chain is sequential by nature, so the parallelism comes from running many INDEPENDENT chains at
// once, which is why the swap path uses a much larger batch. Each thread owns its own K*Q slice
// outright, so the atomics here are only the buffer's type, never contention.
@compute @workgroup_size(${PAIR_WG})
fn swapChain(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= U.batch) { return; }
  if (U.K < 2u || U.Q < 2u) { return; }
  let base = s * U.K * U.Q;
  var rng = mix32(U.seedBase ^ (s * 0x9e3779b9u) ^ 0x5bf03635u);
  var done = 0u;
  var attempts = 0u;
  let cap = 200u * U.swapSteps + 1000u;
  loop {
    if (done >= U.swapSteps || attempts >= cap) { break; }
    attempts = attempts + 1u;
    let r0 = nextRand(&rng) % U.K;
    var r1 = nextRand(&rng) % (U.K - 1u);
    if (r1 >= r0) { r1 = r1 + 1u; }
    let c0 = nextRand(&rng) % U.Q;
    var c1 = nextRand(&rng) % (U.Q - 1u);
    if (c1 >= c0) { c1 = c1 + 1u; }
    let iaa = base + r0 * U.Q + c0;
    let ibb = base + r0 * U.Q + c1;
    let icc = base + r1 * U.Q + c0;
    let idd = base + r1 * U.Q + c1;
    let a = atomicLoad(&counts[iaa]);
    let b = atomicLoad(&counts[ibb]);
    let c = atomicLoad(&counts[icc]);
    let d = atomicLoad(&counts[idd]);
    let minDiag1 = min(a, d);
    let minDiag2 = min(b, c);
    if (minDiag1 == 0u && minDiag2 == 0u) { continue; }
    if (minDiag1 > 0u) {
      let step = 1u + nextRand(&rng) % minDiag1;
      atomicStore(&counts[iaa], a - step);
      atomicStore(&counts[idd], d - step);
      atomicStore(&counts[ibb], b + step);
      atomicStore(&counts[icc], c + step);
    } else {
      let step = 1u + nextRand(&rng) % minDiag2;
      atomicStore(&counts[iaa], a + step);
      atomicStore(&counts[idd], d + step);
      atomicStore(&counts[ibb], b - step);
      atomicStore(&counts[icc], c - step);
    }
    done = done + 1u;
  }
}

// ---- 2. per-row mean and centred norm --------------------------------------------------------------

var<workgroup> red: array<f32, ${MOMENT_WG}>;

fn treeSum(lid: u32, v: f32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  var stride = ${MOMENT_WG}u >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) { red[lid] = red[lid] + red[lid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let out = red[0];
  workgroupBarrier();
  return out;
}

@compute @workgroup_size(${MOMENT_WG})
fn rowMoments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let k = wid.x;
  let s = wid.y;
  let base = (s * U.K + k) * U.Q;
  var acc0 = 0.0;
  var j = lid.x;
  loop {
    if (j >= U.Q) { break; }
    acc0 = acc0 + f32(atomicLoad(&counts[base + j]));
    j = j + ${MOMENT_WG}u;
  }
  let total = treeSum(lid.x, acc0);
  let mean = total / f32(U.Q);
  var acc1 = 0.0;
  j = lid.x;
  loop {
    if (j >= U.Q) { break; }
    let dv = f32(atomicLoad(&counts[base + j])) - mean;
    acc1 = acc1 + dv * dv;
    j = j + ${MOMENT_WG}u;
  }
  let ss = treeSum(lid.x, acc1);
  if (lid.x == 0u) {
    moments[(s * U.K + k) * 2u] = mean;
    // The CPU's 'sd' is sqrt(Σd²) WITHOUT the 1/Q — the same unnormalised quantity that cancels in
    // the Pearson ratio. Matching it here keeps the two implementations comparable term by term.
    moments[(s * U.K + k) * 2u + 1u] = sqrt(ss);
  }
}

// ---- 3. the correlation matrix ---------------------------------------------------------------------

// One workgroup per (row a, shuffle); thread b walks the pair (a, b) over all quadrats. Row a is
// re-read by every thread rather than staged in shared memory, because Q is set by the quadrat size
// and is not bounded — a shared tile would cap how fine a grid the caller may ask for.
@compute @workgroup_size(${PAIR_WG})
fn correlate(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let a = wid.x;
  let s = wid.y;
  let b = lid.x;
  if (b >= U.K) { return; }
  let ma = moments[(s * U.K + a) * 2u];
  let sa = moments[(s * U.K + a) * 2u + 1u];
  let mb = moments[(s * U.K + b) * 2u];
  let sb = moments[(s * U.K + b) * 2u + 1u];
  let baseA = (s * U.K + a) * U.Q;
  let baseB = (s * U.K + b) * U.Q;
  var dot = 0.0;
  for (var j = 0u; j < U.Q; j = j + 1u) {
    dot = dot + (f32(atomicLoad(&counts[baseA + j])) - ma) * (f32(atomicLoad(&counts[baseB + j])) - mb);
  }
  let den = sa * sb;
  corr[(s * U.K + a) * U.K + b] = select(UNDEF, dot / den, den > 0.0);
}

// ---- 4. the partial correlation --------------------------------------------------------------------

var<workgroup> inv: array<f32, ${MAX_TYPES * MAX_TYPES}>;
var<workgroup> bad: u32;

// One workgroup per shuffle: invert the correlation matrix in place and turn the precision matrix
// into partial correlations.
//
// **Degenerate types are neutralised, not gathered out.** The obvious version compacts the matrix
// down to the types with variance and loops over that count — but WGSL requires every
// 'workgroupBarrier' to sit in UNIFORM control flow, and a count derived from a buffer read is not
// uniform to the analyser, so every loop and branch around a barrier has to be bounded by 'U.K'
// from the uniform block. Substituting the identity row and column for a variance-free type gives
// the same answer without any of that: the working matrix becomes block-diagonal, '[R_keep, I]',
// whose inverse is '[R_keep⁻¹, I]', so the partial correlations among the kept types are exactly
// what compaction would have produced. Their own entries are then written as undefined on the way
// out, which is a plain conditional store and needs no barrier at all.
@compute @workgroup_size(${PAIR_WG})
fn partialCorr(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let s = wid.x;
  let K = U.K;
  if (lid.x == 0u) { bad = 0u; }

  for (var idx = lid.x; idx < K * K; idx = idx + ${PAIR_WG}u) {
    let i = idx / K;
    let j = idx - i * K;
    let ok = moments[(s * K + i) * 2u + 1u] > 0.0 && moments[(s * K + j) * 2u + 1u] > 0.0;
    let identity = select(0.0, 1.0, i == j);
    inv[idx] = select(identity, corr[s * K * K + idx], ok);
  }
  workgroupBarrier();

  // Gauss-Jordan in place, no pivoting. The identity is never materialised: writing 1 into the
  // pivot before scaling the row, and 0 into the pivot column before eliminating, leaves exactly
  // the identity's entries to be transformed alongside the matrix. Pivoting is unnecessary because
  // a correlation matrix is symmetric positive definite whenever it is invertible at all.
  for (var k = 0u; k < K; k = k + 1u) {
    let p = inv[k * K + k];
    if (lid.x == 0u) {
      // '!(abs(p) >= eps)' rather than '<', so a NaN pivot from an earlier step is also caught.
      if (!(abs(p) >= 1e-9)) { bad = 1u; }
      inv[k * K + k] = 1.0;
    }
    workgroupBarrier();
    for (var j = lid.x; j < K; j = j + ${PAIR_WG}u) { inv[k * K + j] = inv[k * K + j] / p; }
    workgroupBarrier();
    for (var i = 0u; i < K; i = i + 1u) {
      // f = 0 on the pivot row makes its update a no-op, so the barriers below stay unconditional
      // instead of hiding behind an 'if (i != k)'.
      let f = select(0.0, inv[i * K + k], i != k);
      workgroupBarrier();
      if (lid.x == 0u && i != k) { inv[i * K + k] = 0.0; }
      workgroupBarrier();
      for (var j = lid.x; j < K; j = j + ${PAIR_WG}u) { inv[i * K + j] = inv[i * K + j] - f * inv[k * K + j]; }
      workgroupBarrier();
    }
  }
  workgroupBarrier();

  let failed = bad;
  for (var idx = lid.x; idx < K * K; idx = idx + ${PAIR_WG}u) {
    let i = idx / K;
    let j = idx - i * K;
    let ok = moments[(s * K + i) * 2u + 1u] > 0.0 && moments[(s * K + j) * 2u + 1u] > 0.0;
    let den = sqrt(inv[i * K + i] * inv[j * K + j]);
    var v = UNDEF;
    if (failed == 0u && ok) {
      if (i == j) { v = 1.0; } else if (den > 0.0) { v = -inv[idx] / den; }
    }
    part[s * K * K + idx] = v;
  }
}

// ---- 5. accumulate over the batch ------------------------------------------------------------------

// One thread per (a, b), walking this batch's shuffles. No atomics: each cell of the matrix is owned
// by exactly one thread, and batches are separate dispatches.
@compute @workgroup_size(${TALLY_WG})
fn tally(@builtin(global_invocation_id) gid: vec3u) {
  let K = U.K;
  let idx = gid.x;
  if (idx >= K * K) { return; }
  let obsR = obs[idx];
  let obsP = obs[K * K + idx];
  var sR = acc[idx * 6u]; var qR = acc[idx * 6u + 1u]; var eR = acc[idx * 6u + 2u];
  var sP = acc[idx * 6u + 3u]; var qP = acc[idx * 6u + 4u]; var eP = acc[idx * 6u + 5u];
  for (var s = 0u; s < U.batch; s = s + 1u) {
    let v = corr[s * K * K + idx];
    if (v < UNDEF) {
      sR = sR + v; qR = qR + v * v;
      if (obsR < UNDEF && abs(v) >= abs(obsR)) { eR = eR + 1.0; }
    }
    let w = part[s * K * K + idx];
    if (w < UNDEF) {
      sP = sP + w; qP = qP + w * w;
      if (obsP < UNDEF && abs(w) >= abs(obsP)) { eP = eP + 1.0; }
    }
  }
  acc[idx * 6u] = sR; acc[idx * 6u + 1u] = qR; acc[idx * 6u + 2u] = eR;
  acc[idx * 6u + 3u] = sP; acc[idx * 6u + 4u] = qP; acc[idx * 6u + 5u] = eP;
}
`;

type Root = ReturnType<typeof tgpu.initFromDevice>;

interface Ctx {
  device: GPUDevice;
  root: Root;
  layout: GPUBindGroupLayout;
  clearCounts: GPUComputePipeline;
  scatter: GPUComputePipeline;
  seedChains: GPUComputePipeline;
  swapChain: GPUComputePipeline;
  rowMoments: GPUComputePipeline;
  correlate: GPUComputePipeline;
  partialCorr: GPUComputePipeline;
  tally: GPUComputePipeline;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const module = device.createShaderModule({ code: SHADER });
    const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    // One explicit layout shared by all six entry points. `layout: "auto"` derives a different
    // layout per entry point — only the bindings it happens to touch — so no single bind group
    // could satisfy them all.
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        storage(1, "read-only-storage"),
        storage(2, "read-only-storage"),
        storage(3, "storage"),
        storage(4, "storage"),
        storage(5, "storage"),
        storage(6, "storage"),
        storage(7, "read-only-storage"),
        storage(8, "storage"),
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const mk = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    return {
      device,
      root,
      layout,
      clearCounts: mk("clearCounts"),
      scatter: mk("scatter"),
      seedChains: mk("seedChains"),
      swapChain: mk("swapChain"),
      rowMoments: mk("rowMoments"),
      correlate: mk("correlate"),
      partialCorr: mk("partialCorr"),
      tally: mk("tally"),
    };
  })();
  return ctxCache;
}

// Pooled, grow-only, never destroyed — `.destroy()` segfaults Dawn-on-Node's teardown, the same
// constraint every module in this directory works under.
const bufPool = new Map<string, { buf: GPUBuffer; cap: number }>();

function ensureBuf(device: GPUDevice, key: string, bytes: number, usage: number): GPUBuffer {
  const got = bufPool.get(key);
  if (got && got.cap >= bytes) return got.buf;
  const cap = Math.max(bytes, (got?.cap ?? 0) * 2, 256);
  const buf = device.createBuffer({ size: cap, usage });
  bufPool.set(key, { buf, cap });
  return buf;
}

const readbacks = new Map<string, { raw: GPUBuffer; wrap: ReturnType<Root["createBuffer"]>; cap: number }>();

function ensureReadback(device: GPUDevice, root: Root, key: string, floats: number) {
  const got = readbacks.get(key);
  if (got && got.cap >= floats) return got;
  const cap = Math.max(floats, (got?.cap ?? 0) * 2, 4);
  const raw = device.createBuffer({
    size: cap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const made = { raw, wrap: root.createBuffer(d.arrayOf(d.f32, cap), raw), cap };
  readbacks.set(key, made);
  return made;
}

/** True when this problem fits the GPU path. `MAX_TYPES` is the shared-memory bound on the inverse;
 *  everything else scales. Exported so a caller can choose without catching an exception. */
export function quadratCorrelationGpuSupported(nTypes: number): boolean {
  return nTypes <= MAX_TYPES;
}

/**
 * QCM with its permutation null evaluated on the GPU. Same inputs and same result shape as
 * `quadratCorrelation`; the observed `r` and `pc` are the identical f64 values.
 *
 * Falls back to nothing — the caller checks `quadratCorrelationGpuSupported` first, or catches.
 */
export async function quadratCorrelationGpu(cells: LabelledCells, p: QuadratCorrelationParams): Promise<QuadratCorrelationResult> {
  const countsObs = quadratCounts(cells, p);
  const K = countsObs.nTypes;
  const Q = countsObs.cols * countsObs.rows;
  const r = rowCorrelation(countsObs.counts, K, Q);
  const pc = partialCorrelation(r, K);
  const sims = p.simulations ?? 0;
  const empty = () => new Float64Array(0);
  if (sims <= 0) {
    return { nTypes: K, quadrats: Q, r, pc, ses: empty(), p: empty(), q: empty(), pcSes: empty(), pcP: empty(), pcQ: empty(), simulations: 0 };
  }
  if (!quadratCorrelationGpuSupported(K)) {
    throw new Error(`quadratCorrelationGpu: ${K} types exceeds the ${MAX_TYPES} the in-workgroup inverse allows`);
  }

  const n = cells.xs.length;
  const [minX, minY] = p.bbox;
  const cellQuadrat = new Uint32Array(n);
  const cellLabel = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(countsObs.cols - 1, Math.max(0, Math.floor((cells.xs[i]! - minX) / p.quadratSize)));
    const cy = Math.min(countsObs.rows - 1, Math.max(0, Math.floor((cells.ys[i]! - minY) / p.quadratSize)));
    cellQuadrat[i] = cy * countsObs.cols + cx;
    cellLabel[i] = cells.typeId[i]!;
  }

  const { device, root, layout, ...pipes } = await getCtx();
  // Feistel domain: the smallest balanced power of two that covers n. Cycle-walking folds it back,
  // so the only cost of over-covering is the expected number of laps.
  const bits = Math.max(2, Math.ceil(Math.log2(Math.max(n, 2))));
  const halfBits = Math.ceil(bits / 2);
  const halfMask = (1 << halfBits) - 1;

  const swap = (p.nullModel ?? "label") === "swap";
  const burnIn = p.swapBurnIn ?? SWAP_BURN_IN;
  const between = p.swapBetween ?? SWAP_BETWEEN;
  const batchCap = swap ? SWAP_BATCH : BATCH;

  const quadBuf = ensureBuf(device, "cellQuadrat", n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const labBuf = ensureBuf(device, "cellLabel", n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const countBuf = ensureBuf(device, "counts", batchCap * K * Q * 4, GPUBufferUsage.STORAGE);
  const momBuf = ensureBuf(device, "moments", batchCap * K * 2 * 4, GPUBufferUsage.STORAGE);
  const corrBuf = ensureBuf(device, "corr", batchCap * K * K * 4, GPUBufferUsage.STORAGE);
  const partBuf = ensureBuf(device, "part", batchCap * K * K * 4, GPUBufferUsage.STORAGE);
  // [ r | pc | observed counts ] in one buffer — see `seedChains` for why the counts are not their
  // own binding.
  const obsBuf = ensureBuf(device, "obs", (2 * K * K + K * Q) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const uniBuf = ensureBuf(device, "uni", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const accRb = ensureReadback(device, root, "acc", K * K * 6);

  // The swap chain starts from the observed table, so it needs it on the device as integers.

  device.queue.writeBuffer(quadBuf, 0, cellQuadrat);
  device.queue.writeBuffer(labBuf, 0, cellLabel);
  // The observed matrices go up as f32 only to be compared for extremity; the returned values stay
  // f64. A pair whose observed statistic is undefined is sent as the sentinel, so the kernel counts
  // no exceedances for it and the host reports NaN.
  const obsF32 = new Float32Array(2 * K * K + K * Q);
  for (let i = 0; i < K * K; i++) {
    obsF32[i] = Number.isFinite(r[i]!) ? r[i]! : UNDEFINED;
    obsF32[K * K + i] = Number.isFinite(pc[i]!) ? pc[i]! : UNDEFINED;
  }
  for (let i = 0; i < K * Q; i++) obsF32[2 * K * K + i] = countsObs.counts[i]!;
  device.queue.writeBuffer(obsBuf, 0, obsF32);
  device.queue.writeBuffer(accRb.raw, 0, new Float32Array(K * K * 6));

  const bind = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: { buffer: quadBuf } },
      { binding: 2, resource: { buffer: labBuf } },
      { binding: 3, resource: { buffer: countBuf } },
      { binding: 4, resource: { buffer: momBuf } },
      { binding: 5, resource: { buffer: corrBuf } },
      { binding: 6, resource: { buffer: partBuf } },
      { binding: 7, resource: { buffer: obsBuf } },
      { binding: 8, resource: { buffer: accRb.raw } },
    ],
  });

  const seed = (p.seed ?? 0x9ced) >>> 0;
  for (let done = 0; done < sims; done += batchCap) {
    const batch = Math.min(batchCap, sims - done);
    // Uniforms are rewritten per batch and the batch is submitted before the next write — the
    // writeBuffer-before-submit ordering that a single fused command buffer would get wrong.
    device.queue.writeBuffer(uniBuf, 0, new Uint32Array([n, K, Q, batch, (seed + done) >>> 0, halfBits, halfMask, burnIn]));
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setBindGroup(0, bind);
    if (swap) {
      // Seed ONCE. Each slot is its own continuous chain: it is burnt in on the first batch and only
      // ADVANCED thereafter, so slot j supplies draws j, j+B, j+2B, … spaced `swapBetween` apart —
      // the same autocorrelation structure SpOOx's single chain has, spread over B chains so the
      // device has something to do. B independent chains also mix better than one, which makes this
      // estimate more precise than the published column rather than merely different from it.
      if (done === 0) {
        pass.setPipeline(pipes.seedChains);
        pass.dispatchWorkgroups(Math.ceil((batch * K * Q) / SCATTER_WG));
      }
    } else {
      pass.setPipeline(pipes.clearCounts);
      pass.dispatchWorkgroups(Math.ceil((batch * K * Q) / SCATTER_WG));
      pass.setPipeline(pipes.scatter);
      pass.dispatchWorkgroups(Math.ceil(n / SCATTER_WG), batch);
    }
    // The chain is walked in CHUNKS, each its own dispatch, because a long chain is the one thing
    // here that can run past the ~2 s OS GPU watchdog — which kills the dispatch with no error and
    // leaves the buffer holding a partially-mixed table that still looks like a valid null. Each
    // chunk continues from what the last wrote (the state is the buffer, not the thread), so the
    // only thing that must not repeat is the randomness: `seedBase` is advanced per chunk, or every
    // chunk would replay the same moves and the chain would go nowhere.
    if (swap) {
      const steps = done === 0 ? burnIn : between;
      for (let step = 0; step < steps; step += SWAP_CHUNK) {
        const chunk = Math.min(SWAP_CHUNK, steps - step);
        pass.end();
        device.queue.submit([enc.finish()]);
        device.queue.writeBuffer(
          uniBuf,
          0,
          new Uint32Array([n, K, Q, batch, (seed + done + 0x9e3779b9 * (1 + step / SWAP_CHUNK)) >>> 0, halfBits, halfMask, chunk]),
        );
        const cenc = device.createCommandEncoder();
        const cpass = cenc.beginComputePass();
        cpass.setBindGroup(0, bind);
        cpass.setPipeline(pipes.swapChain);
        cpass.dispatchWorkgroups(Math.ceil(batch / PAIR_WG));
        cpass.end();
        device.queue.submit([cenc.finish()]);
        await device.queue.onSubmittedWorkDone();
        enc = device.createCommandEncoder();
        pass = enc.beginComputePass();
        pass.setBindGroup(0, bind);
      }
    }
    pass.setPipeline(pipes.rowMoments);
    pass.dispatchWorkgroups(K, batch);
    pass.setPipeline(pipes.correlate);
    pass.dispatchWorkgroups(K, batch);
    pass.setPipeline(pipes.partialCorr);
    pass.dispatchWorkgroups(batch);
    pass.setPipeline(pipes.tally);
    pass.dispatchWorkgroups(Math.ceil((K * K) / TALLY_WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const accRaw = await accRb.wrap.read();
  return finish(accRaw as unknown as number[], r, pc, K, Q, sims);
}

/** Turn the six accumulators into effect sizes, p-values and BH q-values — the same arithmetic the
 *  CPU does after its loop, in f64, so the two paths differ only in how the null was sampled. */
function finish(acc: readonly number[], r: Float64Array, pc: Float64Array, K: number, Q: number, sims: number): QuadratCorrelationResult {
  const mk = () => new Float64Array(K * K);
  const ses = mk();
  const pv = mk();
  const pcSes = mk();
  const pcP = mk();
  for (let i = 0; i < K * K; i++) {
    const stat = (off: number, obs: number) => {
      const sum = acc[i * 6 + off]!;
      const sumSq = acc[i * 6 + off + 1]!;
      const ext = acc[i * 6 + off + 2]!;
      const mu = sum / sims;
      const sd = Math.sqrt(Math.max(0, sumSq / sims - mu * mu));
      return {
        ses: Number.isFinite(obs) && sd > 0 ? (obs - mu) / sd : Number.NaN,
        p: Number.isFinite(obs) ? (ext + 1) / (sims + 1) : Number.NaN,
      };
    };
    const a = stat(0, r[i]!);
    const b = stat(3, pc[i]!);
    ses[i] = a.ses;
    pv[i] = a.p;
    pcSes[i] = b.ses;
    pcP[i] = b.p;
  }
  return {
    nTypes: K,
    quadrats: Q,
    r,
    pc,
    ses,
    p: pv,
    q: benjaminiHochbergMatrix(pv, K),
    pcSes,
    pcP,
    pcQ: benjaminiHochbergMatrix(pcP, K),
    simulations: sims,
  };
}
