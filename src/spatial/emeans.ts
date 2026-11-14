// k-means clustering by **e-means** (Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze & Jégou
// 2026, arXiv:2611.09501): mini-batch k-means whose centroid update is an exponential moving
// average, with the batch size and step chosen by fitted scaling laws. The CPU golden for a
// future GPU port, and a usable host implementation in its own right.
//
// Every centroid is a ratio c_j = S_j / N_j of a running sum and count. Lloyd replaces both
// with the full-dataset statistics each pass; e-means blends in one mini-batch at a time:
//
//     S_j ← (1−α)·S_j + α·s_j        N_j ← (1−α)·N_j + α·n_j
//
// where s_j, n_j are the sum and count of the batch points assigned to j. Lloyd is the corner
// B = n, α = 1 of that family (`emeans(..., { batch: n, alpha0: 1, schedule: "constant",
// revive: 0 })` reproduces it), and the authors' hyperparameter sweep puts that corner well
// outside the low-loss band. The rest of the method:
//
//   - **Scaling laws** pick B and α₀ from (n, k, T) — no tuning beyond Lloyd's own inputs.
//     `emeansSchedule` exposes them so a caller (or the GPU port) can read the plan.
//   - **Cosine schedule**: α anneals from α₀ to 0.1·α₀ over the run, like a learning rate.
//   - **Dead-cluster revival** every 5 steps: a cluster whose EMA count falls below
//     `revive × mean count` is replaced by splitting a live cluster drawn with probability
//     proportional to its count (Gumbel top-k), both halves nudged ±1e-4 multiplicatively.
//   - **Finalisation**: the last epoch covers every point once and accumulates the exact
//     per-cluster sum and count of its assignments; each centroid ends on that mean.
//
// **Why it is here.** Not mainly for speed at our usual sizes. The authors report ~10× fewer
// epochs than Lloyd for the same loss at large k (thousands to 65k clusters), but the result
// that matters for spatial biology is at *small* k: on toy mixtures with uneven blob sizes and
// k = 5–10, Lloyd stays in a bad local minimum on 15–46% of seeds and e-means on 0–4%, because
// noisy mini-batch updates escape configurations Lloyd's monotone descent cannot. Cellular-
// neighbourhood ("microenvironment") clustering — k ≈ 5–20 on per-cell composition vectors —
// is exactly that regime. The mini-batch form also suits data that is streamed rather than
// resident (pixel clustering over tiled multiplexed imagery). See `docs/emeans-clustering.md`.
//
// **Provenance.** Ported from the authors' MIT-licensed reference JavaScript
// (`playground.html` in github.com/e-means/e-means.github.io, Copyright (c) Meta Platforms,
// Inc. and affiliates); the library implementation is `faiss/contrib/e_means`. Constants are
// theirs. Differences from the playground: f64 throughout (no f32/bf16 emulation), and the
// returned labels and loss come from one extra full assignment against the final centroids,
// so they are consistent with them (faiss returns last-epoch labels instead).
//
// **Caveat from the authors:** on very low-dimensional data (their 2-D toys) revival *hurts*,
// and their playground runs with it off. So the default here is off for d ≤ 3 — clustering
// raw spatial coordinates — and 0.1 otherwise. That threshold is our reading of their note,
// not a measured cut-off.

import { type Rng, mulberry32 } from "#src/evo/rng";

/** Which fitted law sets the batch size. `"quality"` is 1/8 of `"default"`: lower loss, more
 *  (smaller) steps. On CPU the cost of an epoch barely depends on B, so quality is the
 *  default here; a GPU wants the larger batch to fill the device. */
export type EmeansBatchLaw = "default" | "quality";

export interface EmeansSchedule {
  /** Mini-batch size. */
  batch: number;
  /** Peak EMA step α₀ (the schedule anneals to 0.1·α₀). */
  alpha0: number;
  /** Full batches per epoch, floor(n / B); the tail is visited in the final epoch only. */
  stepsPerEpoch: number;
}

export interface EmeansOptions {
  /** Epochs T (default 5, as in faiss). The last one is the finalisation pass. */
  epochs?: number;
  /** Seed for initialisation, shuffles and revival (default 0). */
  seed?: number;
  /** Override the scaling-law batch size. */
  batch?: number;
  /** Override the scaling-law peak step α₀ (clamped to (0, 1]). */
  alpha0?: number;
  /** Batch law when `batch` is not given (default `"quality"`). */
  batchLaw?: EmeansBatchLaw;
  /** `"cosine"` (default) anneals α₀ → 0.1·α₀; `"constant"` holds α₀. */
  schedule?: "cosine" | "constant";
  /** Revival threshold as a fraction of the mean EMA count; 0 disables. Default 0 when
   *  d ≤ 3, else 0.1. */
  revive?: number;
  /** Optional initial centroids (k × d, row-major) instead of k distinct random points. */
  init?: ArrayLike<number>;
}

export interface KMeansResult {
  /** k × d, row-major. */
  centroids: Float64Array;
  /** Nearest-centroid index of every point against the final centroids. */
  labels: Int32Array;
  /** Points per cluster under `labels`. */
  counts: Int32Array;
  /** Mean squared distance of each point to its centroid (the k-means objective / n). */
  mse: number;
  /** Clusters re-seeded by splitting during training. */
  splits: number;
}

export interface EmeansResult extends KMeansResult {
  schedule: EmeansSchedule;
}

/** Initial EMA count of a seeded centroid: tiny, so the first batch that reaches a cluster
 *  essentially replaces its seed by that batch's mean. */
const INIT_USAGE = 2 ** -16;
const EMA_EPS = 1e-5;
const SPLIT_EPS = 1e-4;
const ALPHA_MIN_FRAC = 0.1;
const REVIVE_EVERY = 5;
/** `"default"` batch law ÷ `"quality"` law: the authors' playground runs at "B/8, where B is
 *  the batch given by the default scaling law", and their table gives 0.035 vs 0.0044. */
const DEFAULT_OVER_QUALITY = 8;

/** The scaling-law batch size, rounded to the nearer power of two (in ratio) and capped at
 *  n/2: B = 0.00439·k^0.135·n^0.876·T^0.813 for the quality law. */
export function emeansBatch(n: number, k: number, epochs: number, law: EmeansBatchLaw = "quality"): number {
  const scale = law === "default" ? DEFAULT_OVER_QUALITY : 1;
  const b = scale * 0.00439175 * k ** 0.134749 * n ** 0.876473 * epochs ** 0.813344;
  const lo = 2 ** Math.floor(Math.log2(Math.max(1, b)));
  const hi = lo * 2;
  const rounded = b / lo < hi / b ? lo : hi;
  return Math.max(1, Math.min(rounded, Math.floor(n / 2)));
}

/** The scaling-law peak step: α₀ = 0.824·k^−0.462·B^0.900·n^−0.450·T^−0.106, in [1e-4, 1]. */
export function emeansAlpha0(batch: number, n: number, k: number, epochs: number): number {
  const a = 0.8244 * k ** -0.4617 * batch ** 0.9003 * n ** -0.4498 * epochs ** -0.1057;
  return Math.min(Math.max(a, 1e-4), 1);
}

/** The plan e-means will run for an (n, k, T) problem. */
export function emeansSchedule(
  n: number,
  k: number,
  epochs = 5,
  opts: Pick<EmeansOptions, "batch" | "alpha0" | "batchLaw"> = {},
): EmeansSchedule {
  const batch = Math.max(1, Math.min(n, Math.floor(opts.batch ?? emeansBatch(n, k, epochs, opts.batchLaw))));
  const alpha0 = opts.alpha0 !== undefined ? Math.min(Math.max(opts.alpha0, Number.MIN_VALUE), 1) : emeansAlpha0(batch, n, k, epochs);
  return { batch, alpha0, stepsPerEpoch: Math.max(1, Math.floor(n / batch)) };
}

/** Index of the nearest centroid to row `xi` (ties → lowest index), and the squared distance. */
function nearest(data: ArrayLike<number>, xi: number, d: number, C: Float64Array, k: number): [number, number] {
  const xOff = xi * d;
  let best = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let j = 0; j < k; j++) {
    const cOff = j * d;
    let s = 0;
    for (let t = 0; t < d; t++) {
      const diff = data[xOff + t]! - C[cOff + t]!;
      s += diff * diff;
      if (s >= bestDist) break;
    }
    if (s < bestDist) {
      bestDist = s;
      best = j;
    }
  }
  return [best, bestDist];
}

/** Assign every point to its nearest centroid. Writes `labels`, returns the MSE. */
export function assignNearest(
  data: ArrayLike<number>,
  n: number,
  d: number,
  centroids: Float64Array,
  k: number,
  labels: Int32Array,
): number {
  let total = 0;
  for (let i = 0; i < n; i++) {
    const [j, dist] = nearest(data, i, d, centroids, k);
    labels[i] = j;
    total += dist;
  }
  return n > 0 ? total / n : 0;
}

function shuffled(rng: Rng, n: number): Int32Array {
  const p = new Int32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = p[i]!;
    p[i] = p[j]!;
    p[j] = tmp;
  }
  return p;
}

/** k distinct points as the starting centroids (partial Fisher–Yates). */
function seedCentroids(data: ArrayLike<number>, n: number, d: number, k: number, rng: Rng, init?: ArrayLike<number>): Float64Array {
  const C = new Float64Array(k * d);
  if (init) {
    if (init.length !== k * d) throw new Error(`k-means init has ${init.length} values; expected k·d = ${k * d}`);
    for (let i = 0; i < k * d; i++) C[i] = init[i]!;
    return C;
  }
  const idx = shuffled(rng, n);
  for (let j = 0; j < k; j++) for (let t = 0; t < d; t++) C[j * d + t] = data[idx[j]! * d + t]!;
  return C;
}

function validate(n: number, d: number, k: number, dataLength: number): void {
  if (!Number.isInteger(k) || k < 1) throw new Error(`k-means needs an integer k ≥ 1 (got ${k})`);
  if (n < k) throw new Error(`k-means needs at least k = ${k} points (got ${n})`);
  if (!Number.isInteger(d) || d < 1) throw new Error(`k-means needs dimension d ≥ 1 (got ${d})`);
  if (dataLength < n * d) throw new Error(`k-means data has ${dataLength} values; expected n·d = ${n * d}`);
}

/** Re-seed the clusters with `N[j] < thr` by splitting live clusters drawn ∝ N (Gumbel
 *  top-k, without replacement). Both halves take N/2 and are nudged ±ε, alternating sign by
 *  coordinate. Returns how many were re-seeded. */
function splitInto(C: Float64Array, N: Float64Array, k: number, d: number, thr: number, rng: Rng): number {
  const dead: number[] = [];
  const keys: [number, number][] = [];
  for (let j = 0; j < k; j++) {
    const Nj = N[j]!;
    if (Nj < thr) dead.push(j);
    else if (Nj > 0) keys.push([Math.log(Nj) - Math.log(-Math.log(Math.max(rng(), 1e-300))), j]);
  }
  const take = Math.min(dead.length, keys.length);
  if (take === 0) return 0;
  keys.sort((p, q) => q[0] - p[0]);
  for (let m = 0; m < take; m++) {
    const src = keys[m]![1];
    const dst = dead[m]!;
    const half = N[src]! / 2;
    for (let t = 0; t < d; t++) {
      const v = C[src * d + t]!;
      const s = t % 2 === 0 ? 1 : -1;
      C[src * d + t] = v * (1 - SPLIT_EPS * s);
      C[dst * d + t] = v * (1 + SPLIT_EPS * s);
    }
    N[src] = half;
    N[dst] = half;
  }
  return take;
}

/**
 * e-means k-means on `n` points of dimension `d` (row-major `data`, length ≥ n·d).
 *
 * Cost: T epochs of n·k·d for assignment (T = 5 by default) plus one final full assignment,
 * plus O(k·d) per step for the EMA blend. Deterministic for a given seed.
 */
export function emeans(data: ArrayLike<number>, n: number, d: number, k: number, opts: EmeansOptions = {}): EmeansResult {
  validate(n, d, k, data.length);
  const epochs = Math.max(1, Math.floor(opts.epochs ?? 5));
  const schedule = emeansSchedule(n, k, epochs, opts);
  const { batch, alpha0, stepsPerEpoch } = schedule;
  const reviveRatio = opts.revive ?? (d <= 3 ? 0 : 0.1);
  const cosine = (opts.schedule ?? "cosine") === "cosine";
  const rng = mulberry32(opts.seed ?? 0);

  const C = seedCentroids(data, n, d, k, rng, opts.init);
  const N = new Float64Array(k).fill(INIT_USAGE);
  const cnt = new Float64Array(k);
  const sum = new Float64Array(k * d);
  const capCnt = new Float64Array(k);
  const capSum = new Float64Array(k * d);
  const scheduleSteps = epochs * stepsPerEpoch;
  let step = 0;
  let untilRevive = REVIVE_EVERY;
  let splits = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const finalEpoch = epoch === epochs - 1;
    const perm = shuffled(rng, n);
    // The final epoch also visits the tail that floor(n / B) full batches leave out.
    const steps = finalEpoch ? Math.ceil(n / batch) : stepsPerEpoch;
    for (let s = 0; s < steps; s++) {
      const off = s * batch;
      const len = Math.min(batch, n - off);
      cnt.fill(0);
      sum.fill(0);
      for (let b = 0; b < len; b++) {
        const xi = perm[off + b]!;
        const [j] = nearest(data, xi, d, C, k);
        cnt[j]! += 1;
        const xOff = xi * d;
        for (let t = 0; t < d; t++) sum[j * d + t]! += data[xOff + t]!;
      }
      if (finalEpoch) {
        for (let j = 0; j < k; j++) capCnt[j]! += cnt[j]!;
        for (let i = 0; i < k * d; i++) capSum[i]! += sum[i]!;
      }
      // EMA blend in S/N form: S_j = c_j·N_j, so c_j ← ((1−α)·c_j·N_j + α·s_j) / N_j'.
      const frac = Math.min(step, scheduleSteps) / Math.max(1, scheduleSteps);
      const alpha = cosine ? alpha0 * (ALPHA_MIN_FRAC + (1 - ALPHA_MIN_FRAC) * 0.5 * (1 + Math.cos(Math.PI * frac))) : alpha0;
      const decay = 1 - alpha;
      for (let j = 0; j < k; j++) {
        const kept = N[j]! * decay;
        const cj = cnt[j]!;
        if (cj === 0) {
          N[j] = kept;
          continue;
        }
        const next = kept + alpha * cj;
        const inv = 1 / Math.max(next, EMA_EPS);
        for (let t = 0; t < d; t++) C[j * d + t] = (C[j * d + t]! * kept + alpha * sum[j * d + t]!) * inv;
        N[j] = next;
      }
      if (--untilRevive === 0) {
        untilRevive = REVIVE_EVERY;
        if (!finalEpoch && reviveRatio > 0) {
          let total = 0;
          for (let j = 0; j < k; j++) total += N[j]!;
          splits += splitInto(C, N, k, d, (reviveRatio * total) / k, rng);
        }
      }
      step++;
    }
  }
  // Finalise: every cluster seen in the last epoch ends on the exact mean of its assignments.
  for (let j = 0; j < k; j++) {
    const cj = capCnt[j]!;
    if (cj > 0) for (let t = 0; t < d; t++) C[j * d + t] = capSum[j * d + t]! / cj;
  }
  return { ...finish(data, n, d, k, C, splits), schedule };
}

export interface LloydOptions {
  /** Full-batch iterations (default 25, the authors' Lloyd baseline). */
  iters?: number;
  seed?: number;
  init?: ArrayLike<number>;
}

/**
 * Lloyd's algorithm — the baseline e-means is measured against, and the classic golden. Empty
 * clusters are re-seeded by splitting a cluster drawn ∝ its size, as faiss does.
 */
export function lloyd(data: ArrayLike<number>, n: number, d: number, k: number, opts: LloydOptions = {}): KMeansResult {
  validate(n, d, k, data.length);
  const iters = Math.max(1, Math.floor(opts.iters ?? 25));
  const rng = mulberry32(opts.seed ?? 0);
  const C = seedCentroids(data, n, d, k, rng, opts.init);
  const labels = new Int32Array(n);
  const N = new Float64Array(k);
  const sum = new Float64Array(k * d);
  let splits = 0;
  for (let it = 0; it < iters; it++) {
    assignNearest(data, n, d, C, k, labels);
    N.fill(0);
    sum.fill(0);
    for (let i = 0; i < n; i++) {
      const j = labels[i]!;
      N[j]! += 1;
      for (let t = 0; t < d; t++) sum[j * d + t]! += data[i * d + t]!;
    }
    for (let j = 0; j < k; j++) {
      const Nj = N[j]!;
      if (Nj > 0) for (let t = 0; t < d; t++) C[j * d + t] = sum[j * d + t]! / Nj;
    }
    splits += splitInto(C, N, k, d, 0.5, rng);
  }
  return finish(data, n, d, k, C, splits);
}

function finish(data: ArrayLike<number>, n: number, d: number, k: number, C: Float64Array, splits: number): KMeansResult {
  const labels = new Int32Array(n);
  const mse = assignNearest(data, n, d, C, k, labels);
  const counts = new Int32Array(k);
  for (let i = 0; i < n; i++) counts[labels[i]!]! += 1;
  return { centroids: C, labels, counts, mse, splits };
}
