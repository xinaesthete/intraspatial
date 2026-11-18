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
// **Fit once, assign anywhere.** `emeansFit` returns a frozen centroid table; `assignCentroids`
// labels any set of points against it — a viewport tile, another slide — so labels are stable
// however the data arrives. `emeans` is the two together on one array. The fit is fed a sample
// that fits in memory (a points-zarr coarse level is a uniform random one); only assignment needs
// to run tile by tile. See "The operating model" in `docs/emeans-clustering.md`.
//
// **Weights and masks.** Every point may carry a weight (area on a lat/lon grid, 1/sample
// fraction, slide balancing): the batch statistics sum `w·x` and `w` instead of `x` and 1, so a
// weight of 2 is the point twice. Weight 0, or a point outside a `mask` domain, leaves the point
// out of the fit altogether — it is never a seed — and `assignCentroids` labels it −1. Weights are
// rescaled to mean 1 over the points kept, which makes the fit independent of their units, and the
// scaling laws take n = the number of points kept (see `emeansFit`).
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
import { type Domain, domainContains } from "./domain";

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

/** Points kept by where they are: point i is in when `(xs[i], ys[i])` lies in `domain`. The
 *  coordinates need not be the clustered ones — a descriptor is clustered, its position is masked. */
export interface PointMask {
  readonly domain: Domain;
  readonly xs: ArrayLike<number>;
  readonly ys: ArrayLike<number>;
}

/** Which points count, and how much. Shared by the fit, Lloyd and assignment. */
export interface SampleWeighting {
  /** Per-point weight, finite and ≥ 0 (length ≥ n). A weight of w counts the point w times;
   *  0 leaves it out. Default: every point 1. */
  readonly weights?: ArrayLike<number>;
  /** Points outside the mask's domain are left out, exactly as with weight 0. */
  readonly mask?: PointMask;
}

export interface EmeansOptions extends SampleWeighting {
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
  /** Optional initial centroids (k × d, row-major) instead of seeding from the data. */
  init?: ArrayLike<number>;
  /** How centroids are seeded from the data when `init` is not given: k distinct random points
   *  (default, as the reference implementation does) or k-means++ (Arthur & Vassilvitskii 2007).
   *  Prefer k-means++ when clusters may be well separated: e-means escapes a bad start only
   *  through points between clusters, so with none there it keeps the start it was given. */
  seeding?: Seeding;
}

export type Seeding = "random" | "kmeans++";

/** A frozen set of k centroids in d dimensions: what `assignCentroids` labels points against. */
export interface CentroidTable {
  readonly k: number;
  readonly d: number;
  /** k × d, row-major. */
  readonly centroids: Float64Array;
}

/** What `emeansFit` returns: the table, and how it was reached. */
export interface EmeansFit extends CentroidTable {
  /** Weight each cluster took in the finalisation epoch, in the caller's weight units (point
   *  counts when unweighted). Under the final centroids `assignCentroids` gives the exact mass. */
  readonly mass: Float64Array;
  readonly schedule: EmeansSchedule;
  /** Clusters re-seeded by splitting during training. */
  readonly splits: number;
}

/** Labels for a set of points against a centroid table. */
export interface Assignment {
  /** Nearest-centroid index of every point; −1 for a point left out (weight 0 or outside the mask). */
  labels: Int32Array;
  /** Points per cluster under `labels`. */
  counts: Int32Array;
  /** Weight per cluster under `labels` (equal to `counts` when unweighted). */
  mass: Float64Array;
  /** Weighted mean squared distance of each kept point to its centroid (the k-means objective
   *  over the total weight); 0 when no point is kept. */
  mse: number;
}

/** A fit and its assignment of the points it was fitted to; a table in its own right. */
export interface KMeansResult extends Assignment, CentroidTable {
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

/** Weight of point `i` under `opts`, or 0 when it is left out. Throws on a negative or non-finite weight. */
function weightOf(opts: SampleWeighting, i: number): number {
  const w = opts.weights ? opts.weights[i]! : 1;
  if (!(w >= 0 && w < Number.POSITIVE_INFINITY)) throw new Error(`k-means weight ${i} is ${w}; weights must be finite and ≥ 0`);
  if (w === 0) return 0;
  const m = opts.mask;
  return m && !domainContains(m.domain, m.xs[i]!, m.ys[i]!) ? 0 : w;
}

function checkWeighting(n: number, opts: SampleWeighting): void {
  if (opts.weights && opts.weights.length < n) throw new Error(`k-means weights have ${opts.weights.length} values for ${n} points`);
  const m = opts.mask;
  if (m && (m.xs.length < n || m.ys.length < n))
    throw new Error(`k-means mask has ${Math.min(m.xs.length, m.ys.length)} positions for ${n} points`);
}

/** The points a fit runs over: their indices, and their weights rescaled to mean 1 (absent when
 *  every kept point weighs the same, so the unweighted arithmetic runs unchanged). */
interface Sample {
  readonly idx: Int32Array;
  /** Indexed by point, not by position in `idx`. */
  readonly w?: Float64Array;
  /** Mean of the caller's weights over the kept points: multiplies a rescaled mass back. */
  readonly scale: number;
}

function sampleOf(n: number, opts: SampleWeighting): Sample {
  checkWeighting(n, opts);
  if (!opts.weights && !opts.mask) return { idx: Int32Array.from({ length: n }, (_, i) => i), scale: 1 };
  const keep: number[] = [];
  let total = 0;
  let uniform = true;
  for (let i = 0; i < n; i++) {
    const w = weightOf(opts, i);
    if (w === 0) continue;
    if (keep.length > 0 && w !== opts.weights?.[keep[0]!]) uniform = false;
    keep.push(i);
    total += w;
  }
  const idx = Int32Array.from(keep);
  // Equal weights are no weights: the same arithmetic, draws and all, as leaving them out.
  if (!opts.weights || idx.length === 0) return { idx, scale: 1 };
  if (uniform) return { idx, scale: opts.weights[idx[0]!]! };
  const scale = total / idx.length;
  const w = new Float64Array(n);
  for (const i of idx) w[i] = opts.weights[i]! / scale;
  return { idx, w, scale };
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

/** First position whose running total of `mass` passes `u`; the last one if none does (all
 *  remaining mass zero, or rounding). */
function drawPosition(mass: (a: number) => number, m: number, u: number): number {
  for (let a = 0; a < m; a++) {
    u -= mass(a);
    if (u < 0) return a;
  }
  return m - 1;
}

/** Starting centroids: the given `init`, k-means++, or k distinct random points (partial
 *  Fisher–Yates) — always drawn from the sample's points, so never a point of weight 0. */
function seedCentroids(
  data: ArrayLike<number>,
  d: number,
  k: number,
  rng: Rng,
  s: Sample,
  init?: ArrayLike<number>,
  seeding: Seeding = "random",
): Float64Array {
  const C = new Float64Array(k * d);
  if (init) {
    if (init.length !== k * d) throw new Error(`k-means init has ${init.length} values; expected k·d = ${k * d}`);
    for (let i = 0; i < k * d; i++) C[i] = init[i]!;
    return C;
  }
  const { idx, w } = s;
  const m = idx.length;
  const put = (j: number, a: number) => {
    for (let t = 0; t < d; t++) C[j * d + t] = data[idx[a]! * d + t]!;
  };
  if (seeding === "kmeans++") {
    // Weighted k-means++: the first seed is drawn ∝ w, each next one ∝ w·(squared distance to the
    // nearest seed so far). Unweighted, the first is uniform — one draw either way.
    const wAt = (a: number) => (w ? w[idx[a]!]! : 1);
    const dist = new Float64Array(m).fill(Number.POSITIVE_INFINITY);
    let pick = w ? drawPosition(wAt, m, rng() * m) : Math.floor(rng() * m);
    for (let j = 0; j < k; j++) {
      put(j, pick);
      if (j === k - 1) break;
      let total = 0;
      for (let a = 0; a < m; a++) {
        const xOff = idx[a]! * d;
        let sq = 0;
        for (let t = 0; t < d; t++) sq += (data[xOff + t]! - C[j * d + t]!) ** 2;
        if (sq < dist[a]!) dist[a] = sq;
        total += dist[a]! * wAt(a);
      }
      // All remaining points coincide with a seed: fall back to the last point.
      pick = drawPosition((a) => dist[a]! * wAt(a), m, rng() * total);
    }
    return C;
  }
  const order = shuffled(rng, m);
  for (let j = 0; j < k; j++) put(j, order[j]!);
  return C;
}

function validate(n: number, d: number, k: number, dataLength: number): void {
  if (!Number.isInteger(k) || k < 1) throw new Error(`k-means needs an integer k ≥ 1 (got ${k})`);
  if (n < k) throw new Error(`k-means needs at least k = ${k} points (got ${n})`);
  if (!Number.isInteger(d) || d < 1) throw new Error(`k-means needs dimension d ≥ 1 (got ${d})`);
  if (dataLength < n * d) throw new Error(`k-means data has ${dataLength} values; expected n·d = ${n * d}`);
}

/** Validate, then the sample, which must still hold k points once weight 0 and the mask are applied. */
function prepare(data: ArrayLike<number>, n: number, d: number, k: number, opts: SampleWeighting): Sample {
  validate(n, d, k, data.length);
  const s = sampleOf(n, opts);
  if (s.idx.length < k) throw new Error(`k-means needs at least k = ${k} points of weight > 0 inside the mask (got ${s.idx.length})`);
  return s;
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
 * Fit an e-means centroid table to `n` points of dimension `d` (row-major `data`, length ≥ n·d),
 * to label points with later through `assignCentroids`.
 *
 * With `weights`, the batch statistics are weighted sums, so a centroid is the weighted mean of
 * its points and revival reads weighted EMA counts. **The scaling laws take n = the number of
 * points kept** (weight > 0, inside the mask), not an effective sample size such as
 * (Σw)²/Σw²: B and α₀ set how many point visits a step averages and how many steps an epoch takes,
 * and weights change neither. Weights are rescaled to mean 1 first, so multiplying them all by a
 * constant changes nothing. With strongly uneven weights a batch is noisier than its size
 * suggests; the laws were not fit for that case.
 *
 * Cost: T epochs of n·k·d for assignment (T = 5 by default), plus O(k·d) per step for the EMA
 * blend. Deterministic for a given seed. A point of weight 0 behaves exactly as if absent.
 */
export function emeansFit(data: ArrayLike<number>, n: number, d: number, k: number, opts: EmeansOptions = {}): EmeansFit {
  const s = prepare(data, n, d, k, opts);
  const { idx, w } = s;
  const m = idx.length;
  const epochs = Math.max(1, Math.floor(opts.epochs ?? 5));
  const schedule = emeansSchedule(m, k, epochs, opts);
  const { batch, alpha0, stepsPerEpoch } = schedule;
  const reviveRatio = opts.revive ?? (d <= 3 ? 0 : 0.1);
  const cosine = (opts.schedule ?? "cosine") === "cosine";
  const rng = mulberry32(opts.seed ?? 0);

  const C = seedCentroids(data, d, k, rng, s, opts.init, opts.seeding);
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
    const perm = shuffled(rng, m);
    // The final epoch also visits the tail that floor(m / B) full batches leave out.
    const steps = finalEpoch ? Math.ceil(m / batch) : stepsPerEpoch;
    for (let st = 0; st < steps; st++) {
      const off = st * batch;
      const len = Math.min(batch, m - off);
      cnt.fill(0);
      sum.fill(0);
      for (let b = 0; b < len; b++) {
        const xi = idx[perm[off + b]!]!;
        const wi = w ? w[xi]! : 1;
        const [j] = nearest(data, xi, d, C, k);
        cnt[j]! += wi;
        const xOff = xi * d;
        for (let t = 0; t < d; t++) sum[j * d + t]! += wi * data[xOff + t]!;
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
  // Finalise: every cluster seen in the last epoch ends on the exact (weighted) mean of its assignments.
  for (let j = 0; j < k; j++) {
    const cj = capCnt[j]!;
    if (cj > 0) for (let t = 0; t < d; t++) C[j * d + t] = capSum[j * d + t]! / cj;
  }
  const mass = capCnt.map((c) => c * s.scale);
  return { k, d, centroids: C, mass, schedule, splits };
}

/**
 * Label `n` points (row-major `data`, length ≥ n·d) against a frozen centroid table. A point of
 * weight 0 or outside the mask gets label −1 and no part in the counts or the MSE. Deterministic
 * and per point, so a tile labelled alone gets the labels it would get as part of the whole.
 */
export function assignCentroids(table: CentroidTable, data: ArrayLike<number>, n: number, opts: SampleWeighting = {}): Assignment {
  const { k, d, centroids } = table;
  if (centroids.length !== k * d) throw new Error(`centroid table has ${centroids.length} values; expected k·d = ${k * d}`);
  if (data.length < n * d) throw new Error(`k-means data has ${data.length} values; expected n·d = ${n * d}`);
  checkWeighting(n, opts);
  const labels = new Int32Array(n);
  const counts = new Int32Array(k);
  const mass = new Float64Array(k);
  let total = 0;
  let weight = 0;
  for (let i = 0; i < n; i++) {
    const wi = weightOf(opts, i);
    if (wi === 0) {
      labels[i] = -1;
      continue;
    }
    const [j, dist] = nearest(data, i, d, centroids, k);
    labels[i] = j;
    counts[j]! += 1;
    mass[j]! += wi;
    total += wi * dist;
    weight += wi;
  }
  return { labels, counts, mass, mse: weight > 0 ? total / weight : 0 };
}

/**
 * e-means k-means on `n` points: `emeansFit`, then `assignCentroids` on the same points, so the
 * labels, counts and MSE are those of the final centroids.
 */
export function emeans(data: ArrayLike<number>, n: number, d: number, k: number, opts: EmeansOptions = {}): EmeansResult {
  const fit = emeansFit(data, n, d, k, opts);
  return { ...assignCentroids(fit, data, n, opts), k, d, centroids: fit.centroids, splits: fit.splits, schedule: fit.schedule };
}

export interface LloydOptions extends SampleWeighting {
  /** Full-batch iterations (default 25, the authors' Lloyd baseline). */
  iters?: number;
  seed?: number;
  init?: ArrayLike<number>;
  seeding?: Seeding;
}

/**
 * Lloyd's algorithm — the baseline e-means is measured against, and the classic golden. Empty
 * clusters are re-seeded by splitting a cluster drawn ∝ its weight, as faiss does. Takes the same
 * weights and mask as `emeansFit`.
 */
export function lloyd(data: ArrayLike<number>, n: number, d: number, k: number, opts: LloydOptions = {}): KMeansResult {
  const s = prepare(data, n, d, k, opts);
  const { idx, w } = s;
  const iters = Math.max(1, Math.floor(opts.iters ?? 25));
  const rng = mulberry32(opts.seed ?? 0);
  const C = seedCentroids(data, d, k, rng, s, opts.init, opts.seeding);
  const N = new Float64Array(k);
  const sum = new Float64Array(k * d);
  let splits = 0;
  for (let it = 0; it < iters; it++) {
    N.fill(0);
    sum.fill(0);
    for (const i of idx) {
      const [j] = nearest(data, i, d, C, k);
      const wi = w ? w[i]! : 1;
      N[j]! += wi;
      for (let t = 0; t < d; t++) sum[j * d + t]! += wi * data[i * d + t]!;
    }
    for (let j = 0; j < k; j++) {
      const Nj = N[j]!;
      if (Nj > 0) for (let t = 0; t < d; t++) C[j * d + t] = sum[j * d + t]! / Nj;
    }
    // Only an empty cluster is dead: a light one is still a cluster.
    splits += splitInto(C, N, k, d, Number.MIN_VALUE, rng);
  }
  const table = { k, d, centroids: C };
  return { ...assignCentroids(table, data, n, opts), ...table, splits };
}
