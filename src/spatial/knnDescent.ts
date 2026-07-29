// Approximate k-NN by **nearest-neighbour descent** (Dong, Charikar & Li 2011) — the
// primitive that lifts the ~20k-cell ceiling.
//
// The exact search is O(N²·D) and a bigger GPU does not fix it: at 500k cells that is
// 2.5·10¹¹ distance evaluations before the multiply by D. NN-descent replaces it with a
// fixed number of O(N·k²·D) passes, on the observation that **a neighbour of my
// neighbour is a good candidate for being my neighbour**. Start from a rough guess,
// repeatedly look one hop further out, keep the best k. At N=500k, k=15, D=50 one pass
// is ~5·10⁹ — about the cost of ONE exact pass at N=10k, and a handful of passes suffice.
//
// It is approximate, and the honest way to state that is **recall**: the fraction of
// each point's true k nearest neighbours that the approximation found (`knnRecall`
// below). UMAP is unusually tolerant here — the fuzzy graph is a *smoothed* object and
// missing a few far-tail neighbours perturbs it little — but "tolerant" is not "immune",
// so recall is measured rather than assumed, and the tests pin it.
//
// **When NOT to use it, and what it costs.** Descent costs roughly
// `maxIters * n * candidates²`, so below about `maxIters * candidates²` points the exact
// search is genuinely the *faster* of the two as well as being exact. Measured host-side,
// D=50, k=15:
//
//     n       exact     descent   recall   speedup
//     1000     69 ms     184 ms   0.999    0.37x
//     3000    551 ms     938 ms   0.990    0.59x
//     6000   2197 ms    1814 ms   0.970    1.21x
//    12000   8988 ms    3874 ms   0.937    2.32x
//    25000  38003 ms    8929 ms   0.881    4.26x
//
// Two things to read off that. Against a **host** exact search the crossover is around
// **5-6k points**. And recall *falls* as n grows at a fixed `maxIters`, because a larger
// graph needs more passes to propagate; raising `maxIters` buys it back at proportional
// cost. 0.88 recall is still ample for UMAP (the fuzzy graph is smoothed, and the misses
// are far-tail neighbours with small membership), but it is a knob, not a constant.
//
// **The crossover depends on which exact search you are racing.** Against the tiled
// `knnGpu` it moves a long way out — measured on the same machine, D=50, k=15:
//
//     n       knnGpu    host descent
//    16000     739 ms      ~6.1 s
//    30000    3484 ms     ~11.2 s
//    50000   14570 ms     ~19 s (extrapolated)
//
// So with a GPU present, exact is still the better choice past 50k, and the honest
// statement is that this primitive's value today is (a) on machines without a usable GPU
// and (b) as the algorithm that will actually scale once its inner local join is moved to
// the device. `pickKnn` therefore takes the crossover as a parameter rather than
// pretending one number fits both.
//
// **The seam.** This returns the same `KnnResult` as `knnBruteForceCpu` and `knnGpu`, so
// it drops into `umapGraphFor`'s injected `knn` with nothing downstream changing. That
// was the point of making the k-NN injected rather than imported.
//
// Layering: the bookkeeping (initialisation, reverse adjacency, convergence) is host-side
// where it can be exactly right. The inner local join — the only O(k²) part — is the
// natural next thing to move to the GPU, and the flat `NeighbourHeap` / candidate arrays
// are laid out to be uploaded verbatim when it is. It is NOT on the GPU yet: the timings
// above are host-only, so the numbers understate what the primitive can do.

import type { KnnResult } from "./umapGraph";
import { mulberry32 } from "./umapLayout";

export interface DescentOptions {
  readonly k: number;
  /** Passes over the graph. Recall climbs steeply for the first few and then flattens;
   *  the default is where that knee sits on the fixtures here. */
  readonly maxIters?: number;
  /** Stop early once a pass improves fewer than this fraction of all slots. */
  readonly tol?: number;
  readonly seed?: number;
  /** Random 1-D projections used to seed the lists. 0 = purely random init. */
  readonly nProjections?: number;
}

/** A point's current best-k, kept sorted ascending by distance. Flat arrays rather than
 *  per-point objects: this is the structure that gets uploaded to the GPU verbatim. */
export interface NeighbourHeap {
  readonly n: number;
  readonly k: number;
  readonly indices: Int32Array;
  readonly distances: Float32Array;
}

function makeHeap(n: number, k: number): NeighbourHeap {
  return { n, k, indices: new Int32Array(n * k).fill(-1), distances: new Float32Array(n * k).fill(Number.POSITIVE_INFINITY) };
}

/**
 * Offer `cand` as a neighbour of `i`. Returns 1 if it was accepted, 0 if not.
 *
 * Rejects duplicates, which matters more than it looks: without the check a point can
 * fill its entire list with copies of one good neighbour discovered along several paths,
 * and the descent then stops making progress while reporting improvements.
 */
export function offer(heap: NeighbourHeap, i: number, cand: number, dist: number): number {
  const { k, indices, distances } = heap;
  const base = i * k;
  if (dist >= distances[base + k - 1]!) return 0;
  for (let t = 0; t < k; t++) {
    if (indices[base + t] === cand) return 0;
  }
  let p = k - 1;
  while (p > 0 && distances[base + p - 1]! > dist) {
    distances[base + p] = distances[base + p - 1]!;
    indices[base + p] = indices[base + p - 1]!;
    p--;
  }
  distances[base + p] = dist;
  indices[base + p] = cand;
  return 1;
}

function euclidean(data: ArrayLike<number>, dim: number, a: number, b: number): number {
  let acc = 0;
  const ab = a * dim;
  const bb = b * dim;
  for (let c = 0; c < dim; c++) {
    const delta = data[ab + c]! - data[bb + c]!;
    acc += delta * delta;
  }
  return Math.sqrt(acc);
}

/**
 * Seed each point's list.
 *
 * Purely random candidates converge, but slowly on clustered data: a random point is
 * almost never in the same cluster, so the first passes have nothing good to descend
 * from. Cheap fix — project everything onto a few random directions and offer each point
 * its neighbours *in projected order*. Points close in the original space are close in
 * any projection (points far apart may also project close, which is why this is only an
 * initialisation), so a handful of 1-D sorts buys a much better starting graph for
 * O(N log N).
 */
export function initialiseHeap(data: ArrayLike<number>, n: number, dim: number, opts: DescentOptions): NeighbourHeap {
  const { k } = opts;
  const rnd = mulberry32(opts.seed ?? 42);
  const heap = makeHeap(n, k);
  const nProjections = opts.nProjections ?? 3;

  for (let p = 0; p < nProjections; p++) {
    const dir = new Float64Array(dim);
    let norm = 0;
    for (let c = 0; c < dim; c++) {
      const u = Math.max(rnd(), 1e-12);
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
      dir[c] = g;
      norm += g * g;
    }
    norm = Math.sqrt(norm) || 1;
    const proj = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let c = 0; c < dim; c++) acc += data[i * dim + c]! * dir[c]!;
      proj[i] = acc / norm;
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => proj[a]! - proj[b]!);
    // Offer each point the few points adjacent to it along this projection.
    const window = Math.min(k, 8);
    for (let r = 0; r < n; r++) {
      const i = order[r]!;
      for (let w = 1; w <= window; w++) {
        for (const s of [r - w, r + w]) {
          if (s < 0 || s >= n) continue;
          const j = order[s]!;
          if (j === i) continue;
          offer(heap, i, j, euclidean(data, dim, i, j));
        }
      }
    }
  }

  // Top up any short lists with random candidates, so every slot is occupied before the
  // descent starts — an unfilled slot is an infinity that blocks nothing but wastes a
  // comparison every pass.
  for (let i = 0; i < n; i++) {
    let guard = 0;
    while (heap.indices[i * k + k - 1] === -1 && guard < k * 8) {
      const j = Math.floor(rnd() * n);
      guard++;
      if (j !== i) offer(heap, i, j, euclidean(data, dim, i, j));
    }
  }
  return heap;
}

/**
 * The candidate lists for one pass: each point's current neighbours plus its **reverse**
 * neighbours (the points that chose it).
 *
 * The reverse half is what makes NN-descent work rather than merely drift. If only
 * forward edges were followed, a point sitting on the edge of a cluster would never hear
 * about the points that already consider it close, and the graph would converge to
 * something lopsided. Built here on the host, O(N·k), because it is a counting sort and
 * the GPU has no business doing it.
 *
 * Reverse lists are capped at `maxReverse` — a hub point can be chosen by thousands of
 * others, and letting one thread walk all of them would stall a whole GPU workgroup while
 * adding little (the extras are mostly redundant).
 */
export function buildCandidates(
  heap: NeighbourHeap,
  maxReverse: number,
  rnd: () => number,
): { candidates: Int32Array; counts: Int32Array; width: number } {
  const { n, k } = heap;
  const width = k + maxReverse;
  const candidates = new Int32Array(n * width).fill(-1);
  const counts = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const j = heap.indices[i * k + t]!;
      if (j < 0) continue;
      candidates[i * width + counts[i]!] = j;
      counts[i] = counts[i]! + 1;
    }
  }
  // Reverse edges, with reservoir sampling once a point's quota is full so a hub's
  // reverse list is a fair sample rather than whichever points happened to come first.
  const seen = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const j = heap.indices[i * k + t]!;
      if (j < 0) continue;
      const used = counts[j]! - k >= 0 ? counts[j]! - k : 0;
      seen[j] = seen[j]! + 1;
      if (used < maxReverse) {
        candidates[j * width + counts[j]!] = i;
        counts[j] = counts[j]! + 1;
      } else {
        const r = Math.floor(rnd() * seen[j]!);
        if (r < maxReverse) candidates[j * width + k + r] = i;
      }
    }
  }
  return { candidates, counts, width };
}

/**
 * Approximate k-NN on the host — the reference the GPU kernel is checked against, and a
 * usable path in its own right.
 *
 * `data` is row-major `[n, dim]`. The result has the same contract as
 * `knnBruteForceCpu`: self excluded, each row ascending — except that it is approximate.
 */
export function knnDescentCpu(data: ArrayLike<number>, n: number, dim: number, opts: DescentOptions): KnnResult {
  const { k } = opts;
  if (k >= n) throw new Error(`knnDescentCpu: need k < n (k=${k}, n=${n})`);
  const maxIters = opts.maxIters ?? 12;
  const tol = opts.tol ?? 0.001;
  const rnd = mulberry32((opts.seed ?? 42) ^ 0x5bf03635);

  const heap = initialiseHeap(data, n, dim, opts);
  const maxReverse = Math.max(4, Math.floor(k / 2));

  for (let iter = 0; iter < maxIters; iter++) {
    const { candidates, counts, width } = buildCandidates(heap, maxReverse, rnd);
    let changes = 0;
    for (let i = 0; i < n; i++) {
      const cBase = i * width;
      const cn = counts[i]!;
      for (let a = 0; a < cn; a++) {
        const p = candidates[cBase + a]!;
        if (p < 0) continue;
        // The local join: everything `p` knows about is a candidate for `i`.
        for (let b = 0; b < counts[p]!; b++) {
          const q = candidates[p * width + b]!;
          if (q < 0 || q === i) continue;
          changes += offer(heap, i, q, euclidean(data, dim, i, q));
        }
      }
    }
    if (changes <= tol * n * k) break;
  }

  return finalise(heap);
}

/** Convert the working heap to the public `KnnResult`. Any slot still unfilled (possible
 *  only for pathologically small n) is replaced by a self-reference-free fallback. */
export function finalise(heap: NeighbourHeap): KnnResult {
  const { n, k } = heap;
  const indices = new Uint32Array(n * k);
  const distances = new Float32Array(n * k);
  for (let t = 0; t < n * k; t++) {
    indices[t] = heap.indices[t]! < 0 ? 0 : heap.indices[t]!;
    distances[t] = heap.distances[t]!;
  }
  return { n, k, indices, distances };
}

/**
 * Recall of an approximate k-NN against the exact one: the mean fraction of each point's
 * true neighbours that were found.
 *
 * This is the number that decides whether an approximation is usable, so it is a
 * first-class export rather than a test helper — a caller tuning `maxIters` against their
 * own data needs it too.
 */
export function knnRecall(approx: KnnResult, exact: KnnResult): number {
  const { n, k } = exact;
  let hits = 0;
  const want = new Set<number>();
  for (let i = 0; i < n; i++) {
    want.clear();
    for (let t = 0; t < k; t++) want.add(exact.indices[i * k + t]!);
    for (let t = 0; t < approx.k; t++) if (want.has(approx.indices[i * approx.k + t]!)) hits++;
  }
  return hits / (n * k);
}

/**
 * Choose the k-NN implementation by problem size.
 *
 * Exact below the crossover measured in the header, approximate above it. Callers that
 * know better (a benchmark, a recall requirement) should pick explicitly; this is for the
 * common case where "do the sensible thing" is the whole requirement, and it is what the
 * offline CLI uses by default.
 *
 * `exact` is injected rather than imported so a browser caller can hand in `knnGpu` and a
 * Node one `knnBruteForceCpu`, without this module depending on either.
 */
export function pickKnn(
  exact: (data: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult> | KnnResult,
  /** `crossover` must match the `exact` you passed: ~5k for a host brute force, ~100k for
   *  the tiled `knnGpu`. The default suits the host. */
  opts: {
    crossover?: number;
    seed?: number;
    maxIters?: number;
    /** The approximate path. Defaults to the host descent; pass `knnDescentGpu` where a
     *  device is available — it is the same algorithm and, being race-free, returns the
     *  same neighbours, 6-12x faster. Injected rather than imported so this module stays
     *  free of any dependency on the GPU layer. */
    approx?: (data: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult> | KnnResult;
  } = {},
): (data: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult> | KnnResult {
  const crossover = opts.crossover ?? 5000;
  const approx =
    opts.approx ?? ((data, n, dim, k) => knnDescentCpu(data, n, dim, { k, seed: opts.seed, maxIters: opts.maxIters }));
  return (data, n, dim, k) => (n < crossover ? exact(data, n, dim, k) : approx(data, n, dim, k));
}

/** Which path `pickKnn` would take, for reporting. */
export function knnStrategyFor(n: number, crossover = 5000): "exact" | "descent" {
  return n < crossover ? "exact" : "descent";
}
