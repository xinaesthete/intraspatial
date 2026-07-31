// UMAP's fuzzy simplicial set — the k-NN graph → weighted edge list half of UMAP
// (McInnes, Healy & Melville 2018), host-side and exact.
//
// This is the CPU golden. The GPU takes over for the two parts that are actually
// expensive at scale — the k-NN search (`src/gpu/spatial/knn.ts`) and the layout SGD
// (`src/gpu/spatial/umapLayout.ts`) — but the graph construction between them is
// O(N·k) with k in the tens, so it stays here where it can be exactly right and
// exactly testable. At N=100k, k=15 that is 1.5M edges of arithmetic: milliseconds.
//
// **Why bit-faithful to the reference implementation matters here.** The offline path
// writes into `obsm/X_umap`, and the whole point of writing into `obsm` is that
// someone else — scanpy, MDV, a colleague's notebook — reads it back and compares it
// to what their own pipeline produces. A graph that is merely "UMAP-like" makes that
// comparison meaningless. So the constants below (`SMOOTH_K_TOLERANCE`,
// `MIN_K_DIST_SCALE`, the log2 target, the `rho` interpolation) are the reference
// implementation's, deliberately, and the tests pin them.
//
// Relation to what was already here: `src/gpu/spatial/fuzzyAdjacencyAdaptive.ts` is
// the same t-conorm symmetrisation with σ_i = scale·ρ_i taken as given rather than
// *calibrated*, on 2-D points, emitting a dense N×N matrix. That is the right shape
// for a persistence sweep over a few hundred points and the wrong one here — see
// `docs/umap-on-anndata.md` §2. This module is sparse throughout.

/** How close the binary search must get to the log2(k) target before it stops. */
const SMOOTH_K_TOLERANCE = 1e-5;
/** σ_i is floored at this fraction of the local (or global) mean distance, so a point
 *  sitting on top of its neighbours cannot produce a degenerate zero-width kernel. */
const MIN_K_DIST_SCALE = 1e-3;
const N_ITER = 64;

/** A k-NN result in the layout every consumer here expects: row-major, `k` entries per
 *  point, **self excluded**, and each row sorted by ascending distance.
 *
 *  Self-exclusion is a real interface decision, not a detail. The reference
 *  implementation carries self as column 0 and then skips it everywhere, which means
 *  its `n_neighbors` is one larger than the number of actual neighbours. We keep the
 *  honest count in the data structure and reconcile at the one place it matters — see
 *  `FuzzyGraphOptions.nNeighbors`. */
export interface KnnResult {
  readonly n: number;
  /** Neighbours per point (excluding self). */
  readonly k: number;
  /** `indices[i * k + t]` — the t-th nearest neighbour of point i. */
  readonly indices: Uint32Array;
  /** `distances[i * k + t]` — its distance. Ascending within each row. */
  readonly distances: Float32Array;
}

export interface FuzzyGraphOptions {
  /**
   * `n_neighbors` in **reference-implementation semantics**, i.e. *including* the
   * point itself. Passing 15 here reproduces `scanpy.pp.neighbors(n_neighbors=15)`,
   * and consumes a `KnnResult` with `k = 14`.
   *
   * The asymmetry is inherited, not invented: the target the σ binary search solves
   * for is `log2(n_neighbors)` while the sum it solves over runs across the
   * `n_neighbors - 1` non-self entries. Reproducing the published numbers means
   * reproducing that off-by-one, so it is named and documented rather than quietly
   * fixed.
   */
  readonly nNeighbors: number;
  /** Number of nearest neighbours assumed locally connected — the manifold is taken
   *  to be at least this densely sampled, so the ρ_i offset removes their distance
   *  entirely. Fractional values interpolate. Reference default 1. */
  readonly localConnectivity?: number;
  /** Scales the target sum; >1 gives a broader kernel. Reference default 1. */
  readonly bandwidth?: number;
  /** Scales the whole graph's weights (UMAP's `set_op_mix_ratio` sibling). 1 = pure
   *  fuzzy union (t-conorm), 0 = pure intersection. Reference default 1. */
  readonly setOpMixRatio?: number;
}

/** Per-point kernel parameters from the smooth-k-NN calibration. */
export interface SmoothKnnResult {
  /** ρ_i — distance to the nearest neighbour (the local-connectivity offset). */
  readonly rho: Float64Array;
  /** σ_i — the calibrated bandwidth. */
  readonly sigma: Float64Array;
}

/**
 * Calibrate the per-point bandwidth σ_i.
 *
 * For each point, find σ_i such that `Σ_j exp(-max(0, d_ij - ρ_i) / σ_i) = log2(n_neighbors)`.
 * The left side is a smooth count of "how many neighbours point i effectively has";
 * fixing it to a constant is what makes UMAP's kernel adapt to local density — the
 * same self-tuning move as `cknn.ts`, but solved for rather than assumed.
 *
 * Bisection, not Newton: the sum is monotone in σ but its derivative vanishes for
 * large σ, and 64 bisection steps are both fast enough (O(N·k·64), still trivial next
 * to the k-NN search) and unconditionally convergent.
 */
export function smoothKnnDist(knn: KnnResult, opts: FuzzyGraphOptions): SmoothKnnResult {
  const { n, k, distances } = knn;
  const localConnectivity = opts.localConnectivity ?? 1;
  const bandwidth = opts.bandwidth ?? 1;
  const target = Math.log2(opts.nNeighbors) * bandwidth;

  const rho = new Float64Array(n);
  const sigma = new Float64Array(n);

  // The fallback floor for points whose ρ_i is 0 (every neighbour coincident) is the
  // mean over the WHOLE distance matrix, not the point's own row — a row of zeros
  // carries no scale of its own to fall back on.
  let globalMean = 0;
  for (let t = 0; t < distances.length; t++) globalMean += distances[t]!;
  globalMean /= Math.max(distances.length, 1);

  for (let i = 0; i < n; i++) {
    const base = i * k;
    let lo = 0;
    let hi = Number.POSITIVE_INFINITY;
    let mid = 1;

    // ρ_i: the distance to the `localConnectivity`-th *non-zero* neighbour, linearly
    // interpolated for fractional values. Duplicate points (distance exactly 0) are
    // skipped rather than counted — they are the same sample, not a neighbour.
    const nonZero: number[] = [];
    for (let t = 0; t < k; t++) {
      const dist = distances[base + t]!;
      if (dist > 0) nonZero.push(dist);
    }
    if (nonZero.length >= localConnectivity) {
      const index = Math.floor(localConnectivity);
      const interpolation = localConnectivity - index;
      if (index > 0) {
        rho[i] = nonZero[index - 1]!;
        // `nonZero[index]` is absent when localConnectivity lands on the last entry;
        // there is nothing to interpolate towards, so the floor stands unadjusted.
        const next = nonZero[index];
        if (interpolation > SMOOTH_K_TOLERANCE && next !== undefined) {
          rho[i] = rho[i]! + interpolation * (next - nonZero[index - 1]!);
        }
      } else {
        rho[i] = interpolation * nonZero[0]!;
      }
    } else if (nonZero.length > 0) {
      rho[i] = nonZero[nonZero.length - 1]!;
    }

    for (let iter = 0; iter < N_ITER; iter++) {
      let psum = 0;
      for (let t = 0; t < k; t++) {
        const dist = distances[base + t]! - rho[i]!;
        psum += dist > 0 ? Math.exp(-(dist / mid)) : 1;
      }
      if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) break;
      if (psum > target) {
        hi = mid;
        mid = (lo + hi) / 2;
      } else {
        lo = mid;
        mid = hi === Number.POSITIVE_INFINITY ? mid * 2 : (lo + hi) / 2;
      }
    }
    sigma[i] = mid;

    // Floor σ_i so the kernel never collapses to a delta.
    if (rho[i]! > 0) {
      let rowMean = 0;
      for (let t = 0; t < k; t++) rowMean += distances[base + t]!;
      rowMean /= Math.max(k, 1);
      if (sigma[i]! < MIN_K_DIST_SCALE * rowMean) sigma[i] = MIN_K_DIST_SCALE * rowMean;
    } else if (sigma[i]! < MIN_K_DIST_SCALE * globalMean) {
      sigma[i] = MIN_K_DIST_SCALE * globalMean;
    }
  }

  return { rho, sigma };
}

/** A symmetric weighted graph in coordinate (COO) form — one entry per directed edge,
 *  both directions present, so a GPU kernel can walk it as a flat edge list without
 *  any per-row indirection. This is the form the layout SGD consumes. */
export interface FuzzyGraph {
  readonly n: number;
  readonly head: Uint32Array;
  readonly tail: Uint32Array;
  /** Membership strength μ_ij ∈ (0, 1]. */
  readonly weight: Float32Array;
  /** Number of directed edges (= `head.length`). */
  readonly nEdges: number;
}

/**
 * Build the symmetric fuzzy simplicial set from a k-NN graph.
 *
 * The directed membership `μ_{j|i} = exp(-max(0, d_ij - ρ_i) / σ_i)` is point i's own
 * view of how strongly j is its neighbour, and it is *not* symmetric — i can be j's
 * nearest neighbour while j is nowhere near i's shortlist. UMAP reconciles the two
 * views with the probabilistic t-conorm (fuzzy union):
 *
 *     μ_ij = μ_{j|i} + μ_{i|j} − μ_{j|i}·μ_{i|j}
 *
 * so an edge survives if *either* endpoint considers the other close. `setOpMixRatio`
 * blends that union against the intersection (the plain product) for callers who want
 * a more conservative graph.
 *
 * Zero-weight edges are dropped: they contribute nothing to the layout and every one
 * kept is a wasted GPU thread.
 */
export function fuzzySimplicialSet(knn: KnnResult, opts: FuzzyGraphOptions): FuzzyGraph {
  const { n, k, indices, distances } = knn;
  const { rho, sigma } = smoothKnnDist(knn, opts);
  const mix = opts.setOpMixRatio ?? 1;

  // Directed memberships, keyed so the transpose can be looked up in one pass. A Map
  // keyed on `i*n + j` is the honest structure for a graph this sparse; building the
  // full N×N transpose to symmetrise it would defeat the point of being sparse at all.
  // (Key arithmetic stays exact while n² < 2^53, i.e. n < 94M — far past any N here.)
  const directed = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const base = i * k;
    for (let t = 0; t < k; t++) {
      const j = indices[base + t]!;
      if (j === i) continue;
      const dist = distances[base + t]! - rho[i]!;
      const mu = dist > 0 ? Math.exp(-(dist / sigma[i]!)) : 1;
      if (mu > 0) directed.set(i * n + j, mu);
    }
  }

  const head: number[] = [];
  const tail: number[] = [];
  const weight: number[] = [];
  // Walk each directed edge once and emit the symmetrised pair only from the (i<j)
  // side, so each undirected edge yields exactly two directed entries.
  const seen = new Set<number>();
  for (const [key, mu] of directed) {
    const i = Math.floor(key / n);
    const j = key % n;
    const lo = Math.min(i, j);
    const hi = Math.max(i, j);
    const pairKey = lo * n + hi;
    if (seen.has(pairKey)) continue;
    seen.add(pairKey);
    const muT = directed.get(j * n + i) ?? 0;
    const union = mu + muT - mu * muT;
    const inter = mu * muT;
    const w = mix * union + (1 - mix) * inter;
    if (w <= 0) continue;
    head.push(i, j);
    tail.push(j, i);
    weight.push(w, w);
  }

  return {
    n,
    head: Uint32Array.from(head),
    tail: Uint32Array.from(tail),
    weight: Float32Array.from(weight),
    nEdges: head.length,
  };
}

/**
 * Exact brute-force k-NN on the host — the golden the GPU kernel is checked against,
 * and a usable path for the few-thousand-point case.
 *
 * `data` is row-major `[n, dim]`. Euclidean distance. O(N²·D), so this is a reference
 * and a small-N convenience, not the production path — `knnGpu` is the same answer
 * with the inner loop on the device.
 */
export function knnBruteForceCpu(data: ArrayLike<number>, n: number, dim: number, k: number): KnnResult {
  if (k >= n) throw new Error(`knnBruteForceCpu: need k < n (k=${k}, n=${n})`);
  const indices = new Uint32Array(n * k);
  const distances = new Float32Array(n * k);
  // One reusable scratch pair; the per-row work is a partial selection, not a sort.
  const bestD = new Float64Array(k);
  const bestI = new Uint32Array(k);

  for (let i = 0; i < n; i++) {
    bestD.fill(Number.POSITIVE_INFINITY);
    bestI.fill(0);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let acc = 0;
      const ai = i * dim;
      const aj = j * dim;
      for (let c = 0; c < dim; c++) {
        const delta = data[ai + c]! - data[aj + c]!;
        acc += delta * delta;
      }
      const dist = Math.sqrt(acc);
      if (dist >= bestD[k - 1]!) continue;
      // Insertion into the sorted-descending tail keeps each row ascending on exit.
      let p = k - 1;
      while (p > 0 && bestD[p - 1]! > dist) {
        bestD[p] = bestD[p - 1]!;
        bestI[p] = bestI[p - 1]!;
        p--;
      }
      bestD[p] = dist;
      bestI[p] = j;
    }
    for (let t = 0; t < k; t++) {
      indices[i * k + t] = bestI[t]!;
      distances[i * k + t] = bestD[t]!;
    }
  }
  return { n, k, indices, distances };
}
