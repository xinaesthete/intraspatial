// PCA over an expression matrix — the standard pre-step to a UMAP, and the reason a
// UMAP over thousands of genes is tractable at all.
//
// Running k-NN directly on raw expression is both slow (the distance loop is O(N²·G)
// in the gene count) and statistically poor (Euclidean distance in a few thousand
// sparse, heteroscedastic dimensions is dominated by noise). Every real pipeline
// reduces to ~50 components first, and so do we.
//
// **The shape of the problem is what makes this cheap.** For an `[N, G]` matrix the
// covariance is `G×G`, and G here is the number of *selected* genes — tens to a few
// hundred, by ADR-0005's position that gene columns are a selection mechanism and the
// full matrix is never densified. So:
//
//   1. centre the columns                      — O(N·G)
//   2. form the G×G covariance CᵀC/(N-1)       — O(N·G²), the only heavy step
//   3. eigendecompose it with `eigenSym`       — O(G³), G in the tens: microseconds
//   4. project the cells onto the top d PCs    — O(N·G·d)
//
// Step 3 is exactly what `eigenSym.ts` was built for ("N is the number of cell types
// (tens) or selected genes (tens)"), so this module is mostly plumbing around it. Note
// this is the *covariance* route, not an SVD of the data matrix: for G ≪ N it is the
// cheaper of the two and, because `eigenSym` is unconditionally-accurate f64 Jacobi,
// the usual numerical objection to squaring the matrix does not bite at these sizes.
//
// **Steps 2 and 4 are on the GPU, and this module is the oracle for them.** Measured
// host-side on an M2 Max, `nComponents = 30`, `standardise`, stage by stage:
//
//     n         G    stats(1)     cov(2)   eigen(3)    proj(4)   host pca
//     20000   150       12 ms      521 ms     78 ms     196 ms     806 ms
//     60000   150       35 ms     1559 ms     76 ms     559 ms    2230 ms
//     60000   300       66 ms     6008 ms    516 ms    1108 ms    7533 ms
//    100000   300      101 ms     9793 ms    480 ms    1861 ms   12199 ms
//    200000   300      195 ms    20165 ms    461 ms    3726 ms   24167 ms
//
// Those are seconds of *synchronous* main-thread work in the page — the browser is frozen
// for the duration, which is what makes this worth moving rather than merely worth
// speeding up. Measured in Chrome on the real 162k-cell Xenium table (100k cells x 377
// genes), this path is **14.4 s in one unbroken block**; the device path is 1.4 s.
//
// `src/gpu/spatial/pcaGpu.ts` runs steps 2 and 4 as tiled matmuls. The functions below are
// factored out so it can substitute them one at a time and be diffed against them; the
// layering is the `pickKnn` one in reverse, in that the host module exports the pieces and
// knows nothing about the device while the GPU module composes them.
//
// Steps 1 and 3 stay here unconditionally — O(N·G) and O(G³) respectively, and step 3 is
// the part where f64 is load-bearing. Worth reading off the table above, though: once 2
// and 4 are on the device, **step 3 is what is left.** The eigensolve does not care how
// many cells there are, so it is invisible at G=150 and about 500 ms at G=300, which is
// most of the remaining device-path cost. If PCA needs to get faster again, the next move
// is a truncated (Lanczos / randomised) solve for the leading d components rather than the
// full G x G Jacobi — not more GPU work on the covariance.

import { eigenSym } from "./eigenSym";

export interface PcaResult {
  /** Row-major `[n, nComponents]` — the cell coordinates in PC space. */
  readonly scores: Float32Array;
  /** Component-major `[nComponents, dim]` — PC `c` is `components[c*dim .. +dim)`. */
  readonly components: Float64Array;
  /** Variance along each retained component, descending. */
  readonly explainedVariance: Float64Array;
  /** Each component's share of the TOTAL variance (including discarded components), so
   *  these sum to ≤ 1 and the shortfall tells you what the reduction threw away. */
  readonly explainedVarianceRatio: Float64Array;
  readonly nComponents: number;
  /** Column means subtracted before the decomposition — needed to project new data. */
  readonly mean: Float64Array;
}

export interface PcaOptions {
  readonly nComponents?: number;
  /**
   * Divide each column by its standard deviation as well as centring it.
   *
   * Off by default, matching scanpy: on log-normalised expression the variance
   * differences between genes are signal (a gene that varies is more informative than
   * one that does not), and standardising deletes exactly that. Turn it on for
   * features on genuinely incomparable scales — e.g. mixing morphology columns like
   * cell area with expression.
   */
  readonly standardise?: boolean;
}

/** How the columns are centred and rescaled before anything else looks at them. Both
 *  arrays are `dim` long and both the covariance and the projection must use the SAME
 *  pair — recomputing them in the second step would let the basis and the scores
 *  disagree, which is exactly the class of bug the GPU path could introduce silently. */
export interface ColumnStats {
  readonly mean: Float64Array;
  /** Per-column multiplier applied after centring: 1 (centre only) or 1/sd. */
  readonly scale: Float64Array;
}

/** The retained eigenbasis, independent of the data it came from. */
export interface PcaBasis {
  readonly components: Float64Array;
  readonly explainedVariance: Float64Array;
  readonly explainedVarianceRatio: Float64Array;
  readonly nComponents: number;
}

/** How many components `pca` would retain for this shape — the clamp is part of the
 *  contract, and a GPU caller sizing buffers needs it before doing any work. */
export function pcaComponentCount(n: number, dim: number, opts: PcaOptions = {}): number {
  return Math.min(opts.nComponents ?? 50, dim, n);
}

/** Step 1 — column means, and standard deviations if asked for. O(N·G). */
export function columnStats(data: ArrayLike<number>, n: number, dim: number, opts: PcaOptions = {}): ColumnStats {
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let c = 0; c < dim; c++) mean[c] = mean[c]! + data[base + c]!;
  }
  for (let c = 0; c < dim; c++) mean[c] = mean[c]! / n;

  const scale = new Float64Array(dim).fill(1);
  if (opts.standardise) {
    const sumSq = new Float64Array(dim);
    for (let i = 0; i < n; i++) {
      const base = i * dim;
      for (let c = 0; c < dim; c++) {
        const dv = data[base + c]! - mean[c]!;
        sumSq[c] = sumSq[c]! + dv * dv;
      }
    }
    for (let c = 0; c < dim; c++) {
      const sd = Math.sqrt(sumSq[c]! / (n - 1));
      scale[c] = sd > 1e-12 ? 1 / sd : 1; // a constant column stays constant, not NaN
    }
  }
  return { mean, scale };
}

/**
 * Step 2 — the `dim x dim` covariance, on the host in f64. O(N·G²), the heavy step.
 *
 * This is the f64 oracle `covarianceGpu` is measured against, and the reason it stays
 * here rather than being deleted in favour of the device version.
 */
export function covarianceHost(data: ArrayLike<number>, n: number, dim: number, stats: ColumnStats): Float64Array {
  const { mean, scale } = stats;
  // Upper triangle, then mirrored. `eigenSym` reads only the upper triangle but
  // `symmetrise` semantics are cleaner if we fill both.
  const cov = new Float64Array(dim * dim);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let p = 0; p < dim; p++) {
      const vp = (data[base + p]! - mean[p]!) * scale[p]!;
      if (vp === 0) continue;
      for (let q = p; q < dim; q++) {
        const vq = (data[base + q]! - mean[q]!) * scale[q]!;
        cov[p * dim + q] = cov[p * dim + q]! + vp * vq;
      }
    }
  }
  return normaliseCovariance(cov, n, dim);
}

/** Divide the accumulated sum of products by `n - 1` and mirror the upper triangle down.
 *  Shared with the GPU path, which returns raw sums for exactly this to finish. */
export function normaliseCovariance(cov: Float64Array, n: number, dim: number): Float64Array {
  const denom = n - 1;
  for (let p = 0; p < dim; p++) {
    for (let q = p; q < dim; q++) {
      const v = cov[p * dim + q]! / denom;
      cov[p * dim + q] = v;
      cov[q * dim + p] = v;
    }
  }
  return cov;
}

/** Step 3 — eigendecompose the covariance and keep the leading `nComponents`. O(G³). */
export function pcaBasis(cov: Float64Array, dim: number, nComponents: number): PcaBasis {
  const eig = eigenSym(cov, dim);
  let totalVariance = 0;
  for (let c = 0; c < dim; c++) totalVariance += Math.max(eig.values[c]!, 0);

  const components = new Float64Array(nComponents * dim);
  const explainedVariance = new Float64Array(nComponents);
  const explainedVarianceRatio = new Float64Array(nComponents);
  for (let c = 0; c < nComponents; c++) {
    explainedVariance[c] = eig.values[c]!;
    explainedVarianceRatio[c] = totalVariance > 0 ? Math.max(eig.values[c]!, 0) / totalVariance : 0;
    for (let t = 0; t < dim; t++) components[c * dim + t] = eig.vectors[c * dim + t]!;
  }
  fixSigns(components, nComponents, dim);
  return { components, explainedVariance, explainedVarianceRatio, nComponents };
}

/** Step 4 — project the cells onto the retained components. O(N·G·d). */
export function projectScoresHost(data: ArrayLike<number>, n: number, dim: number, stats: ColumnStats, basis: PcaBasis): Float32Array {
  const { mean, scale } = stats;
  const { components, nComponents } = basis;
  const scores = new Float32Array(n * nComponents);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let c = 0; c < nComponents; c++) {
      let acc = 0;
      const cb = c * dim;
      for (let t = 0; t < dim; t++) acc += (data[base + t]! - mean[t]!) * scale[t]! * components[cb + t]!;
      scores[i * nComponents + c] = acc;
    }
  }
  return scores;
}

/** Argument checks shared by the host and GPU entry points, so the two reject the same
 *  inputs with the same messages rather than one of them failing later and further away. */
export function checkPcaArgs(n: number, nComponents: number): void {
  if (n < 2) throw new Error("pca: need at least 2 rows");
  if (nComponents < 1) throw new Error("pca: nComponents must be >= 1");
}

/**
 * Principal components of a row-major `[n, dim]` matrix.
 *
 * Components are returned in descending variance order with a deterministic sign
 * convention (see `fixSigns`), so two runs on the same data give byte-identical output
 * — which matters when the result is written into `obsm` and diffed.
 */
export function pca(data: ArrayLike<number>, n: number, dim: number, opts: PcaOptions = {}): PcaResult {
  const nComponents = pcaComponentCount(n, dim, opts);
  checkPcaArgs(n, nComponents);

  const stats = columnStats(data, n, dim, opts);
  const cov = covarianceHost(data, n, dim, stats);
  const basis = pcaBasis(cov, dim, nComponents);
  const scores = projectScoresHost(data, n, dim, stats, basis);
  return { scores, ...basis, mean: stats.mean };
}

/**
 * Pin each eigenvector's sign.
 *
 * An eigenvector is only defined up to sign, and Jacobi's choice depends on rounding —
 * so the same data can produce a PC that is flipped between runs, which then flips the
 * UMAP and makes two `obsm` entries look unrelated when they are identical. Convention
 * here: the component whose magnitude is largest is made positive.
 */
function fixSigns(components: Float64Array, nComponents: number, dim: number): void {
  for (let c = 0; c < nComponents; c++) {
    const base = c * dim;
    let bestAt = 0;
    let best = 0;
    for (let t = 0; t < dim; t++) {
      const m = Math.abs(components[base + t]!);
      if (m > best) {
        best = m;
        bestAt = t;
      }
    }
    if (components[base + bestAt]! < 0) {
      for (let t = 0; t < dim; t++) components[base + t] = -components[base + t]!;
    }
  }
}
