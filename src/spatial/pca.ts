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
// Step 2 is the one worth moving to the GPU if G grows — it is a Gram matrix over
// cells, structurally identical to `src/gpu/spatial/gramMatrix.ts`. Left on the host
// for now because at G ≤ a few hundred it is not the bottleneck; the k-NN is.

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

/**
 * Principal components of a row-major `[n, dim]` matrix.
 *
 * Components are returned in descending variance order with a deterministic sign
 * convention (see `fixSigns`), so two runs on the same data give byte-identical output
 * — which matters when the result is written into `obsm` and diffed.
 */
export function pca(data: ArrayLike<number>, n: number, dim: number, opts: PcaOptions = {}): PcaResult {
  const nComponents = Math.min(opts.nComponents ?? 50, dim, n);
  if (n < 2) throw new Error("pca: need at least 2 rows");
  if (nComponents < 1) throw new Error("pca: nComponents must be >= 1");

  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let c = 0; c < dim; c++) mean[c] = mean[c]! + data[base + c]!;
  }
  for (let c = 0; c < dim; c++) mean[c] = mean[c]! / n;

  // Per-column multiplier applied after centring: 1 (centre only) or 1/sd (standardise).
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

  // Covariance, upper triangle then mirrored. `eigenSym` reads only the upper triangle
  // but `symmetrise` semantics are cleaner if we fill both.
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
  const denom = n - 1;
  for (let p = 0; p < dim; p++) {
    for (let q = p; q < dim; q++) {
      const v = cov[p * dim + q]! / denom;
      cov[p * dim + q] = v;
      cov[q * dim + p] = v;
    }
  }

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

  return { scores, components, explainedVariance, explainedVarianceRatio, nComponents, mean };
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
