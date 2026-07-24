// Symmetric eigendecomposition by cyclic Jacobi rotations — f64, host-side, small n.
//
// This exists for the N-way cell-type / gene matrices: N is the number of cell types (tens) or
// selected genes (tens), never thousands, so an O(n³)-per-sweep method that is *unconditionally
// accurate* beats anything asymptotically better. Jacobi is chosen over the usual
// tridiagonalise-then-QL for one reason that matters here: it computes the small eigenvalues to
// high **relative** accuracy, and the small eigenvalues are exactly what we interrogate. A matrix
// that ought to be positive semi-definite but comes back with a −1e−3 eigenvalue is telling us
// something about the formulation (see `gram.ts`), and we need to trust that number rather than
// wonder whether the solver produced it.
//
// Everything here is f64 on the host. The matrices are n×n with n in the tens; the expensive part
// of the pipeline is the O(P·n) raster work on the GPU, and pushing a 20×20 eigenproblem onto the
// device to save microseconds would trade a reliable answer for an unreliable one.

/** A symmetric eigendecomposition, eigenvalues descending. */
export interface EigenResult {
  /** Eigenvalues, **descending** (`values[0]` is the largest). Length `n`. */
  readonly values: Float64Array;
  /** Eigenvectors, **mode-major**: mode `k`'s components are `vectors[k*n .. k*n+n)`. Unit norm.
   *  Mode-major (rather than the column-of-a-matrix convention) because every consumer here wants
   *  "give me mode k as a vector over channels", and that should be a contiguous slice. */
  readonly vectors: Float64Array;
  /** Number of Jacobi sweeps used. `sweeps === maxSweeps` means it did not converge. */
  readonly sweeps: number;
  /** Largest remaining off-diagonal magnitude, relative to the largest |eigenvalue|. */
  readonly offDiagonal: number;
}

export interface EigenOptions {
  /** Give up after this many sweeps. 50 is far beyond the ~6–10 quadratic convergence needs. */
  readonly maxSweeps?: number;
}

/**
 * Eigendecompose the symmetric `n`×`n` matrix `a` (row-major, `a[i*n + j]`).
 *
 * Only the upper triangle is read — a caller holding a matrix that is symmetric *in exact
 * arithmetic* but has picked up rounding asymmetry (as `MMᵀ` accumulated in a different order
 * does) gets a well-defined answer rather than a silently averaged one. Use `symmetrise` first if
 * you want the average instead.
 *
 * Eigenvector **signs** are canonicalised: the component of largest magnitude is made positive.
 * The sign of an eigenvector is mathematically arbitrary, so without a convention the same data
 * re-analysed flips the colours on a mode map for no reason. Ties (exactly equal magnitudes)
 * resolve to the lowest index, so the rule is deterministic.
 */
export function eigenSym(a: ArrayLike<number>, n: number, opts: EigenOptions = {}): EigenResult {
  const maxSweeps = opts.maxSweeps ?? 50;
  if (n <= 0) return { values: new Float64Array(0), vectors: new Float64Array(0), sweeps: 0, offDiagonal: 0 };

  // Working copy, symmetrised from the upper triangle.
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = a[i * n + j] ?? 0;
      m[i * n + j] = v;
      m[j * n + i] = v;
    }
  }
  // Eigenvector accumulator, starts at the identity.
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  const offNorm = (): number => {
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) s += m[i * n + j]! ** 2;
    return Math.sqrt(2 * s);
  };

  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    if (offNorm() <= 1e-300) break;
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m[p * n + q]!;
        // Skip when the off-diagonal is negligible against the two diagonals it would mix: this is
        // the standard threshold, and it is what stops the rotation angle being computed from noise.
        const app = m[p * n + p]!;
        const aqq = m[q * n + q]!;
        if (Math.abs(apq) <= 1e-18 * (Math.abs(app) + Math.abs(aqq))) continue;
        rotated = true;

        // Rotation that zeroes (p,q). `t` is the smaller root, which keeps the rotation angle
        // under 45° and the accumulated basis well-conditioned.
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const mkp = m[k * n + p]!;
          const mkq = m[k * n + q]!;
          m[k * n + p] = c * mkp - s * mkq;
          m[k * n + q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p * n + k]!;
          const mqk = m[q * n + k]!;
          m[p * n + k] = c * mpk - s * mqk;
          m[q * n + k] = s * mpk + c * mqk;
        }
        m[p * n + q] = 0;
        m[q * n + p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p]!;
          const vkq = v[k * n + q]!;
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    if (!rotated) {
      sweeps++;
      break;
    }
  }

  // Sort descending, and transpose the accumulator into mode-major order as we go.
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => m[y * n + y]! - m[x * n + x]!);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    const src = order[k]!;
    values[k] = m[src * n + src]!;
    let big = 0;
    let bigAt = 0;
    for (let i = 0; i < n; i++) {
      const c = v[i * n + src]!;
      vectors[k * n + i] = c;
      if (Math.abs(c) > big) {
        big = Math.abs(c);
        bigAt = i;
      }
    }
    if (vectors[k * n + bigAt]! < 0) for (let i = 0; i < n; i++) vectors[k * n + i] = -vectors[k * n + i]!;
  }

  let maxAbs = 0;
  for (let k = 0; k < n; k++) maxAbs = Math.max(maxAbs, Math.abs(values[k]!));
  return { values, vectors, sweeps, offDiagonal: maxAbs > 0 ? offNorm() / maxAbs : offNorm() };
}

/** `(A + Aᵀ)/2` — the nearest symmetric matrix in Frobenius norm. Note this does **not** make a
 *  matrix positive semi-definite; symmetry and definiteness are different properties, and
 *  conflating them is the trap `gram.ts` is built to avoid. */
export function symmetrise(a: ArrayLike<number>, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i * n + j] = ((a[i * n + j] ?? 0) + (a[j * n + i] ?? 0)) / 2;
  }
  return out;
}

/**
 * How far a symmetric matrix is from positive semi-definite, as a fraction of its spectral radius:
 * `−λ_min / max|λ|`, clamped at 0. Zero means PSD.
 *
 * This is the number that decides whether an eigen-*projection* means anything. A PSD matrix's
 * eigenvalues are variances — non-negative, summing to a total that each mode takes a share of.
 * An indefinite one has modes with negative "variance", which have no such reading, and any
 * "% variance explained" computed from them is arithmetic without a referent.
 */
export function psdDefect(values: ArrayLike<number>, n: number): number {
  let min = Infinity;
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    const x = values[i] ?? 0;
    if (x < min) min = x;
    maxAbs = Math.max(maxAbs, Math.abs(x));
  }
  if (maxAbs === 0) return 0;
  return Math.max(0, -min / maxAbs);
}
