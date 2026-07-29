// The Fourier side of `kernels.ts` — and the reason the Gram form of the N-way matrix is not
// merely a faster way to compute it.
//
// Write the N-way association matrix in operator form. With `R` the K×P matrix of per-channel
// rasters and `K` the convolution operator of the mark kernel,
//
//     C = R K Rᵀ                                   (the "direct" form)
//
// C is symmetric for any radially symmetric kernel, but symmetry is not definiteness. C is
// positive semi-definite **iff the kernel is positive-definite**, and by Bochner's theorem a
// kernel is positive-definite iff its Fourier transform is non-negative everywhere. So the
// question "can I eigendecompose the association matrix and read the eigenvalues as variances?"
// reduces to a property of the kernel alone, computable in closed form and independent of any
// dataset.
//
// For the 2-D radially symmetric case the transform is the order-0 Hankel transform,
//
//     K̂(k) = 2π ∫₀^∞ K(u) J₀(k u) u du
//
// and for the polynomial family K_n(u) = (n+1)/(πr²)(1 − u²/r²)ⁿ it has the closed form
// (Gradshteyn & Ryzhik 6.567.1), with z = k·r and K̂(0) = 1 since every kernel carries unit mass:
//
//     K̂_n(z) = (n+1) · 2^(n+1) · n! · J_{n+1}(z) / z^(n+1)
//
// J_{n+1} oscillates about zero, so **every member of the polynomial family fails** — including
// the smooth ones. What smoothness buys is not positivity but a smaller violation: the first
// negative lobe both moves outward and is suppressed by the extra powers of z. The measured
// ladder (pinned in `kernelSpectrum.test.ts`) runs −13.2%, −5.9%, −2.9%, −1.6% of the DC value
// for orders 0–3, and the truncated Gaussian at the repo's 3σ default reaches −0.13%.
//
// The consequence for `gram.ts` is not "pick a better kernel" — no available kernel is exactly PD
// at a usable truncation. It is that the **symmetric** Gram form `C = M Mᵀ` with `M = J ⊛ R` is
// PSD by construction for *any* J, exactly and even in f32, because it is a Gram matrix of real
// vectors. That form is what the eigen-projection runs on.

import { GAUSS_TRUNC, type KernelSpec } from "./kernels";

/** Ascending series `J_m(x) = Σ_k (−1)^k/(k!(k+m)!) · (x/2)^(2k+m)`. Exact in relative terms for
 *  small `x`; useless for large `x`, where the intermediate terms grow like `e^x/√x` before they
 *  decay and cancellation eats every significant digit. */
function besselSeries(m: number, x: number): number {
  const h = x / 2;
  let term = 1;
  for (let i = 1; i <= m; i++) term *= h / i; // (x/2)^m / m!
  let sum = term;
  for (let k = 1; k < 60; k++) {
    term *= (-h * h) / (k * (k + m));
    sum += term;
    if (Math.abs(term) < 1e-18 * Math.abs(sum)) break;
  }
  return sum;
}

// Hart/Numerical-Recipes rational approximations to the large-argument forms of J₀ and J₁,
// accurate to a few times 1e-8 absolute. Used only above SERIES_MAX, where that is far more
// accuracy than the caller can use (see `besselJ`).
const TWO_OVER_PI = 0.636619772;

function besselJ0Large(x: number): number {
  const z = 8 / x;
  const y = z * z;
  const xx = x - 0.785398164;
  const p1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  const p2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934935152e-7)));
  return Math.sqrt(TWO_OVER_PI / x) * (Math.cos(xx) * p1 - z * Math.sin(xx) * p2);
}

function besselJ1Large(x: number): number {
  const z = 8 / x;
  const y = z * z;
  const xx = x - 2.356194491;
  const p1 = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
  const p2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  return Math.sqrt(TWO_OVER_PI / x) * (Math.cos(xx) * p1 - z * Math.sin(xx) * p2);
}

/** Below this the ascending series is used; above it, the asymptotic forms plus recurrence. The
 *  split is at 12 because `kernelHankel` divides `J_{n+1}(z)` by `z^(n+1)`, which *amplifies*
 *  absolute error at small `z` (by up to 4·16·6/12⁴ ≈ 2e−2 at the boundary, and unboundedly below
 *  it) and *suppresses* it above — so the series must own precisely the region where the division
 *  is dangerous. Upward recurrence is also unstable for `m ≳ x`, and 12 > 4 keeps it safe. */
const SERIES_MAX = 12;

/**
 * Bessel function of the first kind, integer order `m ∈ [0, 4]`.
 *
 * Two regimes, because no single method is good in both: the ascending series is exact for small
 * argument and catastrophically cancelling for large, the asymptotic form is the reverse. Above
 * the split, `J₂..J₄` come from the standard upward recurrence
 * `J_{m+1}(x) = (2m/x)J_m(x) − J_{m−1}(x)`, which is stable while `m < x`.
 */
export function besselJ(m: number, x: number): number {
  const ax = Math.abs(x);
  const sign = x < 0 && m % 2 === 1 ? -1 : 1; // J_m(−x) = (−1)^m J_m(x)
  if (ax === 0) return m === 0 ? 1 : 0;
  if (ax <= SERIES_MAX) return sign * besselSeries(m, ax);
  let prev = besselJ0Large(ax);
  if (m === 0) return sign * prev;
  let cur = besselJ1Large(ax);
  for (let k = 1; k < m; k++) {
    const next = ((2 * k) / ax) * cur - prev;
    prev = cur;
    cur = next;
  }
  return sign * cur;
}

const FACTORIAL = [1, 1, 2, 6, 24];

/**
 * The kernel's 2-D radial Fourier transform at `z = k·r`, normalised so `K̂(0) = 1`.
 *
 * `z` is dimensionless — the kernel's support radius `r` is the only length, so the whole family
 * is described by one curve per member and the radius never enters. Negative values are the
 * finding: they are the frequencies at which the kernel's convolution operator has negative
 * eigenvalues.
 */
export function kernelHankel(k: KernelSpec, z: number): number {
  if (z === 0) return 1;
  if (k.kind === "poly") {
    const n = k.order;
    return ((n + 1) * 2 ** (n + 1) * FACTORIAL[n]! * besselJ(n + 1, z)) / z ** (n + 1);
  }
  // Truncated Gaussian: no closed form once it is cut off, so integrate. With u′ = u/r on [0,1]
  // the profile is exp(−T²u′²/2) for T = r/σ, and the normalisation cancels in the ratio.
  const T = GAUSS_TRUNC;
  const panels = 400; // the integrand is smooth and completes < 7 periods over [0,1] at zMax = 40
  const h = 1 / panels;
  let num = 0;
  let den = 0;
  for (let i = 0; i <= panels; i++) {
    const u = i * h;
    const w = i === 0 || i === panels ? 1 : i % 2 ? 4 : 2;
    const g = Math.exp((-T * T * u * u) / 2);
    num += w * g * besselJ(0, z * u) * u;
    den += w * g * u;
  }
  return num / den;
}

export interface SpectrumProbe {
  /** The most negative value of `K̂(z)/K̂(0)` found. `0` (or above) means positive-definite. */
  readonly min: number;
  /** The `z = k·r` at which `min` occurs. */
  readonly at: number;
  /** `min >= -tolerance`. */
  readonly positiveDefinite: boolean;
}

/**
 * Scan `K̂` for its most negative value — i.e. measure how far the kernel is from
 * positive-definite, as a fraction of its DC value.
 *
 * `zMax = 40` covers the first several lobes of every member of the family; the lobes decay
 * monotonically after the first, so the global minimum is always found well inside it.
 */
export function kernelSpectrumMin(k: KernelSpec, zMax = 40, steps = 2000, tolerance = 1e-5): SpectrumProbe {
  let min = Infinity;
  let at = 0;
  for (let i = 1; i <= steps; i++) {
    const z = (i * zMax) / steps;
    const v = kernelHankel(k, z);
    if (v < min) {
      min = v;
      at = z;
    }
  }
  return { min, at, positiveDefinite: min >= -tolerance };
}
