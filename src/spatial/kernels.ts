// Radial kernels for 2-D density estimation — the shared vocabulary for the cell-stats front.
//
// Once you see the TCM mark as a sampled KDE (docs/muspan-cell-stats-plan.md; eq 9 rewritten),
//
//     m_ab(a) = (K ⊛ β)(x_a) / ρ_B
//
// the paper's hard disk stops being special: it is one member of a family, and the only thing that
// makes it distinguished is that it is the LEAST smooth member. All kernels here carry UNIT MASS
// (∫K = 1), so `K ⊛ β` is a density estimate in the same units as ρ_B and the mark is
// kernel-agnostic — swapping the kernel changes the estimator, never the meaning of m.
//
// The polynomial family is the ladder we measure along:
//
//     K_n(u) = (n+1)/(πr²) · (1 − |u|²/r²)^n     for |u| < r
//
//   n = 0  top-hat / disk   — the paper's kernel. Discontinuous at |u| = r.
//   n = 1  Epanechnikov     — continuous; AMISE-optimal among 2nd-order kernels.
//   n = 2  quartic/biweight — C¹ at the boundary.
//   n = 3  triweight        — C².
//
// so K and its first n−1 derivatives vanish at the support boundary: `order` IS the smoothness.
// Normalisation follows from ∫(1−t)^n over the disk = πr²/(n+1) (substitute t = d²/r²).
//
// A truncated Gaussian (σ = r/3, cut at r) is included as the non-compact comparison. Its
// constants are the truncated ones, not the ideal-Gaussian ones — the truncation is part of the
// estimator, so pretending otherwise would put a systematic error into every moment below.
//
// CAUTION when comparing kernels: equal `r` does NOT mean equal scale. The second moment
// μ₂ = r²/(n+2) shrinks as the kernel smooths, so a triweight at radius r probes a visibly tighter
// neighbourhood than a top-hat at the same r. `equivalentRadius` is the fair-comparison rescaling;
// every kernel-vs-kernel claim should go through it.

/** A radial kernel: the compactly-supported polynomial family, or a truncated Gaussian. */
export type KernelSpec = { readonly kind: "poly"; readonly order: 0 | 1 | 2 | 3 } | { readonly kind: "gaussian" };

export const TOPHAT: KernelSpec = { kind: "poly", order: 0 };
export const EPANECHNIKOV: KernelSpec = { kind: "poly", order: 1 };
export const QUARTIC: KernelSpec = { kind: "poly", order: 2 };
export const TRIWEIGHT: KernelSpec = { kind: "poly", order: 3 };
export const GAUSSIAN: KernelSpec = { kind: "gaussian" };

/** Every kernel, smoothest-last — the order the comparison harness sweeps. */
export const KERNELS: readonly KernelSpec[] = [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT, GAUSSIAN];

/** Truncation point of the Gaussian, in units of σ: the support radius is `r = TRUNC·σ`. */
export const GAUSS_TRUNC = 3;
const T2 = GAUSS_TRUNC * GAUSS_TRUNC; // 9
const E_HALF = Math.exp(-T2 / 2); // e^-4.5 — mass outside the truncation
const E_FULL = Math.exp(-T2); // e^-9   — appears in ∫K²

export function kernelLabel(k: KernelSpec): string {
  if (k.kind === "gaussian") return `gaussian (${GAUSS_TRUNC}σ)`;
  return (["top-hat", "Epanechnikov", "quartic", "triweight"] as const)[k.order];
}

/** Stable identifier — also the shader's kernel selector (`-1` = gaussian). */
export function kernelCode(k: KernelSpec): number {
  return k.kind === "gaussian" ? -1 : k.order;
}

/** K(u) at squared distance `d2` with support radius `r`. Zero outside the support. */
export function kernelAt(k: KernelSpec, d2: number, r: number): number {
  const r2 = r * r;
  const t = d2 / r2;
  if (t >= 1) return 0;
  if (k.kind === "poly") return ((k.order + 1) / (Math.PI * r2)) * (1 - t) ** k.order;
  const s2 = r2 / T2; // σ²
  return Math.exp(-d2 / (2 * s2)) / (2 * Math.PI * s2 * (1 - E_HALF));
}

/** μ₂/r² — the kernel's squared scale as a fraction of its support radius. Dimensionless, so this
 *  is the number to compare kernels by (see `equivalentRadius`). */
export function moment2Coeff(k: KernelSpec): number {
  if (k.kind === "poly") return 1 / (k.order + 2);
  // ∫|u|²K over the truncated disk, with the truncated normalisation carried through.
  return (2 / T2) * ((1 - (1 + T2 / 2) * E_HALF) / (1 - E_HALF));
}

/** μ₂(K) = ∫|u|²K(u)du — the kernel's effective squared scale. */
export function moment2(k: KernelSpec, r: number): number {
  return moment2Coeff(k) * r * r;
}

/** Roughness R(K) = ∫K(u)²du. With μ₂ it determines the AMISE of the density estimate: lower
 *  roughness at matched scale means less variance for the same bias. */
export function roughness(k: KernelSpec, r: number): number {
  const r2 = r * r;
  if (k.kind === "poly") return (k.order + 1) ** 2 / ((2 * k.order + 1) * Math.PI * r2);
  const s2 = r2 / T2;
  return (1 - E_FULL) / (4 * Math.PI * s2 * (1 - E_HALF) ** 2);
}

/** The radius at which `k` has the SAME second moment as a top-hat of radius `baseRadius` — i.e.
 *  the radius at which the two probe the same spatial scale. Comparing kernels at equal `r`
 *  instead of equal μ₂ silently confounds "smoother" with "more local". */
export function equivalentRadius(k: KernelSpec, baseRadius: number): number {
  return baseRadius * Math.sqrt(moment2Coeff(TOPHAT) / moment2Coeff(k));
}
