// TCM in its kernel-density form — the CONTINUOUS generalisation of `tcm.ts`, and the oracle for
// the GPU render path.
//
// Rewriting eq 9 as a sampled KDE (see kernels.ts) turns the whole statistic into two splats with a
// pointwise nonlinearity between them:
//
//     m_ab(a) = (K_r ⊛ β)(x_a) / ρ_B          — B's local density at the A cell, relative to CSR
//     M_ab(a) = 𝔐(m_ab(a), α)                 — eqs 10–13, pointwise
//     Γ_ab    = G_σ ⊛ (M · α)                 — eq 14: a Gaussian splat of A, weighted per point
//
// With `kernel = TOPHAT` this is EXACTLY the paper: (K_0 ⊛ β)(x_a)/ρ_B = count/(ρ_B·πr²) = eq 9.
// The generalisation is therefore free — no term is dropped, the disk is simply one kernel among
// several, and the smooth ones remove its hard edge (a B cell drifting across |u| = r changes the
// top-hat mark by a full count, and every smoother kernel by less).
//
// Two things stay exact here that the GPU path approximates, which is the point of keeping this:
//   • the B field is evaluated AT x_a, not sampled from a raster;
//   • the arithmetic is f64.
// So this module is the parity anchor between `computeTcmReference` (the paper, exact) and
// `computeTcmRender` (the two-pass GPU render). The Γ splat here uses the render's support — a
// world box of half-extent `radiusSigma·σ` — rather than the reference's cell-quantised ±kr box,
// so that the CPU→GPU link of that chain is tight and the only remaining gap is raster sampling.
//
// What is NOT composable per-pixel, and why this still needs the points: 𝔐 is nonlinear and is
// applied PER CELL before smoothing. G_σ ⊛ (𝔐(m)·α) ≠ (G_σ ⊛ α)·𝔐(m) — the difference is a Jensen
// gap, largest exactly where a neighbourhood mixes clustered and excluded A cells. The per-point
// evaluation is load-bearing; it is just a point sample of a field, not a neighbour search.

import { buildBucketGrid } from "./bucketGrid";
import { type KernelSpec, kernelAt, TOPHAT } from "./kernels";
import { type CellCloud, markToM, type TcmParams } from "./tcm";

export interface TcmKernelParams extends TcmParams {
  /** Mark kernel. Default `TOPHAT` — the paper's hard disk, and the exact eq 9. */
  readonly kernel?: KernelSpec;
  /** Γ Gaussian support half-extent in units of σ (a world box, as the GPU quad is). Default 4. */
  readonly radiusSigma?: number;
}

/** Per-A-cell mark m_ab = (K_r ⊛ β)(x_a) / ρ_B — B's local density at each A cell, in units of the
 *  global density. 1 = CSR, >1 clustered, <1 excluded. Exact (evaluated at x_a, in f64). */
export function kernelMarks(a: CellCloud, b: CellCloud, p: TcmKernelParams): Float64Array {
  const kernel = p.kernel ?? TOPHAT;
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const rhoB = b.xs.length / roiArea;
  const r = p.radius;
  const r2 = r * r;
  const nA = a.xs.length;
  const m = new Float64Array(nA);
  if (rhoB <= 0) return m;

  const grid = buildBucketGrid(b.xs, b.ys, r);
  for (let i = 0; i < nA; i++) {
    const ax = a.xs[i]!;
    const ay = a.ys[i]!;
    const c0 = Math.min(grid.cols - 1, Math.max(0, Math.floor((ax - grid.minX) / grid.cell)));
    const r0 = Math.min(grid.rows - 1, Math.max(0, Math.floor((ay - grid.minY) / grid.cell)));
    let density = 0;
    for (let dRow = -1; dRow <= 1; dRow++) {
      const rr = r0 + dRow;
      if (rr < 0 || rr >= grid.rows) continue;
      for (let dCol = -1; dCol <= 1; dCol++) {
        const cc = c0 + dCol;
        if (cc < 0 || cc >= grid.cols) continue;
        const bucket = rr * grid.cols + cc;
        for (let k = grid.cellOffsets[bucket]!; k < grid.cellOffsets[bucket + 1]!; k++) {
          const j = grid.pointIds[k]!;
          const dx = b.xs[j]! - ax;
          const dy = b.ys[j]! - ay;
          const d2 = dx * dx + dy * dy;
          if (d2 < r2) density += kernelAt(kernel, d2, r);
        }
      }
    }
    m[i] = density / rhoB;
  }
  return m;
}

export interface TcmKernelField {
  /** Row-major `width×height` Γ_ab. Row 0 is at minY (as `computeTcmReference`). */
  readonly gamma: Float32Array;
  /** Per-A-cell transformed mark M_ab ∈ [−1,1] — the per-cell quantity behind the raster. */
  readonly marks: Float32Array;
}

/** Γ_ab and the marks behind it, in one pass. The marks are worth having on their own: they are a
 *  per-cell "is this A cell somewhere B-rich?" score that can be coloured straight onto a scatter,
 *  with no raster in the way. */
export function tcmKernelField(a: CellCloud, b: CellCloud, p: TcmKernelParams): TcmKernelField {
  const [minX, minY, maxX, maxY] = p.bbox;
  const { width: w, height: h, sigma } = p;
  const cw = (maxX - minX) / w;
  const ch = (maxY - minY) / h;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI)); // eq 14's normalisation, verbatim
  const support = (p.radiusSigma ?? 4) * sigma; // world box, matching the GPU quad
  const m = kernelMarks(a, b, p);
  const marks = new Float32Array(m.length);
  const gamma = new Float32Array(w * h);

  for (let idx = 0; idx < a.xs.length; idx++) {
    const M = markToM(m[idx]!, p.alpha);
    marks[idx] = M;
    if (M === 0) continue;
    const ax = a.xs[idx]!;
    const ay = a.ys[idx]!;
    // Cells whose CENTRE lies in the point's world box — the same set the rasteriser fills.
    const i0 = Math.max(0, Math.ceil((ax - support - minX) / cw - 0.5));
    const i1 = Math.min(w - 1, Math.floor((ax + support - minX) / cw - 0.5));
    const j0 = Math.max(0, Math.ceil((ay - support - minY) / ch - 0.5));
    const j1 = Math.min(h - 1, Math.floor((ay + support - minY) / ch - 0.5));
    for (let j = j0; j <= j1; j++) {
      const dy = minY + (j + 0.5) * ch - ay;
      for (let i = i0; i <= i1; i++) {
        const dx = minX + (i + 0.5) * cw - ax;
        gamma[j * w + i]! += M * norm * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      }
    }
  }
  return { gamma, marks };
}

/** Γ_ab(x) alone — the drop-in for `computeTcm` with a choice of mark kernel. */
export function computeTcmKernel(a: CellCloud, b: CellCloud, p: TcmKernelParams): Float32Array {
  return tcmKernelField(a, b, p).gamma;
}
