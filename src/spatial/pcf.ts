// Cross pair correlation function g_AB(r) — MuSpAn / SpOOx, Mode 1 (global ρ_B, edge correction
// deferred). Source: docs/natureCovid-TCMetc-s41467-023-42421-0.pdf eq (8); see
// docs/muspan-cell-stats-plan.md §1.
//
//   g(r_k) = (1/N_A) · Σ_a Σ_b  I[r_k ≤ |x_a−x_b| < r_{k+1}] / (ρ_B · A_{r_k}(x_a))
//
// ρ_B is the GLOBAL density of B over the ROI. A_{r_k} is the annulus area (inner r_k, outer
// r_{k+1}); here the FULL annulus (edge correction deferred — exact for anchors ≥r_max inside the
// ROI, biased near the boundary; the Mode-2 viewport-apron path removes that bias). g>1 clustering
// of A around B at range r, g<1 exclusion, g=1 CSR. Radii are in the clouds' own world units.

import type { CellCloud } from "./tcm";

export interface PcfParams {
  /** ROI `[minX, minY, maxX, maxY]` — sets ρ_B = N_B/|ROI|. */
  readonly bbox: readonly [number, number, number, number];
  /** Maximum radius (world units). Bins are equal width `dr = rMax / nBins`. */
  readonly rMax: number;
  /** Number of radial bins. */
  readonly nBins: number;
}

export interface PcfResult {
  /** Bin centres (world units). */
  readonly r: number[];
  /** g(r_k). */
  readonly g: number[];
  /** Raw A–B pair counts per bin (for the HUD / diagnostics). */
  readonly counts: number[];
}

export function crossPCF(a: CellCloud, b: CellCloud, p: PcfParams): PcfResult {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const rhoB = b.xs.length / roiArea; // global density of B (Mode 1)
  const dr = p.rMax / p.nBins;
  const rMax2 = p.rMax * p.rMax;
  const nA = a.xs.length;
  const nB = b.xs.length;
  const counts = new Float64Array(p.nBins);

  // Bucket grid over B, cell size = rMax: every B within rMax of an anchor lies in the anchor's 3×3
  // bucket neighbourhood (same argument as computeTcm), so the exact per-pair distance test runs
  // over only those candidates.
  let bMinX = Infinity;
  let bMinY = Infinity;
  let bMaxX = -Infinity;
  let bMaxY = -Infinity;
  for (let j = 0; j < nB; j++) {
    const x = b.xs[j]!;
    const y = b.ys[j]!;
    if (x < bMinX) bMinX = x;
    if (x > bMaxX) bMaxX = x;
    if (y < bMinY) bMinY = y;
    if (y > bMaxY) bMaxY = y;
  }
  const cell = Math.max(p.rMax, 1e-9);
  const cols = nB > 0 ? Math.max(1, Math.ceil((bMaxX - bMinX) / cell) + 1) : 1;
  const rows = nB > 0 ? Math.max(1, Math.ceil((bMaxY - bMinY) / cell) + 1) : 1;
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - bMinX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - bMinY) / cell)));
  for (let j = 0; j < nB; j++) buckets[rowOf(b.ys[j]!) * cols + colOf(b.xs[j]!)]!.push(j);

  for (let i = 0; i < nA; i++) {
    const ax = a.xs[i]!;
    const ay = a.ys[i]!;
    const c0 = colOf(ax);
    const r0 = rowOf(ay);
    for (let dRow = -1; dRow <= 1; dRow++) {
      const rr = r0 + dRow;
      if (rr < 0 || rr >= rows) continue;
      for (let dCol = -1; dCol <= 1; dCol++) {
        const cc = c0 + dCol;
        if (cc < 0 || cc >= cols) continue;
        for (const j of buckets[rr * cols + cc]!) {
          const dx = b.xs[j]! - ax;
          const dy = b.ys[j]! - ay;
          const d2 = dx * dx + dy * dy;
          if (d2 >= rMax2) continue;
          const bin = Math.min(p.nBins - 1, Math.floor(Math.sqrt(d2) / dr));
          counts[bin]! += 1;
        }
      }
    }
  }

  const r: number[] = [];
  const g: number[] = [];
  const out: number[] = [];
  for (let k = 0; k < p.nBins; k++) {
    const r0 = k * dr;
    const r1 = (k + 1) * dr;
    const annulus = Math.PI * (r1 * r1 - r0 * r0);
    const expected = nA * rhoB * annulus; // Σ_a of (ρ_B · annulus) under CSR
    r.push((r0 + r1) / 2);
    g.push(expected > 0 ? counts[k]! / expected : 0);
    out.push(counts[k]!);
  }
  return { r, g, counts: out };
}
