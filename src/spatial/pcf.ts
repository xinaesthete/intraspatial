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

export interface LabelledCells {
  readonly xs: readonly number[];
  readonly ys: readonly number[];
  /** Per-cell type id, index-aligned to xs/ys. */
  readonly typeId: readonly number[];
}

export interface PcfMatrixParams {
  readonly bbox: readonly [number, number, number, number];
  /** Contact radius (world units): g_AB is the disk cross-PCF over `[0, radius)`. */
  readonly radius: number;
}

export interface PcfMatrixResult {
  /** Sorted unique type ids — the matrix's row/column order. */
  readonly types: number[];
  /** Cell count per type, index-aligned to `types`. */
  readonly counts: number[];
  /** N×N row-major cross-PCF `g[a*N + b] = g_{types[a]→types[b]}(r<radius)`. Asymmetric. */
  readonly g: Float64Array;
}

/** N-way cross-PCF: g_AB(r<radius) for **all** ordered type pairs at once, in a single batched
 *  bucket-grid pass over every cell (self-pairs excluded), instead of N² separate `crossPCF` calls.
 *  This is the paper's cell-type association matrix (its Fig-4 network is a threshold of it); it is
 *  also the natural producer of an open-axis "pair" tensor (docs/muspan-cell-stats-plan.md §7). */
export function crossPCFMatrix(cells: LabelledCells, p: PcfMatrixParams): PcfMatrixResult {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const n = cells.xs.length;
  const r = p.radius;
  const r2 = r * r;

  // Type id → dense index.
  const types = [...new Set(cells.typeId)].sort((a, b) => a - b);
  const idx = new Map<number, number>();
  types.forEach((t, i) => idx.set(t, i));
  const N = types.length;
  const nPer = new Array<number>(N).fill(0);
  const ti = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const k = idx.get(cells.typeId[i]!)!;
    ti[i] = k;
    nPer[k]!++;
  }

  // Bucket grid over all cells, cell size = radius (in-radius neighbours lie in the 3×3 neighbourhood).
  const cols = Math.max(1, Math.ceil((maxX - minX) / r) + 1);
  const rows = Math.max(1, Math.ceil((maxY - minY) / r) + 1);
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - minX) / r)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / r)));
  for (let i = 0; i < n; i++) buckets[rowOf(cells.ys[i]!) * cols + colOf(cells.xs[i]!)]!.push(i);

  const count = new Float64Array(N * N); // count[a*N + b] = ordered pairs (a-cell, b-cell) within r
  for (let i = 0; i < n; i++) {
    const ax = cells.xs[i]!;
    const ay = cells.ys[i]!;
    const a = ti[i]!;
    const c0 = colOf(ax);
    const r0 = rowOf(ay);
    for (let dRow = -1; dRow <= 1; dRow++) {
      const rr = r0 + dRow;
      if (rr < 0 || rr >= rows) continue;
      for (let dCol = -1; dCol <= 1; dCol++) {
        const cc = c0 + dCol;
        if (cc < 0 || cc >= cols) continue;
        for (const j of buckets[rr * cols + cc]!) {
          if (j === i) continue; // exclude self-pairs
          const dx = cells.xs[j]! - ax;
          const dy = cells.ys[j]! - ay;
          if (dx * dx + dy * dy < r2) count[a * N + ti[j]!]! += 1;
        }
      }
    }
  }

  const diskArea = Math.PI * r2;
  const g = new Float64Array(N * N);
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const rhoB = nPer[b]! / roiArea;
      const expected = nPer[a]! * rhoB * diskArea;
      g[a * N + b] = expected > 0 ? count[a * N + b]! / expected : 0;
    }
  }
  return { types, counts: nPer, g };
}
