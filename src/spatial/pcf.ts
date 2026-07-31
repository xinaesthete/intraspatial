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
//
// The deferral above applies to `crossPCF` and `crossPCFMatrix` only. `crossPCFMatrixBinned` (below)
// takes `edgeCorrected`, which applies the real per-anchor `A_{r_k}(x_a)` from `edgeCorrection.ts`;
// that is what reproduces published numbers on a FIXED ROI, where the uncorrected estimator is low by
// an amount set by the ROI's perimeter-to-area ratio rather than by anything biological. See the
// step-3 validation block in docs/muspan-cell-stats-plan.md §6.

import { annulusAreasInto } from "./edgeCorrection";
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

export interface PcfMatrixBinnedParams {
  /** ROI `[minX, minY, maxX, maxY]`. Also the clipping rectangle when `edgeCorrected`. */
  readonly bbox: readonly [number, number, number, number];
  /** Maximum radius; bins are equal width `dr = rMax / nBins`. */
  readonly rMax: number;
  readonly nBins: number;
  /** Length of the type axis. Type ids must lie in `[0, nTypes)`. Defaults to `max(typeId) + 1`.
   *
   *  Pass it whenever the axis is defined elsewhere — comparing against a precomputed table means
   *  every one of its cell types must occupy its own row and column even in an ROI where that type
   *  has no cells, or the matrix silently shifts. */
  readonly nTypes?: number;
  /** Area used for `ρ_B = N_B / area`. Defaults to the bbox area; pass the ROI's own recorded area
   *  when the source defines one (MDV carries `roi_area`, which is not always the bbox). */
  readonly roiArea?: number;
  /** Clip each anchor's annulus to `bbox` — the paper's `A_{r_k}(x_a)` (eq 8). Default `false`,
   *  which is the full annulus: exact for anchors ≥`rMax` inside the ROI, low near the boundary. */
  readonly edgeCorrected?: boolean;
}

export interface PcfMatrixBinnedResult {
  readonly nTypes: number;
  readonly nBins: number;
  /** Cell count per type id. */
  readonly counts: number[];
  /** Bin centres (world units). */
  readonly r: number[];
  /** `g[a·K·B + b·B + k]` = g_{a→b}(r_k). Asymmetric in (a,b). */
  readonly g: Float64Array;
  /** Raw ordered pair counts, same layout — the diagnostic that says whether a `g` rests on 3
   *  pairs or 3000. */
  readonly pairs: Float64Array;
}

/**
 * N-way cross-PCF **binned in r**: g_{a→b}(r_k) for every ordered type pair and every annulus, in
 * one bucket-grid pass. This is `crossPCFMatrix` (a single cumulative disk) crossed with `crossPCF`
 * (annulus bins for one pair) — the `[type_a][type_b][r_bin]` 3-D histogram of
 * docs/muspan-cell-stats-plan.md §7, and the shape a published g(r=20) table is read out of.
 *
 * With `edgeCorrected`, each pair is weighted by `1/A_{r_k}(x_a)` for its own anchor rather than by
 * a per-bin constant, which is what eq (8) actually says and what closes the gap to the published
 * numbers near an ROI boundary. Anchors further than `rMax` from every edge take a fast path — their
 * annuli are uncut, so the per-anchor geometry is skipped for the bulk of a large ROI.
 */
export function crossPCFMatrixBinned(cells: LabelledCells, p: PcfMatrixBinnedParams): PcfMatrixBinnedResult {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max(p.roiArea ?? (maxX - minX) * (maxY - minY), 1e-12);
  const n = cells.xs.length;
  const B = p.nBins;
  const dr = p.rMax / B;
  const rMax2 = p.rMax * p.rMax;

  let K = p.nTypes ?? 0;
  if (!p.nTypes) for (let i = 0; i < n; i++) K = Math.max(K, cells.typeId[i]! + 1);
  const nPer = new Array<number>(K).fill(0);
  for (let i = 0; i < n; i++) {
    const t = cells.typeId[i]!;
    if (t < 0 || t >= K) throw new Error(`crossPCFMatrixBinned: type id ${t} outside [0, ${K})`);
    nPer[t]!++;
  }

  // Bucket grid over the cells, cell size = rMax, so every in-range neighbour is in the 3×3
  // neighbourhood. Built over the cells' own extent rather than the bbox: an ROI whose recorded
  // rectangle is larger than the data (a padded image) would otherwise allocate empty buckets.
  let gMinX = Infinity;
  let gMinY = Infinity;
  let gMaxX = -Infinity;
  let gMaxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = cells.xs[i]!;
    const y = cells.ys[i]!;
    if (x < gMinX) gMinX = x;
    if (x > gMaxX) gMaxX = x;
    if (y < gMinY) gMinY = y;
    if (y > gMaxY) gMaxY = y;
  }
  const cell = Math.max(p.rMax, 1e-9);
  const cols = n > 0 ? Math.max(1, Math.ceil((gMaxX - gMinX) / cell) + 1) : 1;
  const rows = n > 0 ? Math.max(1, Math.ceil((gMaxY - gMinY) / cell) + 1) : 1;
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - gMinX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - gMinY) / cell)));
  for (let i = 0; i < n; i++) buckets[rowOf(cells.ys[i]!) * cols + colOf(cells.xs[i]!)]!.push(i);

  // Reciprocal annulus areas: the interior (uncut) case once, the per-anchor case into a scratch
  // buffer only for anchors the boundary actually reaches.
  const invFull = new Float64Array(B);
  for (let k = 0; k < B; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    invFull[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const areaScratch = new Float64Array(B);
  const invAnchor = new Float64Array(B);

  const weighted = new Float64Array(K * K * B);
  const pairs = new Float64Array(K * K * B);

  for (let i = 0; i < n; i++) {
    const ax = cells.xs[i]!;
    const ay = cells.ys[i]!;
    const a = cells.typeId[i]!;
    let inv = invFull;
    if (p.edgeCorrected) {
      const interior = ax - minX >= p.rMax && maxX - ax >= p.rMax && ay - minY >= p.rMax && maxY - ay >= p.rMax;
      if (!interior) {
        annulusAreasInto(areaScratch, ax, ay, dr, B, minX, minY, maxX, maxY);
        for (let k = 0; k < B; k++) invAnchor[k] = areaScratch[k]! > 0 ? 1 / areaScratch[k]! : 0;
        inv = invAnchor;
      }
    }
    const rowBase = a * K * B;
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
          const d2 = dx * dx + dy * dy;
          if (d2 >= rMax2) continue;
          const k = Math.min(B - 1, Math.floor(Math.sqrt(d2) / dr));
          const at = rowBase + cells.typeId[j]! * B + k;
          weighted[at]! += inv[k]!;
          pairs[at]! += 1;
        }
      }
    }
  }

  const r: number[] = [];
  for (let k = 0; k < B; k++) r.push((k + 0.5) * dr);
  const g = new Float64Array(K * K * B);
  for (let a = 0; a < K; a++) {
    const nA = nPer[a]!;
    for (let b = 0; b < K; b++) {
      const rhoB = nPer[b]! / roiArea;
      const denom = nA * rhoB;
      if (denom <= 0) continue;
      const base = a * K * B + b * B;
      for (let k = 0; k < B; k++) g[base + k] = weighted[base + k]! / denom;
    }
  }
  return { nTypes: K, nBins: B, counts: nPer, r, g, pairs };
}
