// A bootstrap confidence interval for g_AB(r) — the band the published charts draw.
//
// ## It is not the envelope, and the difference is the whole point
//
// `pcfEnvelope.ts` answers "could a world with no association have produced this curve?" — it
// simulates a NULL, so its band sits where the null lives (around 1 for the shift null) and the
// curve either escapes it or does not. This answers a different question: "how well do we know
// g(r)?" It resamples the DATA, so its band straddles the observed curve and follows it up and down.
//
// Both are two-sided shaded regions around a line and they are constantly confused, so they must be
// drawn differently and read differently. A CI that excludes 1 says the estimate is far from CSR
// with the precision we have; an envelope the curve exits says the arrangement is unlikely under the
// null. Neither implies the other: a huge sample gives a tight CI around g = 1.02 while the envelope
// says nothing is going on, and a tiny sample can escape an envelope with a CI so wide it is useless.
//
// ## Why it costs almost nothing
//
// g(r_k) is a MEAN OVER ANCHORS:
//
//     g(r_k) = (1/N_A) Σ_a  [ Σ_b I[bin] / (ρ_B · A_{r_k}(x_a)) ]
//                            \_______________________________/
//                                    c_a(r_k), one anchor's contribution
//
// so the pair search runs once, producing `c_a(r_k)` for every anchor, and a resample is then just a
// mean over N_A draws from an array that is already in memory. No neighbour search is repeated. That
// is the difference between this and the envelope, where each simulation MOVES the points and so has
// to search again — and it is why a CI can be offered by default while the envelope is gated.
//
// The resampling unit is the anchor cell, which is the standard choice for a mean-over-anchors
// estimator and the one that treats the A cells as the sample. It conditions on B, so it reports the
// uncertainty from having observed these particular A cells; a scheme resampling both would report
// something wider, and the papers' own resampling scheme is not documented in the project this is
// checked against. So this reproduces the KIND of band those charts carry, at the same α — treat
// close agreement in width as a good sign rather than as parity.

import { annulusAreasInto } from "./edgeCorrection";
import { mulberry32 } from "./kernelAnalysis";
import type { PcfParams } from "./pcf";
import type { CellCloud } from "./tcm";

export interface PcfBootstrapOptions {
  /** Resamples. 999 is cheap here and makes the percentile edges stable. */
  readonly resamples?: number;
  /** Two-sided level; the band is the α/2 and 1−α/2 percentiles. */
  readonly alpha?: number;
  readonly seed?: number;
}

export interface PcfBootstrapResult {
  /** Bin centres, matching `crossPCF`. */
  readonly r: number[];
  /** The observed g_AB(r) — identical to `crossPCF` for the same params. */
  readonly g: Float64Array;
  /** Percentile CI bounds, one per bin. */
  readonly lo: Float64Array;
  readonly hi: Float64Array;
  /** Anchors resampled (= N_A), and how many resamples were taken. */
  readonly nA: number;
  readonly resamples: number;
  readonly alpha: number;
}

/**
 * Observed cross-PCF plus a percentile bootstrap CI over anchor cells.
 *
 * Honours the same `edgeCorrected` / `roiArea` as `crossPCF`, and returns the same `g`, so the band
 * and the line are one estimator.
 */
export function crossPCFBootstrap(a: CellCloud, b: CellCloud, p: PcfParams, opts: PcfBootstrapOptions = {}): PcfBootstrapResult {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max(p.roiArea ?? (maxX - minX) * (maxY - minY), 1e-12);
  const nA = a.xs.length;
  const nB = b.xs.length;
  const B = p.nBins;
  const dr = p.rMax / B;
  const rMax2 = p.rMax * p.rMax;
  const rhoB = nB / roiArea;

  const r: number[] = [];
  for (let k = 0; k < B; k++) r.push((k + 0.5) * dr);
  const g = new Float64Array(B);
  const lo = new Float64Array(B);
  const hi = new Float64Array(B);
  const resamples = Math.max(1, opts.resamples ?? 999);
  const alpha = Math.min(0.5, Math.max(1e-6, opts.alpha ?? 0.05));
  if (nA === 0 || nB === 0) return { r, g, lo, hi, nA, resamples, alpha };

  // ---- one pass: per-anchor contributions --------------------------------------------------------
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
  const cols = Math.max(1, Math.ceil((bMaxX - bMinX) / cell) + 1);
  const rows = Math.max(1, Math.ceil((bMaxY - bMinY) / cell) + 1);
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - bMinX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - bMinY) / cell)));
  for (let j = 0; j < nB; j++) buckets[rowOf(b.ys[j]!) * cols + colOf(b.xs[j]!)]!.push(j);

  const invFull = new Float64Array(B);
  for (let k = 0; k < B; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    invFull[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const areaScratch = new Float64Array(B);
  const invAnchor = new Float64Array(B);
  const contrib = new Float64Array(nA * B); // c_a(r_k), already divided by ρ_B

  for (let i = 0; i < nA; i++) {
    const ax = a.xs[i]!;
    const ay = a.ys[i]!;
    let inv = invFull;
    if (p.edgeCorrected) {
      const interior = ax - minX >= p.rMax && maxX - ax >= p.rMax && ay - minY >= p.rMax && maxY - ay >= p.rMax;
      if (!interior) {
        annulusAreasInto(areaScratch, ax, ay, dr, B, minX, minY, maxX, maxY);
        for (let k = 0; k < B; k++) invAnchor[k] = areaScratch[k]! > 0 ? 1 / areaScratch[k]! : 0;
        inv = invAnchor;
      }
    }
    const base = i * B;
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
          const k = Math.min(B - 1, Math.floor(Math.sqrt(d2) / dr));
          contrib[base + k] = contrib[base + k]! + inv[k]! / rhoB;
        }
      }
    }
  }

  for (let i = 0; i < nA; i++) for (let k = 0; k < B; k++) g[k] = g[k]! + contrib[i * B + k]!;
  for (let k = 0; k < B; k++) g[k] = g[k]! / nA;

  // ---- resample the anchors ----------------------------------------------------------------------
  const rnd = mulberry32(opts.seed ?? 0xb0075);
  const draws = new Float64Array(resamples * B);
  const acc = new Float64Array(B);
  for (let s = 0; s < resamples; s++) {
    acc.fill(0);
    for (let d = 0; d < nA; d++) {
      const base = ((rnd() * nA) | 0) * B;
      for (let k = 0; k < B; k++) acc[k] = acc[k]! + contrib[base + k]!;
    }
    const at = s * B;
    for (let k = 0; k < B; k++) draws[at + k] = acc[k]! / nA;
  }

  // Percentiles per bin. Sorting a column at a time keeps this a single small allocation.
  const column = new Float64Array(resamples);
  const loIdx = Math.max(0, Math.min(resamples - 1, Math.floor((alpha / 2) * (resamples - 1))));
  const hiIdx = Math.max(0, Math.min(resamples - 1, Math.ceil((1 - alpha / 2) * (resamples - 1))));
  for (let k = 0; k < B; k++) {
    for (let s = 0; s < resamples; s++) column[s] = draws[s * B + k]!;
    column.sort();
    lo[k] = column[loIdx]!;
    hi[k] = column[hiIdx]!;
  }
  return { r, g, lo, hi, nA, resamples, alpha };
}
