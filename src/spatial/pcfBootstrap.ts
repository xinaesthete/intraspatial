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
// ## The resampling unit is a TILE, not an anchor
//
// SpOOx's `plotPCFWithBootstrappedConfidenceInterval` implements Loh's spatial block bootstrap, and
// it is the published scheme, so it is the default here. Tile the ROI, sum each tile's anchors and
// their contributions, resample TILES with replacement, and form the ratio Σcontributions /
// Σanchors. Two details of it are easy to get wrong and both are deliberate below: the denominator
// is the number of anchors actually drawn (a ratio estimator, not a renormalisation to N_A), and
// EMPTY tiles stay in the pool.
//
// This matters because the obvious scheme — resample anchor cells independently — is wrong whenever
// the pattern is clustered, which is the case the statistic exists for. Nearby anchors see largely
// the same B cells, so their contributions are correlated; drawing them independently destroys that
// and the interval comes out too narrow. Measured over 300 realisations of a Thomas cluster process,
// at a nominal 95%:
//
//                            coverage          mean width
//     clustered   block        98%              0.484
//                 anchor       40%              0.100
//     independent block        99%              0.109
//                 anchor       93%              0.046
//
// The anchor scheme covers 40% of the time while claiming 95% — its band is five times too narrow
// on exactly the patterns tissue produces. The block scheme is mildly CONSERVATIVE instead, on
// independent points as well (99% at a nominal 95%), because with a 400 µm ROI at 100 µm tiles its
// effective sample size is 16 units however many cells there are. That trade is the right way round:
// conservative when the data are independent, correct when they are clustered, and real tissue is
// clustered. `scheme: "anchor"` is kept so the comparison can be made.
//
// The difference is not confined to synthetic clusters. On COVID_SAMPLE_16_ROI_3, Fibroblast →
// Endothelial (2,821 anchors, a 1998 × 1999 µm ROI, so exactly 400 tiles — SpOOx's own comment says
// "sum down each of the 400 boxes"), the block band is 1.4× wider at r = 15 µm rising to 2.6× by
// r = 95 µm. Every CI this module drew before was that much too narrow on real data.
//
// An earlier version of this file said the published resampling scheme "is not documented in the
// project this is checked against". That was true of the paper and false of the code.

import { annulusAreasInto } from "./edgeCorrection";
import { mulberry32 } from "./kernelAnalysis";
import type { PcfParams } from "./pcf";
import type { CellCloud } from "./tcm";

/**
 * What gets resampled.
 *
 * `"block"` — SpOOx's, and the right one. Loh's spatial block bootstrap: tile the ROI, add up each
 * tile's anchors and their contributions, then resample TILES with replacement and form the ratio
 * Σcontributions / Σanchors. Because whole neighbourhoods move together, the spatial dependence
 * between nearby anchors is carried into the resample instead of being destroyed by it.
 *
 * `"anchor"` — resample anchor cells independently. Kept because it is what this module used to do
 * and because the comparison is the argument: on a clustered pattern, nearby anchors see the same
 * neighbours, so treating them as independent draws understates the variance and the interval is
 * too narrow. `pcfBootstrap.test.ts` measures that undercoverage rather than asserting it.
 */
export type PcfBootstrapScheme = "block" | "anchor";

/**
 * SpOOx's tile size, in µm — **100, not the 20 the paper states.**
 *
 * The methods section says confidence intervals come from "resampling grid sites within a 20μm
 * square lattice"; `utils_alt.py` hard-codes `rectangleWidthX = rectangleWidthY = 100`, with a TODO
 * noting the method becomes unsuitable below roughly 300 µm domains. 100 is what produced the
 * published figures, so it is the default here. This is the second place SpOOx's source contradicts
 * its own methods section — see `quadratCorrelation.ts` for the first.
 */
export const LOH_BLOCK_UM = 100;

export interface PcfBootstrapOptions {
  /** Resamples. 999 is cheap here and matches SpOOx. */
  readonly resamples?: number;
  /** Two-sided level. */
  readonly alpha?: number;
  readonly seed?: number;
  /** Defaults to `"block"`. */
  readonly scheme?: PcfBootstrapScheme;
  /** Tile side for the block scheme, in world units. Defaults to `LOH_BLOCK_UM`. */
  readonly blockSize?: number;
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
  readonly scheme: PcfBootstrapScheme;
  /** Tiles in the resampling pool, INCLUDING empty ones. 0 for the anchor scheme. */
  readonly blocks: number;
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
  const scheme = opts.scheme ?? "block";
  const alpha = Math.min(0.5, Math.max(1e-6, opts.alpha ?? 0.05));
  if (nA === 0 || nB === 0) return { r, g, lo, hi, nA, resamples, alpha, scheme, blocks: 0 };

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

  // ---- resample ------------------------------------------------------------------------------------
  const rnd = mulberry32(opts.seed ?? 0xb0075);
  const draws = new Float64Array(resamples * B);
  const acc = new Float64Array(B);
  let blocks = 0;

  if (scheme === "anchor") {
    for (let s = 0; s < resamples; s++) {
      acc.fill(0);
      for (let d = 0; d < nA; d++) {
        const base = ((rnd() * nA) | 0) * B;
        for (let k = 0; k < B; k++) acc[k] = acc[k]! + contrib[base + k]!;
      }
      const at = s * B;
      for (let k = 0; k < B; k++) draws[at + k] = acc[k]! / nA;
    }
  } else {
    const bs = Math.max(1e-9, opts.blockSize ?? LOH_BLOCK_UM);
    const bCols = Math.max(1, Math.ceil((maxX - minX) / bs));
    const bRows = Math.max(1, Math.ceil((maxY - minY) / bs));
    blocks = bCols * bRows;
    const blockContrib = new Float64Array(blocks * B);
    const blockN = new Float64Array(blocks);
    for (let i = 0; i < nA; i++) {
      const bx = Math.min(bCols - 1, Math.max(0, Math.floor((a.xs[i]! - minX) / bs)));
      const by = Math.min(bRows - 1, Math.max(0, Math.floor((a.ys[i]! - minY) / bs)));
      const blk = by * bCols + bx;
      blockN[blk] = blockN[blk]! + 1;
      const src = i * B;
      const dst = blk * B;
      for (let k = 0; k < B; k++) blockContrib[dst + k] = blockContrib[dst + k]! + contrib[src + k]!;
    }
    for (let s = 0; s < resamples; s++) {
      // A RATIO estimator: the denominator is the number of anchors actually drawn, not N_A. That is
      // what makes the tile the sampling unit — a draw that happens to pick sparse tiles has fewer
      // anchors behind it and says so, instead of being renormalised back to the observed count.
      let n = 0;
      acc.fill(0);
      for (let d = 0; d < blocks; d++) {
        // Empty tiles stay in the pool. They contribute nothing to either sum, so drawing one simply
        // means this resample rests on fewer anchors — real variability in a patchy ROI, and
        // excluding them would quietly narrow the interval.
        const blk = (rnd() * blocks) | 0;
        n += blockN[blk]!;
        const src = blk * B;
        for (let k = 0; k < B; k++) acc[k] = acc[k]! + blockContrib[src + k]!;
      }
      const at = s * B;
      if (n > 0) for (let k = 0; k < B; k++) draws[at + k] = acc[k]! / n;
      else for (let k = 0; k < B; k++) draws[at + k] = g[k]!; // every tile drawn was empty
    }
  }

  // The interval is REFLECTED about the resample mean, as SpOOx does it:
  //     lo = 2·mean − q(1−α/2),  hi = 2·mean − q(α/2)
  // not the plain percentile interval. On a symmetric resample distribution the two coincide; on a
  // skewed one this flips the asymmetry, which is the basic-bootstrap idea applied about the
  // resample mean rather than the observed statistic. Reproducing the published band means
  // reproducing that choice.
  const column = new Float64Array(resamples);
  const loIdx = Math.max(0, Math.min(resamples - 1, Math.floor((alpha / 2) * (resamples - 1))));
  const hiIdx = Math.max(0, Math.min(resamples - 1, Math.ceil((1 - alpha / 2) * (resamples - 1))));
  for (let k = 0; k < B; k++) {
    let mean = 0;
    for (let s = 0; s < resamples; s++) {
      const v = draws[s * B + k]!;
      column[s] = v;
      mean += v;
    }
    mean /= resamples;
    column.sort();
    lo[k] = 2 * mean - column[hiIdx]!;
    hi[k] = 2 * mean - column[loIdx]!;
  }
  return { r, g, lo, hi, nA, resamples, alpha, scheme, blocks };
}
