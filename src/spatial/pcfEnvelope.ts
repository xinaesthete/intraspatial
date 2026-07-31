// A global rank envelope for the cross-PCF — the piece that turns g_AB(r) from a picture into a test.
//
// `envelope.ts` builds the envelope from a set of null curves and `permute.ts` argues for the null;
// this supplies the curves, fast enough that the answer arrives while the pointer is still on the
// cell. That speed is the whole point: the published form of this statistic is a static chart with
// pointwise bootstrap CIs, and pointwise bands do not have the coverage they appear to (measured in
// `envelope.ts`: 29% rejection at a nominal 5%). Doing it properly means simulating, and simulating
// is what a precomputed chart cannot afford to redo when you move the slider.
//
// ## The null: random labelling within A ∪ B
//
// Positions are held fixed and the A/B labels are shuffled between the cells that carry them, with
// `n_A` and `n_B` preserved. This is the classical random-labelling test for a bivariate pattern,
// and it conditions on exactly the right thing — the combined A+B pattern is anatomy (epithelium is
// dense, lumen is empty), so holding it fixed asks the question actually of interest: *given where
// these cells are, is the split into A and B spatially arranged?* CSR would instead reject on every
// pair, having detected that the section has structure. That is `permute.ts`'s argument, applied to
// one pair.
//
// **What it conditions on, stated plainly, because it is a choice.** The union null holds the A+B
// pattern fixed; a null that shuffled the two labels across *all* cells in the ROI would instead
// hold the whole tissue fixed and let A land anywhere a cell of any type sits. That is the more
// general construction — one permutation would drive every pair in the N×N matrix at once — and it
// is not what runs here, for a reason that is practical rather than principled: its neighbour
// structure spans every cell in the ROI (≈50M ordered pairs on the largest covid ROI at r=270 µm)
// and cannot be resimulated at interactive speed. The union null spans only A ∪ B, which is smaller
// by the square of the type fraction.
//
// ## Why it is fast
//
// Under random labelling the *positions never move*, so the neighbour structure — which pairs are
// within r, and in which bin — is identical for every simulation. It is built once and each
// permutation only re-reads labels along it. That turns `s` full O(n·k) neighbour searches into one
// search plus `s` linear scans of an integer array, which is the difference between a progress bar
// and a hover.
//
// Pairs are stored once, unordered (`i < j`), and both directions are tested in the accumulate loop:
// g_AB is asymmetric, so (i∈A, j∈B) and (j∈A, i∈B) are different contributions, but they share a
// distance and therefore a bin.

import { annulusAreasInto } from "./edgeCorrection";
import type { GlobalEnvelope } from "./envelope";
import { globalRankEnvelope } from "./envelope";
import { mulberry32 } from "./kernelAnalysis";
import type { CellCloud } from "./tcm";

export interface PcfEnvelopeParams {
  /** ROI `[minX, minY, maxX, maxY]`. Also the clipping rectangle when `edgeCorrected`. */
  readonly bbox: readonly [number, number, number, number];
  readonly rMax: number;
  readonly nBins: number;
  /** Area for ρ_B. Defaults to the bbox area. */
  readonly roiArea?: number;
  /** Clip each anchor's annulus to the bbox (eq 8). Matches `crossPCFMatrixBinned`. */
  readonly edgeCorrected?: boolean;
  /** Null curves to simulate. 199 gives a smallest attainable p of 0.005 at α=0.05. */
  readonly simulations?: number;
  readonly alpha?: number;
  readonly seed?: number;
  /** Refuse rather than hang when the neighbour structure would be this large. The caller gets the
   *  count in the error and can lower `rMax`. */
  readonly maxPairs?: number;
}

export interface PcfEnvelopeResult {
  /** Bin centres. */
  readonly r: number[];
  /** The observed g_AB(r) — identical to what `crossPCF` / `crossPCFMatrixBinned` produce for the
   *  same inputs, so the curve the band is drawn around is the curve the rest of the UI shows. */
  readonly observed: Float64Array;
  readonly envelope: GlobalEnvelope;
  /** Unordered within-range pairs in A ∪ B — the size of the structure being rescanned, and the
   *  honest cost figure to report. */
  readonly pairs: number;
  readonly nA: number;
  readonly nB: number;
  /** Milliseconds spent simulating (excluding the neighbour build). */
  readonly simulateMs: number;
}

const DEFAULT_MAX_PAIRS = 40_000_000;

/**
 * Observed cross-PCF plus a global rank envelope under random labelling within A ∪ B.
 *
 * Throws when A and B are the same population: the union null has no content there (every point
 * carries both labels), so there is no test to run rather than a test that trivially passes.
 */
export function crossPCFEnvelope(a: CellCloud, b: CellCloud, p: PcfEnvelopeParams): PcfEnvelopeResult {
  const nA = a.xs.length;
  const nB = b.xs.length;
  if (nA === 0 || nB === 0) throw new Error("crossPCFEnvelope: both populations must be non-empty");
  if (a.xs === b.xs && a.ys === b.ys) {
    throw new Error("crossPCFEnvelope: A and B are the same population — random labelling within A ∪ B has no content for a self-pair");
  }

  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max(p.roiArea ?? (maxX - minX) * (maxY - minY), 1e-12);
  const B = p.nBins;
  const dr = p.rMax / B;
  const rMax2 = p.rMax * p.rMax;

  // Union point set: A first, then B. `isA` is the observed labelling; a permutation reshuffles it.
  const m = nA + nB;
  const xs = new Float64Array(m);
  const ys = new Float64Array(m);
  for (let i = 0; i < nA; i++) {
    xs[i] = a.xs[i]!;
    ys[i] = a.ys[i]!;
  }
  for (let j = 0; j < nB; j++) {
    xs[nA + j] = b.xs[j]!;
    ys[nA + j] = b.ys[j]!;
  }

  // Per-point reciprocal annulus areas. Constant for an interior point; clipped for one the boundary
  // reaches. Computed for the UNION, because a permutation can make any union point an anchor.
  const invArea = new Float64Array(m * B);
  const full = new Float64Array(B);
  for (let k = 0; k < B; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    full[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const scratch = new Float64Array(B);
  for (let i = 0; i < m; i++) {
    const x = xs[i]!;
    const y = ys[i]!;
    const interior = !p.edgeCorrected || (x - minX >= p.rMax && maxX - x >= p.rMax && y - minY >= p.rMax && maxY - y >= p.rMax);
    if (interior) {
      invArea.set(full, i * B);
    } else {
      annulusAreasInto(scratch, x, y, dr, B, minX, minY, maxX, maxY);
      for (let k = 0; k < B; k++) invArea[i * B + k] = scratch[k]! > 0 ? 1 / scratch[k]! : 0;
    }
  }

  // ---- neighbour structure, built once -----------------------------------------------------------
  let gMinX = Infinity;
  let gMinY = Infinity;
  let gMaxX = -Infinity;
  let gMaxY = -Infinity;
  for (let i = 0; i < m; i++) {
    if (xs[i]! < gMinX) gMinX = xs[i]!;
    if (xs[i]! > gMaxX) gMaxX = xs[i]!;
    if (ys[i]! < gMinY) gMinY = ys[i]!;
    if (ys[i]! > gMaxY) gMaxY = ys[i]!;
  }
  const cell = Math.max(p.rMax, 1e-9);
  const cols = Math.max(1, Math.ceil((gMaxX - gMinX) / cell) + 1);
  const rows = Math.max(1, Math.ceil((gMaxY - gMinY) / cell) + 1);
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - gMinX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - gMinY) / cell)));
  for (let i = 0; i < m; i++) buckets[rowOf(ys[i]!) * cols + colOf(xs[i]!)]!.push(i);

  // Two passes: count, then fill. One growable array of 40M entries reallocates far more than it
  // costs to walk the neighbourhood twice.
  let nPairs = 0;
  const walk = (visit: (i: number, j: number, bin: number) => void) => {
    for (let i = 0; i < m; i++) {
      const x = xs[i]!;
      const y = ys[i]!;
      const c0 = colOf(x);
      const r0 = rowOf(y);
      for (let dRow = -1; dRow <= 1; dRow++) {
        const rr = r0 + dRow;
        if (rr < 0 || rr >= rows) continue;
        for (let dCol = -1; dCol <= 1; dCol++) {
          const cc = c0 + dCol;
          if (cc < 0 || cc >= cols) continue;
          for (const j of buckets[rr * cols + cc]!) {
            if (j <= i) continue; // unordered, each pair once
            const ddx = xs[j]! - x;
            const ddy = ys[j]! - y;
            const d2 = ddx * ddx + ddy * ddy;
            if (d2 >= rMax2) continue;
            visit(i, j, Math.min(B - 1, Math.floor(Math.sqrt(d2) / dr)));
          }
        }
      }
    }
  };
  walk(() => {
    nPairs++;
  });
  const maxPairs = p.maxPairs ?? DEFAULT_MAX_PAIRS;
  if (nPairs > maxPairs) {
    throw new Error(`crossPCFEnvelope: ${nPairs} pairs within rMax=${p.rMax} exceeds maxPairs=${maxPairs} — lower rMax or raise the cap`);
  }
  const pi = new Int32Array(nPairs);
  const pj = new Int32Array(nPairs);
  const pb = new Uint8Array(nPairs);
  if (B > 256) throw new Error(`crossPCFEnvelope: nBins=${B} exceeds the 256 a byte bin index holds`);
  let w = 0;
  walk((i, j, bin) => {
    pi[w] = i;
    pj[w] = j;
    pb[w] = bin;
    w++;
  });

  // ---- accumulate one labelling ------------------------------------------------------------------
  const rhoB = nB / roiArea;
  const denom = nA * rhoB;
  const curve = (label: Uint8Array, out: Float64Array): Float64Array => {
    out.fill(0);
    for (let e = 0; e < nPairs; e++) {
      const i = pi[e]!;
      const j = pj[e]!;
      const li = label[i]!;
      const lj = label[j]!;
      // Labels are binary over the union (1 = A, 0 = B), so a pair contributes exactly when the two
      // differ — same-label pairs are A–A or B–B and are not what g_AB counts. That makes the mixed
      // case unambiguous: whichever endpoint is A is the anchor, and the edge-corrected weight is
      // its own. Both directions are covered without storing the pair twice.
      if (li === lj) continue;
      const k = pb[e]!;
      const anchor = li === 1 ? i : j;
      out[k] = out[k]! + invArea[anchor * B + k]!;
    }
    for (let k = 0; k < B; k++) out[k] = denom > 0 ? out[k]! / denom : 0;
    return out;
  };

  const observedLabel = new Uint8Array(m);
  observedLabel.fill(1, 0, nA); // A first, B after — matches how the union was built
  const observed = curve(observedLabel, new Float64Array(B));

  // ---- simulate ----------------------------------------------------------------------------------
  const s = p.simulations ?? 199;
  const rnd = mulberry32(p.seed ?? 0x5eed);
  const perm = new Uint8Array(m);
  const simulated: Float64Array[] = [];
  const t0 = performance.now();
  for (let sim = 0; sim < s; sim++) {
    // Partial Fisher–Yates directly on the label array: shuffle the observed labelling rather than
    // building an index permutation and gathering through it.
    perm.set(observedLabel);
    for (let i = m - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = perm[i]!;
      perm[i] = perm[j]!;
      perm[j] = t;
    }
    simulated.push(curve(perm, new Float64Array(B)));
  }
  const simulateMs = performance.now() - t0;

  const r: number[] = [];
  for (let k = 0; k < B; k++) r.push((k + 0.5) * dr);
  return {
    r,
    observed,
    envelope: globalRankEnvelope(observed, simulated, { alpha: p.alpha ?? 0.05 }),
    pairs: nPairs,
    nA,
    nB,
    simulateMs,
  };
}
