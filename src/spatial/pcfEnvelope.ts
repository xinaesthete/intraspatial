// A global rank envelope for the cross-PCF — the piece that turns g_AB(r) from a picture into a test.
//
// `envelope.ts` builds the envelope from a set of null curves and `permute.ts` argues for simulating
// rather than assuming; this supplies the curves. The published form of this statistic is a static
// chart with pointwise bootstrap CIs, and pointwise bands do not have the coverage they appear to
// (measured in `envelope.ts`: 29% rejection at a nominal 5%). Doing it properly means simulating,
// which is what a precomputed chart cannot afford to redo when you move a slider.
//
// ## Which null — and the one that looks right and is not
//
// The first version of this file offered only **random labelling within A ∪ B**: hold every position
// fixed, shuffle the A/B labels between them keeping `n_A` and `n_B`. That is the textbook
// random-labelling test and it is genuinely the right null for *some* questions, but it is the wrong
// one for "is A associated with B", and the failure is not subtle once measured. Two point patterns
// that are each strongly self-clustered but generated INDEPENDENTLY of one another — no association
// whatsoever, by construction — were rejected **20 times out of 20**, with the observed curve at
// 0.755 in the first bin against a null band of [3.16, 3.31].
//
// The mechanism is worth stating because it is invisible from the picture. A ∪ B at short range is
// dominated by A–A and B–B pairs, because each type clusters with itself. Shuffling labels
// redistributes those self-pairs into A–B pairs, so the null expects far MORE A–B contact than
// independence does, and any two self-clustered types fail it. What the test actually asks is "are A
// and B one interleaved homogeneous population?", and for two distinct cell types the answer is
// trivially no. On real data it rejected 89 of the top 90 ordered pairs, and the band never touched
// the curve — which is what a null being trivially false looks like from the outside.
//
// So the default is the **random toroidal shift** (Lotwick & Silverman, *Methods for analysing
// spatial processes of several types of points*, JRSS-B 44(3):406–413, 1982). A stays where it is; B
// is translated by a uniform random vector and wrapped around the ROI. Each pattern keeps its own
// internal clustering exactly — the shift is a rigid motion — and only their relative positioning is
// destroyed. That is the null that isolates *association*, which is the question g_AB(r) is asked.
//
// Its known cost is the wrap: a cluster split across the boundary becomes two partial clusters, and
// pairs are created across edges that are not neighbours in the tissue. That bias is toward the null
// (it decorrelates), it is the standard accepted price of the construction, and it shrinks as the
// ROI grows relative to `rMax`.
//
// **Random labelling is kept, because it is the right null when A and B are nested or exchangeable
// subsets of one population** — `Fibroblast` vs `Prolif fibroblast`, or a marker-high vs marker-low
// split. There the question really is "given these cells, is the split spatially arranged?", the
// self-clustering is shared rather than a confound, and the labelling null is exactly right. Pick it
// deliberately; it is not the default because the common case is two distinct types.
//
// ## Why the shift null is affordable
//
// A never moves, so its edge-corrected weights `1/A_{r_k}(x_a)` are computed once and reused by
// every simulation — the correction depends only on the anchor. B moves rigidly, so each simulation
// is one ordinary cross-PCF pass against a rebuilt bucket grid, and the estimator applied to the
// shifted pattern is the SAME estimator applied to the observed one. No metric changes between
// observed and null, which is the trap in implementing this via toroidal distances instead.

import { annulusAreasInto } from "./edgeCorrection";
import type { GlobalEnvelope } from "./envelope";
import { globalRankEnvelope } from "./envelope";
import { mulberry32 } from "./kernelAnalysis";
import type { CellCloud } from "./tcm";

/** Which null the envelope is drawn against. See the header — this choice decides what is tested. */
export type PcfNullModel =
  /** Random toroidal shift of B. Preserves each pattern's own clustering; tests ASSOCIATION. */
  | "shift"
  /** Random labelling within A ∪ B. Tests whether the A/B split of one population is arranged;
   *  correct for nested or exchangeable subsets, wrong for two distinct self-clustered types. */
  | "label";

export interface PcfEnvelopeParams {
  /** ROI `[minX, minY, maxX, maxY]`. The clipping rectangle when `edgeCorrected`, and the torus the
   *  shift null wraps around. */
  readonly bbox: readonly [number, number, number, number];
  readonly rMax: number;
  readonly nBins: number;
  /** Area for ρ_B. Defaults to the bbox area. */
  readonly roiArea?: number;
  /** Clip each anchor's annulus to the bbox (eq 8). Matches `crossPCFMatrixBinned`. */
  readonly edgeCorrected?: boolean;
  /** Default `"shift"`. */
  readonly nullModel?: PcfNullModel;
  /** Null curves to simulate. 199 gives a smallest attainable p of 0.005 at α=0.05. */
  readonly simulations?: number;
  readonly alpha?: number;
  readonly seed?: number;
  /** `"label"` only: refuse rather than hang when the neighbour structure would be this large. */
  readonly maxPairs?: number;
}

export interface PcfEnvelopeResult {
  /** Bin centres. */
  readonly r: number[];
  /** The observed g_AB(r) — identical to what `crossPCF` / `crossPCFMatrixBinned` produce for the
   *  same inputs, so the curve the band is drawn around is the curve the rest of the UI shows. */
  readonly observed: Float64Array;
  readonly envelope: GlobalEnvelope;
  readonly nullModel: PcfNullModel;
  /** `"label"`: unordered pairs in A ∪ B rescanned per simulation. `"shift"`: A–B pairs found in the
   *  observed pass, as an indication of the per-simulation cost. */
  readonly pairs: number;
  readonly nA: number;
  readonly nB: number;
  /** Milliseconds spent simulating. */
  readonly simulateMs: number;
}

const DEFAULT_MAX_PAIRS = 40_000_000;

/** Reciprocal clipped annulus areas for every anchor, `[i*nBins + k]`. The anchor never moves under
 *  either null, so this is computed once and shared by every simulation. */
function anchorWeights(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  n: number,
  p: PcfEnvelopeParams,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): Float64Array {
  const B = p.nBins;
  const dr = p.rMax / B;
  const inv = new Float64Array(n * B);
  const full = new Float64Array(B);
  for (let k = 0; k < B; k++) {
    const r0 = k * dr;
    const r1 = r0 + dr;
    full[k] = 1 / (Math.PI * (r1 * r1 - r0 * r0));
  }
  const scratch = new Float64Array(B);
  for (let i = 0; i < n; i++) {
    const x = xs[i]!;
    const y = ys[i]!;
    const interior = !p.edgeCorrected || (x - minX >= p.rMax && maxX - x >= p.rMax && y - minY >= p.rMax && maxY - y >= p.rMax);
    if (interior) {
      inv.set(full, i * B);
    } else {
      annulusAreasInto(scratch, x, y, dr, B, minX, minY, maxX, maxY);
      for (let k = 0; k < B; k++) inv[i * B + k] = scratch[k]! > 0 ? 1 / scratch[k]! : 0;
    }
  }
  return inv;
}

/** A flat CSR bucket grid, rebuilt in place each simulation. Array-of-arrays would allocate `cols*rows`
 *  arrays per simulation and dominate the cost; this reuses three typed arrays. */
class BucketGrid {
  readonly start: Int32Array;
  readonly items: Int32Array;
  private readonly counts: Int32Array;
  constructor(
    readonly cols: number,
    readonly rows: number,
    n: number,
  ) {
    this.start = new Int32Array(cols * rows + 1);
    this.counts = new Int32Array(cols * rows);
    this.items = new Int32Array(n);
  }
  build(cellOf: (i: number) => number, n: number): void {
    this.counts.fill(0);
    for (let i = 0; i < n; i++) this.counts[cellOf(i)]!++;
    let acc = 0;
    for (let c = 0; c < this.counts.length; c++) {
      this.start[c] = acc;
      acc += this.counts[c]!;
    }
    this.start[this.counts.length] = acc;
    const cursor = Int32Array.from(this.start.subarray(0, this.counts.length));
    for (let i = 0; i < n; i++) this.items[cursor[cellOf(i)]!++] = i;
  }
}

export function crossPCFEnvelope(a: CellCloud, b: CellCloud, p: PcfEnvelopeParams): PcfEnvelopeResult {
  const nA = a.xs.length;
  const nB = b.xs.length;
  if (nA === 0 || nB === 0) throw new Error("crossPCFEnvelope: both populations must be non-empty");
  const nullModel = p.nullModel ?? "shift";
  if (a.xs === b.xs && a.ys === b.ys) {
    throw new Error("crossPCFEnvelope: A and B are the same population — neither null has content for a self-pair");
  }

  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max(p.roiArea ?? (maxX - minX) * (maxY - minY), 1e-12);
  const B = p.nBins;
  if (B > 256) throw new Error(`crossPCFEnvelope: nBins=${B} exceeds the 256 a byte bin index holds`);
  const dr = p.rMax / B;
  const denom = nA * (nB / roiArea);
  const rnd = mulberry32(p.seed ?? 0x5eed);
  const alpha = p.alpha ?? 0.05;
  const s = p.simulations ?? 199;

  const r: number[] = [];
  for (let k = 0; k < B; k++) r.push((k + 0.5) * dr);

  const result = (observed: Float64Array, simulated: Float64Array[], pairs: number, simulateMs: number): PcfEnvelopeResult => ({
    r,
    observed,
    envelope: globalRankEnvelope(observed, simulated, { alpha }),
    nullModel,
    pairs,
    nA,
    nB,
    simulateMs,
  });

  if (nullModel === "shift") return shiftNull();
  return labelNull();

  // ---- shift null: A fixed, B rigidly translated with wraparound ---------------------------------
  function shiftNull(): PcfEnvelopeResult {
    const W = maxX - minX;
    const H = maxY - minY;
    if (!(W > 0 && H > 0)) throw new Error("crossPCFEnvelope: the shift null needs a bbox with positive extent");
    const invA = anchorWeights(a.xs, a.ys, nA, p, minX, minY, maxX, maxY);

    // B in ROI-local coordinates, so wrapping is a modulo. Points outside the declared ROI are
    // folded in rather than dropped: the bbox is the torus, and a stray point would otherwise shift
    // into a region the grid does not cover.
    const b0x = new Float64Array(nB);
    const b0y = new Float64Array(nB);
    for (let j = 0; j < nB; j++) {
      b0x[j] = (((b.xs[j]! - minX) % W) + W) % W;
      b0y[j] = (((b.ys[j]! - minY) % H) + H) % H;
    }
    const bx = new Float64Array(nB);
    const by = new Float64Array(nB);

    const cols = Math.max(1, Math.ceil(W / p.rMax));
    const rows = Math.max(1, Math.ceil(H / p.rMax));
    const cw = W / cols;
    const ch = H / rows;
    const grid = new BucketGrid(cols, rows, nB);
    const cellOf = (j: number) => {
      const c = Math.min(cols - 1, Math.floor(bx[j]! / cw));
      const rr = Math.min(rows - 1, Math.floor(by[j]! / ch));
      return rr * cols + c;
    };
    // Rings needed to cover rMax at this cell size, and whether the grid is small enough that a ring
    // wraps onto itself (then every bucket is a candidate and wrapping would double-count).
    const ringC = Math.min(Math.floor(cols / 2), Math.ceil(p.rMax / cw));
    const ringR = Math.min(Math.floor(rows / 2), Math.ceil(p.rMax / ch));
    const rMax2 = p.rMax * p.rMax;

    const pass = (vx: number, vy: number, out: Float64Array): number => {
      out.fill(0);
      for (let j = 0; j < nB; j++) {
        bx[j] = b0x[j]! + vx >= W ? b0x[j]! + vx - W : b0x[j]! + vx;
        by[j] = b0y[j]! + vy >= H ? b0y[j]! + vy - H : b0y[j]! + vy;
      }
      grid.build(cellOf, nB);
      let found = 0;
      for (let i = 0; i < nA; i++) {
        const ax = a.xs[i]! - minX;
        const ay = a.ys[i]! - minY;
        const c0 = Math.min(cols - 1, Math.max(0, Math.floor(ax / cw)));
        const r0 = Math.min(rows - 1, Math.max(0, Math.floor(ay / ch)));
        const base = i * B;
        for (let dRow = -ringR; dRow <= ringR; dRow++) {
          const rr = (((r0 + dRow) % rows) + rows) % rows;
          for (let dCol = -ringC; dCol <= ringC; dCol++) {
            const cc = (((c0 + dCol) % cols) + cols) % cols;
            const cellIdx = rr * cols + cc;
            const end = grid.start[cellIdx + 1]!;
            for (let e = grid.start[cellIdx]!; e < end; e++) {
              const j = grid.items[e]!;
              // Plain Euclidean distance on the SHIFTED pattern — B' is an ordinary point pattern in
              // the ROI, so the observed estimator applies to it unchanged. (A toroidal metric here
              // instead would silently make the null use a different estimator from the observed.)
              const ddx = bx[j]! - ax;
              const ddy = by[j]! - ay;
              const d2 = ddx * ddx + ddy * ddy;
              if (d2 >= rMax2) continue;
              const k = Math.min(B - 1, (Math.sqrt(d2) / dr) | 0);
              out[k] = out[k]! + invA[base + k]!;
              found++;
            }
          }
        }
      }
      for (let k = 0; k < B; k++) out[k] = denom > 0 ? out[k]! / denom : 0;
      return found;
    };

    const observed = new Float64Array(B);
    const pairs = pass(0, 0, observed);
    const simulated: Float64Array[] = [];
    const t0 = performance.now();
    for (let sim = 0; sim < s; sim++) {
      const out = new Float64Array(B);
      pass(rnd() * W, rnd() * H, out);
      simulated.push(out);
    }
    return result(observed, simulated, pairs, performance.now() - t0);
  }

  // ---- label null: positions fixed, A/B shuffled within the union --------------------------------
  function labelNull(): PcfEnvelopeResult {
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
    // Any union point can become the anchor under a relabelling, so weights cover the union.
    const invU = anchorWeights(xs, ys, m, p, minX, minY, maxX, maxY);

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
    const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - gMinX) / cell)));
    const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - gMinY) / cell)));
    const grid = new BucketGrid(cols, rows, m);
    grid.build((i) => rowOf(ys[i]!) * cols + colOf(xs[i]!), m);
    const rMax2 = p.rMax * p.rMax;

    // Under this null the positions never move, so which pairs are within r — and in which bin — is
    // the same for every simulation. Built once; each relabelling is a linear scan of it.
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
            const idx = rr * cols + cc;
            const end = grid.start[idx + 1]!;
            for (let e = grid.start[idx]!; e < end; e++) {
              const j = grid.items[e]!;
              if (j <= i) continue; // unordered, each pair once
              const ddx = xs[j]! - x;
              const ddy = ys[j]! - y;
              const d2 = ddx * ddx + ddy * ddy;
              if (d2 >= rMax2) continue;
              visit(i, j, Math.min(B - 1, (Math.sqrt(d2) / dr) | 0));
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
    let w = 0;
    walk((i, j, bin) => {
      pi[w] = i;
      pj[w] = j;
      pb[w] = bin;
      w++;
    });

    const curve = (label: Uint8Array, out: Float64Array): Float64Array => {
      out.fill(0);
      for (let e = 0; e < nPairs; e++) {
        const i = pi[e]!;
        const j = pj[e]!;
        const li = label[i]!;
        const lj = label[j]!;
        // Labels are binary over the union (1 = A, 0 = B), so a pair contributes exactly when the
        // two differ, and whichever endpoint is A is the anchor whose weight applies. Both
        // directions are covered without storing the pair twice.
        if (li === lj) continue;
        const k = pb[e]!;
        const anchor = li === 1 ? i : j;
        out[k] = out[k]! + invU[anchor * B + k]!;
      }
      for (let k = 0; k < B; k++) out[k] = denom > 0 ? out[k]! / denom : 0;
      return out;
    };

    const observedLabel = new Uint8Array(m);
    observedLabel.fill(1, 0, nA); // A first, B after — matches how the union was built
    const observed = curve(observedLabel, new Float64Array(B));

    const perm = new Uint8Array(m);
    const simulated: Float64Array[] = [];
    const t0 = performance.now();
    for (let sim = 0; sim < s; sim++) {
      perm.set(observedLabel);
      for (let i = m - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = t;
      }
      simulated.push(curve(perm, new Float64Array(B)));
    }
    return result(observed, simulated, nPairs, performance.now() - t0);
  }
}
