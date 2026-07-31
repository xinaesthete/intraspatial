// The Quadrat Correlation Matrix (QCM) — the paper's coarse-scale companion to the cross-PCF.
//
// Lay a regular grid over the ROI, count each cell type per quadrat, and correlate the count vectors
// between types. Two types that fill the same quadrats correlate positively; two that avoid each
// other correlate negatively. It answers a deliberately blunter question than g(r) — "do these types
// occupy the same regions", not "how close do they get" — and at one fixed scale, set by the quadrat
// size.
//
// ## This one is reproduced exactly
//
// The covid project's `quadratCounts` column (the name is the config's, not the statistic's) is
// **the Pearson correlation of per-type counts in 100 µm quadrats**, recovered by scanning quadrat
// sizes against the stored values: at 100 µm all 2,401 pairs of an ROI agree to within 1e-3, with a
// median absolute difference of 0.0000. Its sibling `MH_PC` is a different, related quantity — it
// correlates 0.93 with the 25 µm QCM but equals no plain form of it tried (Pearson at any size,
// Spearman, presence/absence, sqrt counts, Morisita-Horn), so it is NOT claimed here.
//
// ## Inference
//
// The correlation on its own is not interpretable across pairs: two abundant types correlate more
// stably than two rare ones, so a raw 0.3 means different things in different rows. The standard
// effect size fixes that by referring each correlation to what the same pair would give under a
// null — labels shuffled between cells, positions held, which is `permute.ts`'s argument applied at
// quadrat scale:
//
//     SES = (r_obs − mean(r_null)) / sd(r_null)
//
// **The null is not centred at zero, and that is the point.** Shuffling labels holds each quadrat's
// TOTAL fixed, so every type inherits the tissue's own density pattern: wherever cells are dense,
// both types are dense, and the null correlation is already strongly positive. Measured on a
// synthetic pair built to be perfectly co-located, r ≈ 0.99 scores SES ≈ 2 — it is barely more
// co-located than sharing the dense quadrats already implies — while a segregated pair scores hugely
// negative because it is fighting that shared density. Reading r alone would call both "strong".
//
// and the p-value is the two-sided rank of the observed value among the simulations. With K types
// there are K(K−1)/2 tests on one ROI — 1,176 at K=49 — so a per-pair p-value is not usable as-is,
// and `benjaminiHochberg` returns the q-values that are.

import { mulberry32 } from "./kernelAnalysis";
import type { LabelledCells } from "./pcf";

export interface QuadratParams {
  /** ROI `[minX, minY, maxX, maxY]` — the grid's extent. */
  readonly bbox: readonly [number, number, number, number];
  /** Quadrat side in world units. The covid project's stored column uses 100 (µm). */
  readonly quadratSize: number;
  /** Length of the type axis; ids must lie in `[0, nTypes)`. Defaults to `max(typeId) + 1`. */
  readonly nTypes?: number;
}

export interface QuadratCounts {
  readonly nTypes: number;
  readonly cols: number;
  readonly rows: number;
  /** `counts[type * (cols*rows) + quadrat]`. */
  readonly counts: Float64Array;
}

/** Per-type counts per quadrat. Cells outside the bbox are clamped into the edge quadrat rather than
 *  dropped: the ROI is the stated extent, and a cell fractionally outside it is a rounding artefact
 *  of how the extent was derived, not a cell that is somewhere else. */
export function quadratCounts(cells: LabelledCells, p: QuadratParams): QuadratCounts {
  const [minX, minY, maxX, maxY] = p.bbox;
  const n = cells.xs.length;
  let K = p.nTypes ?? 0;
  if (!p.nTypes) for (let i = 0; i < n; i++) K = Math.max(K, cells.typeId[i]! + 1);
  const cols = Math.max(1, Math.ceil((maxX - minX) / p.quadratSize));
  const rows = Math.max(1, Math.ceil((maxY - minY) / p.quadratSize));
  const q = cols * rows;
  const counts = new Float64Array(K * q);
  for (let i = 0; i < n; i++) {
    const t = cells.typeId[i]!;
    if (t < 0 || t >= K) throw new Error(`quadratCounts: type id ${t} outside [0, ${K})`);
    const cx = Math.min(cols - 1, Math.max(0, Math.floor((cells.xs[i]! - minX) / p.quadratSize)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor((cells.ys[i]! - minY) / p.quadratSize)));
    counts[t * q + cy * cols + cx]! += 1;
  }
  return { nTypes: K, cols, rows, counts };
}

/** Pearson correlation between every pair of rows. A row with no variance (a type absent, or present
 *  in exactly one quadrat at a constant count) has an undefined correlation and yields NaN — which
 *  is reported rather than silently zeroed, because 0 would read as "no association". */
export function rowCorrelation(m: Float64Array, k: number, q: number): Float64Array {
  const mean = new Float64Array(k);
  const sd = new Float64Array(k);
  for (let a = 0; a < k; a++) {
    let s = 0;
    for (let j = 0; j < q; j++) s += m[a * q + j]!;
    const mu = s / q;
    let v = 0;
    for (let j = 0; j < q; j++) {
      const d = m[a * q + j]! - mu;
      v += d * d;
    }
    mean[a] = mu;
    sd[a] = Math.sqrt(v);
  }
  const out = new Float64Array(k * k);
  for (let a = 0; a < k; a++) {
    for (let b = a; b < k; b++) {
      let dot = 0;
      for (let j = 0; j < q; j++) dot += (m[a * q + j]! - mean[a]!) * (m[b * q + j]! - mean[b]!);
      const den = sd[a]! * sd[b]!;
      const r = den > 0 ? dot / den : Number.NaN;
      out[a * k + b] = r;
      out[b * k + a] = r;
    }
  }
  return out;
}

export interface QuadratCorrelationParams extends QuadratParams {
  /** Label shuffles for the null. 0 skips inference and returns correlations only. */
  readonly simulations?: number;
  readonly seed?: number;
}

export interface QuadratCorrelationResult {
  readonly nTypes: number;
  /** Number of quadrats the correlation was computed over. */
  readonly quadrats: number;
  /** `r[a*K + b]`, symmetric, 1 on the diagonal. NaN where a type has no variance across quadrats. */
  readonly r: Float64Array;
  /** `(r_obs − mean_null) / sd_null`. Empty when `simulations` is 0. */
  readonly ses: Float64Array;
  /** Two-sided permutation p-value per pair. Empty when `simulations` is 0. */
  readonly p: Float64Array;
  /** Benjamini–Hochberg q-values over the K(K−1)/2 off-diagonal pairs. Empty when `simulations` is 0. */
  readonly q: Float64Array;
  readonly simulations: number;
}

/**
 * QCM with a permutation null: correlation, standard effect size, p and BH q-value per type pair.
 *
 * The null shuffles type labels between cells with positions fixed, so it preserves the quadrat
 * occupancy of the tissue as a whole and every type's abundance, and destroys only which type sits
 * where. That is the null the effect size is relative to, and it is the one the paper's own
 * 1000-shuffle uses.
 */
export function quadratCorrelation(cells: LabelledCells, p: QuadratCorrelationParams): QuadratCorrelationResult {
  const counts = quadratCounts(cells, p);
  const K = counts.nTypes;
  const q = counts.cols * counts.rows;
  const r = rowCorrelation(counts.counts, K, q);
  const sims = p.simulations ?? 0;
  if (sims <= 0) {
    return { nTypes: K, quadrats: q, r, ses: new Float64Array(0), p: new Float64Array(0), q: new Float64Array(0), simulations: 0 };
  }

  const n = cells.xs.length;
  // Quadrat index per cell, computed once — a shuffle moves labels, never positions.
  const cellQuadrat = new Int32Array(n);
  const [minX, minY, maxX, maxY] = p.bbox;
  void maxX;
  void maxY;
  for (let i = 0; i < n; i++) {
    const cx = Math.min(counts.cols - 1, Math.max(0, Math.floor((cells.xs[i]! - minX) / p.quadratSize)));
    const cy = Math.min(counts.rows - 1, Math.max(0, Math.floor((cells.ys[i]! - minY) / p.quadratSize)));
    cellQuadrat[i] = cy * counts.cols + cx;
  }
  const labels = Int32Array.from(cells.typeId);
  const rnd = mulberry32(p.seed ?? 0x9ced);
  const shuffled = new Int32Array(n);
  const m = new Float64Array(K * q);
  const sum = new Float64Array(K * K);
  const sumSq = new Float64Array(K * K);
  const moreExtreme = new Float64Array(K * K);

  for (let s = 0; s < sims; s++) {
    shuffled.set(labels);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = shuffled[i]!;
      shuffled[i] = shuffled[j]!;
      shuffled[j] = t;
    }
    m.fill(0);
    for (let i = 0; i < n; i++) m[shuffled[i]! * q + cellQuadrat[i]!]! += 1;
    const rs = rowCorrelation(m, K, q);
    for (let i = 0; i < K * K; i++) {
      const v = rs[i]!;
      if (!Number.isFinite(v)) continue;
      sum[i]! += v;
      sumSq[i]! += v * v;
      // Two-sided: |simulated| at least as large as |observed|.
      if (Number.isFinite(r[i]!) && Math.abs(v) >= Math.abs(r[i]!)) moreExtreme[i]! += 1;
    }
  }

  const ses = new Float64Array(K * K);
  const pv = new Float64Array(K * K);
  for (let i = 0; i < K * K; i++) {
    const mu = sum[i]! / sims;
    const varr = Math.max(0, sumSq[i]! / sims - mu * mu);
    const sd = Math.sqrt(varr);
    ses[i] = Number.isFinite(r[i]!) && sd > 0 ? (r[i]! - mu) / sd : Number.NaN;
    pv[i] = Number.isFinite(r[i]!) ? (moreExtreme[i]! + 1) / (sims + 1) : Number.NaN;
  }
  return { nTypes: K, quadrats: q, r, ses, p: pv, q: benjaminiHochbergMatrix(pv, K), simulations: sims };
}

/**
 * Benjamini–Hochberg q-values for a vector of p-values. NaN entries are carried through and excluded
 * from the count, so a matrix with absent types is not penalised for tests it never ran.
 */
export function benjaminiHochberg(p: readonly number[] | Float64Array): Float64Array {
  const idx: number[] = [];
  for (let i = 0; i < p.length; i++) if (Number.isFinite(p[i]!)) idx.push(i);
  idx.sort((a, b) => p[a]! - p[b]!);
  const m = idx.length;
  const out = new Float64Array(p.length).fill(Number.NaN);
  let prev = 1;
  // Walk from the largest p down, carrying the running minimum — that is what makes q monotone.
  for (let rank = m; rank >= 1; rank--) {
    const i = idx[rank - 1]!;
    const q = Math.min(prev, (p[i]! * m) / rank);
    out[i] = q;
    prev = q;
  }
  return out;
}

/** BH over the K(K−1)/2 unordered off-diagonal pairs of a symmetric K×K p-matrix, written back
 *  symmetrically. The diagonal is not a test (r = 1 by construction) and must not inflate `m`. */
export function benjaminiHochbergMatrix(p: Float64Array, k: number): Float64Array {
  const flat: number[] = [];
  const where: [number, number][] = [];
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) {
      flat.push(p[a * k + b]!);
      where.push([a, b]);
    }
  }
  const q = benjaminiHochberg(flat);
  const out = new Float64Array(k * k).fill(Number.NaN);
  for (let i = 0; i < where.length; i++) {
    const [a, b] = where[i]!;
    out[a * k + b] = q[i]!;
    out[b * k + a] = q[i]!;
  }
  return out;
}
