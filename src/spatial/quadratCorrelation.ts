// The Quadrat Correlation Matrix (QCM) — the paper's coarse-scale companion to the cross-PCF.
//
// Lay a regular grid over the ROI, count each cell type per quadrat, and correlate the count vectors
// between types. Two types that fill the same quadrats correlate positively; two that avoid each
// other correlate negatively. It answers a deliberately blunter question than g(r) — "do these types
// occupy the same regions", not "how close do they get" — and at one fixed scale, set by the quadrat
// size.
//
// ## Both published columns are reproduced exactly
//
// The covid project stores two of these per (ROI, A, B), and they are different statistics:
//
//   * `quadratCounts` (the name is the config's, not the statistic's) is the **Pearson correlation**
//     of per-type counts in 100 µm quadrats. Recovered by scanning quadrat sizes against the stored
//     values; at 100 µm, 70,742 rows agree with a median |Δ| of 1.0e-9.
//   * `MH_PC` ("Quadrat Correlation Pair Correlation") is the **partial correlation** of the same
//     counts — see `partialCorrelation`. `MH` is **Morueta-Holme**, not Morisita-Horn: the paper's
//     methods cite Morueta-Holme et al. and SpOOx runs this as `--function morueta-holme`. 4,802
//     rows agree with a median |Δ| of 6.6e-10.
//
// Quadrat size 100 µm is the paper's own ("square quadrats with edge length 100μm, resulting in
// between 100 and 400 quadrats per ROI"), which is what the empirical scan recovered independently.
//
// ## What is NOT matched, measured rather than glossed
//
// The third column, `MH_SES`, standardises the PARTIAL correlation (SES computed on `pc` here
// correlates 0.978 with it, against 0.663 for the plain `r` — so the choice of statistic is
// settled). The residual is the NULL SAMPLER, not noise: both the paper's and this module's nulls
// hold the same margins fixed — each type's abundance and each quadrat's total — but the paper walks
// a Markov chain of 2×2 swaps that is uniform over fixed-margin matrices, while a label shuffle
// weights them by how many labellings produce each one. Raising the shuffles from 199 to 999 moved
// the median |Δ| from 0.154 to 0.146, i.e. essentially not at all, which is how we know it is the
// sampler. Implementing the swap chain is the outstanding piece for exact `MH_SES` parity.
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

/**
 * Partial correlation between every pair of rows — the paper's actual QCM statistic.
 *
 * `MH` in the published columns is **Morueta-Holme**, not Morisita-Horn: the methods cite ref. 38
 * (Morueta-Holme et al., "A network approach for inferring species associations from co-occurrence
 * data") and SpOOx runs it as `--function morueta-holme`. Its defining move is that the association
 * between two types is measured **after conditioning on every other type**, which is what a partial
 * correlation is:
 *
 *     P = R⁻¹                    (the precision matrix)
 *     ρ_ab·rest = −P_ab / √(P_aa P_bb)
 *
 * The distinction is not academic. A plain correlation is transitive by construction — if a
 * macrophage type and a T-cell type both crowd into the same inflamed quadrats, they correlate with
 * each other AND with everything else that crowds in there, so the matrix fills with a single
 * dominant "this is where the cells are" factor and nearly every pair reads positive. The partial
 * correlation asks the narrower question the biology actually wants: does A predict B *beyond* what
 * the other 47 types already say about that quadrat? Direct associations survive it; associations
 * induced by a shared third type do not.
 *
 * Types with no variance across quadrats (absent, or present at a constant count) cannot enter the
 * inverse at all — they would make R singular — so they are dropped, the surviving submatrix is
 * inverted, and their rows come back NaN.
 */
export function partialCorrelation(r: Float64Array, k: number): Float64Array {
  const out = new Float64Array(k * k).fill(Number.NaN);
  // Drop the variance-free types, which would make R singular. The DIAGONAL is the test: a type
  // correlated with itself is 1 unless its own sd is 0. Scanning the whole row instead would be
  // wrong — a NaN off the diagonal says the OTHER type is degenerate, and would take this one down
  // with it.
  const keep: number[] = [];
  for (let a = 0; a < k; a++) if (Number.isFinite(r[a * k + a]!)) keep.push(a);
  const m = keep.length;
  if (m < 2) return out;

  // Gauss-Jordan with partial pivoting on [R | I]. m ≤ ~50 here, so an O(m³) dense inverse is
  // microseconds and there is nothing to gain from a factorisation that preserves symmetry.
  const a = new Float64Array(m * 2 * m);
  const w = 2 * m;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) a[i * w + j] = r[keep[i]! * k + keep[j]!]!;
    a[i * w + m + i] = 1;
  }
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let i = col + 1; i < m; i++) if (Math.abs(a[i * w + col]!) > Math.abs(a[piv * w + col]!)) piv = i;
    // Singular to working precision — collinear types, or fewer quadrats than types. Conditioning on
    // a set that already determines the row has no answer, and a ridge would invent one.
    if (Math.abs(a[piv * w + col]!) < 1e-12) return out;
    if (piv !== col) {
      for (let j = 0; j < w; j++) {
        const t = a[col * w + j]!;
        a[col * w + j] = a[piv * w + j]!;
        a[piv * w + j] = t;
      }
    }
    const d = a[col * w + col]!;
    for (let j = 0; j < w; j++) a[col * w + j]! /= d;
    for (let i = 0; i < m; i++) {
      if (i === col) continue;
      const f = a[i * w + col]!;
      if (f === 0) continue;
      for (let j = 0; j < w; j++) a[i * w + j]! -= f * a[col * w + j]!;
    }
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const pii = a[i * w + m + i]!;
      const pjj = a[j * w + m + j]!;
      const den = Math.sqrt(pii * pjj);
      out[keep[i]! * k + keep[j]!] = i === j ? 1 : den > 0 ? -a[i * w + m + j]! / den : Number.NaN;
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
  /** `r[a*K + b]`, symmetric, 1 on the diagonal. NaN where a type has no variance across quadrats.
   *  The project's `quadratCounts` column. */
  readonly r: Float64Array;
  /** Partial correlation — the project's `MH_PC`, and the statistic the paper actually reports. */
  readonly pc: Float64Array;
  /** `(r_obs − mean_null) / sd_null`. Empty when `simulations` is 0. */
  readonly ses: Float64Array;
  /** Two-sided permutation p-value per pair. Empty when `simulations` is 0. */
  readonly p: Float64Array;
  /** Benjamini–Hochberg q-values over the K(K−1)/2 off-diagonal pairs. Empty when `simulations` is 0. */
  readonly q: Float64Array;
  /** The same three for the PARTIAL correlation — the project's `MH_SES` / `MH_FDR` line. Empty when
   *  `simulations` is 0. These are the ones to read: `ses` says "do A and B share quadrats more than
   *  chance", `pcSes` says "…beyond what every other type already explains". */
  readonly pcSes: Float64Array;
  readonly pcP: Float64Array;
  readonly pcQ: Float64Array;
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
  const pc = partialCorrelation(r, K);
  const sims = p.simulations ?? 0;
  const empty = () => new Float64Array(0);
  if (sims <= 0) {
    return {
      nTypes: K,
      quadrats: q,
      r,
      pc,
      ses: empty(),
      p: empty(),
      q: empty(),
      pcSes: empty(),
      pcP: empty(),
      pcQ: empty(),
      simulations: 0,
    };
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
  /** Running Σ, Σ², and two-sided extremity count for one statistic across the shuffles. */
  const acc = () => ({ sum: new Float64Array(K * K), sumSq: new Float64Array(K * K), moreExtreme: new Float64Array(K * K) });
  const accR = acc();
  const accPc = acc();
  const tally = (a: ReturnType<typeof acc>, sim: Float64Array, obs: Float64Array) => {
    for (let i = 0; i < K * K; i++) {
      const v = sim[i]!;
      if (!Number.isFinite(v)) continue;
      a.sum[i]! += v;
      a.sumSq[i]! += v * v;
      // Two-sided: |simulated| at least as large as |observed|.
      if (Number.isFinite(obs[i]!) && Math.abs(v) >= Math.abs(obs[i]!)) a.moreExtreme[i]! += 1;
    }
  };

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
    tally(accR, rs, r);
    tally(accPc, partialCorrelation(rs, K), pc);
  }

  const finish = (a: ReturnType<typeof acc>, obs: Float64Array) => {
    const ses = new Float64Array(K * K);
    const pv = new Float64Array(K * K);
    for (let i = 0; i < K * K; i++) {
      const mu = a.sum[i]! / sims;
      const varr = Math.max(0, a.sumSq[i]! / sims - mu * mu);
      const sd = Math.sqrt(varr);
      ses[i] = Number.isFinite(obs[i]!) && sd > 0 ? (obs[i]! - mu) / sd : Number.NaN;
      pv[i] = Number.isFinite(obs[i]!) ? (a.moreExtreme[i]! + 1) / (sims + 1) : Number.NaN;
    }
    return { ses, p: pv, q: benjaminiHochbergMatrix(pv, K) };
  };
  const plain = finish(accR, r);
  const partial = finish(accPc, pc);
  return {
    nTypes: K,
    quadrats: q,
    r,
    pc,
    ses: plain.ses,
    p: plain.p,
    q: plain.q,
    pcSes: partial.ses,
    pcP: partial.p,
    pcQ: partial.q,
    simulations: sims,
  };
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
