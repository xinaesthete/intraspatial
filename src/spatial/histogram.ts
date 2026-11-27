// Value histograms of a raster, and quantiles read off them — the f64 golden for the GPU
// `histogram` op (`src/gpu/spatial/histogram.ts`) and the host-side quantile helper both paths use.
//
// ## Binning rule (shared, exactly, with the GPU)
//
// Counts must match the GPU bit for bit, so the bin of a value is defined in f32 arithmetic that
// both sides can evaluate identically — never by a division, since GPU division is not correctly
// rounded (Metal's is ~2.5 ulp), and a value one ulp from an edge would land in different bins:
//
//     range = f32(hi − lo)
//     t     = f32(f32(v − lo) · bins)
//     bin   = the largest b in [0, bins) with f32(b · range) ≤ t
//
// Add, subtract and multiply ARE correctly rounded on every WebGPU backend, so both sides agree.
// The division `t / range` only seeds the search; two fix-up steps make it exact.
//
// - **Auto range** is [min, max] of the finite values; `v == max` goes in the last bin (numpy's
//   closed last edge). A constant field (min == max) gets [v − ½, v + ½], numpy's convention.
// - **Fixed range** `[lo, hi]`: values below `lo` / above `hi` are *under* / *over* — counted per
//   row, not binned. `v == hi` goes in the last bin.
// - **Non-finite** values (NaN, ±Inf) are skipped and counted per row.
// - **Mask**: a row per side. Row 0 = inside (mask non-zero), row 1 = outside. Both rows share one
//   set of edges (the range spans both), so the two histograms are directly comparable.
//
// ## Layout (what R3 will extend)
//
// Everything is row-major `rows × bins`. Today `rows` is 1, or 2 with a mask; a labelled (zonal)
// reduction is the same layout with `rows = labels × channels`.
//
// ## Quantiles — error bound
//
// `histogramQuantile` finds the bin holding fractional rank p = q·(N−1) (numpy's "linear"
// definition) and interpolates linearly inside it. The order statistic x₍⌊p⌋₎ is in that bin, so
// whenever the two order statistics bracketing p share a bin — always, for data dense enough to
// histogram — the estimate is within **one bin width** of the exact quantile. Where data has gaps
// wider than a bin the error is at most one bin width plus the gap. The estimate is clamped to the
// row's [min, max], which makes a constant field exact.
//
// An exact median would be a second pass: with the target bin b known from the CDF, collect the
// values with `bin(v) == b` (stream compaction, `src/gpu/scan/streamCompact.ts`), sort that small
// set, and pick rank p − C_b. Not built — the bin-width bound is already finer than the 300 µm
// patch statistics this serves need.

/** Per-row totals stride: `[inRange, under, over, nonFinite]`. */
export const HISTOGRAM_TOTALS = 4;
/** Per-row moments stride: `[min, max, sum, mean]` over the row's finite values. */
export const HISTOGRAM_MOMENTS = 4;

export interface HistogramOptions {
  /** Bin count. Default 256. */
  readonly bins?: number;
  /** Fixed `[lo, hi]` edges, `hi > lo`. Absent ⇒ auto (finite min/max). */
  readonly range?: readonly [number, number];
}

export interface HistogramResult {
  readonly bins: number;
  /** 1, or 2 when masked (row 0 inside, row 1 outside). */
  readonly rows: number;
  /** The binned range, in f32 — bin b covers `[lo + b·w, lo + (b+1)·w)`, `w = (hi − lo)/bins`. */
  readonly lo: number;
  readonly hi: number;
  /** `rows × bins` counts. */
  readonly counts: Uint32Array;
  /** `rows × bins` INCLUSIVE cumulative counts, per row (not across rows). */
  readonly cdf: Uint32Array;
  /** `rows × 4`: `[inRange, under, over, nonFinite]`. */
  readonly totals: Uint32Array;
  /** `rows × 4`: `[min, max, sum, mean]` of the finite values (NaN for an empty row). */
  readonly moments: Float64Array;
}

const f = Math.fround;

export function histogramBins(opts: HistogramOptions): number {
  const bins = opts.bins ?? 256;
  if (!Number.isInteger(bins) || bins < 1 || bins > 65536) throw new Error(`histogram: bins must be an integer in 1..65536 (got ${bins})`);
  return bins;
}

/** The f32 edges the GPU bins with: fixed if given, else auto from the finite [min, max]
 *  (constant ⇒ [v − ½, v + ½]; no finite values ⇒ [0, 1]). */
export function histogramEdges(min: number, max: number, range?: readonly [number, number]): [number, number] {
  if (range) {
    const lo = f(range[0]);
    const hi = f(range[1]);
    if (!(hi > lo)) throw new Error(`histogram: fixed range needs hi > lo (got [${range[0]}, ${range[1]}])`);
    return [lo, hi];
  }
  if (!(min <= max)) return [0, 1];
  if (min === max) return [f(min - 0.5), f(min + 0.5)];
  return [f(min), f(max)];
}

/** The bin of `v` per the shared rule (header). Assumes `lo ≤ v ≤ hi`. */
export function histogramBinOf(v: number, lo: number, range: number, bins: number): number {
  const t = f(f(v - lo) * bins);
  let b = Math.min(bins - 1, Math.max(0, Math.floor(t / range)));
  if (b > 0 && f(b * range) > t) b--;
  if (b + 1 < bins && f((b + 1) * range) <= t) b++;
  return b;
}

/** Histogram of `values` (any length; f32 semantics) with an optional same-length mask. */
export function histogramCpu(
  values: ArrayLike<number>,
  opts: HistogramOptions & { readonly mask?: ArrayLike<number> } = {},
): HistogramResult {
  const bins = histogramBins(opts);
  const mask = opts.mask;
  if (mask && mask.length !== values.length) throw new Error("histogram: mask length != values length");
  const rows = mask ? 2 : 1;
  const rowOf = (i: number) => (mask && mask[i] === 0 ? 1 : 0);

  const moments = new Float64Array(rows * HISTOGRAM_MOMENTS);
  const count = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    moments[r * 4] = Infinity;
    moments[r * 4 + 1] = -Infinity;
  }
  let gMin = Infinity;
  let gMax = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = f(values[i]!);
    if (!Number.isFinite(v)) continue;
    const r = rowOf(i);
    moments[r * 4] = Math.min(moments[r * 4]!, v);
    moments[r * 4 + 1] = Math.max(moments[r * 4 + 1]!, v);
    moments[r * 4 + 2]! += v;
    count[r]!++;
    gMin = Math.min(gMin, v);
    gMax = Math.max(gMax, v);
  }
  for (let r = 0; r < rows; r++) {
    if (count[r] === 0) moments.fill(Number.NaN, r * 4, r * 4 + 4);
    else moments[r * 4 + 3] = moments[r * 4 + 2]! / count[r]!;
  }

  const [lo, hi] = histogramEdges(gMin, gMax, opts.range);
  const range = f(hi - lo);
  const counts = new Uint32Array(rows * bins);
  const totals = new Uint32Array(rows * HISTOGRAM_TOTALS);
  for (let i = 0; i < values.length; i++) {
    const v = f(values[i]!);
    const r = rowOf(i);
    if (!Number.isFinite(v)) totals[r * 4 + 3]!++;
    else if (v < lo) totals[r * 4 + 1]!++;
    else if (v > hi) totals[r * 4 + 2]!++;
    else {
      counts[r * bins + histogramBinOf(v, lo, range, bins)]!++;
      totals[r * 4]!++;
    }
  }
  return { bins, rows, lo, hi, counts, cdf: histogramCdf(counts, bins), totals, moments };
}

/** Per-row inclusive cumulative counts of a `rows × bins` histogram. */
export function histogramCdf(counts: Uint32Array, bins: number): Uint32Array {
  const cdf = new Uint32Array(counts.length);
  for (let r = 0; r * bins < counts.length; r++) {
    let acc = 0;
    for (let b = 0; b < bins; b++) cdf[r * bins + b] = acc += counts[r * bins + b]!;
  }
  return cdf;
}

/** One row of a histogram, as the quantile helper reads it. `under`/`over` are the fixed-range
 *  out-of-range counts (0 for auto); `min`/`max` clamp the estimate when known. */
export interface HistogramRow {
  readonly counts: ArrayLike<number>;
  readonly lo: number;
  readonly hi: number;
  readonly under?: number;
  readonly over?: number;
  readonly min?: number;
  readonly max?: number;
}

/** Row `r` of a histogram result, ready for `histogramQuantile`. Accepts the GPU result too. */
export function histogramRow(h: Pick<HistogramResult, "bins" | "lo" | "hi" | "counts" | "totals" | "moments">, r = 0): HistogramRow {
  const m = h.moments;
  return {
    counts: h.counts.subarray(r * h.bins, (r + 1) * h.bins),
    lo: h.lo,
    hi: h.hi,
    under: h.totals[r * 4 + 1]!,
    over: h.totals[r * 4 + 2]!,
    min: m[r * 4]!,
    max: m[r * 4 + 1]!,
  };
}

/**
 * The q-quantile (q ∈ [0, 1]) estimated from a histogram row: the bin holding fractional rank
 * q·(N−1), linearly interpolated within it. Within one bin width of the exact quantile for dense
 * data (header). A rank that falls in the fixed-range underflow/overflow is censored: `lo`/`hi`
 * is returned. NaN for an empty row.
 */
export function histogramQuantile(row: HistogramRow, q: number): number {
  const under = row.under ?? 0;
  const bins = row.counts.length;
  let n = under + (row.over ?? 0);
  for (let b = 0; b < bins; b++) n += row.counts[b]!;
  if (n === 0) return Number.NaN;
  const p = Math.min(1, Math.max(0, q)) * (n - 1);
  const w = (row.hi - row.lo) / bins;
  let est: number;
  if (p < under) est = row.lo;
  else {
    let c = under;
    est = row.hi;
    for (let b = 0; b < bins; b++) {
      const k = row.counts[b]!;
      if (k > 0 && p < c + k) {
        // Item j of the bin (0-based) sits at the centre of its 1/k slice of the bin.
        est = row.lo + (b + (p - c + 0.5) / k) * w;
        break;
      }
      c += k;
    }
  }
  if (row.min !== undefined && Number.isFinite(row.min)) est = Math.max(est, row.min);
  if (row.max !== undefined && Number.isFinite(row.max)) est = Math.min(est, row.max);
  return est;
}

export interface HistogramSummary {
  readonly p10: number;
  readonly q1: number;
  readonly median: number;
  readonly q3: number;
  readonly p90: number;
  /** q3 − q1; its error is at most two bin widths. */
  readonly iqr: number;
  /** One bin width: the quantiles' error bound for dense data. */
  readonly binWidth: number;
}

/** Median, quartiles, IQR and p10/p90 of a histogram row. */
export function histogramSummary(row: HistogramRow): HistogramSummary {
  const at = (q: number) => histogramQuantile(row, q);
  const q1 = at(0.25);
  const q3 = at(0.75);
  return { p10: at(0.1), q1, median: at(0.5), q3, p90: at(0.9), iqr: q3 - q1, binWidth: (row.hi - row.lo) / row.counts.length };
}

/** Exact numpy-"linear" quantile of an ASCENDING sorted array — the reference the bound is
 *  stated against. */
export function exactQuantile(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const p = Math.min(1, Math.max(0, q)) * (n - 1);
  const k = Math.floor(p);
  const a = sorted[k]!;
  return k + 1 < n ? a + (p - k) * (sorted[k + 1]! - a) : a;
}
