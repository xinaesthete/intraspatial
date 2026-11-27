// Value histograms of a raster, and quantiles read off them — the f64 golden for the GPU
// `histogram` op (`src/gpu/spatial/histogram.ts`) and the host-side quantile helper both paths use.
//
// ## Binning rule (shared, exactly, with the GPU)
//
// Counts must match the GPU bit for bit, so the bin of a value is defined in f32 arithmetic that
// both sides can evaluate identically — never by a division, since GPU division is not correctly
// rounded (Metal's is ~2.5 ulp), and a value one ulp from an edge would land in different bins:
//
//     s     = 2⁻³² if max(|lo|, |hi|) ≥ 2⁹⁶, else 1
//     range = f32(f32(hi·s) − f32(lo·s))
//     t     = f32(f32(f32(v·s) − f32(lo·s)) · bins)
//     bin   = the largest b in [0, bins) with f32(b · range) ≤ t
//
// Add, subtract and multiply ARE correctly rounded on every WebGPU backend, so both sides agree.
// The division `t / range` only seeds the search; two fix-up steps make it exact. The power-of-two
// prescale `s` is exact, and only exists so that `range` and `t` cannot overflow to +Inf when the
// edges are near ±f32 max (`[-2e38, 2e38]`, say); for every ordinary range s = 1.
//
// - **Auto range** is [min, max] of the finite values; `v == max` goes in the last bin (numpy's
//   closed last edge). A constant field (min == max) gets [v − ½, v + ½], numpy's convention —
//   unless that rounds to a single f32 (|v| ≥ 2²⁴), in which case the edge nearer zero moves one
//   ulp of v toward zero instead, so the range stays non-empty and still contains v.
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
// whenever the two order statistics bracketing p share a bin, the estimate is within **one bin
// width** of the exact quantile. That bound is conditional: the bracketing order statistics can
// land in different bins even when they are close together (on either side of an edge), and with
// gaps wider than a bin or very few values; then the error is at most one bin width plus the gap. A binned
// estimate is clamped to the row's [min, max], which makes a constant field exact; a rank in the
// fixed-range under/overflow is censored and returns `lo`/`hi` as is.
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
  if (range) return histogramFixedRange(range);
  if (!(min <= max)) return [0, 1];
  if (min === max) return constantEdges(f(min));
  return [f(min), f(max)];
}

/** A fixed range as the f32 edges the GPU will see. Validated AFTER rounding: `[1, 1 + 2⁻²⁵]` is
 *  `hi > lo` in f64 but one f32, which would make every bin empty. */
export function histogramFixedRange(range: readonly [number, number]): [number, number] {
  const lo = f(range[0]);
  const hi = f(range[1]);
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    throw new Error(`histogram: fixed range needs finite hi > lo in f32 (got [${range[0]}, ${range[1]}])`);
  }
  return [lo, hi];
}

const bitsView = new DataView(new ArrayBuffer(4));
/** The f32 next to `v` (≠ 0) toward zero: its bit pattern minus one. */
function towardZero(v: number): number {
  bitsView.setFloat32(0, v);
  bitsView.setUint32(0, bitsView.getUint32(0) - 1);
  return bitsView.getFloat32(0);
}

/** [v − ½, v + ½], or — when that collapses to one f32 — v and its neighbour toward zero. */
function constantEdges(v: number): [number, number] {
  const lo = f(v - 0.5);
  const hi = f(v + 0.5);
  if (lo < hi) return [lo, hi];
  return v > 0 ? [towardZero(v), v] : [v, towardZero(v)];
}

/** The exact power-of-two prescale of the shared rule (header). */
export function histogramScale(lo: number, hi: number): number {
  return Math.max(Math.abs(lo), Math.abs(hi)) >= 2 ** 96 ? 2 ** -32 : 1;
}

/** The bin of `v` per the shared rule (header). Assumes `lo ≤ v ≤ hi` (f32 values). */
export function histogramBinOf(v: number, lo: number, hi: number, bins: number): number {
  const s = histogramScale(lo, hi);
  const los = f(lo * s);
  const range = f(f(hi * s) - los);
  const t = f(f(f(v * s) - los) * bins);
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
  const counts = new Uint32Array(rows * bins);
  const totals = new Uint32Array(rows * HISTOGRAM_TOTALS);
  for (let i = 0; i < values.length; i++) {
    const v = f(values[i]!);
    const r = rowOf(i);
    if (!Number.isFinite(v)) totals[r * 4 + 3]!++;
    else if (v < lo) totals[r * 4 + 1]!++;
    else if (v > hi) totals[r * 4 + 2]!++;
    else {
      counts[r * bins + histogramBinOf(v, lo, hi, bins)]!++;
      totals[r * 4]!++;
    }
  }
  return { bins, rows, lo, hi, counts, cdf: histogramCdf(counts, bins), totals, moments };
}

/** Per-row inclusive cumulative counts of a `rows × bins` histogram. */
export function histogramCdf(counts: Uint32Array, bins: number): Uint32Array {
  if (!Number.isInteger(bins) || bins < 1 || counts.length % bins !== 0) {
    throw new Error(`histogramCdf: bins must be a positive integer dividing counts.length (got ${bins} for ${counts.length})`);
  }
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
 * q·(N−1), linearly interpolated within it. Within one bin width of the exact quantile when the
 * order statistics bracketing that rank share a bin (header). A rank that falls in the fixed-range underflow/overflow is censored: `lo`/`hi`
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
  // A censored rank (fixed-range under/overflow) returns the edge as is: clamping it to the row's
  // [min, max] would report a value the histogram never saw binned.
  if (p < under) return row.lo;
  let c = under;
  let est = Number.NaN;
  for (let b = 0; b < bins; b++) {
    const k = row.counts[b]!;
    if (k > 0 && p < c + k) {
      // Item j of the bin (0-based) sits at the centre of its 1/k slice of the bin. The fraction
      // is capped at the bin's upper edge: p in [c + k − ½, c + k) would otherwise overshoot it,
      // past where the next bin's first estimate starts, and q → estimate would not be monotone.
      est = row.lo + (b + Math.min(1, (p - c + 0.5) / k)) * w;
      break;
    }
    c += k;
  }
  if (Number.isNaN(est)) return row.hi;
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
  /** q3 − q1; within two bin widths when both quartiles meet the one-bin condition. */
  readonly iqr: number;
  /** One bin width: the quantiles' error bound when each rank's bracketing order statistics
   *  share a bin; see the header. */
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
