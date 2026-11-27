// Zonal (per-region) statistics of raster channels: the f64 golden for the GPU ops in
// `src/gpu/spatial/zonal.ts`, and the host-side derivation (quantiles, fractions, the exact
// second pass) both paths share. R3 of docs/histogram-op.md; design in docs/zonal-stats.md.
//
// ## Regions are labels
//
// A region is every pixel whose label is its id: label L in 1..regions is region L − 1, label 0
// is "no region", and a label above `regions` is ignored. Polygons become labels by rasterising
// them (`rasterisePolygonsCpu`, the GPU `rasterisePolygons`): polygon k gets label k + 1, a pixel
// is inside when its CENTRE is (`domainContains`, even–odd), and where polygons overlap the
// highest id wins. Overlapping regions (a patch grid with stride < size) go in separate layers
// (`polygonLayers`), each rasterised and reduced on its own.
//
// ## Rows, and what each pixel counts as
//
// Rows are `region × channels + channel`, each a histogram row in R1's layout (`HistogramRow`
// reads one, so `histogramQuantile` / `histogramSummary` apply unchanged). Per channel, every
// pixel of a region lands in exactly one totals slot, in this order:
//
//   unsupported  the channel has a support raster and it is 0 there (NaN counts as supported,
//                as numpy's `img != 0` does)
//   nonFinite    NaN / ±Inf
//   excluded     `valid: "positive"` and v ≤ 0 (the TNBC rasters mark background with 0)
//   under / over outside a fixed range: counted, not binned (quantiles there are censored)
//   inRange      binned, by R1's shared f32 rule (`histogramBinOf`), so the GPU matches exactly
//
// `pixels` is their sum. A channel's values are those in inRange + under + over; `count` is
// their number, `validFraction = count / supported` and `supportFraction = supported / pixels`.
//
// ## Edges are per channel and fixed before binning
//
// Tiles are accumulated into one set of counts, so a channel's edges cannot follow each tile.
// Auto range is R1's rule over the channel's values in ALL regions (one min/max pre-pass, then
// `histogramEdges`); a fixed range skips the pre-pass.
//
// ## Exact quantiles (the second pass)
//
// Binned quantiles are within one bin width under R1's bracketing condition. For an exact value
// (SIMBA's `np.median` / `np.percentile`, numpy "linear"), `zonalExactPlan` reads the CDF and
// names, per row, the bins holding order statistics ⌊p⌋ and ⌊p⌋ + 1 for every requested q. A
// second pass collects the values in exactly those bins (`collect` on the GPU,
// `zonalCollectCpu` here), and `zonalExactQuantiles` sorts each small set and picks the ranks.

import { type Domain, type PolygonDomain, domainContains } from "./domain";
import {
  HISTOGRAM_MOMENTS,
  type HistogramRow,
  exactQuantile,
  histogramBinOf,
  histogramBins,
  histogramCdf,
  histogramEdges,
  histogramFixedRange,
  histogramQuantile,
} from "./histogram";

/** Per-row totals stride: `[inRange, under, over, nonFinite, excluded, unsupported, pixels, 0]`. */
export const ZONAL_TOTALS = 8;
export const ZONAL_SLOT = { inRange: 0, under: 1, over: 2, nonFinite: 3, excluded: 4, unsupported: 5, pixels: 6 } as const;
/** Most channels one zonal reduction takes (the auto-range mask is one u32 on the GPU). */
export const ZONAL_MAX_CHANNELS = 32;

/** Which finite values count. `"positive"`: only v > 0 (0 is background); `"finite"`: all. */
export type ZonalValid = "finite" | "positive";

export interface ZonalChannelOptions {
  /** Fixed `[lo, hi]`. Absent ⇒ auto: R1's rule over this channel's values in every region. */
  readonly range?: readonly [number, number];
  /** Default `"finite"`. */
  readonly valid?: ZonalValid;
}

export interface ZonalOptions {
  /** Region count: labels 1..regions are kept. */
  readonly regions: number;
  /** Bins per row. Default 1024. */
  readonly bins?: number;
  /** One entry per channel (at least one). */
  readonly channels: readonly ZonalChannelOptions[];
}

export interface ZonalHistogram {
  readonly regions: number;
  readonly channels: number;
  readonly bins: number;
  /** `channels × 2`: each channel's f32 `[lo, hi]`. */
  readonly edges: Float32Array;
  /** `rows × bins`, row = region·channels + channel. */
  readonly counts: Uint32Array;
  /** Inclusive per-row CDF. */
  readonly cdf: Uint32Array;
  /** `rows × ZONAL_TOTALS`. */
  readonly totals: Uint32Array;
  /** `rows × 4`: `[min, max, sum, mean]` of the row's values; NaN when it has none. */
  readonly moments: Float64Array;
}

export function zonalBins(opts: { readonly bins?: number }): number {
  return histogramBins({ bins: opts.bins ?? 1024 });
}

export function checkZonalOptions(opts: ZonalOptions): void {
  const { regions, channels } = opts;
  if (!Number.isInteger(regions) || regions < 1 || regions > 2 ** 24)
    throw new Error(`zonal: regions must be an integer in 1..2^24 (got ${regions})`);
  if (channels.length < 1 || channels.length > ZONAL_MAX_CHANNELS) {
    throw new Error(`zonal: 1..${ZONAL_MAX_CHANNELS} channels (got ${channels.length})`);
  }
  for (const c of channels) if (c.range) histogramFixedRange(c.range);
  zonalBins(opts);
}

/** Region index of a label value: 0..regions−1, or −1 for none. Non-integer labels truncate. */
export function zonalRegionOf(label: number, regions: number): number {
  if (!(label >= 1)) return -1;
  const l = Math.floor(label);
  return l <= regions ? l - 1 : -1;
}

/** One channel's host data for the golden: values, and an optional support raster (non-zero ⇒
 *  supported), both the labels' length. */
export interface ZonalChannelData {
  readonly values: ArrayLike<number>;
  readonly support?: ArrayLike<number>;
}

const f = Math.fround;

/** Is pixel i a value of channel c (supported, finite, valid)? Mirrors the GPU kernel's order. */
function classify(v: number, sup: number | undefined, positive: boolean): number {
  if (sup !== undefined && sup === 0) return ZONAL_SLOT.unsupported;
  if (!Number.isFinite(v)) return ZONAL_SLOT.nonFinite;
  if (positive && !(v > 0)) return ZONAL_SLOT.excluded;
  return -1;
}

/** The labelled histogram, in f64 (f32 values): the GPU's golden. Counts, CDF, totals and edges
 *  match it exactly; the GPU's sums are f32 per tile, then compensated across tiles. */
export function zonalHistogramCpu(labels: ArrayLike<number>, data: readonly ZonalChannelData[], opts: ZonalOptions): ZonalHistogram {
  checkZonalOptions(opts);
  const { regions } = opts;
  const C = opts.channels.length;
  if (data.length !== C) throw new Error(`zonal: ${data.length} channel inputs for ${C} channel options`);
  const n = labels.length;
  for (const d of data) {
    if (d.values.length !== n || (d.support && d.support.length !== n)) throw new Error("zonal: channel length != labels length");
  }
  const bins = zonalBins(opts);
  const rows = regions * C;
  const region = new Int32Array(n);
  for (let i = 0; i < n; i++) region[i] = zonalRegionOf(labels[i]!, regions);

  const edges = new Float32Array(C * 2);
  for (let c = 0; c < C; c++) {
    const o = opts.channels[c]!;
    const positive = o.valid === "positive";
    let mn = Infinity;
    let mx = -Infinity;
    if (!o.range) {
      const { values, support } = data[c]!;
      for (let i = 0; i < n; i++) {
        if (region[i]! < 0) continue;
        const v = f(values[i]!);
        if (classify(v, support?.[i], positive) >= 0) continue;
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
      }
    }
    const [lo, hi] = histogramEdges(mn, mx, o.range);
    edges[c * 2] = lo;
    edges[c * 2 + 1] = hi;
  }

  const counts = new Uint32Array(rows * bins);
  const totals = new Uint32Array(rows * ZONAL_TOTALS);
  const moments = new Float64Array(rows * HISTOGRAM_MOMENTS);
  for (let r = 0; r < rows; r++) moments.set([Infinity, -Infinity, 0, 0], r * 4);
  for (let c = 0; c < C; c++) {
    const positive = opts.channels[c]!.valid === "positive";
    const lo = edges[c * 2]!;
    const hi = edges[c * 2 + 1]!;
    const { values, support } = data[c]!;
    for (let i = 0; i < n; i++) {
      const g = region[i]!;
      if (g < 0) continue;
      const row = g * C + c;
      const t = row * ZONAL_TOTALS;
      totals[t + ZONAL_SLOT.pixels]!++;
      const v = f(values[i]!);
      const slot = classify(v, support?.[i], positive);
      if (slot >= 0) {
        totals[t + slot]!++;
        continue;
      }
      const m = row * 4;
      moments[m] = Math.min(moments[m]!, v);
      moments[m + 1] = Math.max(moments[m + 1]!, v);
      moments[m + 2]! += v;
      if (v < lo) totals[t + ZONAL_SLOT.under]!++;
      else if (v > hi) totals[t + ZONAL_SLOT.over]!++;
      else {
        totals[t]!++;
        counts[row * bins + histogramBinOf(v, lo, hi, bins)]!++;
      }
    }
  }
  for (let r = 0; r < rows; r++) {
    const k = zonalCount(totals, r);
    if (k === 0) moments.fill(Number.NaN, r * 4, r * 4 + 4);
    else moments[r * 4 + 3] = moments[r * 4 + 2]! / k;
  }
  return { regions, channels: C, bins, edges, counts, cdf: histogramCdf(counts, bins), totals, moments };
}

/** A row's value count: inRange + under + over. */
export function zonalCount(totals: ArrayLike<number>, row: number): number {
  const t = row * ZONAL_TOTALS;
  return totals[t]! + totals[t + 1]! + totals[t + 2]!;
}

/** Row `region·channels + channel` as R1's `HistogramRow`. */
export function zonalRow(h: ZonalHistogram, region: number, channel: number): HistogramRow {
  const r = region * h.channels + channel;
  const t = r * ZONAL_TOTALS;
  return {
    counts: h.counts.subarray(r * h.bins, (r + 1) * h.bins),
    lo: h.edges[channel * 2]!,
    hi: h.edges[channel * 2 + 1]!,
    under: h.totals[t + 1]!,
    over: h.totals[t + 2]!,
    min: h.moments[r * 4]!,
    max: h.moments[r * 4 + 1]!,
  };
}

/** Add `b` into `a` (same regions, channels, bins and edges) — merging layers or partial runs. */
export function zonalMerge(a: ZonalHistogram, b: ZonalHistogram): ZonalHistogram {
  if (a.regions !== b.regions || a.channels !== b.channels || a.bins !== b.bins || a.edges.some((e, i) => e !== b.edges[i])) {
    throw new Error("zonalMerge: histograms differ in shape or edges");
  }
  // Uint32Array.map would wrap a sum past 2^32 − 1 to a small count: refuse it instead.
  const add = (x: Uint32Array, y: Uint32Array, what: string) => {
    const out = new Uint32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i]! + y[i]!;
      if (v > 0xffffffff) throw new Error(`zonalMerge: ${what} ${i} would exceed 2^32 − 1`);
      out[i] = v;
    }
    return out;
  };
  const counts = add(a.counts, b.counts, "count");
  const totals = add(a.totals, b.totals, "total");
  const moments = new Float64Array(a.moments.length);
  for (let r = 0; r * 4 < moments.length; r++) {
    const k = zonalCount(totals, r);
    const m = r * 4;
    if (k === 0) {
      moments.fill(Number.NaN, m, m + 4);
      continue;
    }
    const pick = (x: number, y: number, fn: (p: number, q: number) => number) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : fn(x, y));
    moments[m] = pick(a.moments[m]!, b.moments[m]!, Math.min);
    moments[m + 1] = pick(a.moments[m + 1]!, b.moments[m + 1]!, Math.max);
    moments[m + 2] = (Number.isNaN(a.moments[m + 2]!) ? 0 : a.moments[m + 2]!) + (Number.isNaN(b.moments[m + 2]!) ? 0 : b.moments[m + 2]!);
    moments[m + 3] = moments[m + 2]! / k;
  }
  return { ...a, counts, cdf: histogramCdf(counts, a.bins), totals, moments };
}

// ---------------------------------------------------------------------------------------------
// Polygons → labels
// ---------------------------------------------------------------------------------------------

/** Where a raster's pixels sit in the polygons' coordinates: pixel (i, j)'s CENTRE is
 *  `(originX + (i + ½)·scaleX, originY + (j + ½)·scaleY)`. Row 0 is j = 0 (no flip): with
 *  polygons in level-0 array space (x = column, y = row), a level-L tile is origin = its level-0
 *  corner and scale = the level's downsample. */
export interface RasterFrame {
  readonly width: number;
  readonly height: number;
  readonly originX?: number;
  readonly originY?: number;
  readonly scaleX?: number;
  readonly scaleY?: number;
}

function frameOf(fr: RasterFrame): [number, number, number, number] {
  const sx = fr.scaleX ?? 1;
  const sy = fr.scaleY ?? 1;
  if (!(sx > 0 && sy > 0)) throw new Error("zonal: frame scale must be positive");
  return [fr.originX ?? 0, fr.originY ?? 0, sx, sy];
}

/** Labels from polygons: polygon k ⇒ label `ids?.[k] ?? k + 1` at every pixel whose centre it
 *  contains (`domainContains`, even–odd), the highest label where they overlap; 0 elsewhere. The
 *  exact geometry the GPU rasteriser is checked against. */
export function rasterisePolygonsCpu(polygons: readonly Domain[], frame: RasterFrame, ids?: ArrayLike<number>): Uint32Array {
  const { width, height } = frame;
  const [ox, oy, sx, sy] = frameOf(frame);
  const out = new Uint32Array(width * height);
  polygons.forEach((p, k) => {
    const id = ids?.[k] ?? k + 1;
    const [minX, minY, maxX, maxY] = p.bbox;
    const i0 = Math.max(0, Math.floor((minX - ox) / sx - 0.5));
    const i1 = Math.min(width - 1, Math.ceil((maxX - ox) / sx - 0.5));
    const j0 = Math.max(0, Math.floor((minY - oy) / sy - 0.5));
    const j1 = Math.min(height - 1, Math.ceil((maxY - oy) / sy - 0.5));
    for (let j = j0; j <= j1; j++) {
      const y = oy + (j + 0.5) * sy;
      for (let i = i0; i <= i1; i++) {
        const q = j * width + i;
        if (out[q]! < id && domainContains(p, ox + (i + 0.5) * sx, y)) out[q] = id;
      }
    }
  });
  return out;
}

/** Polygon edges and a per-polygon table in a raster's PIXEL frame (pixel (i, j)'s centre at
 *  (i + ½, j + ½)), as the GPU rasteriser reads them. Transformed in f64, then rounded to f32 once.
 *  Polygons whose box misses the raster are dropped. */
export interface PolygonRingBuffers {
  /** `E × 4`: `[x0, y0, x1, y1]` per edge, every ring of every kept polygon. */
  readonly edges: Float32Array;
  /** `P × 8`: `[firstEdge, edgeCount, label, 0, minX, minY, maxX, maxY]` (counts exact in f32). */
  readonly polys: Float32Array;
  readonly count: number;
}

export const POLY_STRIDE = 8;

export function polygonRingBuffers(polygons: readonly PolygonDomain[], frame: RasterFrame, ids?: ArrayLike<number>): PolygonRingBuffers {
  const [ox, oy, sx, sy] = frameOf(frame);
  const { width, height } = frame;
  const kept: number[] = [];
  let edgeCount = 0;
  polygons.forEach((p, k) => {
    const [minX, minY, maxX, maxY] = p.bbox;
    const u0 = (minX - ox) / sx;
    const u1 = (maxX - ox) / sx;
    const v0 = (minY - oy) / sy;
    const v1 = (maxY - oy) / sy;
    if (u1 < 0 || v1 < 0 || u0 > width || v0 > height) return;
    kept.push(k);
    for (const ring of p.rings) edgeCount += ring.length / 2;
  });
  if (edgeCount >= 2 ** 24) throw new Error(`zonal: ${edgeCount} polygon edges in one raster, over 2^24`);
  const edges = new Float32Array(Math.max(1, edgeCount) * 4);
  const polys = new Float32Array(Math.max(1, kept.length) * POLY_STRIDE);
  let e = 0;
  kept.forEach((k, slot) => {
    const p = polygons[k]!;
    const first = e;
    for (const ring of p.rings) {
      const m = ring.length;
      for (let a = 0; a < m; a += 2) {
        const b = (a + 2) % m;
        edges.set([(ring[a]! - ox) / sx, (ring[a + 1]! - oy) / sy, (ring[b]! - ox) / sx, (ring[b + 1]! - oy) / sy], e * 4);
        e++;
      }
    }
    const [minX, minY, maxX, maxY] = p.bbox;
    const id = ids?.[k] ?? k + 1;
    if (!(Number.isInteger(id) && id >= 1 && id < 2 ** 24)) throw new Error(`zonal: polygon label ${id} is not an integer in 1..2^24`);
    polys.set([first, e - first, id, 0, (minX - ox) / sx, (minY - oy) / sy, (maxX - ox) / sx, (maxY - oy) / sy], slot * POLY_STRIDE);
  });
  return { edges, polys, count: kept.length };
}

/**
 * Split polygons into layers whose members cannot share a pixel: greedy, by bounding box (boxes
 * that only touch are fine — a polygon contains no centre on its own max edge). One layer for
 * disjoint regions (ST ROIs, a lasso); a patch grid with stride = size / 2 needs four.
 */
export function polygonLayers(polygons: readonly Pick<Domain, "bbox">[]): number[][] {
  const layers: { members: number[]; boxes: (readonly number[])[] }[] = [];
  polygons.forEach((p, k) => {
    const b = p.bbox;
    const fits = (l: { boxes: (readonly number[])[] }) =>
      l.boxes.every((o) => !(b[0] < o[2]! && o[0]! < b[2] && b[1] < o[3]! && o[1]! < b[3]));
    let layer = layers.find(fits);
    if (!layer) {
      layer = { members: [], boxes: [] };
      layers.push(layer);
    }
    layer.members.push(k);
    layer.boxes.push(b);
  });
  return layers.map((l) => l.members);
}

// ---------------------------------------------------------------------------------------------
// Exact quantiles: plan, collect (CPU twin), resolve
// ---------------------------------------------------------------------------------------------

/** Unused target slot. */
export const ZONAL_NO_TARGET = 0xffffffff;
/** Words per target slot in `ZonalExactPlan.table`: `[bin, offset, count]`. */
export const ZONAL_TARGET_STRIDE = 3;

export interface ZonalExactPlan {
  readonly quantiles: readonly number[];
  /** Target slots per row. */
  readonly perRow: number;
  /** `rows × perRow × 3`: `[bin, offset, count]`; bin = `ZONAL_NO_TARGET` when the slot is unused.
   *  `count` is the bin's count from the histogram: what the collect pass must find, and the most
   *  it may write (a pass over different data cannot overrun the next target's segment). */
  readonly table: Uint32Array;
  /** Values the second pass collects (Σ counts of target bins). */
  readonly total: number;
}

/** Per row, the bin of order statistic k (0-based, among the row's values) — or −1 when k is in
 *  the fixed-range under/overflow, which is never binned. */
function binOfRank(h: ZonalHistogram, row: number, k: number): number {
  const under = h.totals[row * ZONAL_TOTALS + 1]!;
  if (k < under) return -1;
  const cdf = h.cdf.subarray(row * h.bins, (row + 1) * h.bins);
  // First b with cdf[b] > k − under: binary search on the non-decreasing CDF.
  const target = k - under;
  let lo = 0;
  let hi = h.bins;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid]! > target) hi = mid;
    else lo = mid + 1;
  }
  return lo < h.bins ? lo : -1;
}

/** Which bins the exact pass must collect for `quantiles` (each in [0, 1]). */
export function zonalExactPlan(h: ZonalHistogram, quantiles: readonly number[]): ZonalExactPlan {
  if (quantiles.length === 0 || quantiles.some((q) => !(q >= 0 && q <= 1))) throw new Error("zonal: quantiles must be in [0, 1]");
  const rows = h.regions * h.channels;
  const perRow = 2 * quantiles.length;
  const table = new Uint32Array(rows * perRow * ZONAL_TARGET_STRIDE).fill(ZONAL_NO_TARGET);
  let total = 0;
  for (let r = 0; r < rows; r++) {
    const n = zonalCount(h.totals, r);
    if (n === 0) continue;
    const want = new Set<number>();
    for (const q of quantiles) {
      const k = Math.floor(q * (n - 1));
      for (const kk of k + 1 < n ? [k, k + 1] : [k]) {
        const b = binOfRank(h, r, kk);
        if (b >= 0) want.add(b);
      }
    }
    let t = 0;
    for (const b of [...want].sort((x, y) => x - y)) {
      const o = (r * perRow + t++) * ZONAL_TARGET_STRIDE;
      const count = h.counts[r * h.bins + b]!;
      table[o] = b;
      table[o + 1] = total;
      table[o + 2] = count;
      total += count;
    }
  }
  if (total > 2 ** 32 - 1) throw new Error(`zonal: the exact pass would collect ${total} values`);
  return { quantiles, perRow, table, total };
}

/** The CPU twin of the GPU `collect` pass: every value in a target bin, at its target's offset
 *  (order within a target is arbitrary on the GPU; `zonalExactQuantiles` sorts). */
export function zonalCollectCpu(
  labels: ArrayLike<number>,
  data: readonly ZonalChannelData[],
  opts: ZonalOptions,
  h: ZonalHistogram,
  plan: ZonalExactPlan,
  out = new Float32Array(Math.max(1, plan.total)),
): Float32Array {
  const C = h.channels;
  const cursor = new Uint32Array(plan.table.length / ZONAL_TARGET_STRIDE);
  for (let c = 0; c < C; c++) {
    const positive = opts.channels[c]!.valid === "positive";
    const lo = h.edges[c * 2]!;
    const hi = h.edges[c * 2 + 1]!;
    const { values, support } = data[c]!;
    for (let i = 0; i < labels.length; i++) {
      const g = zonalRegionOf(labels[i]!, h.regions);
      if (g < 0) continue;
      const v = f(values[i]!);
      if (classify(v, support?.[i], positive) >= 0 || v < lo || v > hi) continue;
      const b = histogramBinOf(v, lo, hi, h.bins);
      const row = g * C + c;
      for (let t = 0; t < plan.perRow; t++) {
        const s = row * plan.perRow + t;
        const o = s * ZONAL_TARGET_STRIDE;
        if (plan.table[o] === b) {
          const k = cursor[s]!++;
          if (k < plan.table[o + 2]!) out[plan.table[o + 1]! + k] = v;
          break;
        }
      }
    }
  }
  checkCollected(plan, cursor);
  return out;
}

/** Throws unless every target received exactly its planned count: the collect pass must see the
 *  same pixels, labels and edges as the bin pass, or the ranks it resolves are wrong. */
export function checkCollected(plan: ZonalExactPlan, cursors: ArrayLike<number>): void {
  for (let s = 0; s * ZONAL_TARGET_STRIDE < plan.table.length; s++) {
    const o = s * ZONAL_TARGET_STRIDE;
    if (plan.table[o] === ZONAL_NO_TARGET) continue;
    const want = plan.table[o + 2]!;
    if (cursors[s] !== want) {
      throw new Error(
        `zonal: the exact pass found ${cursors[s]} values for row ${Math.floor(s / plan.perRow)} bin ${plan.table[o]}, the histogram ${want} (did the passes see different tiles?)`,
      );
    }
  }
}

/**
 * Exact numpy-"linear" quantiles from the collected values: `rows × quantiles.length`, NaN for an
 * empty row. A rank in the fixed-range under/overflow is censored to `lo` / `hi`, as R1's binned
 * estimate is. Throws if a target's collected count disagrees with its bin count (a pass that saw
 * different tiles or edges than the histogram).
 */
export function zonalExactQuantiles(h: ZonalHistogram, plan: ZonalExactPlan, collected: Float32Array): Float64Array {
  const rows = h.regions * h.channels;
  const nq = plan.quantiles.length;
  const out = new Float64Array(rows * nq).fill(Number.NaN);
  for (let r = 0; r < rows; r++) {
    const n = zonalCount(h.totals, r);
    if (n === 0) continue;
    const c = r % h.channels;
    const lo = h.edges[c * 2]!;
    const hi = h.edges[c * 2 + 1]!;
    const sorted = new Map<number, Float32Array>();
    for (let t = 0; t < plan.perRow; t++) {
      const s = (r * plan.perRow + t) * ZONAL_TARGET_STRIDE;
      const b = plan.table[s]!;
      if (b === ZONAL_NO_TARGET) break;
      sorted.set(b, collected.slice(plan.table[s + 1]!, plan.table[s + 1]! + plan.table[s + 2]!).sort());
    }
    const under = h.totals[r * ZONAL_TOTALS + 1]!;
    const valueAt = (k: number): number => {
      const b = binOfRank(h, r, k);
      if (b < 0) return k < under ? lo : hi;
      const vals = sorted.get(b);
      const before = under + (b > 0 ? h.cdf[r * h.bins + b - 1]! : 0);
      const v = vals?.[k - before];
      if (v === undefined || Number.isNaN(v)) throw new Error(`zonal: exact pass is missing rank ${k} of row ${r} (bin ${b})`);
      return v;
    };
    plan.quantiles.forEach((q, j) => {
      const p = q * (n - 1);
      const k = Math.floor(p);
      const a = valueAt(k);
      out[r * nq + j] = k + 1 < n ? a + (p - k) * (valueAt(k + 1) - a) : a;
    });
  }
  return out;
}

/** The exact-quantile reference: numpy-linear over the sorted values of one row (tests). */
export function zonalReferenceQuantile(values: Float32Array, q: number): number {
  return exactQuantile(Float64Array.from(values).sort(), q);
}

// ---------------------------------------------------------------------------------------------
// The regions × features table
// ---------------------------------------------------------------------------------------------

export interface ZonalTableOptions {
  /** Column prefix per channel. Default `c0`, `c1`, …. */
  readonly names?: readonly string[];
  /** Quantiles reported as `<name>_p<100q>`. Default `[0.1, 0.25, 0.5, 0.75, 0.9]`. */
  readonly quantiles?: readonly number[];
  /** Exact quantiles (`zonalExactQuantiles` over `quantiles`); absent ⇒ binned. */
  readonly exact?: Float64Array;
}

export interface ZonalTable {
  /** Label value of each table row (region index + 1). */
  readonly labels: Int32Array;
  /** Column name → one value per region. */
  readonly columns: ReadonlyMap<string, Float64Array>;
}

const pct = (q: number) => `p${Number((q * 100).toFixed(4))}`;

/**
 * One row per region; per channel `<name>_` + `pixels`, `supported`, `count`, `supportFraction`,
 * `validFraction`, `min`, `max`, `mean`, `median`, `iqr`, `binWidth` and `p<q>` per quantile.
 * Median and IQR are the q = 0.5 and q3 − q1 quantiles, binned or exact as the quantiles are.
 */
export function zonalTable(h: ZonalHistogram, opts: ZonalTableOptions = {}): ZonalTable {
  const quantiles = opts.quantiles ?? [0.1, 0.25, 0.5, 0.75, 0.9];
  const extra = [0.25, 0.5, 0.75].filter((q) => !quantiles.includes(q));
  const qs = [...quantiles, ...extra];
  if (opts.exact && opts.exact.length !== h.regions * h.channels * quantiles.length)
    throw new Error("zonalTable: exact has the wrong length");
  if (opts.exact && extra.length) throw new Error("zonalTable: exact quantiles must include 0.25, 0.5 and 0.75");
  // Two channels or two quantiles mapping to one name would silently overwrite each other's columns.
  const names = Array.from({ length: h.channels }, (_, c) => opts.names?.[c] ?? `c${c}`);
  const dup = (keys: string[]) => keys.find((k, i) => keys.indexOf(k) !== i);
  const dupName = dup(names);
  if (dupName !== undefined) throw new Error(`zonalTable: two channels are both named '${dupName}'`);
  const dupQ = dup(quantiles.map(pct));
  if (dupQ !== undefined) throw new Error(`zonalTable: two quantiles both map to column '${dupQ}' (4 decimal places of a percent)`);
  const R = h.regions;
  const columns = new Map<string, Float64Array>();
  const col = (name: string) => {
    let a = columns.get(name);
    if (!a) columns.set(name, (a = new Float64Array(R)));
    return a;
  };
  for (let c = 0; c < h.channels; c++) {
    const name = names[c]!;
    for (let g = 0; g < R; g++) {
      const r = g * h.channels + c;
      const t = r * ZONAL_TOTALS;
      const pixels = h.totals[t + ZONAL_SLOT.pixels]!;
      const supported = pixels - h.totals[t + ZONAL_SLOT.unsupported]!;
      const count = zonalCount(h.totals, r);
      const row = zonalRow(h, g, c);
      const qv = qs.map((q, j) => (opts.exact ? opts.exact[r * quantiles.length + j]! : histogramQuantile(row, q)));
      const at = (q: number) => qv[qs.indexOf(q)]!;
      col(`${name}_pixels`)[g] = pixels;
      col(`${name}_supported`)[g] = supported;
      col(`${name}_count`)[g] = count;
      col(`${name}_supportFraction`)[g] = pixels > 0 ? supported / pixels : Number.NaN;
      col(`${name}_validFraction`)[g] = supported > 0 ? count / supported : Number.NaN;
      col(`${name}_min`)[g] = h.moments[r * 4]!;
      col(`${name}_max`)[g] = h.moments[r * 4 + 1]!;
      col(`${name}_mean`)[g] = h.moments[r * 4 + 3]!;
      col(`${name}_median`)[g] = at(0.5);
      col(`${name}_iqr`)[g] = at(0.75) - at(0.25);
      col(`${name}_binWidth`)[g] = (row.hi - row.lo) / h.bins;
      for (const q of quantiles) col(`${name}_${pct(q)}`)[g] = at(q);
    }
  }
  return { labels: Int32Array.from({ length: R }, (_, g) => g + 1), columns };
}
