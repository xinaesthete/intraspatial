// Channel composition per bin: the descriptor a niche map clusters (docs/emeans-clustering.md).
//
// Bins are a quadtree level over the element's extent — the grid `pointsGridAtLevel` cuts, one level
// finer or more than the tiles — so every points tile at a coarser level is a union of whole bins,
// and a bin's molecules all come from the one tile that owns them. That is what lets a centroid table
// fitted once on a whole-element sample label tiles one at a time as they load: a bin's label depends
// on its own molecules only, never on which other tiles happen to be in view.
//
// Molecules are weighted as `transcriptChannels` weights them: the channel's weight for the gene over
// the share of that gene's points the tile was read from (`pointFraction`). A bin's counts are then
// molecules it stands for, the same at every sampling depth, so a coarse-level sample and a deeper
// tile describe a bin on one scale.

import { type PointsGrid, type PointsTile, pointFraction } from "../datasource/points";
import type { WeightedChannel } from "./transcriptChannels";

/** One tile's bins that hold a channel molecule, with their per-channel counts. */
export interface ChannelBins {
  /** The tile's block of the bin grid: bin columns `x0 … x0 + width − 1`, rows `y0 … y0 + height − 1`. */
  readonly x0: number;
  readonly y0: number;
  readonly width: number;
  readonly height: number;
  /** Bin indices `y · cols + x` on the bin grid, ascending. Bins with no channel molecule are absent. */
  readonly bins: Int32Array;
  readonly channels: number;
  /** `bins.length × channels`, row-major: molecules each bin stands for, per channel. */
  readonly counts: Float64Array;
  /** Channel molecules actually read per bin: how well `counts` is estimated. */
  readonly sampled: Int32Array;
}

/** The bin `edge` falls on, when it is one; throws when it is not (tiles finer than bins, or on another grid). */
function binEdge(edge: number, origin: number, size: number): number {
  const f = (edge - origin) / size;
  const i = Math.round(f);
  if (Math.abs(f - i) > 1e-6)
    throw new Error(`channelBins: tile edge ${edge} is not a bin edge; bins must be a finer level of the tiles' grid`);
  return i;
}

/** Count `tile`'s channel molecules into the bins of `bins` (a finer level over the same extent). */
export function channelBins(tile: PointsTile, bins: PointsGrid, channels: readonly WeightedChannel[]): ChannelBins {
  const { bounds, tileSize: size, cols, rows } = bins;
  const r = tile.rect;
  // A tile on the extent's far edge is clipped there, and so is the bin grid's last bin.
  const x0 = binEdge(r.minX, bounds.minX, size);
  const y0 = binEdge(r.minY, bounds.minY, size);
  const x1 = r.maxX >= bounds.maxX ? cols : binEdge(r.maxX, bounds.minX, size);
  const y1 = r.maxY >= bounds.maxY ? rows : binEdge(r.maxY, bounds.minY, size);
  const width = x1 - x0;
  const height = y1 - y0;
  if (width < 1 || height < 1) throw new Error(`channelBins: tile ${JSON.stringify(r)} spans no whole bin`);

  const C = channels.length;
  const channelOf = new Map<number, number>();
  const weightOf = new Map<number, number>();
  channels.forEach((ch, c) => {
    for (const [code, w] of ch.weights) {
      if (!(w > 0)) continue;
      if (channelOf.has(code)) throw new Error(`channelBins: feature code ${code} is in more than one channel`);
      channelOf.set(code, c);
      weightOf.set(code, w);
    }
  });
  const local = new Float64Array(width * height * C);
  const read = new Int32Array(width * height);
  for (let i = 0; i < tile.count; i++) {
    const code = tile.codes[i]!;
    const c = channelOf.get(code);
    if (c === undefined) continue;
    // Clamped to the tile's own block: a point the tile owns is binned inside it, whatever rounding says.
    const bx = Math.min(width - 1, Math.max(0, Math.floor((tile.xs[i]! - bounds.minX) / size) - x0));
    const by = Math.min(height - 1, Math.max(0, Math.floor((tile.ys[i]! - bounds.minY) / size) - y0));
    const b = by * width + bx;
    local[b * C + c]! += weightOf.get(code)! / pointFraction(tile, code);
    read[b]! += 1;
  }
  let m = 0;
  for (let b = 0; b < width * height; b++) if (read[b]! > 0) m++;
  const out = { bins: new Int32Array(m), counts: new Float64Array(m * C), sampled: new Int32Array(m) };
  let o = 0;
  for (let b = 0; b < width * height; b++) {
    if (read[b] === 0) continue;
    out.bins[o] = (y0 + Math.floor(b / width)) * cols + x0 + (b % width);
    out.counts.set(local.subarray(b * C, b * C + C), o * C);
    out.sampled[o] = read[b]!;
    o++;
  }
  return { x0, y0, width, height, channels: C, ...out };
}

/**
 * `own`'s bins with each one's counts summed over the `(2·radius + 1)²` bins centred on it: a
 * neighbourhood mix, estimated from about 9× the molecules of one bin at radius 1. Bins near the
 * block's edge reach into `neighbours` — the bins of the tiles around it (any others are ignored,
 * and `own` among them is skipped) — so labelling them waits for those tiles; with them, the result
 * equals pooling the whole element at once. Only bins `own` holds molecules in are returned: pooling
 * estimates a bin's mix, it does not invent tissue where none was read (a vessel lumen stays empty).
 */
export function pooledBins(own: ChannelBins, neighbours: Iterable<ChannelBins>, bins: PointsGrid, radius = 1): ChannelBins {
  const { x0, y0, width, height, channels: C } = own;
  const r = radius;
  const W = width + 2 * r;
  const H = height + 2 * r;
  const counts = new Float64Array(W * H * C);
  const read = new Int32Array(W * H);
  const add = (cb: ChannelBins) => {
    if (cb.channels !== C) throw new Error(`pooledBins: a neighbour has ${cb.channels} channels, not ${C}`);
    cb.bins.forEach((b, i) => {
      const lx = (b % bins.cols) - x0 + r;
      const ly = Math.floor(b / bins.cols) - y0 + r;
      if (lx < 0 || lx >= W || ly < 0 || ly >= H) return;
      const o = ly * W + lx;
      for (let c = 0; c < C; c++) counts[o * C + c]! += cb.counts[i * C + c]!;
      read[o]! += cb.sampled[i]!;
    });
  };
  add(own);
  for (const n of neighbours) if (n !== own) add(n);
  const m = own.bins.length;
  const out = { counts: new Float64Array(m * C), sampled: new Int32Array(m) };
  own.bins.forEach((b, i) => {
    const cx = (b % bins.cols) - x0 + r;
    const cy = Math.floor(b / bins.cols) - y0 + r;
    for (let y = cy - r; y <= cy + r; y++)
      for (let x = cx - r; x <= cx + r; x++) {
        const o = y * W + x;
        for (let c = 0; c < C; c++) out.counts[i * C + c]! += counts[o * C + c]!;
        out.sampled[i]! += read[o]!;
      }
  });
  return { ...own, ...out };
}

/** Bins as rows to cluster: `data` is `n × d`, `weights` has one entry per row. */
export interface BinDescriptors {
  readonly n: number;
  readonly d: number;
  readonly data: Float64Array;
  readonly weights: Float64Array;
}

/**
 * Each bin's channel mix as Hellinger coordinates `√(count_c / total)`. Euclidean distance between
 * two rows is then the Hellinger distance between the mixes (×√2), which no single abundant channel
 * dominates the way it would raw proportions. A row's weight is the molecules its bin stands for, so
 * a fit counts molecules rather than area; a bin with fewer than `minSampled` molecules read is
 * weight 0 — too few to say what mix it is — and is left out of a fit and labelled −1.
 */
export function binComposition(b: ChannelBins, minSampled: number): BinDescriptors {
  const n = b.bins.length;
  const d = b.channels;
  const data = new Float64Array(n * d);
  const weights = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let total = 0;
    for (let c = 0; c < d; c++) total += b.counts[i * d + c]!;
    if (!(total > 0)) continue;
    for (let c = 0; c < d; c++) data[i * d + c] = Math.sqrt(b.counts[i * d + c]! / total);
    if (b.sampled[i]! >= minSampled) weights[i] = total;
  }
  return { n, d, data, weights };
}
