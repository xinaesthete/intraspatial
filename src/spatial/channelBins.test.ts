import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { bucketPoints, type PointsTile, pointsGridAtLevel, type Rect } from "../datasource/points";
import { binComposition, type ChannelBins, channelBins, pooledBins } from "./channelBins";
import type { WeightedChannel } from "./transcriptChannels";

const bounds: Rect = { minX: 0, minY: 0, maxX: 100, maxY: 70 };
const channels: WeightedChannel[] = [
  { label: "a", weights: new Map([[0, 1]]) },
  {
    label: "bc",
    weights: new Map([
      [1, 1],
      [2, 2],
    ]),
  },
];

/** Points spread over `bounds`, some exactly on tile and bin edges, codes 0–3 (3 is in no channel). */
function points(n: number) {
  const rnd = mulberry32(1);
  const xs: number[] = [];
  const ys: number[] = [];
  const codes: number[] = [];
  for (let i = 0; i < n; i++) {
    const edge = i % 10 === 0;
    xs.push(edge ? 25 * Math.floor(rnd() * 5) : rnd() * 100);
    ys.push(edge ? 12.5 * Math.floor(rnd() * 6) : rnd() * 70);
    codes.push(Math.floor(rnd() * 4));
  }
  return { xs, ys, codes };
}

/** Bin index → per-channel counts, over all tiles. */
function merged(parts: readonly ChannelBins[]): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (const p of parts)
    p.bins.forEach((b, i) => {
      expect(out.has(b)).toBe(false); // a bin belongs to one tile
      out.set(b, Array.from(p.counts.subarray(i * p.channels, (i + 1) * p.channels)));
    });
  return out;
}

describe("channelBins", () => {
  const raw = points(4000);
  const bins = pointsGridAtLevel(bounds, 4000, 3); // 12.5-unit bins

  it("bins nest in tiles: binning tile by tile equals binning the whole at once", () => {
    const whole = [...bucketPoints(pointsGridAtLevel(bounds, 4000, 0), raw).values()];
    const tiled = [...bucketPoints(pointsGridAtLevel(bounds, 4000, 2), raw).values()];
    expect(tiled.length).toBeGreaterThan(4);
    const a = merged(whole.map((t) => channelBins(t, bins, channels)));
    const b = merged(tiled.map((t) => channelBins(t, bins, channels)));
    expect([...b.keys()].sort((p, q) => p - q)).toEqual([...a.keys()].sort((p, q) => p - q));
    for (const [k, v] of a) expect(b.get(k)).toEqual(v);
    // Every channel molecule is counted once: channel 0 is code 0, channel 1 is codes 1 and 2 (×2).
    const sum = (c: number) => [...a.values()].reduce((s, v) => s + v[c]!, 0);
    expect(sum(0)).toBe(raw.codes.filter((c) => c === 0).length);
    expect(sum(1)).toBe(raw.codes.reduce((s, c) => s + (c === 1 ? 1 : c === 2 ? 2 : 0), 0));
  });

  it("puts each bin where the bin grid says, and reports the tile's block", () => {
    const tile = [...bucketPoints(pointsGridAtLevel(bounds, 4000, 1), raw).values()].find((t) => t.id.x === 1 && t.id.y === 0)!;
    const cb = channelBins(tile, bins, channels);
    expect({ x0: cb.x0, y0: cb.y0, width: cb.width, height: cb.height }).toEqual({ x0: 4, y0: 0, width: 4, height: 4 });
    for (const b of cb.bins) {
      expect(b % bins.cols).toBeGreaterThanOrEqual(4);
      expect(Math.floor(b / bins.cols)).toBeLessThan(4);
    }
  });

  it("weights each molecule by 1 / the share of its gene the tile was read from", () => {
    const [full] = bucketPoints(pointsGridAtLevel(bounds, 4000, 0), raw).values();
    const sampled: PointsTile = { ...full!, sampleFraction: 0.25, featureFractions: new Map([[1, 0.5]]) };
    const a = channelBins(full!, bins, channels);
    const b = channelBins(sampled, bins, channels);
    expect(b.sampled).toEqual(a.sampled);
    const total = (cb: ChannelBins, c: number) => cb.counts.filter((_, i) => i % 2 === c).reduce((s, v) => s + v, 0);
    expect(total(b, 0)).toBeCloseTo(4 * total(a, 0), 9);
    const n1 = raw.codes.filter((c) => c === 1).length;
    const n2 = raw.codes.filter((c) => c === 2).length;
    expect(total(b, 1)).toBeCloseTo(2 * n1 + 4 * 2 * n2, 9);
  });

  it("keeps only molecules at or above a column minimum, and refuses a missing column", () => {
    const [whole] = bucketPoints(pointsGridAtLevel(bounds, 4000, 0), raw).values();
    const qv = Float32Array.from({ length: whole!.count }, (_, i) => (i % 3 === 0 ? 10 : 30));
    const tile: PointsTile = { ...whole!, columns: { ...whole!.columns, qv } };
    const read = (cb: ChannelBins) => cb.sampled.reduce((s, v) => s + v, 0);
    let n = 0; // codes 0–2 are in a channel, 3 is not
    for (let i = 0; i < whole!.count; i++) if (whole!.codes[i]! < 3 && qv[i]! >= 20) n++;
    expect(read(channelBins(tile, bins, channels, { column: "qv", value: 20 }))).toBe(n);
    expect(read(channelBins(tile, bins, channels, { column: "qv", value: 0 }))).toBe(read(channelBins(whole!, bins, channels)));
    expect(() => channelBins(whole!, bins, channels, { column: "nope", value: 1 })).toThrow(/no "nope" column/);
  });

  it("refuses bins coarser than the tile, and overlapping channels", () => {
    const tiles = [...bucketPoints(pointsGridAtLevel(bounds, 4000, 3), raw).values()];
    expect(() => channelBins(tiles[0]!, pointsGridAtLevel(bounds, 4000, 2), channels)).toThrow(/not a bin edge|no whole bin/);
    const overlap = [...channels, { label: "again", weights: new Map([[0, 1]]) }];
    expect(() => channelBins(tiles[0]!, bins, overlap)).toThrow(/more than one channel/);
  });
});

describe("pooledBins", () => {
  const raw = points(4000);
  const bins = pointsGridAtLevel(bounds, 4000, 3);
  const [whole] = bucketPoints(pointsGridAtLevel(bounds, 4000, 0), raw).values();
  const wholeBins = channelBins(whole!, bins, channels);
  const tiles = [...bucketPoints(pointsGridAtLevel(bounds, 4000, 2), raw).values()].map((t) => channelBins(t, bins, channels));

  it("pooled tile by tile, with the tiles around, equals pooling the whole element at once", () => {
    const all = pooledBins(wholeBins, [], bins);
    const want = new Map<number, { c: number[]; s: number }>();
    all.bins.forEach((b, i) => want.set(b, { c: Array.from(all.counts.subarray(i * 2, i * 2 + 2)), s: all.sampled[i]! }));
    let seen = 0;
    for (const t of tiles) {
      const p = pooledBins(t, tiles, bins);
      expect(Array.from(p.bins)).toEqual(Array.from(t.bins));
      p.bins.forEach((b, i) => {
        const w = want.get(b)!;
        expect(p.counts[i * 2]).toBeCloseTo(w.c[0]!, 9);
        expect(p.counts[i * 2 + 1]).toBeCloseTo(w.c[1]!, 9);
        expect(p.sampled[i]).toBe(w.s);
        seen++;
      });
    }
    expect(seen).toBe(all.bins.length);
  });

  it("sums the 3 × 3 block around a bin, and radius 0 is the bin alone", () => {
    expect(pooledBins(wholeBins, [], bins, 0).counts).toEqual(wholeBins.counts);
    const all = pooledBins(wholeBins, [], bins);
    const at = (cb: ChannelBins, b: number) => {
      const i = cb.bins.indexOf(b);
      return i < 0 ? 0 : cb.sampled[i]!;
    };
    const b = 2 * bins.cols + 3;
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += at(wholeBins, b + dy * bins.cols + dx);
    expect(at(all, b)).toBe(s);
  });

  it("leaves out bins with no molecules of their own, however full their neighbours", () => {
    // 4 × 1 bins of side 1, so a bin's index is its column. `cb` owns columns 0–2 and holds
    // molecules in 0 and 2; column 3 is a neighbouring block's, and full.
    const grid = pointsGridAtLevel({ minX: 0, minY: 0, maxX: 4, maxY: 1 }, 1, 2);
    expect(grid.cols).toBe(4);
    const cb: ChannelBins = {
      x0: 0,
      y0: 0,
      width: 3,
      height: 1,
      bins: Int32Array.from([0, 2]),
      channels: 1,
      counts: Float64Array.from([5, 7]),
      sampled: Int32Array.from([5, 7]),
    };
    const full: ChannelBins = {
      x0: 3,
      y0: 0,
      width: 1,
      height: 1,
      bins: Int32Array.from([3]),
      channels: 1,
      counts: Float64Array.from([100]),
      sampled: Int32Array.from([100]),
    };
    const p = pooledBins(cb, [full], grid);
    // Bin 1 sits between 0, 2 and nothing else — it would pool 12 — but holds none of its own.
    expect(Array.from(p.bins)).toEqual([0, 2]);
    expect(Array.from(p.counts)).toEqual([5, 7 + 100]);
    expect(Array.from(p.sampled)).toEqual([5, 7 + 100]);
  });
});

describe("binComposition", () => {
  it("gives Hellinger rows of unit norm, weighted by molecules, and zero weight below the read minimum", () => {
    const cb: ChannelBins = {
      x0: 0,
      y0: 0,
      width: 3,
      height: 1,
      bins: Int32Array.from([0, 1, 2]),
      channels: 2,
      counts: Float64Array.from([3, 1, 0, 8, 2, 2]),
      sampled: Int32Array.from([4, 2, 4]),
    };
    const d = binComposition(cb, 3);
    expect(Array.from(d.data)).toEqual([Math.sqrt(0.75), Math.sqrt(0.25), 0, 1, Math.sqrt(0.5), Math.sqrt(0.5)]);
    expect(Array.from(d.weights)).toEqual([4, 0, 4]);
    for (let i = 0; i < 3; i++) expect(d.data[2 * i]! ** 2 + d.data[2 * i + 1]! ** 2).toBeCloseTo(1, 12);
  });
});
