// Transcript channels from points tiles: membership by weight vector, reach, the quality floor,
// and the refusal of overlapping channels (which gram.ts would otherwise mis-handle silently).
import { describe, expect, it } from "vitest";
import type { PointsTile } from "../datasource/points";
import { gramMatrix, rasterSizeForRadius } from "./gram";
import { transcriptChannels, type WeightedChannel } from "./transcriptChannels";

function tile(x: number, pts: Array<[number, number, number, number]>): PointsTile {
  return {
    id: { level: 0, x, y: 0, z: 0 },
    rect: { minX: 0, minY: 0, maxX: 1, maxY: 1 },
    count: pts.length,
    xs: Float32Array.from(pts.map((p) => p[0])),
    ys: Float32Array.from(pts.map((p) => p[1])),
    codes: Int32Array.from(pts.map((p) => p[2])),
    columns: { qv: Float32Array.from(pts.map((p) => p[3])) },
  };
}

const window = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
const gene = (label: string, code: number): WeightedChannel => ({ label, weights: new Map([[code, 1]]) });

describe("transcriptChannels", () => {
  const tiles = [
    tile(0, [
      [1, 1, 0, 30],
      [2, 2, 1, 30],
      [3, 3, 2, 30],
      [4, 4, 0, 10], // below the qv floor
    ]),
    tile(1, [
      [11, 5, 0, 30], // in the apron (r = 2)
      [15, 5, 1, 30], // beyond reach
      [5, 5, 3, 30], // in no channel
    ]),
  ];

  it("routes each point to its channel, applies reach and the floor, and accounts for every drop", () => {
    const out = transcriptChannels(tiles, [gene("A", 0), gene("B", 1)], { window, apron: 2, minimum: { column: "qv", value: 20 } });
    expect(out.clouds.map((c) => [c.label, c.xs.length])).toEqual([
      ["A", 2],
      ["B", 1],
    ]);
    expect(Array.from(out.clouds[0]?.xs ?? [])).toEqual([1, 11]);
    expect(out.clouds[0]?.weights).toBeUndefined(); // unit weights are left implicit
    expect(out.stats).toEqual({ considered: 7, outOfReach: 1, belowMinimum: 1, unselected: 2, perChannel: [2, 1] });
    expect(out.bbox).toEqual([0, 0, 10, 10]);
  });

  it("a gene set sums its genes, carrying per-code weights", () => {
    const set: WeightedChannel = {
      label: "AB",
      weights: new Map([
        [0, 1],
        [1, 0.5],
      ]),
    };
    const out = transcriptChannels(tiles, [set], { window, apron: 2 });
    expect(out.clouds[0]?.xs.length).toBe(4); // codes 0 (×3 in reach, no floor) and 1 (×1 in reach)
    expect(Array.from(out.clouds[0]?.weights ?? [])).toEqual([1, 0.5, 1, 1]);
  });

  it("refuses overlapping channels rather than letting gram.ts treat them as disjoint", () => {
    const overlap = [
      gene("A", 0),
      {
        label: "A+B",
        weights: new Map([
          [0, 1],
          [1, 1],
        ]),
      },
    ];
    expect(() => transcriptChannels(tiles, overlap, { window, apron: 2 })).toThrow(/code 0 is in both "A" and "A\+B"/);
  });

  it("refuses a floor on a column the tiles do not carry", () => {
    expect(() => transcriptChannels(tiles, [gene("A", 0)], { window, apron: 2, minimum: { column: "nope", value: 1 } })).toThrow(/"nope"/);
  });

  it("feeds gramMatrix directly — disjoint channels give a diagonal-only self term", () => {
    const out = transcriptChannels(tiles, [gene("A", 0), gene("B", 1)], { window, apron: 2 });
    const res = gramMatrix(out.clouds, { bbox: out.bbox, width: 16, height: 16, radius: 2 });
    expect(res.selfTerm[1]).toBe(0);
    expect(res.selfTerm[2]).toBe(0);
    expect(res.mass[0]).toBeGreaterThan(0);
  });
});

describe("rasterSizeForRadius", () => {
  it("puts ~3 px on a radius along the longer side and keeps pixels square", () => {
    expect(rasterSizeForRadius([0, 0, 3000, 1000], 30)).toEqual({ width: 300, height: 100, clamped: false });
    expect(rasterSizeForRadius([0, 0, 1000, 3000], 30)).toEqual({ width: 100, height: 300, clamped: false });
  });

  it("clamps and says so", () => {
    expect(rasterSizeForRadius([0, 0, 10_000, 100], 1)).toMatchObject({ width: 2048, clamped: true });
    expect(rasterSizeForRadius([0, 0, 10, 10], 100)).toEqual({ width: 16, height: 16, clamped: true });
  });
});
