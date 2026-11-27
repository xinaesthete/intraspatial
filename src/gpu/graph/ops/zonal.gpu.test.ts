import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import { polygonDomain } from "#src/spatial/domain";
import { polygonRingBuffers, rasterisePolygonsCpu, zonalHistogramCpu } from "#src/spatial/zonal";
import { Graph, polygonRingSources, pull } from "../index";

// The zonal graph ops against their goldens, and chained: polygons → labels → per-region
// histogram with the labels never leaving the device.

const W = 57;
const H = 43;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

function mismatch(a: ArrayLike<number>, b: ArrayLike<number>, tol = 0): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    if (x === y || (Number.isNaN(x) && Number.isNaN(y))) continue;
    if (!(Math.abs(x - y) <= tol * Math.max(1, Math.abs(y)))) return `index ${i}: got ${x}, want ${y}`;
  }
  return undefined;
}

const polys = [
  polygonDomain([
    [2.2, 3.4],
    [30.7, 1.9],
    [24.1, 25.3],
    [5.5, 20.2],
  ]),
  polygonDomain([
    [20.3, 15.1],
    [54.6, 12.2],
    [50.1, 40.7],
    [26.6, 38.8],
  ]),
  polygonDomain([
    [1.4, 28.2],
    [15.9, 30.1],
    [9.3, 41.6],
  ]),
];

describe("zonal graph ops", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("rasterisePolygons → zonalHistogram on the device matches the goldens", async () => {
    const rnd = lcg(2);
    const field = Float32Array.from({ length: W * H }, (_, i) => (i % 17 === 0 ? 0 : Math.log(rnd() + 1e-3) + 4));
    field[5] = Number.NaN;
    const g = new Graph();
    const [labels] = g.op("rasterisePolygons", polygonRingSources(g, polygonRingBuffers(polys, { width: W, height: H })), {
      width: W,
      height: H,
    });
    const params = { regions: 3, bins: 48, valid: "positive" };
    const outs = g.op("zonalHistogram", { labels: labels!, field: g.grid(field, W, H) }, params);

    const gpuLabels = (await pull(g, labels!)).data!;
    const exact = rasterisePolygonsCpu(polys, { width: W, height: H });
    expect(mismatch(gpuLabels, exact)).toBeUndefined();
    expect(mismatch((await pull(g, labels!, { mode: "cpu" })).data!, exact)).toBeUndefined();

    const cpu = zonalHistogramCpu(exact, [{ values: field }], { regions: 3, bins: 48, channels: [{ valid: "positive" }] });
    const [counts, cdf, edges, totals, moments] = await Promise.all(outs.map(async (o) => (await pull(g, o)).data!));
    expect(mismatch(edges!, cpu.edges)).toBeUndefined();
    expect(mismatch(counts!, cpu.counts)).toBeUndefined();
    expect(mismatch(cdf!, cpu.cdf)).toBeUndefined();
    expect(mismatch(totals!, cpu.totals)).toBeUndefined();
    expect(mismatch(moments!, cpu.moments, 1e-5)).toBeUndefined();
    expect((await pull(g, outs[0]!)).shape).toEqual({ kind: "matrix", rows: 3, cols: 48 });
  });

  it("fixed range and f32 labels agree with cpu mode", async () => {
    const rnd = lcg(3);
    const labels = Float32Array.from({ length: W * H }, () => Math.floor(rnd() * 5));
    const field = Float32Array.from({ length: W * H }, () => rnd() * 4 - 1);
    const g = new Graph();
    const outs = g.op(
      "zonalHistogram",
      { labels: g.grid(labels, W, H), field: g.grid(field, W, H) },
      { regions: 4, bins: 16, range: "fixed", lo: 0, hi: 2 },
    );
    for (const i of [0, 2, 3])
      expect(mismatch((await pull(g, outs[i]!)).data!, (await pull(g, outs[i]!, { mode: "cpu" })).data!)).toBeUndefined();
  });
});
