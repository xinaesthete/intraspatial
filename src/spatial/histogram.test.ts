import { describe, expect, it } from "vite-plus/test";
import {
  exactQuantile,
  histogramBinOf,
  histogramCpu,
  histogramEdges,
  histogramQuantile,
  histogramRow,
  histogramSummary,
} from "./histogram";

/** Deterministic pseudo-random in [0, 1). */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

describe("histogramCpu", () => {
  it("bins signed values, closes the last edge, and conserves the count", () => {
    const v = [-3, -1, 0, 0.5, 1, 2, 3];
    const h = histogramCpu(v, { bins: 6 });
    expect([h.lo, h.hi]).toEqual([-3, 3]);
    expect(Array.from(h.counts)).toEqual([1, 0, 1, 2, 1, 2]);
    expect(Array.from(h.cdf)).toEqual([1, 1, 2, 4, 5, 7]);
    expect(Array.from(h.totals)).toEqual([7, 0, 0, 0]);
    expect(h.moments[0]).toBe(-3);
    expect(h.moments[1]).toBe(3);
    expect(h.moments[2]).toBeCloseTo(2.5, 12);
    expect(h.moments[3]).toBeCloseTo(2.5 / 7, 12);
  });

  it("skips and counts non-finite values; a constant field gets [v-0.5, v+0.5]", () => {
    const h = histogramCpu([2, Number.NaN, 2, Infinity, -Infinity, 2], { bins: 4 });
    expect([h.lo, h.hi]).toEqual([1.5, 2.5]);
    expect(Array.from(h.counts)).toEqual([0, 0, 3, 0]);
    expect(Array.from(h.totals)).toEqual([3, 0, 0, 3]);
    // Clamping to [min, max] makes every quantile of a constant field exact.
    expect(histogramSummary(histogramRow(h)).median).toBe(2);
    expect(histogramSummary(histogramRow(h)).iqr).toBe(0);
  });

  it("fixed range counts under/over separately and censors quantiles in them", () => {
    const h = histogramCpu([-1, 0, 0.25, 0.5, 1, 2, 3], { bins: 4, range: [0, 1] });
    expect(Array.from(h.counts)).toEqual([1, 1, 1, 1]);
    expect(Array.from(h.totals)).toEqual([4, 1, 2, 0]);
    expect(h.moments[0]).toBe(-1);
    const row = histogramRow(h);
    expect(histogramQuantile({ ...row, min: undefined, max: undefined }, 0)).toBe(0);
    expect(histogramQuantile({ ...row, min: undefined, max: undefined }, 1)).toBe(1);
    expect(() => histogramCpu([1], { range: [1, 1] })).toThrow(/hi > lo/);
  });

  it("splits inside/outside by a mask with one shared set of edges", () => {
    const v = [0, 1, 2, 3, 4, 5, 6, 7];
    const mask = [1, 1, 0, 0, 1, 0, Number.NaN, -0];
    const h = histogramCpu(v, { bins: 4, mask });
    expect(h.rows).toBe(2);
    expect([h.lo, h.hi]).toEqual([0, 7]);
    // inside: 0, 1, 4, 6 (NaN mask is non-zero); outside: 2, 3, 5, 7 (-0 is zero).
    expect(Array.from(h.counts)).toEqual([2, 0, 1, 1, 0, 2, 1, 1]);
    expect(Array.from(h.cdf)).toEqual([2, 2, 3, 4, 0, 2, 3, 4]);
    expect(Array.from(h.moments.subarray(4, 6))).toEqual([2, 7]);
  });

  it("the bin rule agrees with exact arithmetic away from edges and is monotone at them", () => {
    const [lo, hi] = histogramEdges(-1.3, 7.9);
    const range = Math.fround(hi - lo);
    let prev = 0;
    let bad = 0;
    for (let i = 0; i <= 20000; i++) {
      const v = Math.fround(lo + (i / 20000) * (hi - lo));
      const b = histogramBinOf(v, lo, range, 97);
      if (b < prev || b > 96) bad++;
      prev = b;
      const exact = Math.min(96, Math.floor(((v - lo) * 97) / (hi - lo)));
      if (Math.abs(b - exact) > 1) bad++;
    }
    expect(bad).toBe(0);
  });

  it("quantiles are within one bin width of exact", () => {
    const rnd = lcg(7);
    const v = new Float32Array(20000);
    for (let i = 0; i < v.length; i++) v[i] = Math.fround(Math.log(rnd() + 1e-6) * -2 - 3 + rnd()); // skewed, signed
    const sorted = Float64Array.from(v).sort();
    for (const bins of [16, 256]) {
      const h = histogramCpu(v, { bins });
      const s = histogramSummary(histogramRow(h));
      let worst = 0;
      for (const [q, est] of [
        [0.1, s.p10],
        [0.25, s.q1],
        [0.5, s.median],
        [0.75, s.q3],
        [0.9, s.p90],
      ] as const) {
        worst = Math.max(worst, Math.abs(est - exactQuantile(sorted, q)) / s.binWidth);
      }
      expect(worst, `bins=${bins}`).toBeLessThanOrEqual(1);
    }
  });
});
