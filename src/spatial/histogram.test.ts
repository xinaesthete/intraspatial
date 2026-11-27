import { describe, expect, it } from "vite-plus/test";
import {
  exactQuantile,
  histogramBinOf,
  histogramCdf,
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

  it("the bin rule equals exact arithmetic away from edges and is monotone at them", () => {
    const [lo, hi] = histogramEdges(-1.3, 7.9);
    let prev = 0;
    let bad = 0;
    for (let i = 0; i <= 20000; i++) {
      const v = Math.fround(lo + (i / 20000) * (hi - lo));
      const b = histogramBinOf(v, lo, hi, 97);
      if (!Number.isInteger(b) || b < 0 || b > 96 || b < prev) bad++;
      else prev = b;
      // Exact real-arithmetic bin; where v is clearly inside a bin (not within 1e-4 of a bin of an
      // edge) the f32 rule must give exactly it, and next to an edge it may differ by one.
      const x = ((v - lo) * 97) / (hi - lo);
      const exact = Math.min(96, Math.floor(x));
      const nearEdge = Math.abs(x - Math.round(x)) < 1e-4;
      if (nearEdge ? Math.abs(b - exact) > 1 : b !== exact) bad++;
    }
    expect(bad).toBe(0);
  });

  it("keeps a constant field binned when v +/- 0.5 rounds to v (|v| >= 2^24)", () => {
    for (const v of [2 ** 24, -(2 ** 25), 3e38]) {
      const h = histogramCpu([v, v, v], { bins: 4 });
      expect(h.lo < h.hi && h.lo <= v && v <= h.hi, `${v}`).toBe(true);
      expect(Array.from(h.totals), `${v}`).toEqual([3, 0, 0, 0]);
      expect(h.cdf[3], `${v}`).toBe(3);
    }
  });

  it("does not overflow near f32 max", () => {
    const h = histogramCpu([0, 1e38, 2e38], { bins: 4 });
    expect(Array.from(h.counts)).toEqual([1, 0, 1, 1]);
    const wide = histogramCpu([-2e38, 0, 2e38], { bins: 4 });
    expect(Array.from(wide.counts)).toEqual([1, 0, 1, 1]);
    expect(wide.totals[0]).toBe(3);
  });

  it("rejects a fixed range that collapses in f32, and a bin count the CDF cannot use", () => {
    expect(() => histogramCpu([1], { range: [1, 1 + 2 ** -25] })).toThrow(/hi > lo in f32/);
    expect(() => histogramCdf(new Uint32Array([1]), 0)).toThrow(/positive integer/);
    expect(() => histogramCdf(new Uint32Array(3), 2)).toThrow(/dividing/);
  });

  it("quantile estimates are monotone in q across bins of different counts", () => {
    const row = histogramRow(histogramCpu([0, 1, 1, 1, 2], { bins: 2 }));
    let prev = -Infinity;
    let bad = 0;
    for (let i = 0; i <= 256; i++) {
      const e = histogramQuantile(row, i / 256);
      if (e < prev) bad++;
      prev = e;
    }
    expect(bad).toBe(0);
  });

  it("a censored rank returns the fixed edge, not the row min/max", () => {
    expect(histogramQuantile(histogramRow(histogramCpu([2], { bins: 4, range: [0, 1] })), 1)).toBe(1);
    expect(histogramQuantile(histogramRow(histogramCpu([-2], { bins: 4, range: [0, 1] })), 0)).toBe(0);
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
