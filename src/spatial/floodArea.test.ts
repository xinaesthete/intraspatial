import { describe, expect, it } from "vite-plus/test";
import { connectedRegion } from "#src/gpu/spatial/wandRegion";
import { bottleneckCosts, floodAreaAt, floodAreaByTolerance, floodJoinTolerance, floodLeaks, type RasterSeed } from "./floodArea";

/** A deterministic uniform [0, 1) stream (LCG), so fields are reproducible without an RNG dep. */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const fillCount = (d: ArrayLike<number>, w: number, h: number, seed: RasterSeed, t: number) =>
  connectedRegion(d, w, h, seed, t).reduce((a, v) => a + v, 0);

/** Every tolerance where the answer could change, and either side of it: the field's own values,
 *  midpoints between consecutive ones, and a value below all and above all. */
function probeTolerances(d: ArrayLike<number>): number[] {
  const v = [...new Set(Array.from(d).filter((x) => !Number.isNaN(x)))].sort((a, b) => a - b);
  const out = [v[0]! - 1, v.at(-1)! + 1, ...v];
  for (let i = 1; i < v.length; i++) out.push((v[i - 1]! + v[i]!) / 2);
  return out;
}

/** Tolerances where the curve's area differs from the fill's pixel count (empty when exact). */
function mismatches(d: ArrayLike<number>, w: number, h: number, seed: RasterSeed, tols = probeTolerances(d)) {
  const curve = floodAreaByTolerance(d, w, h, seed);
  return tols.map((t) => ({ t, curve: floodAreaAt(curve, t), fill: fillCount(d, w, h, seed, t) })).filter((r) => r.curve !== r.fill);
}

/** Two bowls side by side, joined by a ridge whose lowest point (the saddle) is `saddle`. */
function twoBasins(w: number, h: number, saddle: number): Float32Array {
  const d = new Float32Array(w * h);
  // A little jitter breaks the bowls' rings of exactly tied distances, as real fields have none.
  const rnd = lcg(5);
  const mid = w >> 1;
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const cx = c < mid ? mid / 2 : mid + mid / 2;
      const bowl = Math.hypot(c - cx, r - h / 2) / (w / 4);
      // The ridge column is high (3) everywhere except one gap row at `saddle`.
      d[r * w + c] = c === mid ? (r === h >> 1 ? saddle : 3) : Math.min(bowl + rnd() * 1e-3, 2.5);
    }
  return d;
}

describe("bottleneckCosts", () => {
  it("is the least path maximum, not the pixel's own value", () => {
    // 0 1 5 0 : the last pixel is cheap itself but reachable only over the 5.
    expect(Array.from(bottleneckCosts(Float32Array.from([0, 1, 5, 0]), 4, 1, 0))).toEqual([0, 1, 5, 5]);
  });

  it("goes around a high wall when there is a lower way", () => {
    const d = Float32Array.from([
      0,
      9,
      0, //
      1,
      9,
      2,
      1,
      1,
      1,
    ]);
    expect(Array.from(bottleneckCosts(d, 3, 3, 0))).toEqual([0, 9, 2, 1, 9, 2, 1, 1, 1]);
  });

  it("never reaches through NaN", () => {
    const c = bottleneckCosts(Float32Array.from([0, Number.NaN, 0]), 3, 1, 0);
    expect([c[0], Number.isNaN(c[1]!), Number.isNaN(c[2]!)]).toEqual([0, true, true]);
  });
});

describe("floodAreaByTolerance", () => {
  it("matches connectedRegion's pixel count on uniform noise, at every distinct value and between", () => {
    const rnd = lcg(7);
    const w = 23;
    const h = 17;
    const d = Float32Array.from({ length: w * h }, () => rnd() * 4);
    expect(mismatches(d, w, h, { col: 11, row: 8 })).toEqual([]);
  });

  it("matches on a quantised field with many ties, and with NaN holes", () => {
    const rnd = lcg(11);
    const w = 19;
    const h = 21;
    const d = Float32Array.from({ length: w * h }, () => (rnd() < 0.08 ? Number.NaN : Math.floor(rnd() * 6) * 0.5));
    expect(mismatches(d, w, h, { col: 4, row: 15 })).toEqual([]);
  });

  it("matches from a fractional seed whose nearest pixel is out of range at low tolerances", () => {
    const rnd = lcg(3);
    const w = 15;
    const h = 15;
    const d = Float32Array.from({ length: w * h }, () => rnd() * 2);
    // The cell around (7.4, 6.6): make the nearest pixel (7, 7) the highest of its four corners, so
    // low tolerances start the fill from another corner.
    d[7 * w + 7] = 1.9;
    d[6 * w + 7] = 0.1;
    d[6 * w + 8] = 0.3;
    d[7 * w + 8] = 0.2;
    expect(mismatches(d, w, h, { col: 7.4, row: 6.6 })).toEqual([]);
    // …and at a seed on the raster's edge, where corners clamp onto each other.
    expect(mismatches(d, w, h, { col: -0.3, row: 14.2 })).toEqual([]);
  });

  it("leaks into the second basin at exactly the saddle", () => {
    const w = 40;
    const h = 24;
    const saddle = Math.fround(2.75);
    const d = twoBasins(w, h, saddle);
    const seed = { col: 10, row: 12 };
    expect(mismatches(d, w, h, seed)).toEqual([]);
    const curve = floodAreaByTolerance(d, w, h, seed);
    const leaks = floodLeaks(curve);
    expect(leaks.map((k) => curve.tol[k])).toEqual([saddle]);
    // Just below the saddle the region is one basin; at it, both (everything but the ridge wall).
    const below = floodAreaAt(curve, saddle - 1e-6);
    const at = floodAreaAt(curve, saddle);
    expect(below).toBe(fillCount(d, w, h, seed, saddle - 1e-6));
    expect(at).toBe(w * h - (h - 1));
    expect(at).toBeGreaterThan(1.9 * below);
    // The far basin's floor joins at the saddle.
    expect(floodJoinTolerance(curve, 12 * w + 30)).toBe(saddle);
  });

  it("is empty below the seed's own value and the whole raster above every value", () => {
    const d = Float32Array.from([2, 3, 4, 5]);
    const curve = floodAreaByTolerance(d, 2, 2, { col: 0, row: 0 });
    expect([floodAreaAt(curve, 1.9), floodAreaAt(curve, 2), floodAreaAt(curve, 99)]).toEqual([0, 1, 4]);
    expect(Array.from(curve.tol)).toEqual([2, 3, 4, 5]);
  });

  it("takes in +Inf pixels at tolerance +Inf, as the fill does", () => {
    const d = Float32Array.from([0, Number.POSITIVE_INFINITY, 1]);
    expect(mismatches(d, 3, 1, { col: 0, row: 0 }, [0, 1, 1e30, Number.POSITIVE_INFINITY])).toEqual([]);
    expect(floodAreaAt(floodAreaByTolerance(d, 3, 1, { col: 0, row: 0 }), Number.POSITIVE_INFINITY)).toBe(3);
  });

  it("gives each pixel the least tolerance at which the fill includes it, across start switches", () => {
    /** The first probed tolerance at which the fill includes each pixel (NaN if none). */
    const bruteJoins = (d: ArrayLike<number>, w: number, h: number, seed: RasterSeed) => {
      const tols = probeTolerances(d).sort((a, b) => a - b);
      const out = new Float64Array(w * h).fill(Number.NaN);
      for (const t of tols) {
        const r = connectedRegion(d, w, h, seed, t);
        for (let i = 0; i < r.length; i++) if (r[i] && Number.isNaN(out[i]!)) out[i] = t;
      }
      return out;
    };
    const joins = (d: ArrayLike<number>, w: number, h: number, seed: RasterSeed) => {
      const curve = floodAreaByTolerance(d, w, h, seed);
      return Float64Array.from({ length: w * h }, (_, i) => floodJoinTolerance(curve, i));
    };
    // The nearest pixel (d 2) is out of range below 2, so the fill starts at pixel 0 first.
    const small = Float32Array.from([0, 4, 4, 2]);
    expect(Array.from(joins(small, 2, 2, { col: 0.6, row: 0.6 }))).toEqual(Array.from(bruteJoins(small, 2, 2, { col: 0.6, row: 0.6 })));
    expect(floodJoinTolerance(floodAreaByTolerance(small, 2, 2, { col: 0.6, row: 0.6 }), 0)).toBe(0);
    // A NaN nearest pixel: the fill only ever starts from corners.
    const holed = Float32Array.from([0, 1, 3, Number.NaN]);
    expect(Array.from(joins(holed, 2, 2, { col: 0.6, row: 0.6 }))).toEqual(Array.from(bruteJoins(holed, 2, 2, { col: 0.6, row: 0.6 })));
    const rnd = lcg(3);
    const noise = Float32Array.from({ length: 15 * 15 }, () => rnd() * 2);
    noise[7 * 15 + 7] = 1.9;
    noise[6 * 15 + 7] = 0.1;
    const seed = { col: 7.4, row: 6.6 };
    expect(Array.from(joins(noise, 15, 15, seed))).toEqual(Array.from(bruteJoins(noise, 15, 15, seed)));
  });
});
