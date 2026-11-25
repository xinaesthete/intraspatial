import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { rectDomain } from "./domain";
import { assignCentroids, assignNearest, emeans, emeansAlpha0, emeansBatch, emeansFit, emeansSchedule, lloyd } from "./emeans";

interface Blob {
  w: number;
  mu: number[];
  s: number;
}

/** n points from a weighted mixture of isotropic Gaussians, row-major. */
function mixture(seed: number, n: number, d: number, blobs: Blob[]): Float64Array {
  const rnd = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  const total = blobs.reduce((a, b) => a + b.w, 0);
  const x = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    let u = rnd() * total;
    let c = 0;
    while (u > blobs[c]!.w) u -= blobs[c++]!.w;
    const blob = blobs[c]!;
    for (let t = 0; t < d; t++) x[i * d + t] = blob.mu[t]! + blob.s * gauss();
  }
  return x;
}

/** The blog's "five blobs on a line, light middle blob" trap, with overlapping tails. Blobs
 *  far enough apart that no point ever falls between them trap e-means as surely as Lloyd: an
 *  EMA centroid stays in the convex hull of the points assigned to it, so it cannot cross
 *  empty space. The escape needs the overlap. */
const LINE: Blob[] = [0, 1, 2, 3, 4].map((i) => ({ w: i === 2 ? 0.3 : 1, mu: [i * 2.2, 0], s: 0.6 }));

describe("emeansSchedule", () => {
  it("follows the published scaling laws", () => {
    // B = 0.00439·k^0.135·n^0.876·T^0.813 ≈ 7.5·10³ at (10⁶, 1024, 5) → nearer power of two.
    expect(emeansBatch(1e6, 1024, 5)).toBe(8192);
    expect(emeansAlpha0(8192, 1e6, 1024, 5)).toBeCloseTo(0.8244 * 1024 ** -0.4617 * 8192 ** 0.9003 * 1e6 ** -0.4498 * 5 ** -0.1057, 12);
  });

  it("rounds B to a power of two, caps it at n/2, and the default law is 8× quality", () => {
    const cases: [number, number][] = [
      [5e4, 16],
      [1e6, 8192],
      [3e5, 100],
    ];
    for (const [n, k] of cases) {
      const q = emeansBatch(n, k, 5);
      expect(Math.log2(q) % 1).toBe(0);
      expect(emeansBatch(n, k, 5, "default")).toBe(q * 8);
    }
    expect(emeansBatch(100, 1000, 25, "default")).toBe(50);
  });

  it("keeps α₀ in (0, 1] and honours overrides", () => {
    expect(emeansSchedule(1e7, 4, 50).alpha0).toBe(1);
    const s = emeansSchedule(1000, 8, 5, { batch: 100, alpha0: 0.3 });
    expect(s).toEqual({ batch: 100, alpha0: 0.3, stepsPerEpoch: 10 });
  });
});

describe("emeans", () => {
  it("is Lloyd at the corner B = n, α = 1 (constant, no revival)", () => {
    const n = 2000;
    const d = 3;
    const k = 6;
    const x = mixture(
      3,
      n,
      d,
      [0, 1, 2, 3, 4, 5].map((c) => ({ w: 1 + c, mu: [c, c % 2, c % 3], s: 0.7 })),
    );
    const init = x.slice(0, k * d);
    const e = emeans(x, n, d, k, { epochs: 7, batch: n, alpha0: 1, schedule: "constant", revive: 0, init });
    const l = lloyd(x, n, d, k, { iters: 7, init });
    for (let i = 0; i < k * d; i++) expect(e.centroids[i]).toBeCloseTo(l.centroids[i]!, 10);
    expect(e.labels).toEqual(l.labels);
    expect(l.splits).toBe(0);
  });

  it("is deterministic for a seed", () => {
    const x = mixture(
      4,
      3000,
      5,
      [0, 1, 2].map((c) => ({ w: 1, mu: [c, 0, c, 0, c], s: 1 })),
    );
    const a = emeans(x, 3000, 5, 8, { seed: 11 });
    const b = emeans(x, 3000, 5, 8, { seed: 11 });
    const c = emeans(x, 3000, 5, 8, { seed: 12 });
    expect(a.centroids).toEqual(b.centroids);
    expect(a.mse).toBe(b.mse);
    expect(c.centroids).not.toEqual(a.centroids);
  });

  it("returns labels, counts and MSE consistent with its centroids", () => {
    const n = 3000;
    const x = mixture(
      5,
      n,
      4,
      [0, 1, 2, 3].map((c) => ({ w: 1, mu: [c * 3, 0, 0, c], s: 1 })),
    );
    const r = emeans(x, n, 4, 5);
    const labels = new Int32Array(n);
    expect(assignNearest(x, n, 4, r.centroids, 5, labels)).toBe(r.mse);
    expect(r.labels).toEqual(labels);
    expect(r.counts.reduce((a, b) => a + b, 0)).toBe(n);
  });

  it("escapes the local minima Lloyd stays in (5 epochs against Lloyd's 25)", () => {
    const n = 4000;
    const x = mixture(1, n, 2, LINE);
    const seeds = Array.from({ length: 12 }, (_, s) => s);
    const em = seeds.map((seed) => emeans(x, n, 2, 5, { seed }).mse);
    const ll = seeds.map((seed) => lloyd(x, n, 2, 5, { seed }).mse);
    const best = Math.min(...em, ...ll);
    const stuck = (mse: number[]) => mse.filter((m) => m > 1.01 * best).length;
    // Measured: e-means 0/12, Lloyd 3/12 (at n = 20 000 over 30 seeds: 0/30 vs 6/30).
    expect(stuck(em)).toBe(0);
    expect(stuck(ll)).toBeGreaterThanOrEqual(2);
  });

  it("revives dead clusters by splitting live ones", () => {
    const n = 3000;
    const d = 4;
    const k = 8;
    const x = mixture(
      6,
      n,
      d,
      [0, 1, 2, 3].map((c) => ({ w: 1, mu: [c * 2, 0, c, 0], s: 1 })),
    );
    // Half the seeds far from all data: no point is ever nearest to them.
    const init = new Float64Array(k * d);
    for (let j = 0; j < k; j++) for (let t = 0; t < d; t++) init[j * d + t] = j < 4 ? x[j * d + t]! : 1e3 + j;
    const off = emeans(x, n, d, k, { init, revive: 0 });
    const on = emeans(x, n, d, k, { init });
    expect(off.splits).toBe(0);
    expect(Array.from(off.counts).filter((c) => c === 0)).toHaveLength(4);
    expect(on.splits).toBeGreaterThanOrEqual(4);
    expect(Array.from(on.counts).every((c) => c > 0)).toBe(true);
    expect(on.mse).toBeLessThan(off.mse);
  });

  it("leaves revival off by default for d ≤ 3, where the authors found it hurts", () => {
    const n = 2000;
    const x = mixture(7, n, 2, LINE);
    const init = new Float64Array([0, 0, 2, 0, 4, 0, 500, 500, 600, 600]);
    expect(emeans(x, n, 2, 5, { init }).splits).toBe(0);
    expect(emeans(x, n, 2, 5, { init, revive: 0.1 }).splits).toBeGreaterThan(0);
  });

  it("seeds with k-means++ when asked, which finds well-separated clusters random seeding misses", () => {
    // Three tight, far-apart blobs: no point lies between them, so a bad start is never repaired.
    const n = 900;
    const x = mixture(
      8,
      n,
      4,
      [0, 1, 2].map((c) => ({ w: 1, mu: [c * 50, 0, 0, c * 50], s: 0.5 })),
    );
    const seeds = Array.from({ length: 10 }, (_, s) => s);
    const random = seeds.map((seed) => emeans(x, n, 4, 3, { seed }).mse);
    const plusPlus = seeds.map((seed) => emeans(x, n, 4, 3, { seed, seeding: "kmeans++" }).mse);
    const best = Math.min(...random, ...plusPlus);
    expect(plusPlus.every((m) => m < 1.01 * best)).toBe(true);
    expect(random.some((m) => m > 2 * best)).toBe(true);
  });

  it("rejects malformed input", () => {
    const x = new Float64Array(20);
    expect(() => emeans(x, 10, 2, 0)).toThrow(/k ≥ 1/);
    expect(() => emeans(x, 10, 2, 11)).toThrow(/at least k/);
    expect(() => emeans(x, 11, 2, 3)).toThrow(/n·d/);
    expect(() => emeans(x, 10, 2, 3, { init: [1, 2] })).toThrow(/k·d/);
  });
});

/** Rows of `x` (dimension d) with row i repeated `times[i]` times (0 drops it), in order. */
function repeatRows(x: Float64Array, d: number, times: ArrayLike<number>): Float64Array {
  const out: number[] = [];
  for (let i = 0; i < times.length; i++) for (let r = 0; r < times[i]!; r++) for (let t = 0; t < d; t++) out.push(x[i * d + t]!);
  return Float64Array.from(out);
}

describe("weights and masks", () => {
  const d = 3;
  const k = 5;
  const n = 1500;
  const x = mixture(
    9,
    n,
    d,
    [0, 1, 2, 3, 4].map((c) => ({ w: 1 + c, mu: [c * 1.5, c % 2, (c * 7) % 3], s: 0.6 })),
  );
  // Every third point weighs 2, every seventh 0, the rest 1.
  const w = Float64Array.from({ length: n }, (_, i) => (i % 7 === 0 ? 0 : i % 3 === 0 ? 2 : 1));
  const xDup = repeatRows(x, d, w);
  const nDup = xDup.length / d;
  // Seeds that are kept points, so the two runs start from the same centroids.
  const init = x.slice(d, d + k * d);

  it("a weight of 2 is the point twice and 0 is no point (Lloyd)", () => {
    const a = lloyd(x, n, d, k, { weights: w, init, iters: 10 });
    const b = lloyd(xDup, nDup, d, k, { init, iters: 10 });
    for (let i = 0; i < k * d; i++) expect(a.centroids[i]).toBeCloseTo(b.centroids[i]!, 10);
    expect(Array.from(a.mass)).toEqual(Array.from(b.counts));
    expect(a.mse).toBeCloseTo(b.mse, 12);
    expect(Array.from(a.labels).filter((_, i) => w[i] === 0)).toEqual(Array(Math.ceil(n / 7)).fill(-1));
  });

  it("a weight of 2 is the point twice (e-means at the Lloyd corner)", () => {
    const corner = { epochs: 6, alpha0: 1, schedule: "constant", revive: 0, init } as const;
    const a = emeans(x, n, d, k, { ...corner, batch: n, weights: w });
    const b = emeans(xDup, nDup, d, k, { ...corner, batch: nDup });
    for (let i = 0; i < k * d; i++) expect(a.centroids[i]).toBeCloseTo(b.centroids[i]!, 10);
    expect(a.mse).toBeCloseTo(b.mse, 12);
  });

  it("a weight of 0 is exactly the point removed, for every seeding", () => {
    const zeroed = Float64Array.from({ length: n }, (_, i) => (i % 7 === 0 ? 0 : 1));
    const kept = repeatRows(x, d, zeroed);
    const m = kept.length / d;
    for (const seeding of ["random", "kmeans++"] as const) {
      const a = emeans(x, n, d, k, { seed: 3, seeding, weights: zeroed });
      const b = emeans(kept, m, d, k, { seed: 3, seeding });
      expect(a.centroids).toEqual(b.centroids);
      expect(a.mse).toBe(b.mse);
      expect(Array.from(a.labels).filter((l) => l >= 0)).toEqual(Array.from(b.labels));
    }
  });

  it("is still Lloyd at the corner B = n, α = 1 with weights", () => {
    const corner = { epochs: 7, batch: n, alpha0: 1, schedule: "constant", revive: 0, init, weights: w } as const;
    const e = emeans(x, n, d, k, corner);
    const l = lloyd(x, n, d, k, { iters: 7, init, weights: w });
    for (let i = 0; i < k * d; i++) expect(e.centroids[i]).toBeCloseTo(l.centroids[i]!, 10);
    expect(e.labels).toEqual(l.labels);
  });

  it("does not depend on the weights' units", () => {
    const a = emeans(x, n, d, k, { seed: 4, seeding: "kmeans++", weights: w });
    const b = emeans(x, n, d, k, { seed: 4, seeding: "kmeans++", weights: w.map((v) => v * 1000) });
    for (let i = 0; i < k * d; i++) expect(a.centroids[i]).toBeCloseTo(b.centroids[i]!, 9);
    expect(b.mass.reduce((s, v) => s + v, 0)).toBeCloseTo(1000 * w.reduce((s, v) => s + v, 0), 6);
  });

  it("never seeds from a point of weight 0", () => {
    // Far outliers of weight 0: any seed drawn from them would leave a centroid out there.
    const far = Float64Array.from(x);
    const zero = new Float64Array(n).fill(1);
    for (let i = 0; i < n; i += 5) {
      far[i * d] = 1e6;
      zero[i] = 0;
    }
    for (let seed = 0; seed < 8; seed++)
      for (const seeding of ["random", "kmeans++"] as const) {
        const fit = emeansFit(far, n, d, k, { seed, seeding, weights: zero, epochs: 1 });
        for (let j = 0; j < k; j++) expect(Math.abs(fit.centroids[j * d]!)).toBeLessThan(100);
      }
  });

  it("is deterministic for a seed, with weights", () => {
    const a = emeans(x, n, d, k, { seed: 21, weights: w, seeding: "kmeans++" });
    const b = emeans(x, n, d, k, { seed: 21, weights: w, seeding: "kmeans++" });
    expect(a.centroids).toEqual(b.centroids);
    expect(a.labels).toEqual(b.labels);
    expect(a.mse).toBe(b.mse);
  });

  it("assign reproduces the fit's labels, and a tile labelled alone agrees with the whole", () => {
    const fit = emeansFit(x, n, d, k, { seed: 2, weights: w });
    const whole = assignCentroids(fit, x, n, { weights: w });
    expect(whole.labels).toEqual(emeans(x, n, d, k, { seed: 2, weights: w }).labels);
    const nearestOnly = new Int32Array(n);
    assignNearest(x, n, d, fit.centroids, k, nearestOnly);
    expect(Array.from(whole.labels)).toEqual(Array.from(nearestOnly, (l, i) => (w[i] === 0 ? -1 : l)));
    // Points 600–899 as a tile of their own.
    const tile = assignCentroids(fit, x.subarray(600 * d, 900 * d), 300, { weights: w.subarray(600, 900) });
    expect(tile.labels).toEqual(whole.labels.subarray(600, 900));
    // Fit mass is the finalisation epoch's; the final assignment moves only a little of it.
    const total = w.reduce((s, v) => s + v, 0);
    expect(fit.mass.reduce((s, v) => s + v, 0)).toBeCloseTo(total, 6);
    expect(whole.mass.reduce((s, v) => s + v, 0)).toBeCloseTo(total, 6);
  });

  it("masks by position: outside the domain is weight 0 and label −1", () => {
    // Positions independent of the descriptor: a 30 × 50 grid, masked to its left half.
    const xs = Float64Array.from({ length: n }, (_, i) => i % 30);
    const ys = Float64Array.from({ length: n }, (_, i) => Math.floor(i / 30));
    const mask = { domain: rectDomain([0, 0, 14.5, 50]), xs, ys };
    const inside = Float64Array.from(xs, (v) => (v <= 14.5 ? 1 : 0));
    const a = emeans(x, n, d, k, { seed: 5, mask });
    const b = emeans(x, n, d, k, { seed: 5, weights: inside });
    expect(a.centroids).toEqual(b.centroids);
    expect(a.labels).toEqual(b.labels);
    expect(Array.from(a.labels).every((l, i) => (xs[i]! > 14.5 ? l === -1 : l >= 0))).toBe(true);
    expect(a.counts.reduce((s, c) => s + c, 0)).toBe(15 * 50);
    // Mask and weights compose.
    const both = assignCentroids(a, x, n, { mask, weights: w });
    expect(Array.from(both.labels).every((l, i) => (xs[i]! > 14.5 || w[i] === 0 ? l === -1 : l >= 0))).toBe(true);
  });

  it("rejects bad weights and a sample left with fewer than k points", () => {
    expect(() => emeans(x, n, d, k, { weights: w.subarray(0, 10) })).toThrow(/weights have 10/);
    expect(() => emeans(x, n, d, k, { weights: w.map((v, i) => (i === 3 ? -1 : v)) })).toThrow(/weight 3 is -1/);
    expect(() => emeans(x, n, d, k, { weights: w.map((v, i) => (i === 4 ? Number.NaN : v)) })).toThrow(/weight 4 is NaN/);
    expect(() => emeans(x, n, d, k, { weights: new Float64Array(n).map((_, i) => (i < 3 ? 1 : 0)) })).toThrow(
      /weight > 0 inside the mask \(got 3\)/,
    );
  });
});
