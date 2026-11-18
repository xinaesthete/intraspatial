import { describe, expect, it } from "vite-plus/test";
import { mulberry32 } from "#src/evo/rng";
import { assignNearest, emeans, emeansAlpha0, emeansBatch, emeansSchedule, lloyd } from "./emeans";

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
