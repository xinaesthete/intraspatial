import { describe, expect, it } from "vitest";
import { fuzzySimplicialSet, type KnnResult, knnBruteForceCpu, smoothKnnDist } from "./umapGraph";
import { mulberry32 } from "./umapLayout";

/** Two well-separated Gaussian blobs in `dim` dimensions — the standard smoke shape:
 *  any correct UMAP graph keeps within-blob edges and almost no between-blob ones. */
function twoBlobs(nPer: number, dim: number, seed = 7): { data: Float64Array; label: Uint8Array } {
  const rnd = mulberry32(seed);
  const n = nPer * 2;
  const data = new Float64Array(n * dim);
  const label = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const blob = i < nPer ? 0 : 1;
    label[i] = blob;
    for (let c = 0; c < dim; c++) {
      // Box-Muller from the seeded uniform; offset blob 1 far along axis 0.
      const u = Math.max(rnd(), 1e-12);
      const v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      data[i * dim + c] = g + (c === 0 && blob === 1 ? 30 : 0);
    }
  }
  return { data, label };
}

describe("knnBruteForceCpu", () => {
  it("returns each row ascending and excludes self", () => {
    const data = [0, 0, 1, 0, 3, 0, 7, 0];
    const knn = knnBruteForceCpu(data, 4, 2, 2);
    expect(Array.from(knn.indices.slice(0, 2))).toEqual([1, 2]);
    for (let i = 0; i < 4; i++) {
      for (let t = 0; t < 2; t++) expect(knn.indices[i * 2 + t]).not.toBe(i);
      expect(knn.distances[i * 2]!).toBeLessThanOrEqual(knn.distances[i * 2 + 1]!);
    }
  });

  it("finds exact neighbours on a 1-D lattice", () => {
    const n = 20;
    const data = Float64Array.from({ length: n }, (_, i) => i);
    const knn = knnBruteForceCpu(data, n, 1, 2);
    // Point 10's two nearest are 9 and 11, both at distance 1.
    const got = Array.from(knn.indices.slice(10 * 2, 10 * 2 + 2)).sort((a, b) => a - b);
    expect(got).toEqual([9, 11]);
    expect(knn.distances[10 * 2]!).toBeCloseTo(1, 6);
  });

  it("rejects k >= n", () => {
    expect(() => knnBruteForceCpu([0, 1, 2], 3, 1, 3)).toThrow(/k < n/);
  });
});

describe("smoothKnnDist", () => {
  it("solves sigma so the membership sum hits log2(nNeighbors)", () => {
    const { data } = twoBlobs(40, 5);
    const knn = knnBruteForceCpu(data, 80, 5, 14);
    const { rho, sigma } = smoothKnnDist(knn, { nNeighbors: 15 });
    const target = Math.log2(15);
    for (let i = 0; i < knn.n; i++) {
      let psum = 0;
      for (let t = 0; t < knn.k; t++) {
        const dist = knn.distances[i * knn.k + t]! - rho[i]!;
        psum += dist > 0 ? Math.exp(-(dist / sigma[i]!)) : 1;
      }
      // The binary search's own tolerance, plus room for the sigma floor kicking in.
      expect(Math.abs(psum - target)).toBeLessThan(1e-3);
    }
  });

  it("gives rho = the nearest-neighbour distance at localConnectivity 1", () => {
    const { data } = twoBlobs(20, 3, 11);
    const knn = knnBruteForceCpu(data, 40, 3, 10);
    const { rho } = smoothKnnDist(knn, { nNeighbors: 11 });
    for (let i = 0; i < knn.n; i++) expect(rho[i]!).toBeCloseTo(knn.distances[i * knn.k]!, 6);
  });

  it("rho tracks local density — an isolated point has a far-away nearest neighbour", () => {
    // A tight cluster plus one point held far away from everything. Density adaptation
    // lives in rho: it is exactly the k-th-neighbour bandwidth of `kthNeighborDistance`.
    const n = 12;
    const data = new Float64Array(n * 2);
    for (let i = 0; i < n - 1; i++) {
      data[i * 2] = (i % 4) * 0.1;
      data[i * 2 + 1] = Math.floor(i / 4) * 0.1;
    }
    data[(n - 1) * 2] = 100;
    const knn = knnBruteForceCpu(data, n, 2, 5);
    const { rho } = smoothKnnDist(knn, { nNeighbors: 6 });
    const denseMean = Array.from(rho.slice(0, n - 1)).reduce((s, v) => s + v, 0) / (n - 1);
    expect(rho[n - 1]!).toBeGreaterThan(denseMean * 10);
  });

  it("rho scales with the data", () => {
    const tight = new Float64Array([0, 0, 0.1, 0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0]);
    const wide = Float64Array.from(tight, (v) => v * 10);
    const kTight = smoothKnnDist(knnBruteForceCpu(tight, 6, 2, 3), { nNeighbors: 4 });
    const kWide = smoothKnnDist(knnBruteForceCpu(wide, 6, 2, 3), { nNeighbors: 4 });
    for (let i = 0; i < 6; i++) expect(kWide.rho[i]!).toBeCloseTo(kTight.rho[i]! * 10, 6);
  });
});

describe("fuzzySimplicialSet", () => {
  it("is symmetric — every edge appears in both directions with equal weight", () => {
    const { data } = twoBlobs(30, 4, 3);
    const knn = knnBruteForceCpu(data, 60, 4, 10);
    const g = fuzzySimplicialSet(knn, { nNeighbors: 11 });
    const seen = new Map<string, number>();
    for (let e = 0; e < g.nEdges; e++) seen.set(`${g.head[e]}->${g.tail[e]}`, g.weight[e]!);
    for (let e = 0; e < g.nEdges; e++) {
      const back = seen.get(`${g.tail[e]}->${g.head[e]}`);
      expect(back).toBeDefined();
      expect(back!).toBeCloseTo(g.weight[e]!, 6);
    }
  });

  it("has no self-loops and weights in (0, 1]", () => {
    const { data } = twoBlobs(30, 4, 5);
    const knn = knnBruteForceCpu(data, 60, 4, 10);
    const g = fuzzySimplicialSet(knn, { nNeighbors: 11 });
    expect(g.nEdges).toBeGreaterThan(0);
    for (let e = 0; e < g.nEdges; e++) {
      expect(g.head[e]).not.toBe(g.tail[e]);
      expect(g.weight[e]!).toBeGreaterThan(0);
      expect(g.weight[e]!).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it("keeps within-cluster edges and essentially no between-cluster ones", () => {
    const nPer = 40;
    const { data, label } = twoBlobs(nPer, 5, 9);
    const knn = knnBruteForceCpu(data, nPer * 2, 5, 10);
    const g = fuzzySimplicialSet(knn, { nNeighbors: 11 });
    let cross = 0;
    for (let e = 0; e < g.nEdges; e++) if (label[g.head[e]!] !== label[g.tail[e]!]) cross++;
    expect(cross / g.nEdges).toBeLessThan(0.01);
  });

  it("the t-conorm is at least each one-sided membership it unions", () => {
    // The defining property of the fuzzy union: an edge is at least as strong as
    // either endpoint's own view of it. Asserted against the directed memberships
    // recomputed from the same rho/sigma, so it pins the symmetrisation specifically.
    const { data } = twoBlobs(30, 4, 17);
    const knn = knnBruteForceCpu(data, 60, 4, 10);
    const { rho, sigma } = smoothKnnDist(knn, { nNeighbors: 11 });
    const g = fuzzySimplicialSet(knn, { nNeighbors: 11 });

    const directed = new Map<string, number>();
    for (let i = 0; i < knn.n; i++) {
      for (let t = 0; t < knn.k; t++) {
        const j = knn.indices[i * knn.k + t]!;
        const dist = knn.distances[i * knn.k + t]! - rho[i]!;
        directed.set(`${i}->${j}`, dist > 0 ? Math.exp(-(dist / sigma[i]!)) : 1);
      }
    }
    let sawStrictlyGreater = 0;
    for (let e = 0; e < g.nEdges; e++) {
      const i = g.head[e]!;
      const j = g.tail[e]!;
      const forward = directed.get(`${i}->${j}`) ?? 0;
      const backward = directed.get(`${j}->${i}`) ?? 0;
      const strongest = Math.max(forward, backward);
      expect(g.weight[e]!).toBeGreaterThanOrEqual(strongest - 1e-6);
      // A mutual edge where neither side saturates strictly gains from the union.
      if (g.weight[e]! > strongest + 1e-6) sawStrictlyGreater++;
    }
    expect(sawStrictlyGreater).toBeGreaterThan(0);
  });

  it("weights are invariant to a global rescale of the data", () => {
    // Scale-invariance of the *graph* is the property the pipeline depends on — it is
    // why the layout looks the same whether X is raw counts or log1p-scaled.
    //
    // Note it does NOT come from sigma scaling exactly: sigma is found by bisection
    // that stops as soon as |psum - target| < 1e-5, and for a point whose neighbours
    // are equidistant the target is only approached as sigma -> 0, so bisection halts
    // on the dyadic grid (here a 8x ratio across a 10x rescale, not 10x). The
    // memberships still agree to ~5 decimals because they are deep in the tail where
    // the residual sigma difference stops mattering. Asserted on the weights, which
    // is what downstream reads, rather than on sigma, which it does not.
    const tight = new Float64Array([0, 0, 0.1, 0, 0.2, 0, 0.3, 0, 0.4, 0, 0.5, 0]);
    const wide = Float64Array.from(tight, (v) => v * 10);
    const gT = fuzzySimplicialSet(knnBruteForceCpu(tight, 6, 2, 3), { nNeighbors: 4 });
    const gW = fuzzySimplicialSet(knnBruteForceCpu(wide, 6, 2, 3), { nNeighbors: 4 });
    expect(gW.nEdges).toBe(gT.nEdges);
    for (let e = 0; e < gT.nEdges; e++) {
      expect(gW.head[e]).toBe(gT.head[e]);
      expect(gW.tail[e]).toBe(gT.tail[e]);
      expect(gW.weight[e]!).toBeCloseTo(gT.weight[e]!, 4);
    }
  });

  it("setOpMixRatio 0 (intersection) drops one-sided edges below the union", () => {
    const { data } = twoBlobs(25, 3, 13);
    const knn = knnBruteForceCpu(data, 50, 3, 8);
    const union = fuzzySimplicialSet(knn, { nNeighbors: 9, setOpMixRatio: 1 });
    const inter = fuzzySimplicialSet(knn, { nNeighbors: 9, setOpMixRatio: 0 });
    const sum = (g: { weight: Float32Array }) => Array.from(g.weight).reduce((s, v) => s + v, 0);
    expect(sum(inter)).toBeLessThan(sum(union));
  });
});
