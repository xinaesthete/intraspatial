import { describe, expect, it } from "vitest";
import { mulberry32 } from "../../spatial/kernelAnalysis";
import { KNN_NO_NEIGHBOUR, knnGpu } from "./knn";
import { kthNeighborDistanceGpu } from "./kthNeighborDistance";
import { nearestNeighborDistancesGpu } from "./nnDistance";

// Indexed neighbour queries (`cell` given) against their own brute-force kernels, the golden.
// The contract under test (`IndexedQueryOptions`): exact wherever the true k-th neighbour is
// within `cell`; beyond that a neighbour can be missed, so the indexed distance is ≥ brute,
// never <; unfilled slots are +Inf (and index 0xFFFFFFFF for knn).
//
// Comparisons reduce to one scalar and assert once — never `expect()` per element.

function cloud(n: number, seed: number, extent = 100) {
  const rnd = mulberry32(seed);
  const xs = Float32Array.from({ length: n }, () => rnd() * extent);
  const ys = Float32Array.from({ length: n }, () => rnd() * extent);
  // Pin the lattice corners: points exactly on the min and max edges must still find a cell.
  xs[0] = 0;
  ys[0] = 0;
  xs[1] = extent;
  ys[1] = extent;
  return { xs, ys };
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

/** Mismatches split by whether the contract promises equality there. */
function contractViolations(indexed: ArrayLike<number>, brute: ArrayLike<number>, cell: number) {
  let promisedButWrong = 0; // brute ≤ cell, so the stencil must contain the answer
  let below = 0; // indexed < brute can never happen: the stencil is a subset
  let differ = 0;
  for (let i = 0; i < brute.length; i++) {
    const same = indexed[i] === brute[i];
    if (!same) differ++;
    if (!same && brute[i]! <= cell) promisedButWrong++;
    if (indexed[i]! < brute[i]!) below++;
  }
  return { promisedButWrong, below, differ };
}

describe("indexed nearestNeighborDistancesGpu", () => {
  it("equals brute force when cell ≥ every true NN distance", async () => {
    const { xs, ys } = cloud(1500, 0xa11ce);
    const brute = await nearestNeighborDistancesGpu(xs, ys);
    const cell = Math.max(...brute) * 1.01;
    const indexed = await nearestNeighborDistancesGpu(xs, ys, { cell });
    expect(maxAbsDiff(indexed, brute)).toBeLessThanOrEqual(1e-6);
  });

  it("cell too small: misses only where the true NN is beyond cell, never undershoots", async () => {
    const { xs, ys } = cloud(600, 0xb0b);
    const brute = await nearestNeighborDistancesGpu(xs, ys);
    const sorted = Array.from(brute).sort((a, b) => a - b);
    const cell = sorted[Math.floor(sorted.length / 2)]!; // half the points have NN > cell
    const indexed = await nearestNeighborDistancesGpu(xs, ys, { cell });
    const v = contractViolations(indexed, brute, cell);
    expect(v.promisedButWrong).toBe(0);
    expect(v.below).toBe(0);
    expect(v.differ).toBeGreaterThan(0); // the failure mode is real and visible
  });

  it("a point alone in its stencil reports +Infinity", async () => {
    const xs = [0, 0.1, 50];
    const ys = [0, 0, 50];
    const got = await nearestNeighborDistancesGpu(xs, ys, { cell: 1 });
    expect(got[2]).toBe(Number.POSITIVE_INFINITY);
    expect(Math.max(Math.abs(got[0]! - 0.1), Math.abs(got[1]! - 0.1))).toBeLessThan(1e-6);
  });

  it("keeps the brute-force preconditions", async () => {
    await expect(nearestNeighborDistancesGpu([], [], { cell: 1 })).rejects.toThrow(/at least 2/);
    await expect(nearestNeighborDistancesGpu([1, 2], [1, 2], { cell: 0 })).rejects.toThrow(/cell must be > 0/);
  });
});

describe("indexed kthNeighborDistanceGpu", () => {
  const k = 8;

  it("equals brute force when cell ≥ every true k-th distance", async () => {
    const { xs, ys } = cloud(1200, 0xc0de);
    const brute = await kthNeighborDistanceGpu(xs, ys, k);
    const cell = Math.max(...brute) * 1.01;
    const indexed = await kthNeighborDistanceGpu(xs, ys, k, { cell });
    expect(maxAbsDiff(indexed, brute)).toBeLessThanOrEqual(1e-6);
  });

  it("cell too small: misses only where the true k-th is beyond cell, never undershoots", async () => {
    const { xs, ys } = cloud(500, 0xd00d);
    const brute = await kthNeighborDistanceGpu(xs, ys, k);
    const sorted = Array.from(brute).sort((a, b) => a - b);
    const cell = sorted[Math.floor(sorted.length / 2)]!;
    const indexed = await kthNeighborDistanceGpu(xs, ys, k, { cell });
    const v = contractViolations(indexed, brute, cell);
    expect(v.promisedButWrong).toBe(0);
    expect(v.below).toBe(0);
    expect(v.differ).toBeGreaterThan(0);
  });

  it("fewer than k candidates in the stencil reports +Infinity", async () => {
    // Two tight triplets 100 apart; k = 4 needs a point from the other triplet.
    const xs = [0, 0.1, 0.2, 100, 100.1, 100.2];
    const ys = [0, 0, 0, 0, 0, 0];
    const got = await kthNeighborDistanceGpu(xs, ys, 4, { cell: 1 });
    expect(Array.from(got).filter((v) => v === Number.POSITIVE_INFINITY).length).toBe(6);
    const k2 = await kthNeighborDistanceGpu(xs, ys, 2, { cell: 1 });
    expect(Array.from(k2).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("keeps the brute-force preconditions (n ≤ k)", async () => {
    await expect(kthNeighborDistanceGpu([0, 1], [0, 1], 2, { cell: 1 })).rejects.toThrow(/k < N/);
    await expect(kthNeighborDistanceGpu([], [], 1, { cell: 1 })).rejects.toThrow(/k < N/);
  });
});

describe("indexed knnGpu (2-D)", () => {
  const k = 6;

  function rows(xs: Float32Array, ys: Float32Array): Float32Array {
    const out = new Float32Array(2 * xs.length);
    for (let i = 0; i < xs.length; i++) {
      out[2 * i] = xs[i]!;
      out[2 * i + 1] = ys[i]!;
    }
    return out;
  }

  it("equals brute force when cell ≥ every true k-th distance (distances and isolated indices)", async () => {
    const { xs, ys } = cloud(1000, 0xe11e);
    const n = xs.length;
    const data = rows(xs, ys);
    const brute = await knnGpu(data, { n, dim: 2, k });
    let cell = 0;
    for (let i = 0; i < n; i++) cell = Math.max(cell, brute.distances[i * k + k - 1]!);
    const indexed = await knnGpu(data, { n, dim: 2, k, cell: cell * 1.01 });
    // Ties may order either way; count index mismatches only where the brute distance is
    // separated from both ranking neighbours.
    const tol = 1e-3;
    let indexMismatches = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < k; t++) {
        const at = i * k + t;
        const prev = t > 0 ? brute.distances[at - 1]! : Number.NEGATIVE_INFINITY;
        const next = t < k - 1 ? brute.distances[at + 1]! : Number.POSITIVE_INFINITY;
        const isolated = brute.distances[at]! - prev > tol && next - brute.distances[at]! > tol;
        if (isolated && indexed.indices[at] !== brute.indices[at]) indexMismatches++;
      }
    }
    expect(maxAbsDiff(indexed.distances, brute.distances)).toBeLessThanOrEqual(1e-6);
    expect(indexMismatches).toBe(0);
  });

  it("cell too small: every kept distance ≥ brute, exact rows where the k-th is within cell", async () => {
    const { xs, ys } = cloud(400, 0xf00d);
    const n = xs.length;
    const data = rows(xs, ys);
    const brute = await knnGpu(data, { n, dim: 2, k });
    const kth = Array.from({ length: n }, (_, i) => brute.distances[i * k + k - 1]!);
    const sorted = [...kth].sort((a, b) => a - b);
    const cell = sorted[Math.floor(n / 2)]!;
    const indexed = await knnGpu(data, { n, dim: 2, k, cell });
    let promisedButWrong = 0;
    let below = 0;
    let rowsDiffer = 0;
    for (let i = 0; i < n; i++) {
      let same = true;
      for (let t = 0; t < k; t++) {
        const at = i * k + t;
        if (indexed.distances[at] !== brute.distances[at]) same = false;
        if (indexed.distances[at]! < brute.distances[at]!) below++;
      }
      if (!same) rowsDiffer++;
      if (!same && kth[i]! <= cell) promisedButWrong++;
    }
    expect(promisedButWrong).toBe(0);
    expect(below).toBe(0);
    expect(rowsDiffer).toBeGreaterThan(0);
  });

  it("fewer than k candidates: tail padded with KNN_NO_NEIGHBOUR / +Infinity, still ascending", async () => {
    const xs = Float32Array.from([0, 0.1, 0.2, 100, 100.1, 100.2]);
    const ys = new Float32Array(6);
    const n = 6;
    const got = await knnGpu(rows(xs, ys), { n, dim: 2, k: 4, cell: 1 });
    let bad = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < 4; t++) {
        const at = i * 4 + t;
        const filled = t < 2;
        if (filled !== Number.isFinite(got.distances[at]!)) bad++;
        if (filled !== (got.indices[at] !== KNN_NO_NEIGHBOUR)) bad++;
        if (t > 0 && got.distances[at]! < got.distances[at - 1]!) bad++;
      }
    }
    expect(bad).toBe(0);
  });

  it("rejects non-2-D data and keeps the brute-force preconditions", async () => {
    await expect(knnGpu(new Float32Array(30), { n: 10, dim: 3, k: 2, cell: 1 })).rejects.toThrow(/2-D only/);
    await expect(knnGpu(new Float32Array(4), { n: 2, dim: 2, k: 2, cell: 1 })).rejects.toThrow(/k < n/);
    await expect(knnGpu(new Float32Array(0), { n: 0, dim: 2, k: 1, cell: 1 })).rejects.toThrow(/k < n/);
  });
});
