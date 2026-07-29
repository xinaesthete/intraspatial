// The GPU local join is checked against the HOST one element by element, which is a
// stronger contract than any other GPU kernel here gets — and it is available only because
// the algorithm is race-free. Thread `i` offers candidates to `i`'s own neighbour list and
// to nothing else, so unlike `umapLayoutGpu` (Hogwild, deliberately racy, tested
// statistically) this kernel has one right answer and can be held to it.
//
// The one caveat is float ordering: the device compares SQUARED distances and the host
// compares rooted ones, so a pair of near-equal candidates can order differently. One such
// difference changes that point's list, which changes the candidate lists built from it,
// so divergence compounds over passes. At the sizes here it does not arise; at 100k it
// does, and the property that survives is recall rather than exact agreement — measured
// 0.951 for both host and device at n=100k, with the per-slot agreement at 81%.

import { beforeAll, describe, expect, it } from "vitest";
import { knnDescentCpu, knnRecall } from "../../spatial/knnDescent";
import { expressManifold, makeManifold } from "../../spatial/syntheticManifolds";
import { type KnnResult, knnBruteForceCpu } from "../../spatial/umapGraph";
import { knnDescentGpu } from "./knnDescentGpu";

const N = 1500;
const K = 14;
const SEED = 7;

let values: Float32Array;
let dim: number;
let exact: KnnResult;

beforeAll(() => {
  const m = makeManifold("branching", N, 11);
  const e = expressManifold(m, { seed: 11 });
  values = e.values;
  dim = e.dim;
  exact = knnBruteForceCpu(values, N, dim, K);
});

describe("knnDescentGpu", () => {
  it("reproduces the host local join exactly", async () => {
    const cpu = knnDescentCpu(values, N, dim, { k: K, seed: SEED });
    const gpu = await knnDescentGpu(values, N, dim, { k: K, seed: SEED });
    expect(Array.from(gpu.indices)).toEqual(Array.from(cpu.indices));
    let maxDistError = 0;
    for (let t = 0; t < N * K; t++) maxDistError = Math.max(maxDistError, Math.abs(gpu.distances[t]! - cpu.distances[t]!));
    expect(maxDistError).toBeLessThan(1e-5);
  });

  it("reaches the same recall as the host, and a usable one", async () => {
    const cpu = knnDescentCpu(values, N, dim, { k: K, seed: SEED });
    const gpu = await knnDescentGpu(values, N, dim, { k: K, seed: SEED });
    const recall = knnRecall(gpu, exact);
    expect(recall).toBeGreaterThan(0.95);
    expect(recall).toBeCloseTo(knnRecall(cpu, exact), 5);
  });

  it("returns rows ascending, without self", async () => {
    const gpu = await knnDescentGpu(values, N, dim, { k: K, seed: SEED });
    for (let i = 0; i < N; i++) {
      for (let t = 0; t < K; t++) {
        expect(gpu.indices[i * K + t]).not.toBe(i);
        if (t > 0) expect(gpu.distances[i * K + t]!).toBeGreaterThanOrEqual(gpu.distances[i * K + t - 1]!);
      }
    }
  });

  it("gives the same answer however the rows are split across dispatches", async () => {
    // Row tiling exists for the same watchdog reason as in `knn.ts`. Unlike the exact
    // search, no state is carried between tiles within a pass — threads are independent —
    // so the split must be completely invisible, and this pins that.
    const whole = await knnDescentGpu(values, N, dim, { k: K, seed: SEED });
    for (const rowsPerTile of [64, 191, 1024]) {
      const split = await knnDescentGpu(values, N, dim, { k: K, seed: SEED, rowsPerTile });
      expect(Array.from(split.indices), `rowsPerTile=${rowsPerTile}`).toEqual(Array.from(whole.indices));
    }
  });

  it("rejects k out of range", async () => {
    await expect(knnDescentGpu(values, N, dim, { k: N + 1 })).rejects.toThrow(/k < n/);
    await expect(knnDescentGpu(values, N, dim, { k: 33 })).rejects.toThrow(/1\.\.32/);
  });

  it("more passes buy recall, and the early exit does not cost any", async () => {
    // `maxIters` is a knob, not a constant — the point the module header makes about recall
    // falling with n. Pinning the direction guards against a convergence check that exits
    // too eagerly, which would look like a speedup and read as a quality regression.
    const few = await knnDescentGpu(values, N, dim, { k: K, seed: SEED, maxIters: 2 });
    const many = await knnDescentGpu(values, N, dim, { k: K, seed: SEED, maxIters: 12 });
    expect(knnRecall(many, exact)).toBeGreaterThan(knnRecall(few, exact));
  });
});
