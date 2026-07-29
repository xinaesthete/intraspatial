import { describe, expect, it } from "vitest";
import { knnBruteForceCpu } from "../../spatial/umapGraph";
import { mulberry32 } from "../../spatial/umapLayout";
import { knnGpu } from "./knn";

// Structural invariants of `knnGpu` — properties that hold regardless of the input, so
// they need no CPU golden. Golden parity lives in `knn.gpu.test.ts`.
//
// Checks reduce to a scalar and assert once rather than looping `expect()` over
// thousands of entries, which keeps failure output readable.

function randomMatrix(n: number, dim: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  return Float32Array.from({ length: n * dim }, () => rnd() * 10 - 5);
}

describe("knnGpu — structural invariants", () => {
  it("returns each row ascending and never returns self", async () => {
    const n = 150;
    const dim = 8;
    const k = 10;
    const gpu = await knnGpu(randomMatrix(n, dim, 0x4444), { n, dim, k });
    let selfHits = 0;
    let inversions = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < k; t++) {
        if (gpu.indices[i * k + t] === i) selfHits++;
        if (t > 0 && gpu.distances[i * k + t]! < gpu.distances[i * k + t - 1]! - 1e-6) inversions++;
      }
    }
    expect(selfHits).toBe(0);
    expect(inversions).toBe(0);
  });

  it("finds the exact neighbours of a 1-D lattice", async () => {
    const n = 64;
    const data = Float32Array.from({ length: n }, (_, i) => i);
    const gpu = await knnGpu(data, { n, dim: 1, k: 2 });
    let wrong = 0;
    let maxDistError = 0;
    for (let i = 1; i < n - 1; i++) {
      const got = Array.from(gpu.indices.slice(i * 2, i * 2 + 2)).sort((a, b) => a - b);
      if (got[0] !== i - 1 || got[1] !== i + 1) wrong++;
      maxDistError = Math.max(maxDistError, Math.abs(gpu.distances[i * 2]! - 1));
    }
    expect(wrong).toBe(0);
    expect(maxDistError).toBeLessThan(1e-4);
  });

  it("separates two blobs — neighbours stay within a cluster", async () => {
    const nPer = 70;
    const n = nPer * 2;
    const dim = 10;
    const rnd = mulberry32(0x5555);
    const data = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < dim; c++) data[i * dim + c] = rnd() * 2 - 1 + (c === 0 && i >= nPer ? 40 : 0);
    }
    const k = 10;
    const gpu = await knnGpu(data, { n, dim, k });
    let cross = 0;
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < k; t++) if (i < nPer !== gpu.indices[i * k + t]! < nPer) cross++;
    }
    expect(cross).toBe(0);
  });

  it("rejects k >= n and k > MAX_K", async () => {
    await expect(knnGpu(new Float32Array(10), { n: 5, dim: 2, k: 5 })).rejects.toThrow(/k < n/);
    await expect(knnGpu(new Float32Array(200), { n: 100, dim: 2, k: 33 })).rejects.toThrow(/1\.\.32/);
  });

  it("reuses its pooled buffers across calls of differing size", async () => {
    // The pool grows and is never shrunk, so a small call after a large one must still
    // read back only its own prefix — a regression guard on the pooling.
    const data = randomMatrix(40, 4, 0x7777);
    const cpu = knnBruteForceCpu(data, 40, 4, 3);
    const big = await knnGpu(randomMatrix(180, 12, 0x6666), { n: 180, dim: 12, k: 6 });
    const small = await knnGpu(data, { n: 40, dim: 4, k: 3 });
    let maxDistError = 0;
    for (let t = 0; t < 40 * 3; t++) {
      maxDistError = Math.max(maxDistError, Math.abs(small.distances[t]! - cpu.distances[t]!));
    }
    expect(big.indices.length).toBe(180 * 6);
    expect(small.indices.length).toBe(40 * 3);
    expect(maxDistError).toBeLessThan(2e-3);
  });
});
