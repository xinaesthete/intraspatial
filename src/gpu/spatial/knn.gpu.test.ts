import { describe, expect, it } from "vitest";
import { type KnnResult, knnBruteForceCpu } from "../../spatial/umapGraph";
import { mulberry32 } from "../../spatial/umapLayout";
import { knnGpu } from "./knn";

// Parity of `knnGpu` against the CPU golden. Structural invariants live in
// `knnInvariants.gpu.test.ts`.
//
// N is kept modest here simply because an exact O(N^2) golden is the slow part; the
// kernel itself has no such limit. (An earlier version of this comment claimed Dawn was
// unstable past a few hundred points and that fixture placement mattered — both were
// symptoms of an Instance-lifetime bug in `src/gpu/device.ts`, since fixed. The GPU
// suite is now deterministic and runs files in parallel.)
//
// Comparisons still reduce to a scalar and assert once rather than looping `expect()`
// over thousands of entries — that keeps failure output readable, and is worth doing on
// its own merits.

function randomMatrix(n: number, dim: number, seed: number): Float32Array {
  const rnd = mulberry32(seed);
  return Float32Array.from({ length: n * dim }, () => rnd() * 10 - 5);
}

interface Agreement {
  maxDistError: number;
  /** Mismatched indices among entries whose distance is unambiguously separated from
   *  its neighbours in the ranking — ties may legitimately order either way. */
  isolatedIndexMismatches: number;
}

function compare(gpu: KnnResult, cpu: KnnResult, n: number, k: number, tol = 2e-3): Agreement {
  let maxDistError = 0;
  let isolatedIndexMismatches = 0;
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const at = i * k + t;
      const err = Math.abs(gpu.distances[at]! - cpu.distances[at]!);
      if (err > maxDistError) maxDistError = err;
      const prev = t > 0 ? cpu.distances[at - 1]! : Number.NEGATIVE_INFINITY;
      const next = t < k - 1 ? cpu.distances[at + 1]! : Number.POSITIVE_INFINITY;
      const isolated = cpu.distances[at]! - prev > tol && next - cpu.distances[at]! > tol;
      if (isolated && gpu.indices[at] !== cpu.indices[at]) isolatedIndexMismatches++;
    }
  }
  return { maxDistError, isolatedIndexMismatches };
}

interface Case {
  readonly label: string;
  readonly n: number;
  readonly dim: number;
  readonly k: number;
  readonly data: Float32Array;
  cpu: KnnResult;
}

const CASES: Case[] = [
  { label: "2-D", n: 200, dim: 2, k: 8, data: randomMatrix(200, 2, 0x1111), cpu: undefined as unknown as KnnResult },
  // 50-D is the regime UMAP actually runs in — PCA-reduced expression, not raw genes.
  { label: "50-D", n: 160, dim: 50, k: 14, data: randomMatrix(160, 50, 0x2222), cpu: undefined as unknown as KnnResult },
  { label: "k=1", n: 120, dim: 6, k: 1, data: randomMatrix(120, 6, 0x3333), cpu: undefined as unknown as KnnResult },
  { label: "k=MAX_K", n: 120, dim: 6, k: 32, data: randomMatrix(120, 6, 0x3333), cpu: undefined as unknown as KnnResult },
];

for (const c of CASES) c.cpu = knnBruteForceCpu(c.data, c.n, c.dim, c.k);

describe("knnGpu", () => {
  for (const c of CASES) {
    it(`matches the CPU golden — ${c.label}`, async () => {
      const got = compare(await knnGpu(c.data, { n: c.n, dim: c.dim, k: c.k }), c.cpu, c.n, c.k);
      expect(got.maxDistError).toBeLessThan(2e-3);
      expect(got.isolatedIndexMismatches).toBe(0);
    });
  }
});
