import { describe, expect, it } from "vitest";
import { knnGpu } from "./spatial/knn";

// Regression guard for the Instance-lifetime bug that made Dawn-on-Node look unreliable
// for months (see the comment in `device.ts` and the one in `vitest.gpu.config.ts`).
//
// `create([])` returns Dawn's Instance, which owns the native event loop and the mutexes
// every device call takes. It used to be a local inside `getDevice()`, so it became
// unreachable the moment the promise resolved; V8 collected it at some arbitrary later
// point and the N-API finaliser destroyed it out from under a live device. The next
// dispatch then hit a destroyed mutex.
//
// The trigger is **allocation**, not elapsed time or CPU work — allocation is what
// provokes a collection. That is the specific thing these tests reproduce, and it is why
// a plain "call the kernel twice" test would NOT catch a regression: without heap
// pressure between the calls the Instance is never collected and everything passes.
//
// If someone removes the module-level references in `device.ts`, these fail by killing
// the fork (SIGSEGV / `mutex lock failed`) rather than by a clean assertion. That is the
// nature of the fault; a dead fork here means the bug is back.

/** Churn enough short-lived objects to make a major GC very likely. */
function provokeGc(): number {
  let held: unknown[] = [];
  for (let round = 0; round < 4; round++) {
    held = [];
    for (let i = 0; i < 120_000; i++) held.push({ i, pair: [i, i + 1], tag: `k${i & 1023}` });
  }
  return held.length;
}

const N = 128;
const DIM = 12;
const K = 6;
const data = Float32Array.from({ length: N * DIM }, (_, i) => Math.sin(i * 0.37) * 4);

describe("Dawn Instance lifetime", () => {
  it("survives heavy allocation between dispatches", async () => {
    const first = await knnGpu(data, { n: N, dim: DIM, k: K });
    expect(provokeGc()).toBeGreaterThan(0);
    const second = await knnGpu(data, { n: N, dim: DIM, k: K });

    // Same input, same answer — and crucially, we got here at all.
    let mismatches = 0;
    for (let t = 0; t < N * K; t++) {
      if (first.indices[t] !== second.indices[t]) mismatches++;
    }
    expect(mismatches).toBe(0);
  });

  it("survives allocation churn across many dispatches", async () => {
    // The single-collection case above can pass by luck if the collector happens not to
    // run. Repeating it makes a miss far less likely.
    let lastFirstIndex = -1;
    for (let round = 0; round < 4; round++) {
      provokeGc();
      const res = await knnGpu(data, { n: N, dim: DIM, k: K });
      expect(res.indices.length).toBe(N * K);
      if (lastFirstIndex >= 0) expect(res.indices[0]).toBe(lastFirstIndex);
      lastFirstIndex = res.indices[0]!;
    }
  });
});
