import { describe, expect, it } from "vitest";
import { nearestNeighborDistancesGpu } from "./nnDistance";

// CPU golden: brute-force nearest-neighbour distance in f64.
function nnCpu(xs: number[], ys: number[]): Float32Array {
  const n = xs.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = xs[j]! - xs[i]!;
      const dy = ys[j]! - ys[i]!;
      const dd = Math.hypot(dx, dy);
      if (dd < best) best = dd;
    }
    out[i] = best;
  }
  return out;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("nearestNeighborDistancesGpu", () => {
  it("matches the CPU golden on a fixed small set", async () => {
    const xs = [0, 1, 5, 5.5, 10];
    const ys = [0, 0, 5, 5, 0];
    const gpu = await nearestNeighborDistancesGpu(xs, ys);
    const cpu = nnCpu(xs, ys);
    for (let i = 0; i < xs.length; i++) {
      expect(gpu[i]!).toBeCloseTo(cpu[i]!, 4);
    }
  });

  it("matches the CPU golden on a random cloud (grows the pool)", async () => {
    const rnd = mulberry32(0xc0ffee);
    // Was 160, capped because "the process-exit teardown segfaults once a process
    // has done enough GPU work (~>=256 points here)". That ceiling was the Dawn
    // Instance-lifetime bug in src/gpu/device.ts, fixed 2026-07-29; re-tested
    // 2026-08-01 to 8192 points, 5 runs of 5 clean. See ADR-0003 and
    // test/adr3-limits.gpu.test.ts. 2048 keeps the CPU golden (O(n²)) quick.
    const n = 2048;
    const xs = Array.from({ length: n }, () => rnd() * 100);
    const ys = Array.from({ length: n }, () => rnd() * 100);
    const gpu = await nearestNeighborDistancesGpu(xs, ys);
    const cpu = nnCpu(xs, ys);
    let maxErr = 0;
    for (let i = 0; i < n; i++) maxErr = Math.max(maxErr, Math.abs(gpu[i]! - cpu[i]!));
    expect(maxErr).toBeLessThan(1e-3);
  });
});
