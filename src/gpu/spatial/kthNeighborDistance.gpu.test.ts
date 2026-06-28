import { describe, it, expect } from "vitest";
import { kthNeighborDistanceGpu } from "./kthNeighborDistance";

function kthCpu(xs: number[], ys: number[], k: number): Float32Array {
  const n = xs.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const ds: number[] = [];
    for (let j = 0; j < n; j++) if (j !== i) ds.push(Math.hypot(xs[j]! - xs[i]!, ys[j]! - ys[i]!));
    ds.sort((a, b) => a - b);
    out[i] = ds[k - 1]!;
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

describe("kthNeighborDistanceGpu", () => {
  it("k=1 equals the nearest-neighbour distance", async () => {
    const xs = [0, 1, 5, 5.5, 10];
    const ys = [0, 0, 5, 5, 0];
    const got = await kthNeighborDistanceGpu(xs, ys, 1);
    const cpu = kthCpu(xs, ys, 1);
    for (let i = 0; i < xs.length; i++) expect(got[i]!).toBeCloseTo(cpu[i]!, 4);
  });

  it("matches the CPU k-th distance for k=5 on a random cloud", async () => {
    const rnd = mulberry32(0x1234);
    const n = 140;
    const xs = Array.from({ length: n }, () => rnd() * 100);
    const ys = Array.from({ length: n }, () => rnd() * 100);
    const k = 5;
    const got = await kthNeighborDistanceGpu(xs, ys, k);
    const cpu = kthCpu(xs, ys, k);
    let maxErr = 0;
    for (let i = 0; i < n; i++) maxErr = Math.max(maxErr, Math.abs(got[i]! - cpu[i]!));
    expect(maxErr).toBeLessThan(1e-3);
  });

  it("is larger in sparse regions than dense ones (local density estimate)", async () => {
    // dense cluster near origin, one far outlier
    const xs = [0, 0.2, 0.4, 0.1, 0.3, 50];
    const ys = [0, 0.1, 0.0, 0.3, 0.2, 50];
    const rho = await kthNeighborDistanceGpu(xs, ys, 2);
    const denseAvg = (rho[0]! + rho[1]! + rho[2]! + rho[3]! + rho[4]!) / 5;
    expect(rho[5]!).toBeGreaterThan(denseAvg * 10); // outlier ρ much larger
  });
});
