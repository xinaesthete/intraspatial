import { describe, it, expect } from "vitest";
import { anniGpu } from "./anni";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("anniGpu", () => {
  it("reads a uniform-random cloud as approximately random (R ~ 1)", async () => {
    const rnd = mulberry32(0x5eed);
    const n = 150; // kept small for Dawn-on-Node stability
    const xs = Array.from({ length: n }, () => rnd() * 100);
    const ys = Array.from({ length: n }, () => rnd() * 100);
    const r = await anniGpu(xs, ys, { area: 100 * 100 });
    expect(r.index).toBeGreaterThan(0.85);
    expect(r.index).toBeLessThan(1.15);
    expect(r.interpretation).toBe("random");
  });

  it("reads a tightly-clustered pattern as clustered (R << 1, z < 0)", async () => {
    const rnd = mulberry32(0xbeef);
    const n = 150;
    // Three tight blobs inside a large region -> clustered.
    const centers = [[10, 10], [90, 20], [50, 80]];
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const c = centers[i % 3]!;
      xs.push(c[0]! + (rnd() - 0.5) * 4);
      ys.push(c[1]! + (rnd() - 0.5) * 4);
    }
    const r = await anniGpu(xs, ys, { area: 100 * 100 });
    expect(r.index).toBeLessThan(0.5);
    expect(r.zScore).toBeLessThan(-1.96);
    expect(r.interpretation).toBe("clustered");
  });

  it("reads a jittered grid as dispersed (R > 1, z > 0)", async () => {
    const rnd = mulberry32(0xa11ce);
    const xs: number[] = [], ys: number[] = [];
    const g = 12; // 12x12 = 144 points
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        xs.push(i * 8 + (rnd() - 0.5) * 1.5);
        ys.push(j * 8 + (rnd() - 0.5) * 1.5);
      }
    const r = await anniGpu(xs, ys, { area: (g * 8) * (g * 8) });
    expect(r.index).toBeGreaterThan(1.2);
    expect(r.zScore).toBeGreaterThan(1.96);
    expect(r.interpretation).toBe("dispersed");
  });
});
