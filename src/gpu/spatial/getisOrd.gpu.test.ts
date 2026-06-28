import { describe, it, expect } from "vitest";
import { getisOrdGpu, pointHotspotsGpu } from "./getisOrd";

// CPU golden for Gi* with a clamp-to-edge box window (W constant).
function getisCpu(grid: number[], w: number, h: number, radius: number): Float32Array {
  const n = w * h;
  const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
  let sum = 0, sumSq = 0;
  for (const v of grid) { sum += v; sumSq += v * v; }
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  const W = (2 * radius + 1) ** 2;
  const denom = std * Math.sqrt((W * (n - W)) / (n - 1));
  const z = new Float32Array(n);
  for (let row = 0; row < h; row++)
    for (let col = 0; col < w; col++) {
      let local = 0;
      for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++)
          local += grid[clamp(row + dy, h - 1) * w + clamp(col + dx, w - 1)]!;
      z[row * w + col] = denom > 0 ? (local - mean * W) / denom : 0;
    }
  return z;
}

describe("getisOrdGpu", () => {
  it("matches the CPU Gi* golden", async () => {
    const w = 20, h = 18, radius = 2;
    const grid = Array.from({ length: w * h }, (_, i) => ((i * 1103515245 + 12345) % 100) / 100);
    const { z } = await getisOrdGpu(grid, w, h, { radius });
    const golden = getisCpu(grid, w, h, radius);
    let m = 0;
    for (let i = 0; i < z.length; i++) m = Math.max(m, Math.abs(z[i]! - golden[i]!));
    expect(m).toBeLessThan(2e-3);
  });

  it("flags a dense region as a hot spot (high positive z)", async () => {
    const w = 30, h = 30, radius = 3;
    const grid = new Array(w * h).fill(0.0);
    // a bright 6x6 block near the centre
    for (let row = 12; row < 18; row++)
      for (let col = 12; col < 18; col++) grid[row * w + col] = 1.0;
    const { z } = await getisOrdGpu(grid, w, h, { radius });
    const centreZ = z[15 * w + 15]!;
    const cornerZ = z[2 * w + 2]!;
    expect(centreZ).toBeGreaterThan(3); // strongly significant hot spot
    expect(cornerZ).toBeLessThan(0); // empty corner reads cold
  });

  it("composes splat -> hotspots end to end (pointHotspotsGpu)", async () => {
    // one tight cluster + sparse background
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < 40; i++) { xs.push(50 + (i % 7) * 0.6); ys.push(50 + Math.floor(i / 7) * 0.6); }
    for (let i = 0; i < 20; i++) { xs.push((i * 13) % 100); ys.push((i * 29) % 100); }
    const field = await pointHotspotsGpu(xs, ys, {
      width: 40, height: 40, sigma: 2, radius: 2, bbox: [0, 0, 100, 100],
    });
    // the hottest cell should sit near the cluster centre (world ~ (52,52))
    let best = -Infinity, bi = -1;
    for (let i = 0; i < field.z.length; i++) if (field.z[i]! > best) { best = field.z[i]!; bi = i; }
    const col = bi % field.width, row = Math.floor(bi / field.width);
    const wx = field.bbox[0] + ((col + 0.5) / field.width) * (field.bbox[2] - field.bbox[0]);
    const wy = field.bbox[3] - ((row + 0.5) / field.height) * (field.bbox[3] - field.bbox[1]);
    expect(best).toBeGreaterThan(3);
    expect(Math.hypot(wx - 52, wy - 52)).toBeLessThan(12);
  });
});
