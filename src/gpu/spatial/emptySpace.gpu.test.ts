import { describe, expect, it } from "vitest";
import { emptySpaceGpu } from "./emptySpace";

function nearestCpu(qx: number, qy: number, xs: number[], ys: number[]) {
  let best = Infinity;
  for (let j = 0; j < xs.length; j++) best = Math.min(best, Math.hypot(xs[j]! - qx, ys[j]! - qy));
  return best;
}

describe("emptySpaceGpu", () => {
  it("matches a CPU min-distance golden at the sampled locations", async () => {
    const xs = [10, 20, 30, 40, 60, 80];
    const ys = [10, 50, 20, 70, 30, 90];
    const bbox: [number, number, number, number] = [0, 0, 100, 100];
    const r = await emptySpaceGpu(xs, ys, { numSamples: 200, bbox, seed: 7 });
    // recompute the same samples (same seed/mulberry32) on the CPU
    let a = 7 >>> 0;
    const next = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let maxErr = 0;
    for (let i = 0; i < r.distances.length; i++) {
      const qx = bbox[0] + next() * (bbox[2] - bbox[0]);
      const qy = bbox[1] + next() * (bbox[3] - bbox[1]);
      maxErr = Math.max(maxErr, Math.abs(r.distances[i]! - nearestCpu(qx, qy, xs, ys)));
    }
    expect(maxErr).toBeLessThan(1e-3);
  });

  it("reports larger empty space for a clustered pattern than a spread-out one", async () => {
    const bbox: [number, number, number, number] = [0, 0, 100, 100];
    // clustered: all points in one corner → big voids elsewhere
    const cx: number[] = [],
      cy: number[] = [];
    for (let i = 0; i < 30; i++) {
      cx.push((i % 6) * 1.5);
      cy.push(Math.floor(i / 6) * 1.5);
    }
    // spread: a coarse grid covering the region → small voids
    const sx: number[] = [],
      sy: number[] = [];
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++) {
        sx.push(i * 18 + 5);
        sy.push(j * 18 + 5);
      }
    const clustered = await emptySpaceGpu(cx, cy, { numSamples: 256, bbox, seed: 3 });
    const spread = await emptySpaceGpu(sx, sy, { numSamples: 256, bbox, seed: 3 });
    expect(clustered.mean).toBeGreaterThan(spread.mean * 2);
  });
});
