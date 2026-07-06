import { describe, expect, it } from "vitest";
import { cknnGraph, cknnRescaledDistanceGpu, selfTuningWeights } from "./cknn";

function rhoCpu(xs: number[], ys: number[], k: number): Float32Array {
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

describe("cknnRescaledDistanceGpu", () => {
  it("matches the CPU rescaled-distance golden", async () => {
    const xs = [0, 1, 2, 10, 11, 12];
    const ys = [0, 0.5, 0, 8, 8.5, 8];
    const k = 2;
    const { rescaled, n } = await cknnRescaledDistanceGpu(xs, ys, { k });
    const rho = rhoCpu(xs, ys, k);
    let maxErr = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const exp = i === j ? 0 : Math.hypot(xs[j]! - xs[i]!, ys[j]! - ys[i]!) / Math.sqrt(rho[i]! * rho[j]!);
        maxErr = Math.max(maxErr, Math.abs(rescaled[i * n + j]! - exp));
      }
    expect(maxErr).toBeLessThan(2e-3);
  });

  it("normalises density: dense and sparse clusters get comparable internal d̃", async () => {
    // dense cluster (spacing ~1) and sparse cluster (spacing ~6), well separated
    const xs: number[] = [],
      ys: number[] = [];
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++) {
        xs.push(i);
        ys.push(j);
      } // dense, idx 0..24
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++) {
        xs.push(100 + i * 6);
        ys.push(j * 6);
      } // sparse, idx 25..49
    const n = xs.length;
    const { rescaled } = await cknnRescaledDistanceGpu(xs, ys, { k: 4 });

    // nearest-neighbour *raw* distance differs ~6x between clusters, but the nearest
    // rescaled d̃ should be similar (density-normalised).
    const nearestRaw = (a: number, b: number) => {
      let m = Infinity;
      for (let i = a; i < b; i++) for (let j = a; j < b; j++) if (i !== j) m = Math.min(m, Math.hypot(xs[j]! - xs[i]!, ys[j]! - ys[i]!));
      return m;
    };
    const nearestTilde = (a: number, b: number) => {
      let m = Infinity;
      for (let i = a; i < b; i++) for (let j = a; j < b; j++) if (i !== j) m = Math.min(m, rescaled[i * n + j]!);
      return m;
    };
    const rawRatio = nearestRaw(25, 50) / nearestRaw(0, 25);
    const tildeRatio = nearestTilde(25, 50) / nearestTilde(0, 25);
    expect(rawRatio).toBeGreaterThan(4); // raw distances really do differ a lot
    expect(Math.abs(Math.log2(tildeRatio))).toBeLessThan(0.6); // d̃ within ~1.5x → normalised
  });

  it("cknnGraph + selfTuningWeights are consistent readouts of d̃", async () => {
    const xs = [0, 1, 2, 8, 9];
    const ys = [0, 0, 0, 0, 0];
    const { rescaled, n } = await cknnRescaledDistanceGpu(xs, ys, { k: 1 });
    const g = cknnGraph(rescaled, n, 1.5);
    const w = selfTuningWeights(rescaled, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        // an edge in the graph (d̃<δ) must have larger weight than a strong non-edge
        if (g[i * n + j] === 1) expect(w[i * n + j]!).toBeGreaterThan(Math.exp(-1.5 * 1.5));
      }
  });
});
