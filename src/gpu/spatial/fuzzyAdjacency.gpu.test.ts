import { describe, expect, it } from "vitest";
import { fuzzyAdjacencyGpu } from "./fuzzyAdjacency";

function cpu(xs: number[], ys: number[], sigma: number, radiusSigma: number): Float32Array {
  const n = xs.length;
  const inv = 1 / (2 * sigma * sigma);
  const maxD2 = (radiusSigma * sigma) ** 2;
  const m = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = xs[j]! - xs[i]!,
        dy = ys[j]! - ys[i]!;
      const d2 = dx * dx + dy * dy;
      m[i * n + j] = d2 <= maxD2 ? Math.exp(-d2 * inv) : 0;
    }
  return m;
}

describe("fuzzyAdjacencyGpu", () => {
  it("matches the CPU membership matrix; diagonal is 0; symmetric", async () => {
    const xs = [0, 1, 2, 5, 5.5];
    const ys = [0, 0, 0, 5, 5];
    const sigma = 1.5,
      radiusSigma = 3;
    const { membership: m, n } = await fuzzyAdjacencyGpu(xs, ys, { sigma, radiusSigma });
    const g = cpu(xs, ys, sigma, radiusSigma);
    let maxErr = 0;
    for (let i = 0; i < n * n; i++) maxErr = Math.max(maxErr, Math.abs(m[i]! - g[i]!));
    expect(maxErr).toBeLessThan(1e-4);
    for (let i = 0; i < n; i++) expect(m[i * n + i]!).toBe(0); // diagonal
    // membership is symmetric (distance is)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) expect(Math.abs(m[i * n + j]! - m[j * n + i]!)).toBeLessThan(1e-5);
  });

  it("membership decays with distance and truncates past the support radius", async () => {
    const xs = [0, 1, 10];
    const ys = [0, 0, 0];
    const { membership: m, n } = await fuzzyAdjacencyGpu(xs, ys, { sigma: 1, radiusSigma: 3 });
    // near pair (0,1): strong membership; far point (10): beyond 3σ → 0
    expect(m[0 * n + 1]!).toBeGreaterThan(0.5);
    expect(m[0 * n + 2]!).toBe(0);
  });
});
