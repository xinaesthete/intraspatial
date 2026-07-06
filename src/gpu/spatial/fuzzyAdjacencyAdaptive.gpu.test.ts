import { describe, expect, it } from "vitest";
import { fuzzyAdjacencyAdaptiveFromRhoGpu } from "./fuzzyAdjacencyAdaptive";

function adaptiveCpu(xs: number[], ys: number[], rho: number[], scale: number, minSigma: number): Float32Array {
  const n = xs.length;
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    const si = Math.max(scale * rho[i]!, minSigma);
    const invI = 1 / (2 * si * si);
    for (let j = 0; j < n; j++) {
      if (i === j) {
        out[i * n + j] = 0;
        continue;
      }
      const dx = xs[j]! - xs[i]!,
        dy = ys[j]! - ys[i]!;
      const d2 = dx * dx + dy * dy;
      const sj = Math.max(scale * rho[j]!, minSigma);
      const invJ = 1 / (2 * sj * sj);
      const a = Math.exp(-d2 * invI),
        b = Math.exp(-d2 * invJ);
      out[i * n + j] = a + b - a * b;
    }
  }
  return out;
}

describe("fuzzyAdjacencyAdaptive (UMAP-style)", () => {
  it("matches the t-conorm CPU golden for given per-point bandwidths", async () => {
    const n = 12;
    const xs: number[] = [],
      ys: number[] = [],
      rho: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      xs.push(Math.cos(a) * 3 + ((i * 7) % 5) * 0.1);
      ys.push(Math.sin(a) * 3 + ((i * 11) % 5) * 0.1);
      rho.push(0.5 + ((i * 13) % 7) * 0.1); // varied local bandwidths
    }
    const scale = 1.0,
      minSigma = 1e-6;
    const { membership } = await fuzzyAdjacencyAdaptiveFromRhoGpu(xs, ys, rho, { scale, minSigma });
    const golden = adaptiveCpu(xs, ys, rho, scale, minSigma);

    let maxAbs = 0;
    for (let i = 0; i < membership.length; i++) maxAbs = Math.max(maxAbs, Math.abs(membership[i]! - golden[i]!));
    expect(maxAbs).toBeLessThan(1e-5);
  });

  it("is symmetric and zero on the diagonal", async () => {
    const n = 10;
    const xs: number[] = [],
      ys: number[] = [],
      rho: number[] = [];
    for (let i = 0; i < n; i++) {
      xs.push((i * 17) % 13);
      ys.push((i * 29) % 11);
      rho.push(1 + ((i * 5) % 4) * 0.3);
    }
    const { membership } = await fuzzyAdjacencyAdaptiveFromRhoGpu(xs, ys, rho, { scale: 1 });
    let maxAsym = 0,
      maxDiag = 0;
    for (let i = 0; i < n; i++) {
      maxDiag = Math.max(maxDiag, Math.abs(membership[i * n + i]!));
      for (let j = 0; j < n; j++) {
        maxAsym = Math.max(maxAsym, Math.abs(membership[i * n + j]! - membership[j * n + i]!));
      }
    }
    expect(maxDiag).toBe(0);
    expect(maxAsym).toBeLessThan(1e-6);
  });
});
