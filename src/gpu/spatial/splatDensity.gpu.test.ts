import { describe, expect, it } from "vitest";
import { splatDensityGpu } from "./splatDensity";

// CPU golden: full (untruncated) Gaussian KDE sampled at the SAME texel-centre
// world positions the GPU rasteriser uses. Row 0 is the top of the bbox (maxY).
function kdeCpu(
  xs: number[],
  ys: number[],
  w: number,
  h: number,
  sigma: number,
  bbox: [number, number, number, number],
  weights?: number[],
): Float32Array {
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const out = new Float32Array(w * h);
  for (let row = 0; row < h; row++) {
    const wy = maxY - ((row + 0.5) / h) * spanY;
    for (let col = 0; col < w; col++) {
      const wx = minX + ((col + 0.5) / w) * spanX;
      let acc = 0;
      for (let p = 0; p < xs.length; p++) {
        const dx = wx - xs[p]!;
        const dy = wy - ys[p]!;
        acc += (weights ? weights[p]! : 1) * Math.exp(-(dx * dx + dy * dy) * inv2s2);
      }
      out[row * w + col] = acc;
    }
  }
  return out;
}

function maxAbsErr(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe("splatDensityGpu", () => {
  it("matches a CPU Gaussian KDE golden", async () => {
    const xs = [2, 5, 5.4, 8];
    const ys = [2, 6, 6.2, 3];
    const sigma = 1.0;
    const w = 48,
      h = 48;
    const bbox: [number, number, number, number] = [-2, -2, 12, 12];
    const field = await splatDensityGpu(xs, ys, { width: w, height: h, sigma, radiusSigma: 4, bbox });
    expect(field.width).toBe(w);
    expect(field.height).toBe(h);
    const golden = kdeCpu(xs, ys, w, h, sigma, bbox);
    // GPU truncates the kernel at 4 sigma; the dropped tail bounds the error.
    expect(maxAbsErr(field.data, golden)).toBeLessThan(3e-3);
  });

  it("places density where the points are (peak near a known point)", async () => {
    const xs = [10];
    const ys = [10];
    const sigma = 2;
    const w = 40,
      h = 40;
    const bbox: [number, number, number, number] = [0, 0, 20, 20];
    const field = await splatDensityGpu(xs, ys, { width: w, height: h, sigma, bbox });
    // The point at world (10,10) is the centre of the bbox -> texel (~20,~20).
    let best = -1,
      bestIdx = -1;
    for (let i = 0; i < field.data.length; i++) {
      if (field.data[i]! > best) {
        best = field.data[i]!;
        bestIdx = i;
      }
    }
    const row = Math.floor(bestIdx / w);
    const col = bestIdx % w;
    // Weight 1 → peak Gaussian value ~1, but a touch under since no texel centre
    // lands exactly on the point.
    expect(best).toBeGreaterThan(0.95);
    expect(best).toBeLessThanOrEqual(1.0 + 1e-4);
    expect(Math.abs(col - 20)).toBeLessThanOrEqual(1);
    expect(Math.abs(row - 20)).toBeLessThanOrEqual(1);
  });

  it("respects per-point weights (additive)", async () => {
    const xs = [5, 5];
    const ys = [5, 5];
    const sigma = 1.5;
    const w = 32,
      h = 32;
    const bbox: [number, number, number, number] = [0, 0, 10, 10];
    const field = await splatDensityGpu(xs, ys, { width: w, height: h, sigma, bbox, weights: [2, 3] });
    let peak = 0;
    for (const v of field.data) peak = Math.max(peak, v);
    // Two coincident points weight 2 + 3 -> peak ~5 (a touch under: no texel
    // centre sits exactly on the point).
    expect(peak).toBeGreaterThan(4.8);
    expect(peak).toBeLessThanOrEqual(5.0 + 1e-3);
  });
});
