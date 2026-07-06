import { describe, expect, it } from "vitest";
import { boxKernel, convolveSeparableGpu, gaussianKernel } from "./convolveSeparable";

// CPU golden: separable convolution with clamp-to-edge.
function convCpu(grid: number[], w: number, h: number, kernel: number[]): Float32Array {
  const r = (kernel.length - 1) / 2;
  const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
  const tmp = new Float32Array(w * h);
  for (let row = 0; row < h; row++)
    for (let col = 0; col < w; col++) {
      let acc = 0;
      for (let t = 0; t < kernel.length; t++) {
        const c = clamp(col + (t - r), w - 1);
        acc += grid[row * w + c]! * kernel[t]!;
      }
      tmp[row * w + col] = acc;
    }
  const out = new Float32Array(w * h);
  for (let row = 0; row < h; row++)
    for (let col = 0; col < w; col++) {
      let acc = 0;
      for (let t = 0; t < kernel.length; t++) {
        const rr = clamp(row + (t - r), h - 1);
        acc += tmp[rr * w + col]! * kernel[t]!;
      }
      out[row * w + col] = acc;
    }
  return out;
}

function maxAbsErr(a: ArrayLike<number>, b: ArrayLike<number>) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe("convolveSeparableGpu", () => {
  it("box kernel matches CPU (local sum) on a random grid", async () => {
    const w = 24,
      h = 20;
    const grid = Array.from({ length: w * h }, (_, i) => ((i * 2654435761) % 97) / 97);
    const k = Array.from(boxKernel(2));
    const gpu = await convolveSeparableGpu(grid, w, h, k);
    const cpu = convCpu(grid, w, h, k);
    expect(maxAbsErr(gpu, cpu)).toBeLessThan(1e-3);
  });

  it("Gaussian kernel matches CPU and preserves total mass (sums to ~1)", async () => {
    const w = 28,
      h = 28;
    const grid = new Array(w * h).fill(0);
    grid[14 * w + 14] = 1; // unit impulse
    const k = Array.from(gaussianKernel(2));
    const gpu = await convolveSeparableGpu(grid, w, h, k);
    const cpu = convCpu(grid, w, h, k);
    expect(maxAbsErr(gpu, cpu)).toBeLessThan(1e-4);
    let mass = 0;
    for (const v of gpu) mass += v;
    expect(mass).toBeCloseTo(1.0, 3); // normalised Gaussian conserves mass
  });
});
