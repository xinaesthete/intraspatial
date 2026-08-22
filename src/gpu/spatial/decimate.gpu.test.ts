import { describe, expect, it } from "vitest";
import { decimateCpu, decimatedSize, decimateGpu } from "./decimate";

// Aggregated assertions only (no per-element expect loops — they kill the Dawn fork).
function firstMismatch(a: ArrayLike<number>, b: ArrayLike<number>, tol = 0): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i]! - b[i]!) > tol) return `index ${i}: got ${a[i]}, want ${b[i]}`;
  }
  return undefined;
}

const hash = (i: number) => ((i * 2654435761) % 97) / 97;

describe("decimateGpu (block mean/min/max)", () => {
  it("matches the CPU golden on divisible and non-divisible sizes", async () => {
    for (const [w, h, f] of [
      [64, 48, 4],
      [37, 23, 3], // partial blocks on both edges
      [100, 70, 8],
      [128, 128, 64],
      [5, 5, 7], // factor larger than the grid: a single partial block
    ] as const) {
      const grid = Array.from({ length: w * h }, (_, i) => hash(i));
      const size = decimatedSize(w, h, f);
      expect(size).toEqual({ width: Math.ceil(w / f), height: Math.ceil(h / f) });
      for (const mode of ["mean", "min", "max"] as const) {
        const gpu = await decimateGpu(grid, w, h, f, mode);
        expect(gpu.length).toBe(size.width * size.height);
        // min/max are bit-exact; the mean is a fixed-order f32 sum on both sides, so a few ulps.
        const tol = mode === "mean" ? 1e-5 : 0;
        expect(firstMismatch(gpu, decimateCpu(grid, w, h, f, mode), tol), `${w}x${h} f=${f} ${mode}`).toBeUndefined();
      }
    }
  });

  it("partial edge blocks average over the cells that exist, and factor 1 is the identity", async () => {
    // 5x3 grid of ones with factor 2: the right column block has width 1, the bottom row block
    // height 1 — a zero-padded mean would read 0.5/0.25 there; ours reads 1 everywhere.
    const w = 5;
    const h = 3;
    const ones = new Array(w * h).fill(1);
    const mean = await decimateGpu(ones, w, h, 2, "mean");
    expect(Array.from(mean)).toEqual([1, 1, 1, 1, 1, 1]);
    // A known block: 4x4 ramp, factor 2 → means of each 2x2 block.
    const ramp = Array.from({ length: 16 }, (_, i) => i);
    const m = await decimateGpu(ramp, 4, 4, 2, "mean");
    expect(Array.from(m)).toEqual([2.5, 4.5, 10.5, 12.5]);
    expect(Array.from(await decimateGpu(ramp, 4, 4, 2, "min"))).toEqual([0, 2, 8, 10]);
    expect(Array.from(await decimateGpu(ramp, 4, 4, 2, "max"))).toEqual([5, 7, 13, 15]);
    const id = await decimateGpu(ramp, 4, 4, 1, "mean");
    expect(Array.from(id)).toEqual(ramp);
  });

  it("rejects a bad factor", async () => {
    await expect(decimateGpu([1, 2, 3, 4], 2, 2, 0, "mean")).rejects.toThrow(/factor/);
    await expect(decimateGpu([1, 2, 3, 4], 2, 2, 1.5, "mean")).rejects.toThrow(/factor/);
  });
});
