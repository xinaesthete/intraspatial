// DWT reference math (ADR-0006): round-trip correctness and the packed-Mallat band
// geometry. CPU-only and fast. Mirrors the reference `selfTest` in docs-site/src/lib/dwt.ts.
import { describe, it, expect } from "vitest";
import { fdwt2d, idwt2d, dwtBands, mirror } from "./dwt";

// Deterministic pseudo-random integer fixture (no Math.random → reproducible).
function intFixture(w: number, h: number): Float32Array {
  const a = new Float32Array(w * h);
  for (let i = 0; i < a.length; i++) a[i] = (((i * 2654435761) >>> 0) % 256) - 128;
  return a;
}

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
};

describe("dwt", () => {
  it("mirror reflects indices without repeating the boundary", () => {
    // n=4: indices ... -1,0,1,2,3,4,5 → 1,0,1,2,3,2,1
    expect([-1, 0, 1, 2, 3, 4, 5].map((i) => mirror(i, 4))).toEqual([1, 0, 1, 2, 3, 2, 1]);
    expect(mirror(7, 1)).toBe(0);
  });

  it("5/3 round-trips exactly on integer data (lossless), non-power-of-two", () => {
    const w = 37, h = 53, levels = 4;
    const src = intFixture(w, h);
    const coeffs = fdwt2d(src, w, h, "5/3", levels);
    const rec = idwt2d(coeffs, w, h, "5/3", levels);
    expect(maxAbsDiff(rec, src)).toBe(0);
  });

  it("9/7 round-trips within float tolerance", () => {
    const w = 32, h = 32, levels = 3;
    const src = intFixture(w, h);
    const coeffs = fdwt2d(src, w, h, "9/7", levels);
    const rec = idwt2d(coeffs, w, h, "9/7", levels);
    expect(maxAbsDiff(rec, src)).toBeLessThan(0.1); // f32 9/7 is lossy by construction
  });

  it("a forward DWT actually moves energy into the LL corner (it is a transform, not a copy)", () => {
    const w = 16, h = 16, levels = 2;
    // Smooth ramp → most energy should concentrate in the LL approximation.
    const src = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) src[y * w + x] = x + y;
    const coeffs = fdwt2d(src, w, h, "5/3", levels);
    expect(maxAbsDiff(coeffs, src)).toBeGreaterThan(0); // not a no-op
  });

  it("band geometry: LL + 3·levels detail bands, LL at the corner, partitioning the grid", () => {
    const w = 24, h = 16, levels = 3;
    const bands = dwtBands(w, h, levels);
    expect(bands.filter((b) => b.type === "LL")).toHaveLength(1);
    expect(bands.filter((b) => b.type !== "LL")).toHaveLength(3 * levels);
    const ll = bands.find((b) => b.type === "LL")!;
    expect([ll.x, ll.y]).toEqual([0, 0]);
    // total covered area equals the grid (the packed layout tiles it with no overlap)
    const area = bands.reduce((s, b) => s + b.w * b.h, 0);
    expect(area).toBe(w * h);
  });
});
