import { describe, expect, it } from "vite-plus/test";
import { distanceTransform, floodFrom, mixtureDistance } from "./regions";

/** A deterministic pseudo-random 0/1 raster. */
function noise(width: number, height: number, density: number, seed = 1): Float32Array {
  let s = seed;
  const out = new Float32Array(width * height);
  for (let i = 0; i < out.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = s / 2 ** 32 < density ? 1 : 0;
  }
  return out;
}

/** Brute-force distance to the nearest pixel below 0.5, the outside ring counting when `border`. */
function bruteDistance(src: Float32Array, width: number, height: number, border: boolean): number[] {
  const out: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let best = Number.POSITIVE_INFINITY;
      for (let yy = border ? -1 : 0; yy < height + (border ? 1 : 0); yy++) {
        for (let xx = border ? -1 : 0; xx < width + (border ? 1 : 0); xx++) {
          const outside = xx < 0 || yy < 0 || xx >= width || yy >= height;
          if (outside || src[yy * width + xx]! < 0.5) best = Math.min(best, Math.hypot(xx - x, yy - y));
        }
      }
      out.push(best);
    }
  }
  return out;
}

describe("distanceTransform", () => {
  it("is the exact Euclidean distance to the nearest pixel below 0.5", () => {
    const src = noise(23, 17, 0.8);
    const got = distanceTransform(src, 23, 17);
    const want = bruteDistance(src, 23, 17, false);
    expect(Math.max(...want.map((w, i) => Math.abs(w - got[i]!)))).toBeLessThan(1e-5);
  });

  it("measures from the raster's edge too with `border`", () => {
    const src = new Float32Array(9 * 7).fill(1);
    const got = distanceTransform(src, 9, 7, true);
    const want = bruteDistance(src, 9, 7, true);
    expect(Math.max(...want.map((w, i) => Math.abs(w - got[i]!)))).toBeLessThan(1e-5);
    expect(got[3 * 9 + 4]).toBe(4); // the centre is 4 from the top and bottom rings
  });
});

describe("floodFrom", () => {
  // Two blocks, joined only diagonally (not 4-connected).
  const width = 6;
  const src = Float32Array.from([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]);
  it("keeps the 4-connected piece holding the seed", () => {
    expect(Array.from(floodFrom(src, width, 4, 0.4, 0.6))).toEqual([
      1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
  });
  it("is empty off the mask, unless the seed is put in", () => {
    expect(floodFrom(src, width, 4, 5, 0).every((v) => v === 0)).toBe(true);
    expect(Array.from(floodFrom(src, width, 4, 5, 0, true)).reduce((a, v) => a + v, 0)).toBe(1);
  });
});

describe("mixtureDistance", () => {
  // Two planes on a 3×2 raster.
  const u = Float32Array.from([0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 1, 1]);
  it("is the distance to the seed's pixel at a pixel centre", () => {
    const d = mixtureDistance(u, 2, 3, 2, 1, 0);
    expect(d[1]).toBe(0);
    expect(d[5]).toBeCloseTo(Math.hypot(5 - 1, 1 - 0), 6);
  });
  it("interpolates the reference between pixels", () => {
    // Halfway between pixels 0 and 1: reference (0.5, 0).
    const d = mixtureDistance(u, 2, 3, 2, 0.5, 0);
    expect(d[0]).toBeCloseTo(0.5, 6);
    expect(d[1]).toBeCloseTo(0.5, 6);
  });
});
