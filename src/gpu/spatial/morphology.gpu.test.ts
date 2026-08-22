import { describe, expect, it } from "vitest";
import { morphologyGpu, openingGpu } from "./morphology";

// CPU oracle: the NON-separable definition — a direct (2r+1)² square window with
// clamp-to-edge — so the test also proves the two-pass separable GPU form is exact.
// Mirrors psychogeo `codec_eval/foliage.py::binary_reduce` (np.pad mode="edge"), which is the
// validated reference for the LIDAR foliage layer; grey-scale min/max generalises it and the
// binary case is the 0/1 special case.
function morphCpu(grid: ArrayLike<number>, w: number, h: number, r: number, mode: "erode" | "dilate"): Float32Array {
  const out = new Float32Array(w * h);
  const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let acc = mode === "erode" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const v = grid[clamp(y + dy, h - 1) * w + clamp(x + dx, w - 1)]!;
          acc = mode === "erode" ? Math.min(acc, v) : Math.max(acc, v);
        }
      out[y * w + x] = acc;
    }
  return out;
}

function firstMismatch(a: ArrayLike<number>, b: ArrayLike<number>): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return `index ${i}: got ${a[i]}, want ${b[i]}`;
  return undefined;
}

describe("morphologyGpu (separable min/max)", () => {
  it("erode and dilate are bit-exact against the square-window CPU oracle", async () => {
    const w = 37,
      h = 23; // deliberately not multiples of the workgroup size
    const grid = Array.from({ length: w * h }, (_, i) => ((i * 2654435761) % 101) / 101);
    for (const r of [1, 2, 5]) {
      for (const mode of ["erode", "dilate"] as const) {
        const gpu = await morphologyGpu(grid, w, h, r, mode);
        expect(firstMismatch(gpu, morphCpu(grid, w, h, r, mode)), `${mode} r=${r}`).toBeUndefined();
      }
    }
  });

  it("opening removes a one-pixel building perimeter but keeps the canopy blob", async () => {
    // The foliage.py finding: a roof's one-pixel outline reads as canopy (FZ on the roof edge,
    // LZ on the ground beside it). It is thin; canopy is not. An r=1 opening separates them.
    const w = 32,
      h = 32;
    const mask = new Float32Array(w * h);
    // hollow square "building perimeter": rows/cols 4..15, one pixel wide
    for (let i = 4; i <= 15; i++) {
      mask[4 * w + i] = 1;
      mask[15 * w + i] = 1;
      mask[i * w + 4] = 1;
      mask[i * w + 15] = 1;
    }
    // solid 6x6 "canopy" blob at 20..25
    for (let y = 20; y <= 25; y++) for (let x = 20; x <= 25; x++) mask[y * w + x] = 1;

    const opened = await openingGpu(mask, w, h, 1);
    let perimeter = 0;
    let blob = 0;
    for (let i = 4; i <= 15; i++) perimeter += opened[4 * w + i]! + opened[15 * w + i]! + opened[i * w + 4]! + opened[i * w + 15]!;
    for (let y = 20; y <= 25; y++) for (let x = 20; x <= 25; x++) blob += opened[y * w + x]!;
    expect(perimeter).toBe(0);
    expect(blob).toBe(36);
    // and nothing was invented: opening is anti-extensive
    for (let i = 0; i < mask.length; i++) expect(opened[i]! <= mask[i]!).toBe(true);
  });

  it("grey-scale opening at large radius flattens a narrow spike (bare-earth estimate)", async () => {
    // LZ-style surface: a gentle plane with a 3-pixel-wide 'tree' spike; an r=8 opening is the
    // morphological ground estimate and should remove the spike while tracking the plane exactly.
    const w = 40,
      h = 40;
    const plane = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) plane[y * w + x] = 100 + 0.1 * x + 0.05 * y;
    const lz = Float32Array.from(plane);
    for (let y = 19; y <= 21; y++) for (let x = 19; x <= 21; x++) lz[y * w + x] = 130;

    const ground = await openingGpu(lz, w, h, 8);
    // spike gone: at its centre the estimate is back near the plane (below the spike by >25 m)
    expect(lz[20 * w + 20]! - ground[20 * w + 20]!).toBeGreaterThan(25);
    // and the estimate never exceeds the input (anti-extensive) nor drops far below the plane
    for (let i = 0; i < lz.length; i++) {
      expect(ground[i]! <= lz[i]!).toBe(true);
      expect(plane[i]! - ground[i]!).toBeLessThan(0.1 * 16 + 0.05 * 16 + 1e-3); // one window's slope at most
    }
  });
});
