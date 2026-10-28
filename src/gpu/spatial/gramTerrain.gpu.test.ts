import { describe, expect, it } from "vite-plus/test";
import { channelsFromLabels, coLocationModes } from "#src/spatial/gram";
import { EPANECHNIKOV } from "#src/spatial/kernels";
import { gramMatrixGpu } from "./gramMatrix";
import { modeParams } from "./gramModes";
import { probeChannels, standardise, whiteningMatrix } from "./gramTerrain";
import { wandDistance } from "./wandRegion";

// Two overlapping clouds and one apart, as in gramMatrix.gpu.test.ts — formula-built, no RNG.
const cloud = (n: number, cx: number, cy: number) => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(cx + (i % 17) * 1.3);
    ys.push(cy + (i % 13) * 1.7);
  }
  return { xs, ys };
};
const a = cloud(120, 40, 40);
const b = cloud(120, 52, 46);
const c = cloud(120, 140, 140);
const CHANNELS = channelsFromLabels(
  [...a.xs, ...b.xs, ...c.xs],
  [...a.ys, ...b.ys, ...c.ys],
  [...new Array(120).fill(0), ...new Array(120).fill(1), ...new Array(120).fill(2)],
);
const PARAMS = { bbox: [0, 0, 200, 200], width: 128, height: 128, radius: 14, kernel: EPANECHNIKOV } as const;

describe("probeChannels", () => {
  it("reads the same densities from f16 rasters as from f32, at even and odd columns", async () => {
    // Both columns sit inside the a/b overlap (x ≈ 55, y ≈ 155 in rows from the top), so every
    // channel but the far one is well above zero. Odd and even exercise both halves of a packed word.
    const at = [
      [35, 99],
      [36, 99],
    ] as const;
    // The rasters live in one pooled buffer the next call overwrites, so each precision is probed
    // before the other is computed.
    const full = await gramMatrixGpu(CHANNELS, PARAMS);
    const f32 = [await probeChannels(full, ...at[0]), await probeChannels(full, ...at[1])];
    const half = await gramMatrixGpu(CHANNELS, { ...PARAMS, precision: "f16" });
    const f16 = [await probeChannels(half, ...at[0]), await probeChannels(half, ...at[1])];

    let peak = 0;
    let diff = 0;
    for (let p = 0; p < 2; p++)
      for (let k = 0; k < 3; k++) {
        peak = Math.max(peak, Math.abs(f32[p]![k]!));
        diff = Math.max(diff, Math.abs(f32[p]![k]! - f16[p]![k]!));
      }
    expect(peak).toBeGreaterThan(0);
    // f16 keeps ~11 bits, so a density moves by about 1e-3 of itself — not by noise.
    expect(diff / peak).toBeLessThan(2e-3);
  });

  it("interpolates between pixel centres, so the sample follows the pointer", async () => {
    const res = await gramMatrixGpu(CHANNELS, PARAMS);
    const left = await probeChannels(res, 35, 99);
    const right = await probeChannels(res, 36, 99);
    const mid = await probeChannels(res, 35.5, 99);
    const quarter = await probeChannels(res, 35.25, 99);
    let peak = 0;
    let off = 0;
    for (let k = 0; k < 3; k++) {
      peak = Math.max(peak, Math.abs(left[k]!), Math.abs(right[k]!));
      off = Math.max(off, Math.abs(mid[k]! - (left[k]! + right[k]!) / 2), Math.abs(quarter[k]! - (0.75 * left[k]! + 0.25 * right[k]!)));
    }
    // The neighbours must differ, or a nearest-pixel probe would pass too.
    let step = 0;
    for (let k = 0; k < 3; k++) step = Math.max(step, Math.abs(left[k]! - right[k]!));
    expect(step / peak).toBeGreaterThan(1e-3);
    expect(off / peak).toBeLessThan(1e-5);
  });
});

describe("wandDistance", () => {
  it("is the whitened distance the probe implies, at every pixel", async () => {
    const res = await gramMatrixGpu(CHANNELS, PARAMS);
    const { vectors, values } = coLocationModes(res);
    const K = res.labels.length;
    const zAt = async (col: number, row: number) => standardise(await probeChannels(res, col, row), res.resident.mean, res.resident.sd);
    const ref = await zAt(35, 99);
    const d = await wandDistance(res, modeParams(res, { vectors, reference: ref, modesUsed: K }));
    const A = whiteningMatrix(vectors, values, K, K);
    let worst = 0;
    for (const [col, row] of [
      [35, 99],
      [20, 110],
      [90, 40],
    ] as const) {
      const dz = (await zAt(col, row)).map((z, a) => z - ref[a]!);
      let d2 = 0;
      for (let k = 0; k < K; k++) {
        let u = 0;
        for (let a = 0; a < K; a++) u += A[k * K + a]! * dz[a]!;
        d2 += u * u;
      }
      worst = Math.max(worst, Math.abs(d[row * res.width + col]! - Math.sqrt(d2)) / Math.max(1, Math.sqrt(d2)));
    }
    expect(d.length).toBe(res.width * res.height);
    expect(worst).toBeLessThan(1e-3);
  });
});
