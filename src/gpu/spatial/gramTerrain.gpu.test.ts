import { describe, expect, it } from "vite-plus/test";
import { channelsFromLabels, coLocationModes } from "#src/spatial/gram";
import { EPANECHNIKOV } from "#src/spatial/kernels";
import { getDevice } from "../device";
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

  it("interpolates between pixel centres on both axes, so the sample follows the pointer", async () => {
    const res = await gramMatrixGpu(CHANNELS, PARAMS);
    // One probe at a time: the output buffer is shared. The cell from (35, 99) to (36, 100): its four
    // corners, then points inside it.
    const at = async (col: number, row: number) => probeChannels(res, col, row);
    const c00 = await at(35, 99);
    const c10 = await at(36, 99);
    const c01 = await at(35, 100);
    const c11 = await at(36, 100);
    const cases = [
      [35.5, 99],
      [35.25, 99],
      [35, 99.5],
      [35, 99.75],
      [35.25, 99.75],
    ] as const;
    const got: Float64Array[] = [];
    for (const [col, row] of cases) got.push(await at(col, row));
    let peak = 0;
    let off = 0;
    cases.forEach(([col, row], i) => {
      const fx = col - 35;
      const fy = row - 99;
      for (let k = 0; k < 3; k++) {
        const want = (1 - fy) * ((1 - fx) * c00[k]! + fx * c10[k]!) + fy * ((1 - fx) * c01[k]! + fx * c11[k]!);
        peak = Math.max(peak, Math.abs(c00[k]!), Math.abs(c11[k]!));
        off = Math.max(off, Math.abs(got[i]![k]! - want));
      }
    });
    // Neighbours must differ along each axis, or rounding that axis would pass too.
    let across = 0;
    let down = 0;
    for (let k = 0; k < 3; k++) {
      across = Math.max(across, Math.abs(c00[k]! - c10[k]!));
      down = Math.max(down, Math.abs(c00[k]! - c01[k]!));
    }
    expect(across / peak).toBeGreaterThan(1e-3);
    expect(down / peak).toBeGreaterThan(1e-3);
    expect(off / peak).toBeLessThan(1e-5);
  });

  it("reads a snapshot of the rasters after the next compute has overwritten the pool", async () => {
    const res = await gramMatrixGpu(CHANNELS, PARAMS);
    const want = await probeChannels(res, 35, 99);
    // Copy the rasters aside, as a page does, then let another compute take the pooled buffer.
    const device = await getDevice();
    const bytes = res.labels.length * res.height * res.resident.rowWords * 4;
    const kept = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(res.resident.buffer, 0, kept, 0, bytes);
    device.queue.submit([enc.finish()]);
    await gramMatrixGpu(CHANNELS, { ...PARAMS, radius: 30 });
    const got = await probeChannels(res, 35, 99, kept);
    const pooled = await probeChannels(res, 35, 99);
    kept.destroy();
    let off = 0;
    let moved = 0;
    let peak = 0;
    for (let k = 0; k < 3; k++) {
      peak = Math.max(peak, Math.abs(want[k]!));
      off = Math.max(off, Math.abs(got[k]! - want[k]!));
      moved = Math.max(moved, Math.abs(pooled[k]! - want[k]!));
    }
    expect(peak).toBeGreaterThan(0);
    // The pooled buffer now holds the other compute's rasters; the snapshot still holds these.
    expect(moved / peak).toBeGreaterThan(1e-3);
    expect(off / peak).toBeLessThan(1e-6);
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
