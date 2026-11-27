import { describe, expect, it } from "vite-plus/test";
import { channelsFromLabels, coLocationModes } from "#src/spatial/gram";
import { histogramCpu } from "#src/spatial/histogram";
import { EPANECHNIKOV } from "#src/spatial/kernels";
import { gramMatrixGpu } from "./gramMatrix";
import { modeParams } from "./gramModes";
import { probeChannels, standardise } from "./gramTerrain";
import { modeHistograms, modeProjection } from "./modeHistograms";

// The fixture of gramTerrain.gpu.test.ts: two overlapping clouds and one apart, formula-built.
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
// An odd width, so a plane's pixel count is not a multiple of 64 and the 256-byte plane padding matters.
const PARAMS = { bbox: [0, 0, 200, 200], width: 125, height: 128, radius: 14, kernel: EPANECHNIKOV } as const;

describe("modeHistograms", () => {
  it("projects every pixel onto the painted modes and bins each plane exactly", async () => {
    const res = await gramMatrixGpu(CHANNELS, PARAMS);
    const { vectors } = coLocationModes(res);
    const K = res.labels.length;
    const params = modeParams(res, { vectors });
    const planes = await modeProjection(res, params);
    const n = res.width * res.height;

    // The projection at a few pixels, against the host's z·v from the probe.
    let worst = 0;
    let peak = 0;
    for (const [col, row] of [
      [35, 99],
      [20, 110],
      [90, 40],
      [124, 0],
    ] as const) {
      const z = standardise(await probeChannels(res, col, row), res.resident.mean, res.resident.sd);
      for (let k = 0; k < 3; k++) {
        let y = 0;
        for (let i = 0; i < K; i++) y += z[i]! * vectors[k * K + i]!;
        peak = Math.max(peak, Math.abs(y));
        worst = Math.max(worst, Math.abs(planes[k * n + row * res.width + col]! - y));
      }
    }
    expect(peak).toBeGreaterThan(0.1);
    expect(worst / peak).toBeLessThan(1e-4);

    const got = await modeHistograms(res, params, undefined, { bins: 32, readback: true });
    expect(got.planes?.every((v, i) => Object.is(v, planes[i]))).toBe(true);
    expect(got.modes.length).toBe(3);
    const mismatched = got.modes
      .map((h, k) => {
        const want = histogramCpu(planes.subarray(k * n, (k + 1) * n), { bins: 32 });
        const same = h.lo === want.lo && h.hi === want.hi && h.counts.every((v, i) => v === want.counts[i]);
        return same ? -1 : k;
      })
      .filter((k) => k >= 0);
    expect(mismatched).toEqual([]);
    expect(got.halfRanges.every((r) => r > 0)).toBe(true);

    // `ranges` replaces saturate·σ: the scale is the reciprocal half-range.
    const ranged = modeParams(res, { vectors, ranges: got.halfRanges });
    expect(ranged.scales.map((s, k) => s * got.halfRanges[k]!).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([1, 1, 1]);
  });
});
