import { beforeAll, describe, expect, it } from "vitest";
import type { Affine3 } from "../../coords";
import { getDevice } from "../device";
import { splatDensityGpu } from "../spatial/splatDensity";
import type { FieldValue } from "./handle";
import { Graph, pull } from "./index";

// Slice 2 (ADR-0018): the placement `splatDensity` constructs must reach the *runtime* value (not
// just the build-time handle), and adding it must not perturb the density it computes. Two claims
// in one GPU pull:
//   1. the pulled grid carries the points' `system` and the expected bbox/resolution worldFromArray;
//   2. its density still matches the host GPU splat exactly (compared against the GPU path, not the
//      cpuGolden — the render splat has its own truncation error a cpuGolden compare would mask).
// The grid width is 24, so a row is 96 bytes — NOT 256-aligned — which forces the resident de-pad
// pass to actually strip padding (the integer-division lesson). The value check is aggregated into
// a single max-abs-error assertion (per-element expect() loops crash the Dawn fork).

const W = 24;
const H = 24;
const SIGMA = 2;
const RADIUS_SIGMA = 4;
const BBOX: [number, number, number, number] = [10, 20, 34, 44]; // 24×24 world over 24×24 cells ⇒ cellSize 1
const XS = [15, 22, 28];
const YS = [26, 33, 39];

const expectedWfa: Affine3 = {
  origin: [BBOX[0], BBOX[1], 0],
  axes: [
    [(BBOX[2] - BBOX[0]) / W, 0, 0],
    [0, (BBOX[3] - BBOX[1]) / H, 0],
    [0, 0, 1],
  ],
};

function pointsSource(g: Graph, placed: boolean) {
  const data = new Float32Array(XS.length * 2);
  for (let i = 0; i < XS.length; i++) {
    data[2 * i] = XS[i]!;
    data[2 * i + 1] = YS[i]!;
  }
  const value: FieldValue = { shape: { kind: "points", n: XS.length }, dtype: "f32", data };
  if (placed)
    value.placement = {
      system: "global",
      worldFromArray: {
        origin: [0, 0, 0],
        axes: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      },
    };
  return g.source(value, "points");
}

function maxAbsErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe("splatDensity placement through execution (ADR-0018 slice 2)", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("placed points ⇒ grid carries system + worldFromArray, and density matches the host GPU splat", async () => {
    const g = new Graph();
    const pts = pointsSource(g, true);
    const out = await pull(
      g,
      g.op1("splatDensity", { points: pts }, { width: W, height: H, sigma: SIGMA, radiusSigma: RADIUS_SIGMA, bbox: BBOX }),
      {
        mode: "gpu",
      },
    );

    // (1) the constructed placement reached the runtime value.
    expect(out.placement?.system).toBe("global");
    expect(out.placement?.worldFromArray).toEqual(expectedWfa);
    expect(out.shape).toEqual({ kind: "grid", width: W, height: H });

    // (2) the density is unchanged vs the host GPU path (same renderToStaging under the hood).
    const host = await splatDensityGpu(XS, YS, { width: W, height: H, sigma: SIGMA, radiusSigma: RADIUS_SIGMA, bbox: BBOX });
    expect(out.data).toBeInstanceOf(Float32Array);
    expect(maxAbsErr(out.data!, host.data)).toBeLessThan(1e-4); // ONE aggregated assertion
  });

  it("unplaced points ⇒ grid placement absent (array space in ⇒ array space out)", async () => {
    const g = new Graph();
    const pts = pointsSource(g, false);
    const out = await pull(
      g,
      g.op1("splatDensity", { points: pts }, { width: W, height: H, sigma: SIGMA, radiusSigma: RADIUS_SIGMA, bbox: BBOX }),
      {
        mode: "gpu",
      },
    );
    expect(out.placement).toBeUndefined();
  });
});
