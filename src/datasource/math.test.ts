// Geometry sanity: frustum corners and the AABB frustum test.
import { describe, expect, it } from "vitest";
import {
  type Affine3,
  aabbOutsideFrustum,
  applyAffine,
  boxOutsideFrustum,
  type Camera,
  frustumCorners,
  frustumPlanes,
  invertAffine,
  length,
  orientedBoxCorners,
  sub,
  type Vec3,
} from "./math";

const cam: Camera = {
  eye: [0, 0, 0],
  forward: [0, 0, 1],
  up: [0, 1, 0],
  fovY: Math.PI / 3,
  aspect: 1.5,
  near: 0.1,
  far: 100,
  viewportHeightPx: 600,
};

describe("frustumCorners", () => {
  it("returns 8 corners, near ones closer to the eye than far ones", () => {
    const c = frustumCorners(cam, 1, 10);
    expect(c.length).toBe(8);
    const nearAvg = c.slice(0, 4).reduce((s, p) => s + length(sub(p, cam.eye)), 0) / 4;
    const farAvg = c.slice(4, 8).reduce((s, p) => s + length(sub(p, cam.eye)), 0) / 4;
    expect(nearAvg).toBeLessThan(farAvg);
  });

  it("widens with distance (far plane spans more than near)", () => {
    const c = frustumCorners(cam, 1, 10);
    const nearW = length(sub(c[1] ?? [0, 0, 0], c[0] ?? [0, 0, 0]));
    const farW = length(sub(c[5] ?? [0, 0, 0], c[4] ?? [0, 0, 0]));
    expect(farW).toBeGreaterThan(nearW * 5);
  });
});

describe("aabbOutsideFrustum", () => {
  const planes = frustumPlanes(cam);
  it("keeps a box in front, culls one behind", () => {
    expect(aabbOutsideFrustum({ min: [-1, -1, 5], max: [1, 1, 7] }, planes)).toBe(false);
    expect(aabbOutsideFrustum({ min: [-1, -1, -20], max: [1, 1, -18] }, planes)).toBe(true);
  });
});

describe("invertAffine", () => {
  // A rotated (about y), anisotropically-scaled, translated placement.
  const ct = Math.cos(0.6),
    st = Math.sin(0.6);
  const rot: Affine3 = {
    origin: [3, -2, 5],
    axes: [
      [2 * ct, 0, 2 * -st],
      [0, 1.5, 0],
      [0.5 * st, 0, 0.5 * ct],
    ],
  };
  it("round-trips: applyAffine(inv, applyAffine(a, p)) === p", () => {
    const inv = invertAffine(rot);
    for (const p of [
      [1, 2, 3],
      [-4, 0, 2],
      [0, 0, 0],
    ] as Vec3[]) {
      const back = applyAffine(inv, applyAffine(rot, p));
      for (let i = 0; i < 3; i++) expect(back[i] ?? 0).toBeCloseTo(p[i] ?? 0, 10);
    }
  });
});

describe("boxOutsideFrustum (oriented)", () => {
  const planes = frustumPlanes(cam);
  it("keeps an oriented box in front, culls one behind", () => {
    const front = orientedBoxCorners(
      {
        origin: [0, 0, 6],
        axes: [
          [1, 0, 1],
          [0, 1, 0],
          [-1, 0, 1],
        ],
      },
      [0, 0, 0],
      [1, 1, 1],
    );
    const behind = orientedBoxCorners(
      {
        origin: [0, 0, -19],
        axes: [
          [1, 0, 1],
          [0, 1, 0],
          [-1, 0, 1],
        ],
      },
      [0, 0, 0],
      [1, 1, 1],
    );
    expect(boxOutsideFrustum(front, planes)).toBe(false);
    expect(boxOutsideFrustum(behind, planes)).toBe(true);
  });
});
