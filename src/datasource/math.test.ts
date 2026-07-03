// Geometry sanity: frustum corners and the AABB frustum test.
import { describe, expect, it } from "vitest";
import { aabbOutsideFrustum, frustumCorners, frustumPlanes, length, sub, type Camera } from "./math";

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
