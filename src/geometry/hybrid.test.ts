// Hybrid decomposition (`hybrid.ts`, ADR-0013): node extent reporting, planar/non-planar split, and
// the "region where the BSP won't work". That region's whole point is its range — nothing (planar),
// a finite box (local growth), or unbounded (a displaced/blended half-space).

import { describe, expect, it } from "vitest";
import { type AABB, aabbFinite, aabbOverlaps, boundsSdf, isPlanar, nonPlanarRegions, planarSkeleton, UNBOUNDED } from "./hybrid";
import { box, evalSdf, plane, type Sdf, sphere } from "./index";

const contains = (a: AABB, p: [number, number, number], eps = 1e-9) =>
  p[0] >= a.min[0] - eps &&
  p[0] <= a.max[0] + eps &&
  p[1] >= a.min[1] - eps &&
  p[1] <= a.max[1] + eps &&
  p[2] >= a.min[2] - eps &&
  p[2] <= a.max[2] + eps;

describe("boundsSdf — node extents", () => {
  it("bounds primitives tightly", () => {
    expect(boundsSdf(sphere(1.5).node)).toEqual({ min: [-1.5, -1.5, -1.5], max: [1.5, 1.5, 1.5] });
    expect(boundsSdf(box(2, 1, 0.5).node)).toEqual({ min: [-2, -1, -0.5], max: [2, 1, 0.5] });
  });

  it("carries translate and scale", () => {
    expect(boundsSdf(sphere(1).translate(3, 0, -1).node)).toEqual({ min: [2, -1, -2], max: [4, 1, 0] });
    expect(boundsSdf(box(1).scale(2).node)).toEqual({ min: [-2, -2, -2], max: [2, 2, 2] });
  });

  it("is honestly unbounded for a half-space, but an intersection clamps it back", () => {
    expect(boundsSdf(plane([0, 1, 0], 0).node)).toEqual(UNBOUNDED);
    expect(aabbFinite(boundsSdf(plane([0, 1, 0], 0).node))).toBe(false);
    const roofed = box(1).intersect(plane([0, 1, 0], 0.5)); // a box capped by a plane
    expect(aabbFinite(boundsSdf(roofed.node))).toBe(true);
    expect(boundsSdf(roofed.node).max[1]).toBeLessThanOrEqual(1 + 1e-9);
  });

  it("widens by a displace amplitude and a smooth-union blend", () => {
    expect(boundsSdf(sphere(1).displace(0.3, 2).node)).toEqual({ min: [-1.3, -1.3, -1.3], max: [1.3, 1.3, 1.3] });
    const s = boundsSdf(box(1).smoothUnion(sphere(1).translate(2, 0, 0), 0.4).node);
    expect(s.min[0]).toBeCloseTo(-1.4, 9); // union [-1,3] expanded by k=0.4
    expect(s.max[0]).toBeCloseTo(3.4, 9);
  });
});

describe("isPlanar / planarSkeleton", () => {
  it("classifies planar vs non-planar trees", () => {
    expect(isPlanar(box(1).intersect(plane([1, 1, 0], 0.5)).node)).toBe(true);
    expect(isPlanar(sphere(1).node)).toBe(false);
    expect(isPlanar(box(1).smoothUnion(sphere(0.5), 0.3).node)).toBe(false);
  });

  it("pulls a planar skeleton out of a non-planar model", () => {
    const house = box(1, 0.6, 0.8).intersect(plane([0, 1, 1], 0.9)); // planar
    const model = house.smoothUnion(sphere(0.4).displace(0.2, 3).translate(0.5, 0.7, 0), 0.3);
    const skel = planarSkeleton(model.node);
    expect(skel).toBeDefined();
    expect(skel && isPlanar(skel)).toBe(true);
    expect(skel && evalSdf(skel, [0, 0, 0])).toBeLessThan(0); // a point deep in the house is inside the skeleton
  });
});

describe("nonPlanarRegions — where the BSP won't work", () => {
  it("is nothing for a purely planar model", () => {
    expect(nonPlanarRegions(box(1).intersect(plane([0, 1, 0], 0.5)).node)).toEqual([]);
  });

  it("is a local box for a growth smooth-unioned onto a house", () => {
    const house = box(1.2, 0.6, 0.9); // the big planar bulk
    const growth = sphere(0.3).displace(0.18, 3).translate(0.6, 0.8, 0.1);
    const regions = nonPlanarRegions(house.smoothUnion(growth, 0.25).node);
    expect(regions).toHaveLength(1); // one growth → one merged region
    const r = regions[0] as AABB;
    expect(aabbFinite(r)).toBe(true);
    expect(contains(r, [0.6, 0.8, 0.1])).toBe(true); // contains the growth
    expect(r.max[0] - r.min[0]).toBeLessThan(1.6); // local, not the whole 2.4-wide house
  });

  it("is unbounded when a half-space is displaced or blended", () => {
    const wavyGround = plane([0, 1, 0], 0).displace(0.2, 2); // an infinite noisy plane
    const rg = nonPlanarRegions(wavyGround.node);
    expect(rg).toHaveLength(1);
    expect(aabbFinite(rg[0] as AABB)).toBe(false);

    const blendedHalfSpaces = plane([0, 1, 0], 0).smoothUnion(plane([1, 0, 0], 0), 0.3); // seam runs forever
    expect(nonPlanarRegions(blendedHalfSpaces.node).some((r) => !aabbFinite(r))).toBe(true);
  });

  it("keeps two separated growths as two finite regions", () => {
    const model = box(2, 0.5, 0.5)
      .smoothUnion(sphere(0.3).translate(-1.5, 0.4, 0), 0.2)
      .smoothUnion(sphere(0.3).translate(1.5, 0.4, 0), 0.2);
    const regions = nonPlanarRegions(model.node);
    expect(regions.length).toBe(2);
    expect(regions.some((r) => aabbOverlaps(r, { min: [-1.6, 0, -0.1], max: [-1.4, 0.5, 0.1] }))).toBe(true);
    expect(regions.some((r) => aabbOverlaps(r, { min: [1.4, 0, -0.1], max: [1.6, 0.5, 0.1] }))).toBe(true);
  });
});
