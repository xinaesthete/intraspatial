import { describe, expect, it } from "vitest";
import {
  BODY_BLOCK_COUNT,
  INTEGRATE_DEFAULTS,
  integrateBody,
  readBodyState,
  seedSwarmBody,
  tapBlock,
  writeBodyState,
  type BodyState,
} from "./body";
import { length, unpack, vec3, type Vec3In } from "./vec3";

const ZERO = vec3(0, 0, 0);
const bodyOf = (pos: Vec3In, vel: Vec3In): BodyState => ({
  pos: vec3(...unpack(pos)),
  vel: vec3(...unpack(vel)),
  accel: ZERO,
  angPos: ZERO,
  angVel: ZERO,
});

describe("body field layout", () => {
  it("round-trips a body state through the 5-block field", () => {
    const n = 4;
    const data = new Float32Array(n * 3 * BODY_BLOCK_COUNT);
    // f32-exact values so the Float32 round-trip is bit-equal.
    const b: BodyState = {
      pos: vec3(1, 2, 3),
      vel: vec3(4, 5, 6),
      accel: vec3(7, 8, 9),
      angPos: vec3(0.5, 0.25, -0.125),
      angVel: vec3(-1, -2, -3),
    };
    writeBodyState(data, 2, n, b);
    const got = readBodyState(data, 2, n);
    expect(Array.from(got.pos)).toEqual(Array.from(b.pos));
    expect(Array.from(got.vel)).toEqual(Array.from(b.vel));
    expect(Array.from(got.accel)).toEqual(Array.from(b.accel));
    expect(Array.from(got.angPos)).toEqual(Array.from(b.angPos));
    expect(Array.from(got.angVel)).toEqual(Array.from(b.angVel));
  });

  it("tapBlock slices the right block (length 3N)", () => {
    const n = 3;
    const data = seedSwarmBody(n, 1);
    const pos = tapBlock(data, n, "pos");
    expect(pos.length).toBe(n * 3);
    // pos block is the first 3N of the field
    expect(Array.from(pos)).toEqual(Array.from(data.subarray(0, n * 3)));
  });
});

describe("seedSwarmBody", () => {
  it("is deterministic and finite", () => {
    const a = seedSwarmBody(50, 7);
    const b = seedSwarmBody(50, 7);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(a.length).toBe(50 * 3 * BODY_BLOCK_COUNT);
    expect(a.every((x) => Number.isFinite(x))).toBe(true);
  });
});

describe("integrateBody", () => {
  it("is deterministic and pure", () => {
    const b = bodyOf([1, 0, 0], [0, 0.1, 0]);
    const f: Vec3In = [0.02, 0, 0];
    const r1 = integrateBody(b, f, ZERO, INTEGRATE_DEFAULTS);
    const r2 = integrateBody(b, f, ZERO, INTEGRATE_DEFAULTS);
    expect(Array.from(r1.pos)).toEqual(Array.from(r2.pos));
    expect(Array.from(r1.vel)).toEqual(Array.from(r2.vel));
    expect(Array.from(b.pos)).toEqual([1, 0, 0]); // input untouched
  });

  it("jerk-limits acceleration (cannot jump to the full target in one step)", () => {
    const b = bodyOf([0, 0, 0], [0, 0, 0]);
    const bigForce: Vec3In = [10, 0, 0];
    const r = integrateBody(b, bigForce, ZERO, INTEGRATE_DEFAULTS);
    expect(length(r.accel)).toBeLessThanOrEqual(INTEGRATE_DEFAULTS.jerkLimit + 1e-9);
  });

  it("damps to rest with no force", () => {
    let b = bodyOf([2, 0, 0], [0.5, 0.3, -0.2]);
    for (let i = 0; i < 400; i++) b = integrateBody(b, ZERO, ZERO, INTEGRATE_DEFAULTS);
    expect(length(b.vel)).toBeLessThan(1e-3);
  });

  it("respects the speed limit and stays bounded under a wild force", () => {
    let b = bodyOf([0, 0, 0], [0, 0, 0]);
    const params = { ...INTEGRATE_DEFAULTS, jerkLimit: 5 };
    for (let i = 0; i < 500; i++) b = integrateBody(b, [9, 9, 9], ZERO, params);
    expect(length(b.vel)).toBeLessThanOrEqual(params.speedLimit + 1e-6);
    expect(length(b.pos)).toBeLessThanOrEqual(params.maxRadius + 1e-6);
  });

  it("turns to face its motion when face > 0 (angPos moves)", () => {
    let b = bodyOf([0, 0, 0], [1, 0, 0]); // moving +x, facing +z initially
    for (let i = 0; i < 60; i++) b = integrateBody(b, ZERO, ZERO, { ...INTEGRATE_DEFAULTS, linDamp: 1, speedLimit: 2 });
    expect(length(b.angPos)).toBeGreaterThan(0); // it rotated toward facing +x
  });
});
