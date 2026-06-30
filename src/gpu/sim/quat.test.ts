import { describe, expect, it } from "vitest";
import { forward, fromAxisAngle, IDENTITY, integrateQuat, mulQuat, normalizeQuat, rotateVec3 } from "./quat";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, p = 10) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i]!, p);
};

describe("quat", () => {
  it("identity rotates nothing", () => {
    close(rotateVec3(IDENTITY, [1, 2, 3]), [1, 2, 3]);
  });

  it("rotates a vector 90° about z", () => {
    const q = fromAxisAngle([0, 0, 1], Math.PI / 2);
    close(rotateVec3(q, [1, 0, 0]), [0, 1, 0]);
  });

  it("composes rotations (two 90° about z = 180°)", () => {
    const q90 = fromAxisAngle([0, 0, 1], Math.PI / 2);
    close(rotateVec3(mulQuat(q90, q90), [1, 0, 0]), [-1, 0, 0]);
  });

  it("stays unit under integration", () => {
    let q = IDENTITY;
    for (let i = 0; i < 100; i++) q = integrateQuat(q, [0.3, 0.1, -0.2], 0.1);
    expect(Math.hypot(q[0], q[1], q[2], q[3])).toBeCloseTo(1, 9);
  });

  it("integrateQuat about z by π/2 matches the closed form", () => {
    const q = integrateQuat(IDENTITY, [0, 0, 1], Math.PI / 2);
    close(rotateVec3(q, [1, 0, 0]), [0, 1, 0], 6);
  });

  it("forward axis of identity is +z", () => {
    close(forward(IDENTITY), [0, 0, 1]);
    close(normalizeQuat([2, 0, 0, 0]), IDENTITY);
  });
});
