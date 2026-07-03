import { describe, expect, it } from "vitest";
import {
  cohereForce,
  constrainForce,
  orbitForce,
  partnerOrbitForce,
  separateForce,
  solenoidForce,
  springForce,
  swimForce,
  vortexForce,
} from "./forces";
import { dot, length, type Vec3 } from "./vec3";

const finite = (v: Vec3) => v.every((x) => Number.isFinite(x));

describe("field forces", () => {
  it("constrain points back toward the origin", () => {
    const p: Vec3 = [3, 0, 0];
    const f = constrainForce(p, 1);
    expect(f[0]).toBeLessThan(0); // pulled in −x
    expect(dot(f, p)).toBeLessThan(0); // opposes position
    expect(constrainForce([0, 0, 0], 1)).toEqual([0, 0, 0]); // no NaN at origin
  });

  it("swim pushes outward along the radius", () => {
    const f = swimForce([2, 0, 0], 1);
    expect(f[0]).toBeGreaterThan(0);
  });

  it("vortex is tangential about y (perpendicular to the radial x/z)", () => {
    const p: Vec3 = [1, 0, 0];
    const f = vortexForce(p, 1);
    expect(f[0] * p[0] + f[2] * p[2]).toBeCloseTo(0, 9); // ⟂ radius in x/z
    expect(length(f)).toBeGreaterThan(0);
  });

  it("orbit is perpendicular to position (stays in the orbital plane)", () => {
    const p: Vec3 = [3, 0, 0];
    const v: Vec3 = [0, 0, 1];
    const f = orbitForce(p, v, 1);
    expect(dot(f, p)).toBeCloseTo(0, 9);
  });

  it("solenoid is finite everywhere, including on the y-axis", () => {
    expect(finite(solenoidForce([0, 1, 0], 1))).toBe(true); // lxz = 0 branch
    expect(finite(solenoidForce([2.5, 0.5, 1], 1))).toBe(true);
  });
});

describe("pairwise forces", () => {
  // three agents in a row on x
  const pos = new Float32Array([0, 0, 0, 1, 0, 0, 5, 0, 0]);
  const n = 3;

  it("cohere pulls toward the neighbour centroid within radius", () => {
    const f = cohereForce(0, pos, n, 1, 2); // only agent 1 (x=1) is within r=2
    expect(f[0]).toBeGreaterThan(0); // pulled toward +x
  });

  it("separate pushes away from a close neighbour", () => {
    const f = separateForce(0, pos, n, 1, 2); // agent 1 at x=1 is within r=2
    expect(f[0]).toBeLessThan(0); // pushed to −x, away from the neighbour
    // agent 2 (x=5) is outside the radius, so contributes nothing
    expect(separateForce(2, pos, n, 1, 2)).toEqual([0, 0, 0]);
  });

  it("spring attracts when too far and repels when too close", () => {
    // neighbour at distance 1; restLength 2 ⇒ too close ⇒ repel (−x)
    expect(springForce(0, pos, n, 1, 2, 3)[0]).toBeLessThan(0);
    // neighbour at distance 1; restLength 0.5 ⇒ too far ⇒ attract (+x)
    expect(springForce(0, pos, n, 1, 0.5, 3)[0]).toBeGreaterThan(0);
  });

  it("partnerOrbit is perpendicular to the agent's offset from the couple midpoint", () => {
    const f = partnerOrbitForce([1, 0, 0], [0, 0, 1], [-1, 0, 0], 1);
    expect(dot(f, [1, 0, 0])).toBeCloseTo(0, 9); // ⟂ the offset from midpoint (origin)
    expect(length(f)).toBeGreaterThan(0);
  });
});
