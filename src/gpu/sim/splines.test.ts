import { describe, expect, it } from "vitest";
import { calcGoalWeight, costvDown, costvUp, goalTightness, posVelSpline, velSpline } from "./splines";
import type { Vec3In } from "./vec3";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, p = 6) => {
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i]!, p);
};

describe("activity envelopes", () => {
  it("costvUp ramps 0→1, costvDown ramps 1→0", () => {
    expect(costvUp(0, 10, -1)).toBe(0);
    expect(costvUp(0, 10, 11)).toBe(1);
    expect(costvUp(0, 10, 5)).toBeCloseTo(0.25, 6); // (0.5)² at the midpoint
    expect(costvDown(0, 10, -1)).toBe(1);
    expect(costvDown(0, 10, 11)).toBe(0);
  });

  it("calcGoalWeight is the free-motion weight: 0 at the goal, 1 at the window edges", () => {
    const goal = { goalTime: 100, startTight: 50, finishAct: 150 };
    expect(calcGoalWeight(goal, 100)).toBeCloseTo(0, 6); // goal fully tight here
    expect(calcGoalWeight(goal, 50)).toBeCloseTo(1, 6); // free before the window
    expect(calcGoalWeight(goal, 150)).toBeCloseTo(1, 6); // free after
    expect(calcGoalWeight(goal, 75)).toBeGreaterThan(0);
    expect(calcGoalWeight(goal, 75)).toBeLessThan(1);
  });

  it("goalTightness is the inverse: peaks (→1) at the figure onset", () => {
    const goal = { goalTime: 100, startTight: 50, finishAct: 150 };
    expect(goalTightness(goal, 100)).toBeCloseTo(1, 6);
    expect(goalTightness(goal, 50)).toBeCloseTo(0, 6);
  });
});

describe("velSpline (reach a target velocity, C²)", () => {
  const p1: Vec3In = [0, 0, 0];
  const v1: Vec3In = [1, 0, 0];
  const w1: Vec3In = [0, 0, 0];
  const v2: Vec3In = [0, 2, 0];
  const T = 10;

  it("starts at the current state at t=0", () => {
    const s = velSpline(p1, v1, w1, v2, T, 0);
    close(s.pos, p1);
    close(s.vel, v1);
    close(s.accel, w1);
  });

  it("reaches the target velocity at t=T2", () => {
    const s = velSpline(p1, v1, w1, v2, T, T);
    close(s.vel, v2, 5);
  });

  it("holds the current state for a degenerate horizon", () => {
    const s = velSpline(p1, v1, w1, v2, 0, 1);
    close(s.vel, v1);
  });
});

describe("posVelSpline (reach target pos AND vel, C²)", () => {
  const p1: Vec3In = [0, 0, 0];
  const v1: Vec3In = [0, 0, 0];
  const w1: Vec3In = [0, 0, 0];
  const p2: Vec3In = [5, 0, 0];
  const v2: Vec3In = [0, 1, 0];
  const T = 8;

  it("starts at the current state", () => {
    const s = posVelSpline(p1, v1, w1, p2, v2, T, 0);
    close(s.pos, p1);
    close(s.vel, v1);
  });

  it("reaches target position and velocity at t=T2 (accel→0)", () => {
    const s = posVelSpline(p1, v1, w1, p2, v2, T, T);
    close(s.pos, p2, 4);
    close(s.vel, v2, 4);
    close(s.accel, [0, 0, 0], 4); // quintic boundary condition: zero accel at T
  });
});
