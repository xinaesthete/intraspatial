// Goal splines and activity envelopes — ported from DANCERL (docs/DANCERL.ESME). These
// are what make a dancer *reach a state of motion* rather than snap to a keyframe: a goal
// is a target velocity (and optionally position) at a target time, approached along a
// polynomial spline that threads the *current acceleration* (`LastAcc`) so the motion is
// C²-continuous — no jerk when a Ceilidh figure is called. The cosine envelopes ramp a
// goal's influence up to its onset and ease it down after (DANCERL `COSTVUP`/`COSTVDOWN`).
import type { Vec3 } from "./vec3";

/** DANCERL `COSTVUP` — a squared raised-cosine ramp from 0 (at/below `from`) to 1
 *  (at/above `to`). */
export function costvUp(from: number, to: number, x: number): number {
  let r: number;
  if (x < from) r = 0;
  else if (x > to) r = 1;
  else r = 0.5 - 0.5 * Math.cos(((x - from) * Math.PI) / (to - from));
  return r * r;
}

/** DANCERL `COSTVDOWN` — a squared raised-cosine ramp from 1 (at/below `from`) to 0
 *  (at/above `to`). */
export function costvDown(from: number, to: number, x: number): number {
  let r: number;
  if (x < from) r = 1;
  else if (x > to) r = 0;
  else r = 0.5 + 0.5 * Math.cos(((x - from) * Math.PI) / (to - from));
  return r * r;
}

export interface Goal {
  /** When the dancer should be in the target state of motion. */
  goalTime: number;
  /** When the goal starts tightening (DANCERL `StartTight`, ~goalTime − window). */
  startTight: number;
  /** When the goal's influence has fully eased off afterwards (DANCERL `FinishAct`). */
  finishAct: number;
}

/** DANCERL `CalcGoalWeight` — the weight of *free motion* (physics), which DANCERL scales
 *  down as a goal tightens. It is **0 at the goal time** (the goal fully dominates — the
 *  dancer is in the called state of motion) and ramps to **1 at the activity-window edges**
 *  (`startTight` before, `finishAct` after — free motion, goal loose). So the goal's
 *  *tightness* is `1 − calcGoalWeight`, peaking at the figure onset: the scramble. */
export function calcGoalWeight(goal: Goal, time: number): number {
  return time <= goal.goalTime
    ? costvDown(goal.startTight, goal.goalTime, time)
    : costvUp(goal.goalTime, goal.finishAct, time);
}

/** Goal *tightness* — `1 − calcGoalWeight`, i.e. how strongly the called state of motion
 *  pulls right now. Peaks (→1) at the goal time, eases to 0 at the window edges. */
export function goalTightness(goal: Goal, time: number): number {
  return 1 - calcGoalWeight(goal, time);
}

export interface SplineState {
  pos: Vec3;
  vel: Vec3;
  accel: Vec3;
}

/** DANCERL `VELSPLINE` — reach target velocity `v2` at `t2`, from current state
 *  (`p1`,`v1`,`w1` = pos/vel/accel), evaluated at time `t`. Returns the smooth
 *  (pos, vel, accel) along the way; the caller applies the returned `accel`. Threading
 *  `w1` (LastAcc) is what gives C² continuity. Degenerate `t2 ≤ 0` ⇒ hold current state.
 *  A quartic in each component: P(t)=a t⁴+b t³+w1 t²+v1 t+p1. */
export function velSpline(p1: Vec3, v1: Vec3, w1: Vec3, v2: Vec3, t2: number, t: number): SplineState {
  if (t2 <= 1e-9) return { pos: p1, vel: v1, accel: w1 };
  const dt2 = t2 * t2;
  const dt3 = dt2 * t2;
  const comp = (k: 0 | 1 | 2) => {
    const a = (v1[k] - v2[k]) / (2 * dt3) + w1[k] / (2 * dt2);
    const b = (v2[k] - v1[k]) / dt2 - (4 * w1[k]) / (3 * t2);
    const pos = (((a * t + b) * t + w1[k]) * t + v1[k]) * t + p1[k];
    const vel = ((4 * a * t + 3 * b) * t + 2 * w1[k]) * t + v1[k];
    const accel = (12 * a * t + 6 * b) * t + 2 * w1[k]; // true P''(t)
    return { pos, vel, accel };
  };
  const cx = comp(0), cy = comp(1), cz = comp(2);
  return {
    pos: [cx.pos, cy.pos, cz.pos],
    vel: [cx.vel, cy.vel, cz.vel],
    accel: [cx.accel, cy.accel, cz.accel],
  };
}

/** DANCERL `POSVELSPLINE` — reach target pos `p2` AND vel `v2` (with zero acceleration)
 *  at `t2`, from current (`p1`,`v1`,`w1`), evaluated at `t`. A quintic threading
 *  `w1`=LastAcc for C² continuity. Returns true derivatives (the script carried Acc/2). */
export function posVelSpline(p1: Vec3, v1: Vec3, w1: Vec3, p2: Vec3, v2: Vec3, t2: number, t: number): SplineState {
  if (t2 <= 1e-9) return { pos: p1, vel: v1, accel: w1 };
  const dt2 = t2 * t2, dt3 = dt2 * t2, dt4 = dt3 * t2, dt5 = dt4 * t2;
  const comp = (k: 0 | 1 | 2) => {
    const a = (-3 * v1[k] - 3 * v2[k]) / dt4 + (6 * (p2[k] - p1[k])) / dt5 - w1[k] / dt3;
    const b = (8 * v1[k] + 7 * v2[k]) / dt3 - (15 * (p2[k] - p1[k])) / dt4 + (3 * w1[k]) / dt2;
    const c = (10 * (p2[k] - p1[k])) / dt3 - (6 * v1[k] + 4 * v2[k]) / dt2 - (3 * w1[k]) / t2;
    const d = w1[k], e = v1[k], f = p1[k];
    const pos = ((((a * t + b) * t + c) * t + d) * t + e) * t + f;
    const vel = (((5 * a * t + 4 * b) * t + 3 * c) * t + 2 * d) * t + e;
    const accel = ((20 * a * t + 12 * b) * t + 6 * c) * t + 2 * d; // true P''(t)
    return { pos, vel, accel };
  };
  const cx = comp(0), cy = comp(1), cz = comp(2);
  return {
    pos: [cx.pos, cy.pos, cz.pos],
    vel: [cx.vel, cy.vel, cz.vel],
    accel: [cx.accel, cy.accel, cz.accel],
  };
}
