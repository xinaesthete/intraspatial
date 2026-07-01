// The swarm body — the stateful core the force-graph integrates.
//
// Faithful to DANCERL, an agent is a little rigid body: linear position/velocity/
// acceleration AND angular position/velocity. DANCERL carried orientation as a rotation
// *vector* (angle-axis: direction = axis, magnitude = angle — see `AngPos` and
// `rot(length(AngPos)…, AngPos)` in the script), so we do too — which lets the whole body
// ride the op-graph as ONE uniform `vec3` points field of n = 5·N rows, blocks laid out
// as [pos | vel | accel | angPos | angVel]. One feedback node carries it; a `bodyTap` op
// exposes pos/vel so the force ops stay decoupled from this layout.
//
// The integrator is higher-order and on DANCERL's slow `timeFactor` timescale: the summed
// force is a *target acceleration*, approached through the carried `accel` under a jerk
// limit (C²-smooth — no snap at a figure onset), then accel → vel → pos. Angular motion
// optionally turns each dancer to *face* its travel.
import { addScaled, clampLength, cross, length, normalize, readVec3, scale, sub, writeVec3, ZERO3, type Vec3 } from "./vec3";
import { forward, fromAxisAngle } from "./quat";

/** The five vec3 blocks, in field order. */
export const BODY_BLOCKS = ["pos", "vel", "accel", "angPos", "angVel"] as const;
export type BodyBlock = (typeof BODY_BLOCKS)[number];
export const BODY_BLOCK_COUNT = BODY_BLOCKS.length; // 5

export interface BodyState {
  pos: Vec3;
  vel: Vec3;
  accel: Vec3;
  /** Orientation as an angle-axis rotation vector (DANCERL `AngPos`). */
  angPos: Vec3;
  /** Angular velocity as a rotation vector (DANCERL `AngVel`). */
  angVel: Vec3;
}

/** Agent `i`'s offset within block `b` of an `N`-agent body field. */
const blockOffset = (block: number, i: number, n: number): number => (block * n + i) * 3;

export function readBodyState(data: ArrayLike<number>, i: number, n: number): BodyState {
  return {
    pos: readVec3(data, blockOffset(0, i, n) / 3),
    vel: readVec3(data, blockOffset(1, i, n) / 3),
    accel: readVec3(data, blockOffset(2, i, n) / 3),
    angPos: readVec3(data, blockOffset(3, i, n) / 3),
    angVel: readVec3(data, blockOffset(4, i, n) / 3),
  };
}

export function writeBodyState(data: { [index: number]: number; length: number }, i: number, n: number, b: BodyState): void {
  writeVec3(data, blockOffset(0, i, n) / 3, b.pos);
  writeVec3(data, blockOffset(1, i, n) / 3, b.vel);
  writeVec3(data, blockOffset(2, i, n) / 3, b.accel);
  writeVec3(data, blockOffset(3, i, n) / 3, b.angPos);
  writeVec3(data, blockOffset(4, i, n) / 3, b.angVel);
}

/** Slice one block ([x,y,z]×N) out of a body field — the `bodyTap` extraction. */
export function tapBlock(data: ArrayLike<number>, n: number, block: BodyBlock): Float32Array {
  const b = BODY_BLOCKS.indexOf(block);
  const out = new Float32Array(n * 3);
  const base = b * n * 3;
  for (let k = 0; k < n * 3; k++) {
    const v = data[base + k];
    if (v === undefined) throw new RangeError(`tapBlock(${block}): index ${base + k} out of range for length ${data.length}`);
    out[k] = v;
  }
  return out;
}

/** A reproducible swarm: agents on a jittered spherical shell with a small tangential
 *  kick and tiny random spin. Returns a body field (length 5·N·3). Deterministic. */
export function seedSwarmBody(n: number, seed: number, shell = 4.5): Float32Array {
  const data = new Float32Array(n * 3 * BODY_BLOCK_COUNT);
  let a = seed >>> 0;
  const rnd = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    const u = rnd() * 2 - 1;
    const phi = rnd() * Math.PI * 2;
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    const r = shell * (0.7 + 0.6 * rnd());
    const x = r * s * Math.cos(phi);
    const y = r * u;
    const z = r * s * Math.sin(phi);
    const k = 0.06;
    writeBodyState(data, i, n, {
      pos: [x, y, z],
      vel: [-z * k + (rnd() - 0.5) * 0.05, (rnd() - 0.5) * 0.05, x * k + (rnd() - 0.5) * 0.05],
      accel: ZERO3,
      angPos: [(rnd() - 0.5) * 0.2, (rnd() - 0.5) * 0.2, (rnd() - 0.5) * 0.2],
      angVel: ZERO3,
    });
  }
  return data;
}

export interface IntegrateParams {
  dt: number;
  /** DANCERL's global timescale (~0.2): smaller ⇒ slower, more deliberate. */
  timeFactor: number;
  /** Max change in acceleration per step (the C² jerk limit). */
  jerkLimit: number;
  /** Linear velocity retained per step (DANCERL LinDamp). */
  linDamp: number;
  /** Angular velocity retained per step (DANCERL AngDamp). */
  angDamp: number;
  speedLimit: number;
  maxRadius: number;
  /** How strongly a dancer turns to face its motion (0 = no facing). */
  face: number;
}

export const INTEGRATE_DEFAULTS: IntegrateParams = {
  dt: 1,
  timeFactor: 0.2,
  jerkLimit: 0.05,
  linDamp: 0.96,
  angDamp: 0.9,
  speedLimit: 1.2,
  maxRadius: 40,
  face: 0.3,
};

/** Move `cur` toward `target` by at most `maxStep` (the jerk limiter). */
function approach(cur: Vec3, target: Vec3, maxStep: number): Vec3 {
  const d = sub(target, cur);
  const dist = length(d);
  if (dist <= maxStep || dist < 1e-9) return target;
  return addScaled(cur, d, maxStep / dist);
}

/** One rigid-body step for an agent given the summed `force` (target acceleration) and an
 *  optional `torque`. Pure; returns the next state. */
export function integrateBody(b: BodyState, force: Vec3, torque: Vec3, p: IntegrateParams): BodyState {
  const tf = p.dt * p.timeFactor;

  // Linear: jerk-limited approach to the target accel, then accel → vel → pos.
  const accel = approach(b.accel, force, p.jerkLimit * p.dt);
  let vel = scale(addScaled(b.vel, accel, tf), p.linDamp);
  vel = clampLength(vel, p.speedLimit);
  let pos = addScaled(b.pos, vel, tf);
  const r = length(pos);
  if (r > p.maxRadius && r > 1e-9) {
    pos = scale(pos, p.maxRadius / r);
    vel = scale(vel, 0.5);
  }

  // Angular: an optional torque that turns the dancer to face its travel, plus any
  // external torque; then angVel → angPos (angle-axis, DANCERL-style).
  let tq = torque;
  if (p.face > 0) {
    const speed = length(vel);
    if (speed > 1e-3) {
      const fwd = forward(fromAxisAngle(normalize(b.angPos, 1e-9, [0, 0, 1]), length(b.angPos)));
      tq = addScaled(tq, cross(fwd, scale(vel, 1 / speed)), p.face);
    }
  }
  const angVel = scale(addScaled(b.angVel, tq, tf), p.angDamp);
  const angPos = addScaled(b.angPos, angVel, tf);

  return { pos, vel, accel, angPos, angVel };
}
