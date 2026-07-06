// The kinematic state of a dancer.
//
// A trajectory's state is carried as a **2-jet** — position, velocity AND acceleration
// (`Jet3`). Carrying `accel` is what lets goals be reached *smoothly*: the caller's
// goal-splines (DANCERL `PosVelSpline`, which threads the previous acceleration `LastAcc`)
// produce C²-continuous approaches, so a dancer "scrambles to a state of motion" without a
// jerk — never snapping to a keyframe. (A k-jet is the truncated Taylor series of the
// trajectory; this is the k=2 case.)
//
// A `Body` adds rigid-body orientation (DANCERL's AngPos/AngVel), so couples can *swing*
// and dancers can *face* their motion. Inertia is left isotropic (scalar) for now — the
// script's "should be tensor" is a noted future deepening.
//
// Kept as plain high-level TS types over Struct-of-Arrays Float32 buffers (the layout the
// op-graph carries as parallel `points` fields and the GPU artefact carries as storage
// buffers). Could be promoted to a runtime ElementType later.

import { IDENTITY, type Quat, readQuat, writeQuat } from "./quat";
import { readVec3, type Vec3, writeVec3, ZERO3 } from "./vec3";

export interface Jet3 {
  pos: Vec3;
  vel: Vec3;
  accel: Vec3;
}

export interface Body {
  pos: Vec3;
  vel: Vec3;
  accel: Vec3;
  orient: Quat;
  angVel: Vec3;
}

export function neutralBody(pos: Vec3 = ZERO3): Body {
  return { pos, vel: ZERO3, accel: ZERO3, orient: IDENTITY, angVel: ZERO3 };
}

/** Struct-of-arrays storage for a swarm of `n` bodies — the same field layout the op
 *  graph carries (parallel vec3/quat `points` fields) and the GPU artefact mirrors as
 *  storage buffers. */
export interface SwarmBuffers {
  n: number;
  pos: Float32Array; // 3n
  vel: Float32Array; // 3n
  accel: Float32Array; // 3n
  orient: Float32Array; // 4n (x,y,z,w) — wgpu-matrix's quat layout
  angVel: Float32Array; // 3n
}

export function makeSwarmBuffers(n: number): SwarmBuffers {
  const buffers: SwarmBuffers = {
    n,
    pos: new Float32Array(n * 3),
    vel: new Float32Array(n * 3),
    accel: new Float32Array(n * 3),
    orient: new Float32Array(n * 4),
    angVel: new Float32Array(n * 3),
  };
  for (let i = 0; i < n; i++) writeQuat(buffers.orient, i, IDENTITY);
  return buffers;
}

export function readBody(s: SwarmBuffers, i: number): Body {
  return {
    pos: readVec3(s.pos, i),
    vel: readVec3(s.vel, i),
    accel: readVec3(s.accel, i),
    orient: readQuat(s.orient, i),
    angVel: readVec3(s.angVel, i),
  };
}

export function writeBody(s: SwarmBuffers, i: number, b: Body): void {
  writeVec3(s.pos, i, b.pos);
  writeVec3(s.vel, i, b.vel);
  writeVec3(s.accel, i, b.accel);
  writeQuat(s.orient, i, b.orient);
  writeVec3(s.angVel, i, b.angVel);
}

/** True when every component of every buffer is finite — a cheap sanity gate. */
export function swarmFinite(s: SwarmBuffers): boolean {
  for (const arr of [s.pos, s.vel, s.accel, s.orient, s.angVel]) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v === undefined || !Number.isFinite(v)) return false;
    }
  }
  return true;
}
