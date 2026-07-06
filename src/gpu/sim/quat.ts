// Quaternions — just enough for rigid-body orientation, over `wgpu-matrix`'s `quat`.
// DANCERL carried AngPos/AngVel and converted via Mat2Quat; we keep orientation as a unit
// quaternion and integrate it from angular velocity, which is the same idea with less
// matrix machinery (the script's "should be tensor" inertia is left isotropic for now).
//
// wgpu-matrix stores a quaternion as (x, y, z, w) — w LAST, not first like the original
// hand-rolled version. Nothing outside this module and `jet.ts` (which has no other
// consumers — see its own comment) depended on the old (w, x, y, z) wire order, so this
// migration adopts the library's layout outright rather than translating at the boundary.
import { quat as Q, vec3 as V3 } from "wgpu-matrix";
import { unpack, type Vec3In } from "./vec3";

export type Quat = Float32Array;
export type QuatIn = Quat | number[];

/** (x, y, z, w) with w the scalar part. */
export const IDENTITY: Quat = Q.identity();

export const normalizeQuat = (q: QuatIn): Quat => Q.normalize(q);

/** Hamilton product a⊗b (apply b then a). */
export const mulQuat = (a: QuatIn, b: QuatIn): Quat => Q.multiply(a, b);

/** Unit quaternion for a rotation of `angle` radians about `axis`. `quat.fromAxisAngle`
 *  assumes a pre-normalized, non-zero axis; DANCERL's angle-axis state doesn't guarantee
 *  either (see `integrateQuat`), so this keeps the original normalize-and-guard wrapper. */
export function fromAxisAngle(axis: Vec3In, angle: number): Quat {
  const l = V3.length(axis);
  if (l < 1e-12) return Q.clone(IDENTITY);
  return Q.fromAxisAngle(V3.scale(axis, 1 / l), angle);
}

/** Rotate a vector by a (unit) quaternion: v' = q v q⁻¹. */
export const rotateVec3 = (q: QuatIn, v: Vec3In) => V3.transformQuat(v, q);

/** Integrate orientation `q` by angular velocity `w` (rad/s) over `dt`, exactly:
 *  advance by the rotation whose axis is `w` and angle `|w|·dt`. Renormalised. */
export function integrateQuat(q: QuatIn, w: Vec3In, dt: number): Quat {
  const speed = V3.length(w);
  if (speed < 1e-12) return Q.clone(q);
  const dq = fromAxisAngle(w, speed * dt);
  return Q.normalize(Q.multiply(dq, q));
}

/** The forward (local +z) axis of an orientation — useful for "facing" and rendering. */
export const forward = (q: QuatIn) => rotateVec3(q, V3.create(0, 0, 1));

/** Destructure a Quat into four checked numbers — see `vec3.ts`'s `unpack` for why. */
export function unpackQuat(q: QuatIn): readonly [number, number, number, number] {
  const x = q[0];
  const y = q[1];
  const z = q[2];
  const w = q[3];
  if (x === undefined || y === undefined || z === undefined || w === undefined) {
    throw new RangeError("unpackQuat: expected a 4-component quaternion");
  }
  return [x, y, z, w];
}

export function readQuat(buf: ArrayLike<number>, i: number): Quat {
  const o = i * 4;
  const x = buf[o];
  const y = buf[o + 1];
  const z = buf[o + 2];
  const w = buf[o + 3];
  if (x === undefined || y === undefined || z === undefined || w === undefined) {
    throw new RangeError(`readQuat: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  return Q.fromValues(x, y, z, w);
}

export function writeQuat(buf: { [index: number]: number; length: number }, i: number, q: QuatIn): void {
  const o = i * 4;
  if (o < 0 || o + 3 >= buf.length) {
    throw new RangeError(`writeQuat: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  const [x, y, z, w] = unpackQuat(q);
  buf[o] = x;
  buf[o + 1] = y;
  buf[o + 2] = z;
  buf[o + 3] = w;
}
