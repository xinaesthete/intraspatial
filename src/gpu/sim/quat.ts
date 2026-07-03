// Quaternions — just enough for rigid-body orientation. DANCERL carried AngPos/AngVel
// and converted via Mat2Quat; we keep orientation as a unit quaternion and integrate it
// from angular velocity, which is the same idea with less matrix machinery (the script's
// "should be tensor" inertia is left isotropic for now). Pure tuple math, no `!`.
import type { Vec3 } from "./vec3";

/** (w, x, y, z) with w the scalar part. */
export type Quat = readonly [number, number, number, number];

export const IDENTITY: Quat = [1, 0, 0, 0];

export function normalizeQuat(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return l < 1e-12 ? IDENTITY : [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/** Hamilton product a⊗b (apply b then a). */
export function mulQuat(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/** Unit quaternion for a rotation of `angle` radians about `axis`. */
export function fromAxisAngle(axis: Vec3, angle: number): Quat {
  const l = Math.hypot(axis[0], axis[1], axis[2]);
  if (l < 1e-12) return IDENTITY;
  const h = angle / 2;
  const s = Math.sin(h) / l;
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** Rotate a vector by a (unit) quaternion: v' = q v q⁻¹, via the standard
 *  t = 2·(qxyz × v); v' = v + w·t + qxyz × t. */
export function rotateVec3(q: Quat, v: Vec3): Vec3 {
  const qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ];
}

/** Integrate orientation `q` by angular velocity `w` (rad/s) over `dt`, exactly:
 *  advance by the rotation whose axis is `w` and angle `|w|·dt`. Renormalised. */
export function integrateQuat(q: Quat, w: Vec3, dt: number): Quat {
  const speed = Math.hypot(w[0], w[1], w[2]);
  if (speed < 1e-12) return q;
  const dq = fromAxisAngle(w, speed * dt);
  return normalizeQuat(mulQuat(dq, q));
}

/** The forward (local +z) axis of an orientation — useful for "facing" and rendering. */
export const forward = (q: Quat): Vec3 => rotateVec3(q, [0, 0, 1]);

export function readQuat(buf: ArrayLike<number>, i: number): Quat {
  const o = i * 4;
  const w = buf[o];
  const x = buf[o + 1];
  const y = buf[o + 2];
  const z = buf[o + 3];
  if (w === undefined || x === undefined || y === undefined || z === undefined) {
    throw new RangeError(`readQuat: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  return [w, x, y, z];
}

export function writeQuat(buf: { [index: number]: number; length: number }, i: number, q: Quat): void {
  const o = i * 4;
  if (o < 0 || o + 3 >= buf.length) {
    throw new RangeError(`writeQuat: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  buf[o] = q[0];
  buf[o + 1] = q[1];
  buf[o + 2] = q[2];
  buf[o + 3] = q[3];
}
