// Placements — the rigid(+uniform-scale) transforms a structural op instances a Geometry by
// (ADR-0010). A structural node (`stack`/`branch`) doesn't warp its child pointwise; it places N
// copies of it, each under one `Placement`. A whole structural *tree* over one leaf flattens to a
// flat list of Placements (each the product of transforms along its path), so lowering is: one
// base mesh + N placements — on the CPU a concatenation, on the GPU a single instanced draw.
//
// A Placement is a column-major 4×4. We restrict to **rotation + uniform scale + translation**, so
// a normal transforms by `normalize(mat3(M) · n)` (the uniform scale cancels under normalisation) —
// the same `M` serves positions and normals. This module is pure and shared by the CPU golden and
// the GPU instance buffer, so the two can't drift.

import type { Vec3 } from "./superellipsoid";

/** A column-major 4×4 transform (16 numbers). */
export type Mat4 = number[];

export const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function translate(x: number, y: number, z: number): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

/** Uniform scale by `s`. */
export function scaleUniform(s: number): Mat4 {
  return [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1];
}

export function rotX(a: number): Mat4 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}

export function rotZ(a: number): Mat4 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** Matrix product `a · b` (apply `b` first, then `a`). */
export function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Array<number>(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      o[col * 4 + row] =
        (a[row] ?? 0) * (b[col * 4] ?? 0) +
        (a[4 + row] ?? 0) * (b[col * 4 + 1] ?? 0) +
        (a[8 + row] ?? 0) * (b[col * 4 + 2] ?? 0) +
        (a[12 + row] ?? 0) * (b[col * 4 + 3] ?? 0);
    }
  }
  return o;
}

/** Compose a stack of transforms left-to-right: `compose([A, B, C])` = `A · B · C` (C applied
 *  first). Empty ⇒ identity. */
export function compose(ms: Mat4[]): Mat4 {
  let out = IDENTITY;
  for (const m of ms) out = mul(out, m);
  return out;
}

/** Transform a position by `m` (drops the homogeneous `w`). */
export function applyPoint(m: Mat4, p: Vec3): Vec3 {
  return [
    (m[0] ?? 0) * p[0] + (m[4] ?? 0) * p[1] + (m[8] ?? 0) * p[2] + (m[12] ?? 0),
    (m[1] ?? 0) * p[0] + (m[5] ?? 0) * p[1] + (m[9] ?? 0) * p[2] + (m[13] ?? 0),
    (m[2] ?? 0) * p[0] + (m[6] ?? 0) * p[1] + (m[10] ?? 0) * p[2] + (m[14] ?? 0),
  ];
}

/** Transform a normal by `m`'s rotation part, renormalised (valid for rotation + uniform scale). */
export function applyNormal(m: Mat4, n: Vec3): Vec3 {
  const x = (m[0] ?? 0) * n[0] + (m[4] ?? 0) * n[1] + (m[8] ?? 0) * n[2];
  const y = (m[1] ?? 0) * n[0] + (m[5] ?? 0) * n[1] + (m[9] ?? 0) * n[2];
  const z = (m[2] ?? 0) * n[0] + (m[6] ?? 0) * n[1] + (m[10] ?? 0) * n[2];
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}
