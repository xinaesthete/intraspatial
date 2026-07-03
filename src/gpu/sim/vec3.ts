// Vec3 — a thin project-glue layer over `wgpu-matrix`'s `vec3`, so the force kernels still
// read like the equations in DANCERL rather than a soup of `dst`-threading. `Vec3`
// (Float32Array) is the canonical, stored representation: it matches wgpu-matrix's own
// default output type and WGSL's f32 semantics, which matters because
// dancerGpu.gpu.test.ts diffs this CPU golden against real f32 GPU output. `Vec3In` is what
// pure functions accept as *input* (mirrors wgpu-matrix's own `Vec3Arg`) so call sites can
// still pass a plain `[x, y, z]` literal without an extra allocation.
//
// A Float32Array's length isn't known to the type-checker the way a 3-tuple's was, so under
// this repo's `noUncheckedIndexedAccess` every `v[0]` is `number | undefined` now. `unpack`
// is the one guarded place that turns "three components" into three checked numbers (throws
// on the invariant violation instead of reaching for `!`); everything that needs raw
// component access goes through it once, up front.
import { vec3 as V3 } from "wgpu-matrix";

export type Vec3 = Float32Array;
export type Vec3In = Vec3 | number[];

export const ZERO3: Vec3 = V3.create(0, 0, 0);

export const vec3 = (x: number, y: number, z: number): Vec3 => V3.create(x, y, z);

export const add = (a: Vec3In, b: Vec3In): Vec3 => V3.add(a, b);
export const sub = (a: Vec3In, b: Vec3In): Vec3 => V3.subtract(a, b);
export const neg = (a: Vec3In): Vec3 => V3.negate(a);
export const scale = (a: Vec3In, s: number): Vec3 => V3.scale(a, s);

/** a + b·s — the fused "accumulate a scaled vector" used all over the integrators. */
export const addScaled = (a: Vec3In, b: Vec3In, s: number): Vec3 => V3.addScaled(a, b, s);

export const dot = (a: Vec3In, b: Vec3In): number => V3.dot(a, b);
export const cross = (a: Vec3In, b: Vec3In): Vec3 => V3.cross(a, b);

export const length = (a: Vec3In): number => V3.length(a);
export const lengthSq = (a: Vec3In): number => V3.lengthSq(a);

/** Unit vector, or `fallback` (default zero) when `a` is shorter than `eps` — so a
 *  degenerate direction never produces NaN. `vec3.normalize` only special-cases an exact
 *  zero length; this eps guard also catches the near-zero directions finite-precision
 *  force sums produce. */
export function normalize(a: Vec3In, eps = 1e-9, fallback: Vec3In = ZERO3): Vec3 {
  return V3.length(a) < eps ? V3.clone(fallback) : V3.normalize(a);
}

export const lerp = (a: Vec3In, b: Vec3In, t: number): Vec3 => V3.lerp(a, b, t);

/** Clamp a vector's magnitude to `max` (leaves shorter vectors untouched). */
export const clampLength = (a: Vec3In, max: number): Vec3 => V3.truncate(a, max);

/** Destructure a Vec3 into three checked numbers. `Vec3`/`Vec3In` are always exactly 3
 *  long at runtime, but a `Float32Array`'s length isn't part of its type, so this is the
 *  one guarded place that turns indexed access into plain `number`s (throws rather than
 *  letting `undefined` silently reach the math). */
export function unpack(v: Vec3In): readonly [number, number, number] {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  if (x === undefined || y === undefined || z === undefined) {
    throw new RangeError("unpack: expected a 3-component vector");
  }
  return [x, y, z];
}

export const isFiniteVec3 = (a: Vec3In): boolean =>
  Number.isFinite(a[0] ?? Number.NaN) && Number.isFinite(a[1] ?? Number.NaN) && Number.isFinite(a[2] ?? Number.NaN);

/** Read agent `i`'s vec3 from a flat interleaved buffer ([x0,y0,z0,x1,…]). Bounds- and
 *  hole-checked: the `=== undefined` guard both reports out-of-range access and narrows
 *  `number | undefined` to `number` for the type-checker — no `!`. */
export function readVec3(buf: ArrayLike<number>, i: number): Vec3 {
  const o = i * 3;
  const x = buf[o];
  const y = buf[o + 1];
  const z = buf[o + 2];
  if (x === undefined || y === undefined || z === undefined) {
    throw new RangeError(`readVec3: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  return V3.create(x, y, z);
}

/** Write agent `i`'s vec3 into a flat interleaved buffer. Bounds-checked. */
export function writeVec3(buf: { [index: number]: number; length: number }, i: number, v: Vec3In): void {
  const o = i * 3;
  if (o < 0 || o + 2 >= buf.length) {
    throw new RangeError(`writeVec3: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  const [x, y, z] = unpack(v);
  buf[o] = x;
  buf[o + 1] = y;
  buf[o + 2] = z;
}
