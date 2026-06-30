// Vec3 — small, pure 3-vector math on immutable tuples, so the force kernels read
// like the equations in DANCERL rather than a soup of indexed array writes. Tuple
// elements are statically known to be present (a fixed-length tuple), so this module
// needs no non-null assertions; the only place undefined can sneak in is reading from a
// flat buffer, which `readVec3` guards explicitly (the repo prefers checked access over
// `!`).

export type Vec3 = readonly [number, number, number];

export const ZERO3: Vec3 = [0, 0, 0];

export const vec3 = (x: number, y: number, z: number): Vec3 => [x, y, z];

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const neg = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];

/** a + b·s — the fused "accumulate a scaled vector" used all over the integrators. */
export const addScaled = (a: Vec3, b: Vec3, s: number): Vec3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];

export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const lengthSq = (a: Vec3): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

/** Unit vector, or `fallback` (default zero) when `a` is shorter than `eps` — so a
 *  degenerate direction never produces NaN. */
export function normalize(a: Vec3, eps = 1e-9, fallback: Vec3 = ZERO3): Vec3 {
  const l = length(a);
  return l < eps ? fallback : [a[0] / l, a[1] / l, a[2] / l];
}

export const lerp = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/** Clamp a vector's magnitude to `max` (leaves shorter vectors untouched). */
export function clampLength(a: Vec3, max: number): Vec3 {
  const l = length(a);
  return l > max && l > 1e-12 ? scale(a, max / l) : a;
}

export const isFiniteVec3 = (a: Vec3): boolean => Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]);

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
  return [x, y, z];
}

/** Write agent `i`'s vec3 into a flat interleaved buffer. Bounds-checked. */
export function writeVec3(buf: { [index: number]: number; length: number }, i: number, v: Vec3): void {
  const o = i * 3;
  if (o < 0 || o + 2 >= buf.length) {
    throw new RangeError(`writeVec3: agent ${i} (offset ${o}) out of range for buffer length ${buf.length}`);
  }
  buf[o] = v[0];
  buf[o + 1] = v[1];
  buf[o + 2] = v[2];
}
