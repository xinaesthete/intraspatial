// Minimal, self-contained 3-D geometry for the datasource core: vectors, an
// axis-aligned box, an affine placement of array space into world, and a
// perspective camera reduced to the two things `Select` actually needs —
// six inward frustum planes (for culling) and enough basis to compute the
// projected world-per-pixel at a depth.
//
// Deliberately local (mirrors src/gpu/sim/vec3.ts in style) rather than importing
// it, so the datasource module carries no dependency on the sim/dancer code and
// can graduate cleanly (ADR-0008 §layering). No non-null assertions: fixed-length
// tuples are statically complete, and the only buffer reads are bounds-checked.

export type Vec3 = readonly [number, number, number];

export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);

export function normalize(a: Vec3, eps = 1e-12): Vec3 {
  const l = length(a);
  return l < eps ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l];
}

/** An axis-aligned bounding box in world space. */
export interface Aabb {
  readonly min: Vec3;
  readonly max: Vec3;
}

/** The point of `box` closest to `p` (clamp per axis). Equals `p` when inside. */
export function closestPointOnAabb(box: Aabb, p: Vec3): Vec3 {
  const c = (lo: number, hi: number, x: number): number => (x < lo ? lo : x > hi ? hi : x);
  return [c(box.min[0], box.max[0], p[0]), c(box.min[1], box.max[1], p[1]), c(box.min[2], box.max[2], p[2])];
}

/**
 * An affine placement of *array space* into world. Array coordinates are in
 * **level-0 voxel units**; one voxel step along array axis i moves `axes[i]` in
 * world. So `|axes[i]|` is the world size of a level-0 voxel along axis i, and a
 * general (rotated/anisotropic) `axes` lets a plane sit obliquely in 3-D.
 *
 *   world = origin + a0·axes[0] + a1·axes[1] + a2·axes[2]
 */
export interface Affine3 {
  readonly origin: Vec3;
  readonly axes: readonly [Vec3, Vec3, Vec3];
}

/** Map an array-space position (level-0 voxel units) to world. */
export function applyAffine(a: Affine3, p: Vec3): Vec3 {
  return add(a.origin, add(scale(a.axes[0], p[0]), add(scale(a.axes[1], p[1]), scale(a.axes[2], p[2]))));
}

/** The world AABB enclosing an array-space box (its 8 transformed corners). For a
 *  rotated placement this is a loose bound — fine for culling and nearest-point. */
export function worldAabbOfArrayBox(a: Affine3, min: Vec3, max: Vec3): Aabb {
  let lo: Vec3 = [Infinity, Infinity, Infinity];
  let hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i++) {
    const corner: Vec3 = [i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]];
    const w = applyAffine(a, corner);
    lo = [Math.min(lo[0], w[0]), Math.min(lo[1], w[1]), Math.min(lo[2], w[2])];
    hi = [Math.max(hi[0], w[0]), Math.max(hi[1], w[1]), Math.max(hi[2], w[2])];
  }
  return { min: lo, max: hi };
}

/** A perspective camera. `fovY` is the vertical field of view in radians; the
 *  viewport height in pixels drives the projected pixel pitch. */
export interface Camera {
  readonly eye: Vec3;
  /** View direction (need not be unit; normalised internally). */
  readonly forward: Vec3;
  /** Approximate up (re-orthogonalised against `forward`). */
  readonly up: Vec3;
  readonly fovY: number;
  readonly aspect: number;
  readonly near: number;
  readonly far: number;
  readonly viewportHeightPx: number;
}

/** A plane `n·x + d ≥ 0` for points on the inside of the frustum. */
export interface Plane {
  readonly n: Vec3;
  readonly d: number;
}

/** The orthonormal camera basis (forward, right, up). */
export function cameraBasis(cam: Camera): { fwd: Vec3; right: Vec3; up: Vec3 } {
  const fwd = normalize(cam.forward);
  const right = normalize(cross(fwd, cam.up));
  const up = cross(right, fwd); // already unit: right ⟂ fwd, both unit
  return { fwd, right, up };
}

/** The six inward-facing frustum planes (near, far, and four sides through the eye). */
export function frustumPlanes(cam: Camera): Plane[] {
  const { fwd, right, up } = cameraBasis(cam);
  const tanV = Math.tan(cam.fovY / 2);
  const tanH = tanV * cam.aspect;

  // Side planes pass through the eye; orient each so the view centre is inside.
  const sidePlane = (edgeDir: Vec3, inPlane: Vec3): Plane => {
    let n = normalize(cross(edgeDir, inPlane));
    if (dot(n, fwd) < 0) n = scale(n, -1); // inward: forward must be on the inside
    return { n, d: -dot(n, cam.eye) };
  };
  const dL = sub(fwd, scale(right, tanH));
  const dR = add(fwd, scale(right, tanH));
  const dB = sub(fwd, scale(up, tanV));
  const dT = add(fwd, scale(up, tanV));

  const nearPt = add(cam.eye, scale(fwd, cam.near));
  const farPt = add(cam.eye, scale(fwd, cam.far));
  return [
    { n: fwd, d: -dot(fwd, nearPt) }, // near
    { n: scale(fwd, -1), d: -dot(scale(fwd, -1), farPt) }, // far
    sidePlane(dL, up), // left
    sidePlane(dR, up), // right
    sidePlane(dB, right), // bottom
    sidePlane(dT, right), // top
  ];
}

/** True if `box` is entirely outside any single frustum plane (safe to cull). Uses
 *  the box's positive vertex per plane — a conservative, standard AABB test. */
export function aabbOutsideFrustum(box: Aabb, planes: readonly Plane[]): boolean {
  for (const pl of planes) {
    const pv: Vec3 = [
      pl.n[0] >= 0 ? box.max[0] : box.min[0],
      pl.n[1] >= 0 ? box.max[1] : box.min[1],
      pl.n[2] >= 0 ? box.max[2] : box.min[2],
    ];
    if (dot(pl.n, pv) + pl.d < 0) return true; // even the farthest-inward corner is outside
  }
  return false;
}

/** World distance one screen pixel spans at optical-axis depth `depth`. */
export function worldPerPixel(cam: Camera, depth: number): number {
  return (2 * depth * Math.tan(cam.fovY / 2)) / cam.viewportHeightPx;
}

/** The 8 world-space corners of `cam`'s frustum between `nearDist` and `farDist`
 *  (for drawing it as an Overlay). Order: near BL, BR, TR, TL, then far BL, BR, TR, TL. */
export function frustumCorners(cam: Camera, nearDist: number, farDist: number): Vec3[] {
  const { fwd, right, up } = cameraBasis(cam);
  const tanV = Math.tan(cam.fovY / 2);
  const at = (dist: number, sv: number, sh: number): Vec3 => {
    const h = tanV * dist,
      w = h * cam.aspect;
    return add(add(cam.eye, scale(fwd, dist)), add(scale(up, h * sv), scale(right, w * sh)));
  };
  return [
    at(nearDist, -1, -1),
    at(nearDist, -1, 1),
    at(nearDist, 1, 1),
    at(nearDist, 1, -1),
    at(farDist, -1, -1),
    at(farDist, -1, 1),
    at(farDist, 1, 1),
    at(farDist, 1, -1),
  ];
}
