// Geometry of a Multiscale: how a `(level, x, y, z)` chunk maps to level-L voxel
// extents, to a level-0 array box, and to a world AABB — plus the scalar world
// voxel size that the Nyquist selection compares against the projected pixel pitch.

import { type Aabb, length, type Vec3, worldAabbOfArrayBox } from "./math";
import { bytesPerSample, type Multiscale } from "./types";

const ceilDiv = (a: number, b: number): number => Math.ceil(a / b);

/** Voxel dimensions of level L (`ceil(dims0 / 2^L)`, ≥ 1). */
export function levelVoxelDims(ms: Multiscale, level: number): [number, number, number] {
  const f = 2 ** level;
  return [Math.max(1, ceilDiv(ms.voxelDims0[0], f)), Math.max(1, ceilDiv(ms.voxelDims0[1], f)), Math.max(1, ceilDiv(ms.voxelDims0[2], f))];
}

/** Number of chunks along each axis at level L. */
export function chunkCounts(ms: Multiscale, level: number): [number, number, number] {
  const v = levelVoxelDims(ms, level);
  return [ceilDiv(v[0], ms.chunkShape[0]), ceilDiv(v[1], ms.chunkShape[1]), ceilDiv(v[2], ms.chunkShape[2])];
}

/** Actual voxel extent of one chunk (clamped at the level's border). */
export function chunkVoxelExtent(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): [number, number, number] {
  const v = levelVoxelDims(ms, id.level);
  const c = ms.chunkShape;
  const e = (axisVoxels: number, cs: number, idx: number): number => Math.max(0, Math.min(cs, axisVoxels - idx * cs));
  return [e(v[0], c[0], id.x), e(v[1], c[1], id.y), e(v[2], c[2], id.z)];
}

/** The chunk's box in **level-0 array units** (one level-L voxel = `2^L` level-0
 *  voxels), clamped to the full-resolution extent. Returns `[min, max]`. */
export function chunkArrayBox(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): [Vec3, Vec3] {
  const f = 2 ** id.level;
  const c = ms.chunkShape;
  const d0 = ms.voxelDims0;
  const lo: Vec3 = [id.x * c[0] * f, id.y * c[1] * f, id.z * c[2] * f];
  const hi: Vec3 = [Math.min(d0[0], (id.x + 1) * c[0] * f), Math.min(d0[1], (id.y + 1) * c[1] * f), Math.min(d0[2], (id.z + 1) * c[2] * f)];
  return [lo, hi];
}

/** The world AABB of a chunk. */
export function chunkWorldAabb(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): Aabb {
  const [lo, hi] = chunkArrayBox(ms, id);
  return worldAabbOfArrayBox(ms.worldFromArray, lo, hi);
}

/** World size of a level-0 voxel — the largest axis spacing (conservative; the
 *  coarsest-sampled axis governs, so the picked level never under-resolves it).
 *  Anisotropic per-axis LOD is deferred (ADR-0008 §4). */
export function worldVoxelSize0(ms: Multiscale): number {
  const a = ms.worldFromArray.axes;
  return Math.max(length(a[0]), length(a[1]), length(a[2]));
}

/** World sample spacing at level L. */
export function worldVoxelSize(ms: Multiscale, level: number): number {
  return worldVoxelSize0(ms) * 2 ** level;
}

/** Decoded byte size of a chunk (its actual voxel extent × sample size). */
export function chunkApproxBytes(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): number {
  const e = chunkVoxelExtent(ms, id);
  return e[0] * e[1] * e[2] * bytesPerSample(ms.dtype, ms.element);
}
