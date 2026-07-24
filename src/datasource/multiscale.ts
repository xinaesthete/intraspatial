// Geometry of a Multiscale: how a `(level, x, y, z)` chunk maps to level-L voxel
// extents, to a level-0 array box, and to a world AABB — plus the scalar world
// voxel size that the Nyquist selection compares against the projected pixel pitch.

import { type Aabb, length, type Vec3, worldAabbOfArrayBox } from "./math";
import { bytesPerSample, type Multiscale, worldFromArrayOf } from "./types";

const ceilDiv = (a: number, b: number): number => Math.ceil(a / b);

/** Voxel dimensions of level L. Uses the Resource's real per-level dims when known (`ms.levelDims`,
 *  e.g. a floor-halving OME-Zarr pyramid), else the `ceil(dims0 / 2^L)` idealisation. */
export function levelVoxelDims(ms: Multiscale, level: number): [number, number, number] {
  const real = ms.levelDims?.[level];
  if (real) return [Math.max(1, real[0]), Math.max(1, real[1]), Math.max(1, real[2])];
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

/** The chunk's box in **level-0 array units**, from the chunk's *fractional* coverage of its level
 *  scaled to the full extent — so every level maps to the same world box and cross-level tiles align
 *  exactly (rather than a level-L chunk being placed at `2^L·chunkShape`, which drifts from the real
 *  data when the pyramid isn't exactly halving). Returns `[min, max]`. */
export function chunkArrayBox(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): [Vec3, Vec3] {
  const [cx, cy, cz] = ms.chunkShape;
  const [d0x, d0y, d0z] = ms.voxelDims0;
  const [dLx, dLy, dLz] = levelVoxelDims(ms, id.level);
  // level-voxel span of this chunk (clamped to the level) → fraction of the level → level-0 units.
  const edge = (i: number, cw: number, dL: number, d0: number): [number, number] => [
    (Math.min(i * cw, dL) / dL) * d0,
    (Math.min((i + 1) * cw, dL) / dL) * d0,
  ];
  const [lox, hix] = edge(id.x, cx, dLx, d0x);
  const [loy, hiy] = edge(id.y, cy, dLy, d0y);
  const [loz, hiz] = edge(id.z, cz, dLz, d0z);
  return [
    [lox, loy, loz],
    [hix, hiy, hiz],
  ];
}

/** The world AABB of a chunk. */
export function chunkWorldAabb(ms: Multiscale, id: { level: number; x: number; y: number; z: number }): Aabb {
  const [lo, hi] = chunkArrayBox(ms, id);
  return worldAabbOfArrayBox(worldFromArrayOf(ms), lo, hi);
}

/** World size of a level-0 voxel — the largest axis spacing (conservative; the
 *  coarsest-sampled axis governs, so the picked level never under-resolves it).
 *  Anisotropic per-axis LOD is deferred (ADR-0008 §4). */
export function worldVoxelSize0(ms: Multiscale): number {
  const a = worldFromArrayOf(ms).axes;
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
