// `SyntheticLoader` — Milestone-1's dep-free `Loader` (ADR-0008): an in-memory,
// deterministic pyramid so the viewer + Select + Decision view work with no zarr,
// no OpenJPH, no network. It stays permanently as the no-network / golden-fixture
// path. A chunk is filled by sampling an analytic `ScalarField` at the requested
// level's voxel centres, so coarser levels are the same continuous signal sampled
// more sparsely — a valid (point-prefiltered) multiscale.

import type { Vec3 } from "./math";
import { chunkVoxelExtent, levelVoxelDims } from "./multiscale";
import type { ChunkId, Loader, Multiscale, Tile } from "./types";

export type ScalarField = (u: Vec3) => number;

/** Power-8 Mandelbulb escape fraction over the unit cube (0 = escaped fast, 1 = inside). */
export const mandelbulbField: ScalarField = (u) => {
  const cx = (u[0] * 2 - 1) * 1.2,
    cy = (u[1] * 2 - 1) * 1.2,
    cz = (u[2] * 2 - 1) * 1.2;
  const power = 8,
    maxIter = 8;
  let x = 0,
    y = 0,
    z = 0,
    i = 0;
  for (; i < maxIter; i++) {
    const r = Math.hypot(x, y, z);
    if (r > 2) break;
    const rr = r < 1e-9 ? 1e-9 : r;
    const theta = Math.acos(z / rr) * power;
    const phi = Math.atan2(y, x) * power;
    const zr = r ** power;
    x = cx + zr * Math.sin(theta) * Math.cos(phi);
    y = cy + zr * Math.sin(theta) * Math.sin(phi);
    z = cz + zr * Math.cos(theta);
  }
  return i / maxIter;
};

/** Mandelbrot escape fraction over the unit square (u[2] ignored) — the plane signal. */
export const mandelbrotField: ScalarField = (u) => {
  const cx = u[0] * 3 - 2.1,
    cy = u[1] * 2.4 - 1.2;
  const maxIter = 48;
  let x = 0,
    y = 0,
    i = 0;
  for (; i < maxIter; i++) {
    const x2 = x * x - y * y + cx;
    y = 2 * x * y + cy;
    x = x2;
    if (x * x + y * y > 4) break;
  }
  return i / maxIter;
};

/** An axis-aligned Multiscale: `voxelSizeWorld` per level-0 voxel, centred (by
 *  default) on the world origin. `element` is scalar (the synthetic field is scalar). */
export function axisAlignedMultiscale(opts: {
  voxelDims0: readonly [number, number, number];
  chunkShape: readonly [number, number, number];
  levelCount: number;
  voxelSizeWorld?: number;
  origin?: Vec3;
}): Multiscale {
  const s = opts.voxelSizeWorld ?? 1;
  const d = opts.voxelDims0;
  const origin: Vec3 = opts.origin ?? [(-d[0] * s) / 2, (-d[1] * s) / 2, (-d[2] * s) / 2];
  return {
    voxelDims0: d,
    chunkShape: opts.chunkShape,
    levelCount: opts.levelCount,
    placements: [
      {
        system: "global",
        worldFromArray: {
          origin,
          axes: [
            [s, 0, 0],
            [0, s, 0],
            [0, 0, s],
          ],
        },
      },
    ],
    element: { kind: "scalar" },
    dtype: "f32",
  };
}

/** A `Loader` that materialises chunks of `ms` by sampling `field`. */
export function syntheticLoader(ms: Multiscale, field: ScalarField): Loader {
  if (ms.element.kind !== "scalar") throw new Error("syntheticLoader: only scalar elements are supported");
  return {
    async getChunk(id: ChunkId): Promise<Tile> {
      const extent = chunkVoxelExtent(ms, id);
      const dims = levelVoxelDims(ms, id.level);
      const [ex, ey, ez] = extent;
      const data = new Float32Array(ex * ey * ez);
      const base: Vec3 = [id.x * ms.chunkShape[0], id.y * ms.chunkShape[1], id.z * ms.chunkShape[2]];
      let o = 0;
      for (let k = 0; k < ez; k++) {
        const w = (base[2] + k + 0.5) / dims[2];
        for (let j = 0; j < ey; j++) {
          const v = (base[1] + j + 0.5) / dims[1];
          for (let i = 0; i < ex; i++) {
            const uu = (base[0] + i + 0.5) / dims[0];
            data[o++] = field([uu, v, w]);
          }
        }
      }
      return { id, dims: [ex, ey, ez], element: ms.element, dtype: ms.dtype, data };
    },
  };
}

export interface SyntheticSource {
  readonly ms: Multiscale;
  readonly loader: Loader;
}

/** A synthetic Mandelbulb **volume** source. */
export function syntheticVolume(opts?: { size?: number; chunk?: number; levelCount?: number; voxelSizeWorld?: number }): SyntheticSource {
  const size = opts?.size ?? 256;
  const chunk = opts?.chunk ?? 32;
  const ms = axisAlignedMultiscale({
    voxelDims0: [size, size, size],
    chunkShape: [chunk, chunk, chunk],
    levelCount: opts?.levelCount ?? 4,
    voxelSizeWorld: opts?.voxelSizeWorld ?? 1,
  });
  return { ms, loader: syntheticLoader(ms, mandelbulbField) };
}

/** A synthetic Mandelbrot **plane** source (a 2-D image, `D === 1`, placed in 3-D). */
export function syntheticPlane(opts?: {
  width?: number;
  height?: number;
  chunk?: number;
  levelCount?: number;
  voxelSizeWorld?: number;
}): SyntheticSource {
  const w = opts?.width ?? 1024;
  const h = opts?.height ?? 1024;
  const chunk = opts?.chunk ?? 64;
  const ms = axisAlignedMultiscale({
    voxelDims0: [w, h, 1],
    chunkShape: [chunk, chunk, 1],
    levelCount: opts?.levelCount ?? 5,
    voxelSizeWorld: opts?.voxelSizeWorld ?? 1,
  });
  return { ms, loader: syntheticLoader(ms, mandelbrotField) };
}
