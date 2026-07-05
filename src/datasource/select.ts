// `Select` — the pure heuristic (ADR-0008 §4): given a Multiscale's metadata and a
// Camera, name the coarsest chunks that still meet Nyquist for the on-screen pixel
// pitch. No I/O. The intellectual core of "how it decides what to download".
//
// Method: descend the pyramid as an oct-tree from the coarsest level. At each chunk,
// frustum-cull, then compute the level its *nearest* point demands; if the current
// level is already coarse enough, emit it, otherwise descend to the eight finer
// children. This yields fine chunks near the camera and coarse ones far away — the
// receding-resolution gradient — from one formula.

import {
  aabbOutsideFrustum,
  type Camera,
  cameraBasis,
  closestPointOnAabb,
  dot,
  frustumPlanes,
  type Plane,
  sub,
  worldPerPixel,
} from "./math";
import { chunkApproxBytes, chunkCounts, chunkWorldAabb, worldVoxelSize0 } from "./multiscale";
import type { ChunkId, Multiscale, Result, SelectedChunk, Selection } from "./types";

export interface SelectOptions {
  /** Detail budget: `q = 1` ≈ one prefiltered sample per screen pixel; `q > 1`
   *  supersamples; `q < 1` trades sharpness for bandwidth. */
  q?: number;
  /** A global floor on level (never pick finer than this). Used by the degrade-to-fit
   *  budget policy; defaults to 0. */
  minLevel?: number;
}

/** Coarsest level whose world sample-spacing ≤ `worldPerPixel / q` at `depth`,
 *  clamped to `[minLevel, maxLevel]`. */
function levelForDepth(ms: Multiscale, cam: Camera, depth: number, q: number, minLevel: number, maxLevel: number): number {
  const threshold = worldPerPixel(cam, depth) / q; // world units we may leave between samples
  const ratio = threshold / worldVoxelSize0(ms); // = 2^L at the boundary
  const l = ratio <= 0 ? 0 : Math.floor(Math.log2(ratio));
  return Math.max(minLevel, Math.min(maxLevel, l));
}

/** Optical-axis depth of a chunk's nearest point (clamped to the near plane). */
function nearestDepth(ms: Multiscale, cam: Camera, fwd: Camera["forward"], id: ChunkId): number {
  const box = chunkWorldAabb(ms, id);
  const p = closestPointOnAabb(box, cam.eye);
  return Math.max(cam.near, dot(sub(p, cam.eye), fwd));
}

/**
 * Select the chunks the Camera needs. Pure. `q` is the detail budget; `minLevel`
 * floors the resolution (used by `selectWithinBudget`).
 */
export function select(ms: Multiscale, cam: Camera, opts: SelectOptions = {}): Selection {
  const q = opts.q ?? 1;
  const minLevel = Math.max(0, opts.minLevel ?? 0);
  const maxLevel = ms.levelCount - 1;
  const planes = frustumPlanes(cam);
  const { fwd } = cameraBasis(cam);

  const out: SelectedChunk[] = [];
  const countByLevel: number[] = new Array(ms.levelCount).fill(0);

  const emit = (id: ChunkId, depth: number): void => {
    out.push({ id, nearestDepth: depth, approxBytes: chunkApproxBytes(ms, id) });
    const c = countByLevel[id.level];
    countByLevel[id.level] = (c ?? 0) + 1;
  };

  const visit = (id: ChunkId): void => {
    const box = chunkWorldAabb(ms, id);
    if (aabbOutsideFrustum(box, planes as Plane[])) return; // frustum cull

    const depth = nearestDepth(ms, cam, fwd, id);
    const desired = levelForDepth(ms, cam, depth, q, minLevel, maxLevel);

    if (desired >= id.level || id.level === minLevel) {
      emit(id, depth); // this level is coarse enough (or we've hit the floor)
      return;
    }
    // Need finer: descend into the (up to eight) children at the next level down.
    const child = id.level - 1;
    const counts = chunkCounts(ms, child);
    for (let dz = 0; dz < 2; dz++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const cx = id.x * 2 + dx,
            cy = id.y * 2 + dy,
            cz = id.z * 2 + dz;
          if (cx < counts[0] && cy < counts[1] && cz < counts[2]) visit({ level: child, x: cx, y: cy, z: cz });
        }
      }
    }
  };

  const rootLevel = maxLevel;
  const rootCounts = chunkCounts(ms, rootLevel);
  for (let z = 0; z < rootCounts[2]; z++) {
    for (let y = 0; y < rootCounts[1]; y++) {
      for (let x = 0; x < rootCounts[0]; x++) {
        visit({ level: rootLevel, x, y, z });
      }
    }
  }

  const totalApproxBytes = out.reduce((s, c) => s + c.approxBytes, 0);
  return { chunks: out, totalApproxBytes, countByLevel };
}

/**
 * Degrade-to-fit policy over `select` (ADR-0008 §5): if the working set exceeds
 * `ceilingBytes`, raise a global `minLevel` floor (coarsen everything) and retry,
 * up to the coarsest level. Returns `Err('out of memory')` only when even the
 * coarsest selection cannot fit — the honest floor, not a crash.
 */
export function selectWithinBudget(ms: Multiscale, cam: Camera, ceilingBytes: number, opts: SelectOptions = {}): Result<Selection> {
  const maxLevel = ms.levelCount - 1;
  const startFloor = Math.max(0, opts.minLevel ?? 0);
  for (let floor = startFloor; floor <= maxLevel; floor++) {
    const sel = select(ms, cam, { ...opts, minLevel: floor });
    if (sel.totalApproxBytes <= ceilingBytes) return { ok: true, value: sel };
  }
  return { ok: false, error: "out of memory" };
}
