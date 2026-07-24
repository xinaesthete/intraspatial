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
  type Affine3,
  applyAffine,
  boxOutsideFrustum,
  type Camera,
  cameraBasis,
  dot,
  frustumPlanes,
  invertAffine,
  orientedBoxCorners,
  sub,
  type Vec3,
  worldPerPixel,
} from "./math";
import { chunkApproxBytes, chunkArrayBox, chunkCounts, worldVoxelSize0 } from "./multiscale";
import type { ChunkId, Multiscale, Result, SelectedChunk, Selection } from "./types";
import { worldFromArrayOf } from "./types";

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

/** Optical-axis depth of a chunk's nearest point (clamped to the near plane). We find the
 *  nearest point on the *oriented* chunk box by clamping the eye into the box in array space
 *  and mapping back — exact for a rotation+uniform-scale placement, a close approximation
 *  under shear/anisotropy (good enough for the coarse LOD quantisation). */
function nearestDepth(wfa: Affine3, worldToArray: Affine3, cam: Camera, fwd: Vec3, lo: Vec3, hi: Vec3): number {
  const e = applyAffine(worldToArray, cam.eye);
  const clamped: Vec3 = [
    Math.min(Math.max(e[0], lo[0]), hi[0]),
    Math.min(Math.max(e[1], lo[1]), hi[1]),
    Math.min(Math.max(e[2], lo[2]), hi[2]),
  ];
  const p = applyAffine(wfa, clamped);
  return Math.max(cam.near, dot(sub(p, cam.eye), fwd));
}

/**
 * Select the chunks the Camera needs. Pure. `q` is the detail budget; `minLevel`
 * floors the resolution (used by `selectWithinBudget`).
 *
 * SCALING (very large images): the returned Selection is UNBOUNDED — a whole-volume view that
 * demands the finest level yields one chunk per finest cell in-frustum (tens of thousands at
 * 2048³+), which then drives every downstream cost (page-table rebuild, atlas residency, overlay
 * geometry). To scale, bound it with the resource ceiling: raise `minLevel` via `selectWithinBudget`
 * (degrade-to-fit, ADR-0008 §5) so the working set fits the atlas. The renderer demo doesn't drive
 * this yet — it relies on the LOD fallback to stay correct (just blurrier) past capacity.
 */
export function select(ms: Multiscale, cam: Camera, opts: SelectOptions = {}): Selection {
  const q = opts.q ?? 1;
  const minLevel = Math.max(0, opts.minLevel ?? 0);
  const maxLevel = ms.levelCount - 1;
  const planes = frustumPlanes(cam);
  const { fwd } = cameraBasis(cam);
  const wfa = worldFromArrayOf(ms);
  const worldToArray = invertAffine(wfa); // once — reused for every chunk's nearest-point

  const out: SelectedChunk[] = [];
  const countByLevel: number[] = new Array(ms.levelCount).fill(0);

  const emit = (id: ChunkId, depth: number): void => {
    out.push({ id, nearestDepth: depth, approxBytes: chunkApproxBytes(ms, id) });
    const c = countByLevel[id.level];
    countByLevel[id.level] = (c ?? 0) + 1;
  };

  const visit = (id: ChunkId): void => {
    const [lo, hi] = chunkArrayBox(ms, id);
    if (boxOutsideFrustum(orientedBoxCorners(wfa, lo, hi), planes)) return; // frustum cull (oriented box)

    const depth = nearestDepth(wfa, worldToArray, cam, fwd, lo, hi);
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
