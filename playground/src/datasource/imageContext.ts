// One pyramid level of a SpatialData image, composited to a single RGBA texture — the context
// layer the mode views are drawn against.
//
// ## Why a whole level rather than a tile pyramid
//
// `tileRenderer.ts` already does proper level-of-detail streaming for the scene editor. Nothing
// here reuses it, and the reason is scope rather than duplication: the mode views draw ONE fixed
// analysis window that only changes when the statistic is recomputed, so there is no camera-driven
// LOD problem to solve. Fetching the finest level that fits a pixel budget, compositing it once and
// uploading one texture is the whole job, and it makes the cost obvious — a bounded number of
// chunks at load, then nothing per frame.
//
// The honest limit of that choice: zoom past the level's resolution and the overlay goes soft. It
// is a context layer, not a viewer. Wiring the tiled path in would be the fix and is not done.
//
// ## Coordinates are the hard part, not the pixels
//
// The centroids are placed by `cellTable`, which resolves the *table's* transform into the store's
// declared coordinate system. For an overlay to mean anything the image has to land in that same
// system, and `ms.placements[0]` will NOT do it — the loader synthesises a demo-normalised,
// axis-aligned placement that centres each image on the origin at a fixed world span, which is
// right for the scene editor's staggered layout and wrong for anything that has to agree with
// other elements. The real transform is `globalFromArray` (element ∘ dataset, straight from
// sd.js), and it is optional: a store that carries none cannot be aligned at all. That case is
// reported rather than papered over, because an overlay that is silently in the wrong place is
// worse than no overlay.

import type { SpatialData } from "@spatialdata/core";
import type { Affine3 } from "../../../src/datasource";
import { getDevice } from "../../../src/gpu/device";
import { imageHandle, type SpatialDataImage } from "./spatialDataLoader";
import type { ChannelSettings } from "./tileChannelMaterial";

/** Long-side cap for the composited texture. 2048 keeps the load to at most a 4×4 grid of 512²
 *  chunks per channel while still out-resolving the mode raster, which is a few hundred pixels. */
const DEFAULT_MAX_SIDE = 2048;

export interface ContextImage {
  readonly texture: GPUTexture;
  readonly width: number;
  readonly height: number;
  /** Which pyramid level was taken, and out of how many. */
  readonly level: number;
  readonly levelCount: number;
  /** Level-0 array→world for the store's real coordinate system. Absent means the store carries no
   *  stored transform for this element, and `aligned` is false. */
  readonly worldFromArray?: Affine3;
  readonly aligned: boolean;
  /** Level-0 voxel dims, which `worldFromArray` is expressed against. */
  readonly dims0: readonly [number, number];
  readonly label: string;
  readonly channels: readonly ChannelSettings[];
}

/** Names of the image elements in a store, for a dropdown. Takes the shared `SpatialData`. */
export async function listImageElements(sdata: SpatialData): Promise<string[]> {
  return (await imageHandle(sdata)).imageNames;
}

/** Finest level whose long side fits the budget — quality first, subject to a bounded fetch. */
function pickLevel(img: SpatialDataImage, maxSide: number): number {
  const dims = img.ms.levelDims;
  const n = img.ms.levelCount;
  for (let L = 0; L < n; L++) {
    const d = dims?.[L] ?? [Math.ceil(img.ms.voxelDims0[0] / 2 ** L), Math.ceil(img.ms.voxelDims0[1] / 2 ** L), 1];
    if (Math.max(d[0], d[1]) <= maxSide) return L;
  }
  return n - 1; // every level is bigger than the budget: take the coarsest and accept it
}

/**
 * Fetch one level whole and composite it to an RGBA texture.
 *
 * Compositing happens on the host rather than in the mode shaders, and that is deliberate: the
 * channel colours and contrast windows are load-time metadata that never change while the view is
 * being explored, so folding them in once costs one pass over the level and saves carrying N
 * channel planes and their settings into two more shaders.
 */
export async function loadContextImage(sdata: SpatialData, element: string, opts: { maxSide?: number } = {}): Promise<ContextImage> {
  const device = await getDevice();
  const img = await (await imageHandle(sdata)).image(element);
  const maxSide = opts.maxSide ?? DEFAULT_MAX_SIDE;
  const level = pickLevel(img, maxSide);
  const dims = img.ms.levelDims?.[level] ?? [Math.ceil(img.ms.voxelDims0[0] / 2 ** level), Math.ceil(img.ms.voxelDims0[1] / 2 ** level), 1];
  const W = Math.max(1, dims[0]);
  const H = Math.max(1, dims[1]);
  const tile = img.ms.chunkShape[0];
  const nx = Math.ceil(W / tile);
  const ny = Math.ceil(H / tile);

  const lanes = img.ms.element.kind === "vec" ? img.ms.element.n : 1;
  const vis = img.channels.map((c) => c.visible !== false);
  const rgba = new Uint8Array(W * H * 4);
  // Opaque by default: a chunk that fails to load stays black rather than punching a hole through
  // to whatever is behind, which would read as tissue structure.
  for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;

  const jobs: Promise<void>[] = [];
  for (let cy = 0; cy < ny; cy++) {
    for (let cx = 0; cx < nx; cx++) {
      jobs.push(
        img.loader.getChunk({ level, x: cx, y: cy, z: 0 }).then((t) => {
          const [tw, th] = t.dims;
          for (let y = 0; y < th; y++) {
            const gy = cy * tile + y;
            if (gy >= H) break;
            for (let x = 0; x < tw; x++) {
              const gx = cx * tile + x;
              if (gx >= W) break;
              let r = 0;
              let g = 0;
              let b = 0;
              for (let c = 0; c < lanes; c++) {
                if (!vis[c]) continue;
                const ch = img.channels[c];
                if (!ch) continue;
                const [lo, hi] = ch.contrastLimits;
                const raw = t.data[(y * tw + x) * lanes + c] ?? 0;
                // Additive across channels, matching how the scene editor composites — for a
                // 3-channel RGB image with unit colours this is the identity.
                const v = hi > lo ? Math.min(Math.max((raw - lo) / (hi - lo), 0), 1) : 0;
                r += v * (ch.color[0] ?? 0);
                g += v * (ch.color[1] ?? 0);
                b += v * (ch.color[2] ?? 0);
              }
              const o = (gy * W + gx) * 4;
              rgba[o] = Math.min(255, r * 255);
              rgba[o + 1] = Math.min(255, g * 255);
              rgba[o + 2] = Math.min(255, b * 255);
            }
          }
        }),
      );
    }
  }
  await Promise.all(jobs);

  const texture = device.createTexture({
    size: { width: W, height: H },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.writeTexture({ texture }, rgba, { bytesPerRow: W * 4, rowsPerImage: H }, { width: W, height: H });

  return {
    texture,
    width: W,
    height: H,
    level,
    levelCount: img.ms.levelCount,
    worldFromArray: img.globalFromArray,
    aligned: img.globalFromArray !== undefined,
    dims0: [img.ms.voxelDims0[0], img.ms.voxelDims0[1]],
    label: `${element} · level ${level}/${img.ms.levelCount - 1} · ${W}×${H}`,
    channels: img.channels,
  };
}

/**
 * The 2×3 affine taking a **world** point to image UV in [0,1]².
 *
 * Inverts the image's own array→world placement and divides through by the level-0 extent, so the
 * result is level-independent — the composited texture always spans the same world quad whatever
 * level it came from.
 *
 * Returns null when the placement is degenerate (a zero-determinant 2×2, which a malformed or
 * purely-3D transform can produce) rather than emitting silent NaNs into a shader.
 */
export function uvFromWorld(img: ContextImage): Float64Array | null {
  const m = img.worldFromArray;
  if (!m) return null;
  const a = m.axes[0];
  const b = m.axes[1];
  // world = origin + a*ax + b*ay, restricted to the XY plane.
  const det = a[0] * b[1] - b[0] * a[1];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-30) return null;
  // Inverse of [[a0, b0], [a1, b1]].
  const i00 = b[1] / det;
  const i01 = -b[0] / det;
  const i10 = -a[1] / det;
  const i11 = a[0] / det;
  const ox = m.origin[0];
  const oy = m.origin[1];
  const sx = 1 / Math.max(img.dims0[0], 1);
  const sy = 1 / Math.max(img.dims0[1], 1);
  // uv = S · M⁻¹ · (world − origin), flattened row-major as [m00, m01, tx, m10, m11, ty].
  return new Float64Array([i00 * sx, i01 * sx, -(i00 * ox + i01 * oy) * sx, i10 * sy, i11 * sy, -(i10 * ox + i11 * oy) * sy]);
}
