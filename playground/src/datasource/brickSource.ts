// The BrickSource seam — what the brick-atlas VolumeRenderer pulls a chunk's B³ voxel
// brick from, decoupled from *how* the voxels are produced. Two implementations today:
//
//  • `mandelbulbBrickSource` — GPU-generated: a compute pass (src/gpu/fields/mandelbulb)
//    runs the Mandelbulb per voxel on the device and reads the brick back, so the synthetic
//    volume never blocks the main thread computing chunks on the CPU. This is the ADR-0009
//    "data as an op" direction; the readback is async (masked by the load scheduler) and the
//    zero-readback resident bridge is the flagged follow-up (see AGENTS.md).
//  • `loaderBrickSource` — wraps a `Loader` (real OME-Zarr, or the CPU golden loader) and
//    nearest-upsamples the decoded tile into a full B³ brick. The path real data will take.
//
// Both yield a `Uint8Array` of exactly B³ voxels (the atlas is R8). The seam is where the
// load scheduler + latency simulation will wrap in the next steps.

import { type ChunkId, chunkVoxelExtent, type Loader, levelVoxelDims, type Multiscale } from "../../../src/datasource";
import { type MandelbulbRegion, mandelbulbBrickGpu } from "../../../src/gpu/fields/mandelbulb";
import type { GpuBackend } from "../../../src/gpu/graph/backend";

/** Produces a chunk's fully-populated B³ voxel brick (R8), ready for the atlas. */
export interface BrickSource {
  brick(id: ChunkId): Promise<Uint8Array>;
}

/** Mutable latency model for the network simulation. Each brick waits `base + rand·jitter` ms
 *  before the underlying source runs. Mutated live by the UI, so decorators read it per call. */
export interface LatencyModel {
  base: number;
  jitter: number;
}

/** Wrap a source so each brick is delayed per `model` — the "slow network" simulation. Since
 *  GPU generation is otherwise near-instant, this is what makes the loading process observable
 *  (chunks streaming in through the scheduler, LOD-fallback, loading-colour). */
export function slowBrickSource(inner: BrickSource, model: LatencyModel): BrickSource {
  return {
    async brick(id: ChunkId): Promise<Uint8Array> {
      const delay = model.base + Math.random() * model.jitter;
      if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
      return inner.brick(id);
    },
  };
}

const quantize = (f: Float32Array): Uint8Array => {
  const out = new Uint8Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const v = f[i] ?? 0;
    out[i] = Math.round((v < 0 ? 0 : v > 1 ? 1 : v) * 255);
  }
  return out;
};

/** The normalised unit-cube sub-box a chunk covers, sampled at B³ voxel centres. Collapses
 *  to the CPU loader's `(base + idx + 0.5)/levelDims` for an interior chunk (extent === B). */
function chunkRegion(ms: Multiscale, id: ChunkId, B: number): MandelbulbRegion {
  const levelDims = levelVoxelDims(ms, id.level);
  const extent = chunkVoxelExtent(ms, id);
  const base: [number, number, number] = [id.x * ms.chunkShape[0], id.y * ms.chunkShape[1], id.z * ms.chunkShape[2]];
  return {
    dims: [B, B, B],
    origin: [base[0] / levelDims[0], base[1] / levelDims[1], base[2] / levelDims[2]],
    step: [extent[0] / (B * levelDims[0]), extent[1] / (B * levelDims[1]), extent[2] / (B * levelDims[2])],
  };
}

/** GPU-generated Mandelbulb bricks. `backend` should be one bound to the *renderer's* device
 *  (via `adoptDevice`) so the compute shares the device with the atlas. */
export function mandelbulbBrickSource(ms: Multiscale, backend: GpuBackend, B: number): BrickSource {
  return {
    async brick(id: ChunkId): Promise<Uint8Array> {
      const f = await mandelbulbBrickGpu(backend, chunkRegion(ms, id, B));
      return quantize(f);
    },
  };
}

/** Wraps a `Loader`: decode the tile, then nearest-upsample (dims ≤ B) into a full B³ brick
 *  so the page-table UV mapping is uniform regardless of the chunk's own resolution. */
export function loaderBrickSource(loader: Loader, B: number): BrickSource {
  return {
    async brick(id: ChunkId): Promise<Uint8Array> {
      const tile = await loader.getChunk(id);
      const [ex, ey, ez] = tile.dims;
      const out = new Uint8Array(B * B * B);
      for (let z = 0; z < B; z++) {
        const sz = Math.min(ez - 1, ((z * ez) / B) | 0);
        for (let y = 0; y < B; y++) {
          const sy = Math.min(ey - 1, ((y * ey) / B) | 0);
          for (let x = 0; x < B; x++) {
            const sx = Math.min(ex - 1, ((x * ex) / B) | 0);
            out[(z * B + y) * B + x] = Math.round((tile.data[(sz * ey + sy) * ex + sx] ?? 0) * 255);
          }
        }
      }
      return out;
    },
  };
}
