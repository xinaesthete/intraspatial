// Tiles → transcript channels → the GPU Gram → co-location modes, for one window.
//
// `gramMatrixGpu` leaves its rasters in ONE pooled device buffer that the next call overwrites. A
// deck layer redraws every frame, whenever deck likes, including while the next compute is writing,
// so it cannot read that buffer. Each result therefore carries `rasters`, its own GPU-side copy,
// taken inside the GPU queue.
//
// Those copies are pooled, and a copy is reused only when no one can be drawing it: when its result
// is older than both the one on screen and the one before it (deck may still be drawing that for a
// frame). Results still in flight, or finished but not yet on screen, are never reused. Counting
// "the result before" only by order breaks: `useAsync` drops results whose inputs changed while they
// ran, but their GPU work has happened, so while a slider moves, superseded results can come and
// go in between the one on screen and its successor.

import { useEffect, useRef } from "react";
import type { Rect } from "../../../src/datasource/points";
import { getDevice } from "../../../src/gpu/device";
import { type GramMatrixGpuResult, gramMatrixGpu } from "../../../src/gpu/spatial/gramMatrix";
import { alignModeSigns, type CoLocationModes, coLocationModes, rasterSizeForRadius } from "../../../src/spatial/gram";
import { type TranscriptChannels, transcriptChannels, type WeightedChannel } from "../../../src/spatial/transcriptChannels";
import { useAsync } from "../hooks/useAsync";
import type { WindowTiles } from "../hooks/useSpatialData";

let queue: Promise<unknown> = Promise.resolve();
let generation = 0;

/** Run GPU work one task at a time. */
function onGpu<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.catch(() => undefined);
  return run;
}

interface Snapshot {
  readonly buffer: GPUBuffer;
  /** The generation whose rasters it holds. */
  gen: number;
}

/**
 * Copy `res`'s rasters into a pooled buffer no one can be drawing — see the note at the top. `shown`
 * holds the generations on screen now and just before. Call inside `onGpu`.
 */
async function snapshot(pool: Snapshot[], shown: readonly number[], res: GramMatrixGpuResult, gen: number): Promise<GPUBuffer> {
  const device = await getDevice();
  const bytes = res.labels.length * res.height * res.resident.rowWords * 4;
  const newest = shown.length ? Math.max(...shown) : Number.NEGATIVE_INFINITY;
  const free = (s: Snapshot) => s.gen < newest && !shown.includes(s.gen);
  let slot = pool.find((s) => free(s) && s.buffer.size >= bytes);
  if (!slot) {
    // A free buffer too small for this result is replaced. Free means undrawn, so destroying is safe.
    const small = pool.findIndex(free);
    if (small >= 0) pool.splice(small, 1)[0]?.buffer.destroy();
    slot = {
      buffer: device.createBuffer({
        size: Math.max(bytes, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: "gram snapshot",
      }),
      gen,
    };
    pool.push(slot);
  }
  slot.gen = gen;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(res.resident.buffer, 0, slot.buffer, 0, bytes);
  device.queue.submit([enc.finish()]);
  return slot.buffer;
}

export interface GramParamsUi {
  readonly radius: number;
  /** Transcripts below this quality are dropped; 0 keeps all. */
  readonly qvMin: number;
  /** Store the per-channel rasters at half precision — halves the raster buffer, which is both
   *  the memory and the `maxStorageBufferBindingSize` ceiling this path hits first. */
  readonly halfRasters: boolean;
  /** Raster long side; 0 derives it from the radius. */
  readonly rasterSide: number;
}

export interface TranscriptGram {
  readonly generation: number;
  readonly res: GramMatrixGpuResult;
  /** This result's rasters, laid out as `res.resident`, kept intact while it or its successor is on screen. */
  readonly rasters: GPUBuffer;
  readonly modes: CoLocationModes;
  readonly stats: TranscriptChannels["stats"];
  readonly window: Rect;
  /** The neighbourhood radius this result was computed at — the caller's, which may differ from the UI's. */
  readonly radius: number;
  readonly raster: { readonly width: number; readonly height: number; readonly clamped: boolean };
  readonly ms: { readonly channels: number; readonly gram: number };
}

/** A raster whose long side is `side`, pixels kept square. */
function fixedRaster(bbox: readonly [number, number, number, number], side: number) {
  const w = bbox[2] - bbox[0];
  const h = bbox[3] - bbox[1];
  const short = Math.max(1, Math.round((side * Math.min(w, h)) / Math.max(w, h)));
  return { width: w >= h ? side : short, height: w >= h ? short : side, clamped: false };
}

/**
 * The Gram and modes for `tiles`' window. The last result stays on screen while the next computes —
 * within `scope` (the points source, say): another store's result is never shown for this one's.
 */
export function useTranscriptGram(tiles: WindowTiles | undefined, channels: readonly WeightedChannel[], p: GramParamsUi, scope?: unknown) {
  const prevModes = useRef<CoLocationModes | null>(null);
  const pool = useRef<Snapshot[]>([]);
  /** Generations on screen: the one before, and the current one. */
  const shown = useRef<number[]>([]);
  const state = useAsync<TranscriptGram>(
    (signal) => {
      if (!tiles || channels.length === 0) return undefined;
      const superseded = () => {
        if (signal.aborted) throw new Error("superseded");
      };
      return onGpu(async () => {
        superseded();
        const t0 = performance.now();
        const ch = transcriptChannels(tiles.tiles, channels, {
          window: tiles.window,
          apron: p.radius,
          ...(p.qvMin > 0 ? { minimum: { column: "qv", value: p.qvMin } } : {}),
        });
        superseded(); // the channels take a while; a moving slider has usually moved on by now
        const t1 = performance.now();
        const raster = p.rasterSide > 0 ? fixedRaster(ch.bbox, p.rasterSide) : rasterSizeForRadius(ch.bbox, p.radius);
        const res = await gramMatrixGpu(ch.clouds, {
          bbox: ch.bbox,
          width: raster.width,
          height: raster.height,
          radius: p.radius,
          precision: p.halfRasters ? "f16" : "f32",
        });
        superseded(); // nothing to copy or align for a result no one will see
        const modes = alignModeSigns(coLocationModes(res), prevModes.current);
        prevModes.current = modes;
        const gen = ++generation;
        const rasters = await snapshot(pool.current, shown.current, res, gen);
        return {
          generation: gen,
          res,
          rasters,
          modes,
          stats: ch.stats,
          window: tiles.window,
          radius: p.radius,
          raster,
          ms: { channels: t1 - t0, gram: performance.now() - t1 },
        };
      });
    },
    [tiles, channels, p.radius, p.qvMin, p.rasterSide, p.halfRasters],
    scope,
  );
  const gen = state.value?.generation;
  useEffect(() => {
    if (gen !== undefined) shown.current = [...shown.current.slice(-1), gen];
  }, [gen]);
  return state;
}
