// Tiles → transcript channels → the GPU Gram → co-location modes, for one window.
//
// `gramMatrixGpu` leaves its rasters in ONE pooled device buffer that the next call overwrites. A
// deck layer redraws every frame, whenever deck likes, including while the next compute is writing,
// so it cannot read that buffer. Each result therefore carries `rasters`, a GPU-side copy taken
// inside the GPU queue, in one of two buffers used in turn. A result's copy stays intact until the
// result after next — by then the layer has long moved on.

import { useRef } from "react";
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

const copies: (GPUBuffer | undefined)[] = [undefined, undefined];

/** Copy `res`'s rasters into the next of the two snapshot buffers, on the GPU. Call inside `onGpu`. */
async function snapshot(res: GramMatrixGpuResult, slot: number): Promise<GPUBuffer> {
  const device = await getDevice();
  const bytes = res.labels.length * res.height * res.resident.rowFloats * 4;
  let buf = copies[slot];
  if (!buf || buf.size < bytes) {
    // Not destroyed: a layer may still bind it. Dropped, it is collected once nothing holds it.
    buf = device.createBuffer({
      size: Math.max(bytes, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `gram snapshot ${slot}`,
    });
    copies[slot] = buf;
  }
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(res.resident.buffer, 0, buf, 0, bytes);
  device.queue.submit([enc.finish()]);
  return buf;
}

export interface GramParamsUi {
  readonly radius: number;
  /** Transcripts below this quality are dropped; 0 keeps all. */
  readonly qvMin: number;
  /** Raster long side; 0 derives it from the radius. */
  readonly rasterSide: number;
}

export interface TranscriptGram {
  readonly generation: number;
  readonly res: GramMatrixGpuResult;
  /** This result's rasters, laid out as `res.resident`, safe to draw from until the result after next. */
  readonly rasters: GPUBuffer;
  readonly modes: CoLocationModes;
  readonly stats: TranscriptChannels["stats"];
  readonly window: Rect;
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

export function useTranscriptGram(tiles: WindowTiles | undefined, channels: readonly WeightedChannel[], p: GramParamsUi) {
  const prevModes = useRef<CoLocationModes | null>(null);
  return useAsync<TranscriptGram>(
    (signal) => {
      if (!tiles || channels.length === 0) return undefined;
      return onGpu(async () => {
        if (signal.aborted) throw new Error("superseded");
        const t0 = performance.now();
        const ch = transcriptChannels(tiles.tiles, channels, {
          window: tiles.window,
          apron: p.radius,
          ...(p.qvMin > 0 ? { minimum: { column: "qv", value: p.qvMin } } : {}),
        });
        const t1 = performance.now();
        const raster = p.rasterSide > 0 ? fixedRaster(ch.bbox, p.rasterSide) : rasterSizeForRadius(ch.bbox, p.radius);
        const res = await gramMatrixGpu(ch.clouds, { bbox: ch.bbox, width: raster.width, height: raster.height, radius: p.radius });
        const modes = alignModeSigns(coLocationModes(res), prevModes.current);
        prevModes.current = modes;
        const gen = ++generation;
        const rasters = await snapshot(res, gen % 2);
        return {
          generation: gen,
          res,
          rasters,
          modes,
          stats: ch.stats,
          window: tiles.window,
          raster,
          ms: { channels: t1 - t0, gram: performance.now() - t1 },
        };
      });
    },
    [tiles, channels, p.radius, p.qvMin, p.rasterSide],
  );
}
