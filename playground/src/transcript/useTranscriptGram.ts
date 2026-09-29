// Tiles → transcript channels → the GPU Gram → co-location modes, for one window.
//
// `gramMatrixGpu` leaves its rasters in ONE pooled device buffer that the next call overwrites, and
// `paintGramModes` reads that buffer. So compute and paint share a queue, and a paint only runs for
// the latest result (`isLatest`) — an older one's rasters are already gone.

import { useRef } from "react";
import type { Rect } from "../../../src/datasource/points";
import { type GramMatrixGpuResult, gramMatrixGpu } from "../../../src/gpu/spatial/gramMatrix";
import { alignModeSigns, type CoLocationModes, coLocationModes, rasterSizeForRadius } from "../../../src/spatial/gram";
import { type TranscriptChannels, transcriptChannels, type WeightedChannel } from "../../../src/spatial/transcriptChannels";
import { useAsync } from "../hooks/useAsync";
import type { WindowTiles } from "../hooks/useSpatialData";

let queue: Promise<unknown> = Promise.resolve();
let generation = 0;

/** Run GPU work one task at a time. */
export function onGpu<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.catch(() => undefined);
  return run;
}

export const isLatest = (gen: number): boolean => gen === generation;

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
        return {
          generation: ++generation,
          res,
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
