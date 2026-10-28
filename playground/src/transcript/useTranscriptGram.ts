// Tiles → transcript channels → the GPU Gram → co-location modes, for one window. Each result
// carries its own snapshot of the rasters for the deck layer to draw (see `modes/gramSnapshots`).

import { useEffect, useRef } from "react";
import type { Rect } from "../../../src/datasource/points";
import { type GramMatrixGpuResult, gramMatrixGpu } from "../../../src/gpu/spatial/gramMatrix";
import { alignModeSigns, type CoLocationModes, coLocationModes, rasterSizeForRadius } from "../../../src/spatial/gram";
import { type TranscriptChannels, transcriptChannels, type WeightedChannel } from "../../../src/spatial/transcriptChannels";
import { useAsync } from "../hooks/useAsync";
import type { WindowTiles } from "../hooks/useSpatialData";
import { onGpu, useGramSnapshots } from "../modes/gramSnapshots";

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
  const snaps = useGramSnapshots();
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
        const { generation, rasters } = await snaps.take(res);
        return {
          generation,
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
  useEffect(() => snaps.show(gen), [gen, snaps]);
  return state;
}
