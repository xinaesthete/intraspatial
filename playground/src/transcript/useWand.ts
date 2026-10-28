// The wand: pick a place on the mode map, and every other place is measured by how far its mixture
// of channels is from that one (the whitened distance `similarityWgsl` computes), with an outline
// where the distance reaches the tolerance. cellModesMain.ts has the same tool for one fixed canvas.
//
// The wand is held as a PLACE, in the transcripts' own frame, not as a pixel: each new result (a
// zoom, a new radius) re-reads the channels there, so the reference always means "the mixture at
// that spot, standardised as this map is". A place the new window does not cover drops the wand.
//
// Reading the channels is a GPU dispatch plus a buffer map, too slow to keep up with pointer moves
// and not to be queued: fifty pending readbacks would land in order, redrawing for positions the
// pointer left long ago. So the latest place is held and one probe runs at a time; intermediate
// places are dropped, and the last one is always read because the loop re-checks after each probe.
// The connected region's distance field is read back the same way.

import { type PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import { type ModeBasis, modeParams } from "../../../src/gpu/spatial/gramModes";
import { probeChannels, standardise } from "../../../src/gpu/spatial/gramTerrain";
import { connectedRegion, wandDistance } from "../../../src/gpu/spatial/wandRegion";

/** What the wand reads from: a Gram result, a copy of its rasters that stays intact while it is
 *  on screen, and the per-channel mean and sd its map is standardised with (a locked basis's, or
 *  its own). */
export interface WandSource {
  readonly res: GramMatrixGpuResult;
  readonly rasters: GPUBuffer;
  readonly mean: ArrayLike<number>;
  readonly sd: ArrayLike<number>;
  /** The modes the map is painted with, and the locked basis if any — what the metric is built from. */
  readonly vectors: Float64Array;
  readonly basis?: ModeBasis;
}

export interface WandSample {
  /** Standardised channel values at the place — the reference the layer measures from. */
  readonly z: Float64Array;
  /** The kernel-smoothed densities `z` was standardised from. */
  readonly raw: Float64Array;
  /** The place, in raster pixels (pixel centres at whole numbers, fractions between) — the layer's marker. */
  readonly col: number;
  readonly row: number;
  readonly source: WandSource;
}

/** A place in the raster's frame (row 0 at the bbox's top), or undefined outside it. */
function toRaster(res: GramMatrixGpuResult, x: number, y: number): { col: number; row: number } | undefined {
  const [minX, minY, maxX, maxY] = res.bbox;
  const col = ((x - minX) / (maxX - minX)) * res.width - 0.5;
  const row = ((maxY - y) / (maxY - minY)) * res.height - 0.5;
  if (col < -0.5 || row < -0.5 || col > res.width - 0.5 || row > res.height - 0.5) return undefined;
  // Not rounded: the probe interpolates, so the sample follows the pointer, not the pixel grid.
  return { col, row };
}

/**
 * Run `task` for the latest job only, one at a time: a job scheduled while one runs replaces any
 * other waiting job, and the loop re-checks after each run so the last job is never lost.
 */
function coalescing<T>(task: (job: T) => Promise<void>) {
  let waiting: { job: T } | null = null;
  let busy = false;
  return {
    schedule(job: T) {
      waiting = { job };
      if (busy) return;
      busy = true;
      void (async () => {
        try {
          while (waiting) {
            const { job: next } = waiting;
            waiting = null;
            await task(next);
          }
        } finally {
          busy = false;
        }
      })();
    },
    cancel() {
      waiting = null;
    },
  };
}

/** The wand's distance at every pixel, for the sample and metric it was computed with. */
interface Field {
  readonly d: Float32Array;
  readonly sample: WandSample;
  readonly m: number;
}

export function useWand(source: WandSource | undefined) {
  const [tolerance, setTolerance] = useState(1.2);
  /** 0 = every mode: full Mahalanobis distance. */
  const [modesUsed, setModesUsed] = useState(3);
  /** Outline only the fragment the sample sits in, not every place within tolerance. */
  const [connected, setConnected] = useState(false);
  const [place, setPlace] = useState<readonly [number, number]>();
  const [sample, setSample] = useState<WandSample>();
  const [field, setField] = useState<Field>();
  const [error, setError] = useState<Error>();

  const latest = useRef(source);
  const fail = (err: unknown) => setError(err instanceof Error ? err : new Error(String(err)));

  const [probe] = useState(() =>
    coalescing(async ({ at: [x, y], src }: { at: readonly [number, number]; src: WandSource | undefined }) => {
      if (!src) return;
      // Outside this window: nothing to read, and the last sample (another result's) is not drawn.
      const at = toRaster(src.res, x, y);
      if (!at) return;
      try {
        const raw = await probeChannels(src.res, at.col, at.row, src.rasters);
        setSample({ z: standardise(raw, src.mean, src.sd), raw, ...at, source: src });
        setError(undefined);
      } catch (err) {
        fail(err);
      }
    }),
  );
  const [distance] = useState(() =>
    coalescing(async ({ s, m }: { s: WandSample; m: number }) => {
      const { res, rasters, vectors, basis } = s.source;
      try {
        const params = modeParams(res, { vectors, reference: s.z, modesUsed: m, ...(basis ? { basis } : {}) });
        setField({ d: await wandDistance(res, params, rasters), sample: s, m });
      } catch (err) {
        fail(err);
      }
    }),
  );

  const pick = useCallback(
    (x: number, y: number) => {
      setPlace([x, y]);
      probe.schedule({ at: [x, y], src: latest.current });
    },
    [probe],
  );
  const clear = useCallback(() => {
    probe.cancel();
    distance.cancel();
    setPlace(undefined);
    setSample(undefined);
  }, [probe, distance]);

  // A new result re-reads the place: the old z was standardised for another map.
  useEffect(() => {
    latest.current = source;
    if (place) probe.schedule({ at: place, src: source });
  }, [source, place, probe]);

  // A sample from another result is not drawn on this one: it would mark the wrong pixel.
  const current = sample && sample.source === source ? sample : undefined;
  const m = modesUsed || (source?.res.labels.length ?? 3);

  // The connected region needs the whole distance field, read back once per sample and metric; the
  // fill itself is redone on the host whenever the tolerance moves.
  useEffect(() => {
    if (connected && current) distance.schedule({ s: current, m });
  }, [connected, current, m, distance]);
  const region = useMemo(() => {
    // While a newer field is on its way, the last one for this map stands in rather than flashing
    // every fragment back for a frame.
    if (!connected || !current || !field || field.sample.source !== current.source || field.m !== m) return undefined;
    const { width, height } = current.source.res;
    return connectedRegion(field.d, width, height, field.sample, tolerance);
  }, [connected, current, field, m, tolerance]);

  /** Where a plain press went down, to tell a click (sample) from a drag (the map's pan). */
  const pressed = useRef<{ x: number; y: number } | null>(null);
  /** A shift-drag is moving the sample. */
  const moving = useRef(false);

  /**
   * Pointer handlers for the element around the map. A click samples; a shift-drag moves the sample,
   * and the map underneath never sees it, so it does not pan; a plain drag pans and the wheel zooms
   * as usual. Capture phase, so they run before the map's own listeners. `toPlace` turns a position
   * across the element (0–1 from its left and top) into a place in the transcripts' frame.
   */
  const pointer = useCallback(
    (toPlace: (u: number, v: number) => readonly [number, number] | undefined) => {
      const track = (e: PointerEvent<HTMLElement>) => {
        const box = e.currentTarget.getBoundingClientRect();
        if (box.width <= 0 || box.height <= 0) return;
        const p = toPlace((e.clientX - box.left) / box.width, (e.clientY - box.top) / box.height);
        if (p) pick(p[0], p[1]);
      };
      const swallow = (e: PointerEvent<HTMLElement>) => {
        e.stopPropagation();
        // Also suppresses the compatibility mouse events, which a map may listen to instead.
        e.preventDefault();
      };
      return {
        onPointerDownCapture: (e: PointerEvent<HTMLElement>) => {
          if (e.button !== 0) return;
          if (!e.shiftKey) {
            // Left to the map, which pans if this turns into a drag.
            pressed.current = { x: e.clientX, y: e.clientY };
            return;
          }
          swallow(e);
          moving.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          track(e);
        },
        onPointerMoveCapture: (e: PointerEvent<HTMLElement>) => {
          if (!moving.current) return;
          swallow(e);
          track(e);
        },
        onPointerUpCapture: (e: PointerEvent<HTMLElement>) => {
          if (moving.current) {
            moving.current = false;
            swallow(e);
            return;
          }
          const down = pressed.current;
          pressed.current = null;
          // A press that barely moved is a click, not a pan.
          if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) track(e);
        },
      };
    },
    [pick],
  );

  return {
    tolerance,
    setTolerance,
    connected,
    setConnected,
    region,
    modesUsed,
    setModesUsed,
    place,
    sample: current,
    error,
    clear,
    pointer,
  };
}

export type Wand = ReturnType<typeof useWand>;
