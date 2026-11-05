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

import { useCallback, useEffect, useMemo, useState } from "react";
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
 * other waiting job, and the loop re-checks after each run so the last job is never lost. `cancel`
 * drops the waiting job, and `live()` turns false for the running one, so it can land without effect.
 */
function coalescing<T>(task: (job: T, live: () => boolean) => Promise<void>) {
  let waiting: { job: T } | null = null;
  let busy = false;
  let era = 0;
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
            const at = era;
            await task(next, () => at === era);
          }
        } finally {
          busy = false;
        }
      })();
    },
    cancel() {
      waiting = null;
      era++;
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

  const fail = (err: unknown) => setError(err instanceof Error ? err : new Error(String(err)));

  const [probe] = useState(() =>
    coalescing(async ({ at: [x, y], src }: { at: readonly [number, number]; src: WandSource | undefined }, live) => {
      if (!src) return;
      const at = toRaster(src.res, x, y);
      // Outside this window: nothing to read, and the last sample is not drawn.
      if (!at) {
        if (live()) setSample(undefined);
        return;
      }
      try {
        const raw = await probeChannels(src.res, at.col, at.row, src.rasters);
        if (!live()) return;
        setSample({ z: standardise(raw, src.mean, src.sd), raw, ...at, source: src });
        setError(undefined);
      } catch (err) {
        if (live()) fail(err);
      }
    }),
  );
  const [distance] = useState(() =>
    coalescing(async ({ s, m }: { s: WandSample; m: number }, live) => {
      const { res, rasters, vectors, basis } = s.source;
      try {
        const params = modeParams(res, { vectors, reference: s.z, modesUsed: m, ...(basis ? { basis } : {}) });
        const d = await wandDistance(res, params, rasters);
        if (!live()) return;
        setField({ d, sample: s, m });
        setError(undefined);
      } catch (err) {
        if (live()) fail(err);
      }
    }),
  );

  // The place is all a pick sets: the effect below reads the channels there.
  const pick = useCallback((x: number, y: number) => setPlace([x, y]), []);
  const clear = useCallback(() => {
    probe.cancel();
    distance.cancel();
    setPlace(undefined);
    setSample(undefined);
    setField(undefined);
    setError(undefined);
  }, [probe, distance]);

  // A new place, or a new result, reads the channels there: the old z was standardised for another map.
  useEffect(() => {
    if (place) probe.schedule({ at: place, src: source });
  }, [source, place, probe]);

  // A sample from another result is not drawn on this one: it would mark the wrong pixel.
  const current = sample && sample.source === source ? sample : undefined;
  const m = modesUsed || (source?.res.labels.length ?? 3);
  /** Known at once, not after a probe: a place this window does not cover. */
  const outside = !!place && !!source && !toRaster(source.res, place[0], place[1]);

  // The connected region needs the whole distance field, read back once per sample and metric; the
  // fill itself is redone on the host whenever the tolerance moves.
  useEffect(() => {
    if (connected && current) distance.schedule({ s: current, m });
  }, [connected, current, m, distance]);
  // While a newer field is on its way, the last one for this map stands in rather than flashing every
  // fragment back for a frame — and the outline is measured from that field's sample, so the
  // reference and the patch it was filled from are always drawn together.
  const shownField = connected && current && field && field.sample.source === current.source && field.m === m ? field : undefined;
  const region = useMemo(() => {
    if (!shownField) return undefined;
    const { width, height } = shownField.sample.source.res;
    return connectedRegion(shownField.d, width, height, shownField.sample, tolerance);
  }, [shownField, tolerance]);
  const measured = shownField?.sample ?? current;

  // One object per state, not per render, so a page's layers can depend on the wand as a whole.
  return useMemo(
    () => ({
      tolerance,
      setTolerance,
      connected,
      setConnected,
      region,
      modesUsed,
      setModesUsed,
      place,
      outside,
      sample: current,
      /** The sample the outline is measured from: the patch's, while a newer one's field computes. */
      measured,
      error,
      /** Sample the place `x, y`, in the frame of the result's bbox. */
      pick,
      clear,
    }),
    [tolerance, connected, region, modesUsed, place, outside, current, measured, error, pick, clear],
  );
}

export type Wand = ReturnType<typeof useWand>;

/** The `GramModesLayer` props that draw `wand`: its reference, marker, tolerance, metric and patch. */
export function wandLayerProps(wand: Wand, K: number) {
  const s = wand.sample;
  return {
    reference: wand.measured?.z ?? null,
    marker: s ? { col: s.col, row: s.row } : null,
    tolerance: wand.tolerance,
    // "every mode" is held as 0 so the option need not know K.
    modesUsed: wand.modesUsed || K,
    region: wand.region ?? null,
  };
}
