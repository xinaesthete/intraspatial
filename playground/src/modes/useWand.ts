// The wand: a co-location profile sampled from one pixel of the mode map, as the reference every
// other pixel is measured against (the selection outline in `GramModesLayer`).
//
// Sampling is a GPU dispatch plus a buffer map, so it cannot keep up with pointer events one-to-one
// and must not be allowed to queue: fifty pending readbacks would land in order and repaint for
// positions the pointer left long ago. The latest position is held and one sample runs at a time —
// a coalescing loop, which drops intermediate positions rather than falling behind, and always
// samples the final one because it re-checks after each readback.

import { useCallback, useEffect, useRef, useState } from "react";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import { probeChannels, standardise } from "../../../src/gpu/spatial/gramTerrain";
import { onSnapshot } from "./gramSnapshots";

export interface WandSample {
  /** The Gram result it was taken from; a sample from any other is meaningless. */
  readonly generation: number;
  /** Raster pixel. */
  readonly col: number;
  readonly row: number;
  /** Kernel-smoothed density per channel at that pixel. */
  readonly raw: Float64Array;
  /** `raw` standardised the way `corr` was built — the reference vector. */
  readonly z: Float64Array;
}

export interface WandSource {
  readonly generation: number;
  readonly res: GramMatrixGpuResult;
  readonly rasters: GPUBuffer;
}

/** `sample(col, row)` sets the reference, or `error` says why it could not; `wand` is it, but only while `source` is the result it came
 *  from — a new splat standardises against new means, so an old `z` no longer means what it says. */
export function useWand(source: WandSource | undefined) {
  const [sample, setSample] = useState<WandSample>();
  const [error, setError] = useState<Error>();
  const latest = useRef(source);
  useEffect(() => {
    latest.current = source;
  }, [source]);
  const queued = useRef<{ col: number; row: number } | null>(null);
  const busy = useRef(false);

  const take = useCallback((col: number, row: number) => {
    queued.current = { col, row };
    if (busy.current) return;
    busy.current = true;
    void (async () => {
      try {
        while (queued.current && latest.current) {
          const at = queued.current;
          queued.current = null;
          const { res, rasters, generation } = latest.current;
          try {
            // From the snapshot, not `res.resident`: a later compute or a null run may have overwritten that.
            const raw = await probeChannels(onSnapshot(res, rasters), at.col, at.row);
            const z = standardise(raw, res.resident.mean, res.resident.sd);
            setSample({ generation, col: Math.round(at.col), row: Math.round(at.row), raw, z });
            setError(undefined);
          } catch (e) {
            // Reported, not thrown: a probe can fail (f16 rasters, a lost device) and the loop must
            // still take the next position.
            setError(e instanceof Error ? e : new Error(String(e)));
          }
        }
      } finally {
        busy.current = false;
      }
    })();
  }, []);

  const wand = sample && source && sample.generation === source.generation ? sample : undefined;
  return { wand, error, sample: take };
}
