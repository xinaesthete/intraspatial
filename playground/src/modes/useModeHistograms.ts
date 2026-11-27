// Histograms of the three painted modes over the map's window (`modeHistograms`, on the GPU), for
// the legend and for the map's contrast. Read from the result's snapshot rather than its pooled
// rasters, inside `onGpu`, so a compute running meanwhile cannot overwrite what is being binned.
//
// While a new result's histograms compute, the last ones stand in (`forCurrent` false), so the
// map's contrast does not flash back to the σ rule for a frame on every pan.

import { useEffect, useState } from "react";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import { type ModeBasis, modeParams } from "../../../src/gpu/spatial/gramModes";
import { type ModeHistograms, modeHistograms } from "../../../src/gpu/spatial/modeHistograms";
import { onGpu } from "./gramSnapshots";

export interface ModeHistogramState {
  readonly value: ModeHistograms;
  /** The window's raster size the histograms were taken over. */
  readonly width: number;
  readonly height: number;
  /** False while these are the previous result's, standing in. */
  readonly forCurrent: boolean;
}

export function useModeHistograms(
  g: { readonly res: GramMatrixGpuResult; readonly rasters: GPUBuffer } | undefined,
  vectors: Float64Array | undefined,
  basis?: ModeBasis,
): { readonly hist?: ModeHistogramState; readonly error?: Error } {
  const [state, setState] = useState<{ key: object; value: ModeHistograms; width: number; height: number }>();
  const [error, setError] = useState<Error>();
  useEffect(() => {
    if (!g || !vectors) return;
    let live = true;
    const key = g;
    void onGpu(async () => {
      if (!live) return;
      const params = modeParams(g.res, { vectors, ...(basis ? { basis } : {}) });
      const value = await modeHistograms(g.res, params, g.rasters, { bins: 512 });
      if (!live) return;
      setState({ key, value, width: g.res.width, height: g.res.height });
      setError(undefined);
    }).catch((err: unknown) => {
      if (live) setError(err instanceof Error ? err : new Error(String(err)));
    });
    return () => {
      live = false;
    };
  }, [g, vectors, basis]);
  const hist = state && { value: state.value, width: state.width, height: state.height, forCurrent: state.key === g };
  return { ...(hist ? { hist } : {}), ...(error ? { error } : {}) };
}

/** Where each mode's ramp saturates as the map is painted (1 / the scale `modeParams` applies, so
 *  the chroma weighting is included), for the legend's dashed lines. 0 for a missing mode. */
export function rampHalfRanges(
  res: GramMatrixGpuResult,
  opts: {
    vectors: Float64Array;
    basis?: ModeBasis | undefined;
    saturate: number;
    chromaWeight: number;
    ranges?: readonly [number, number, number] | undefined;
  },
): [number, number, number] {
  const { scales } = modeParams(res, {
    vectors: opts.vectors,
    saturate: opts.saturate,
    chromaWeight: opts.chromaWeight,
    ...(opts.basis ? { basis: opts.basis } : {}),
    ...(opts.ranges ? { ranges: opts.ranges } : {}),
  });
  return scales.map((s) => (s > 0 ? 1 / s : 0)) as [number, number, number];
}
