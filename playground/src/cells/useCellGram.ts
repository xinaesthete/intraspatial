// Cell channels → the GPU Gram → co-location modes, over one analysis window. Recomputes whenever
// the channels or the splat parameters change; each result carries its own raster snapshot for the
// deck layer and the wand (see `modes/gramSnapshots`).

import { useEffect, useRef } from "react";
import { type GramMatrixGpuResult, gramMatrixGpu } from "../../../src/gpu/spatial/gramMatrix";
import { alignModeSigns, type ChannelCloud, type CoLocationModes, coLocationModes, type GramParams } from "../../../src/spatial/gram";
import { useAsync } from "../hooks/useAsync";
import { onGpu, useGramSnapshots } from "../modes/gramSnapshots";

export interface CellGram {
  readonly generation: number;
  readonly res: GramMatrixGpuResult;
  /** This result's rasters, laid out as `res.resident`, kept intact while it or its successor is on screen. */
  readonly rasters: GPUBuffer;
  readonly modes: CoLocationModes;
  /** The channels and splat parameters, kept so the null envelope re-splats exactly what this did. */
  readonly channels: readonly ChannelCloud[];
  readonly params: GramParams;
  readonly ms: number;
}

/** `channels` and `params` are compared by identity: memoise them. */
export function useCellGram(channels: readonly ChannelCloud[] | undefined, params: GramParams | undefined, scope?: unknown) {
  const prevModes = useRef<CoLocationModes | null>(null);
  const snaps = useGramSnapshots();
  const state = useAsync<CellGram>(
    (signal) => {
      if (!channels || channels.length < 2 || !params) return undefined;
      return onGpu(async () => {
        if (signal.aborted) throw new Error("superseded");
        const t0 = performance.now();
        const res = await gramMatrixGpu(channels, params);
        if (signal.aborted) throw new Error("superseded");
        // Signs kept consistent with the last result, so a nudged range does not flip the colours.
        const modes = alignModeSigns(coLocationModes(res), prevModes.current);
        prevModes.current = modes;
        const { generation, rasters } = await snaps.take(res);
        return { generation, res, rasters, modes, channels, params, ms: performance.now() - t0 };
      });
    },
    [channels, params],
    scope,
  );
  const gen = state.value?.generation;
  useEffect(() => snaps.show(gen), [gen, snaps]);
  return state;
}
