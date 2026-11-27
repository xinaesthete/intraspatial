// The painted modes' values at the place under the pointer, for marking it on the legend's
// histograms. One small probe (the K channel densities there, `probeChannels`), projected on the
// host the way `WandReadout` does: nothing per pixel leaves the GPU.
//
// Pointer moves outrun probes, so they coalesce: one runs at a time and only the latest place is
// read next (see `coalescing`). The last values stand while the next probe is on its way.

import { useEffect, useState } from "react";
import { probeChannels, standardise } from "../../../src/gpu/spatial/gramTerrain";
import { coalescing, type WandSource } from "./useWand";

/** A place in raster pixels (pixel centres at whole numbers; fractions interpolate, as the map does). */
export interface RasterPoint {
  readonly col: number;
  readonly row: number;
}

export function useModeValuesAt(source: WandSource | undefined, at: RasterPoint | undefined): readonly number[] | undefined {
  const [got, setGot] = useState<{ source: WandSource; values: number[] }>();
  const [probe] = useState(() =>
    coalescing(async ({ src, p }: { src: WandSource; p: RasterPoint }, live) => {
      const raw = await probeChannels(src.res, p.col, p.row, src.rasters);
      if (!live()) return;
      const z = standardise(raw, src.mean, src.sd);
      const K = src.res.labels.length;
      const values = [0, 1, 2]
        .filter((k) => k < K)
        .map((k) => {
          let y = 0;
          for (let a = 0; a < K; a++) y += (z[a] ?? 0) * (src.vectors[k * K + a] ?? 0);
          return y;
        });
      setGot({ source: src, values });
    }),
  );
  useEffect(() => {
    if (source && at) probe.schedule({ src: source, p: at });
    else probe.cancel();
  }, [source, at, probe]);
  return at && got && got.source === source ? got.values : undefined;
}
