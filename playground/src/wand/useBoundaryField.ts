// PROTOTYPE (docs/wand-metric-and-regions.md, experiment 1): the boundary field of a Gram result in the
// PPCA metric, and of a broader scale's result when the patch is judged at two scales.

import { useEffect, useRef, useState } from "react";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";
import type { ModeBasis } from "../../../src/gpu/spatial/gramModes";
import type { CoLocationModes } from "../../../src/spatial/gram";
import { type BoundaryField, boundaryStrength, ppcaMetric } from "./boundaryStrength";
import type { WandSettings } from "./useWandSettings";

/** What the field is computed from: a Gram result, its raster snapshot and its modes. */
export interface GramView {
  readonly res: GramMatrixGpuResult;
  readonly rasters: GPUBuffer;
  readonly radius: number;
  /** Identifies the result: a field is shown only with the result it was computed from. */
  readonly generation: number;
  readonly modes: CoLocationModes;
}

export interface BoundaryNow {
  readonly field: BoundaryField;
  /** The noise floor the modes after q are counted at. */
  readonly sigma2: number;
  readonly q: number;
  /** Bumped per computation, so a layer reuploading the shared buffer knows it changed. */
  readonly version: number;
  readonly generation: number;
}

/**
 * The boundary field of `g` while `on`, measured with `modes` (a lock's, or g's own) and painted with
 * `basis`'s standardisation when one is locked; and of `broad` when given. Undefined until the field for
 * the current result is ready.
 */
export function useBoundaryField(
  on: boolean,
  g: GramView | undefined,
  modes: CoLocationModes | undefined,
  basis: ModeBasis | undefined,
  broad: GramView | undefined,
  s: Pick<WandSettings, "q" | "w" | "residualSmooth">,
) {
  const [bounds, setBounds] = useState<BoundaryNow>();
  const chain = useRef(Promise.resolve());
  const version = useRef(0);
  useEffect(() => {
    if (!on || !g || !modes) return;
    const K = g.res.labels.length;
    const q = Math.min(s.q, K);
    const { A, sigma2 } = ppcaMetric(modes.vectors, modes.values, K, q, s.w);
    const sd = basis?.sd ?? g.res.resident.sd;
    const mean = basis?.mean ?? g.res.resident.mean;
    // One at a time: the strength buffers are shared between calls.
    chain.current = chain.current
      .then(() => boundaryStrength(g.res, g.rasters, mean, sd, A, g.radius, q, s.residualSmooth))
      .then((field) => setBounds({ field, sigma2, q, version: ++version.current, generation: g.generation }))
      .catch((err: unknown) => console.error("boundary strength", err));
  }, [on, g, modes, basis, s.q, s.w, s.residualSmooth]);
  const [boundsB, setBoundsB] = useState<{ field: BoundaryField; generation: number }>();
  useEffect(() => {
    if (!broad) return;
    const K = broad.res.labels.length;
    const q = Math.min(s.q, K);
    const { A } = ppcaMetric(broad.modes.vectors, broad.modes.values, K, q, s.w);
    chain.current = chain.current
      .then(() =>
        boundaryStrength(broad.res, broad.rasters, broad.res.resident.mean, broad.res.resident.sd, A, broad.radius, q, 0, "broad"),
      )
      .then((field) => setBoundsB({ field, generation: broad.generation }))
      .catch((err: unknown) => console.error("boundary strength (broad)", err));
  }, [broad, s.q, s.w]);
  return {
    main: on && bounds && g && bounds.generation === g.generation ? bounds : undefined,
    broad: broad && boundsB && boundsB.generation === broad.generation ? boundsB.field : undefined,
  };
}
