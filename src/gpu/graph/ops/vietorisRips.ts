// Sink node wrapping the CPU Vietoris–Rips persistence reducer. Takes a symmetric
// n×n distance matrix (e.g. the membership-sweep distance from membershipToDistance,
// or CkNN-rescaled distance) and produces a persistence diagram (H0 components, H1
// loops). The reduction is inherently sequential, so it stays on the CPU — the GPU's
// job ended at the distance matrix (see docs/concepts/fuzzy-tda.md).

import type { PersistenceResult } from "../../../spatial/persistence";
import { vietorisRipsPersistence } from "../../../spatial/persistence";
import type { Shape } from "../handle";
import type { OpType } from "../op";

function squareN(s: Shape): number {
  if (s.kind !== "matrix" || s.rows !== s.cols) throw new Error("vietorisRips: input must be a square distance matrix");
  return s.rows;
}

export const vietorisRipsOp: OpType = {
  name: "vietorisRipsPersistence",
  label: "Vietoris–Rips persistence",
  describe: "Persistent homology (H0/H1) of a distance matrix -> a persistence diagram.",
  inputs: [{ name: "distance", kind: "matrix" }],
  outputs: [{ name: "diagram", kind: "opaque", dtype: "f32" }],
  params: [{ name: "maxScale", type: "number", default: 0, min: 0, max: 100, step: 0.01, describe: "cap simplex filtration (0 = auto)" }],
  inferShapes() {
    return [{ kind: "opaque", name: "persistence" }];
  },
  async execute(_ctx, inputs, params) {
    const n = squareN(inputs[0]!.shape);
    const maxScale = params.maxScale as number;
    const result: PersistenceResult = vietorisRipsPersistence(inputs[0]!.data!, n, maxScale > 0 ? { maxScale } : {});
    return [{ shape: { kind: "opaque", name: "persistence" }, dtype: "f32", payload: result }];
  },
  sanity() {
    return true; // CPU op; nothing to validate against
  },
};
