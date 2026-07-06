// Graph node for UMAP-style adaptive fuzzy adjacency: points + a per-point
// bandwidth ρ (from kthNeighborDistance) -> n×n symmetrised membership.
import { fuzzyAdjacencyAdaptiveFromRhoGpu } from "../../spatial/fuzzyAdjacencyAdaptive";
import type { Shape } from "../handle";
import { unpackPoints } from "../handle";
import type { OpType } from "../op";

function pointCount(s: Shape): number {
  if (s.kind !== "points") throw new Error("fuzzyAdjacencyAdaptive: 'points' input must be points");
  return s.n;
}

export const fuzzyAdjacencyAdaptiveOp: OpType = {
  name: "fuzzyAdjacencyAdaptive",
  label: "Fuzzy adjacency (adaptive)",
  describe: "Per-point-bandwidth fuzzy union μ_ij = a + b − a·b (points + ρ -> n×n).",
  inputs: [
    { name: "points", kind: "points" },
    { name: "rho", kind: "matrix" },
  ],
  outputs: [{ name: "membership", kind: "matrix", dtype: "f32" }],
  params: [
    { name: "scale", type: "number", default: 1, min: 0.1, max: 8, step: 0.1, describe: "σ_i = scale·ρ_i" },
    { name: "minSigma", type: "number", default: 1e-6, min: 1e-9, max: 1, step: 1e-6 },
  ],
  inferShapes(inputs) {
    const n = pointCount(inputs[0]!);
    return [{ kind: "matrix", rows: n, cols: n }];
  },
  async execute(_ctx, inputs, params) {
    const { xs, ys, n } = unpackPoints(inputs[0]!);
    const rho = inputs[1]!.data!;
    const { membership } = await fuzzyAdjacencyAdaptiveFromRhoGpu(xs, ys, rho, {
      scale: params.scale as number,
      minSigma: params.minSigma as number,
    });
    return [{ shape: { kind: "matrix", rows: n, cols: n }, dtype: "f32", data: membership }];
  },
};
