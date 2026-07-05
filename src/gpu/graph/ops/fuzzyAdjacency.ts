// Tier-1 graph node wrapping `fuzzyAdjacencyGpu` — the global-σ fuzzy (kernel-
// weighted) adjacency matrix, the weighted 1-skeleton for fuzzier TDA.
import { fuzzyAdjacencyGpu } from "../../spatial/fuzzyAdjacency";
import type { Shape } from "../handle";
import { unpackPoints } from "../handle";
import type { OpType } from "../op";

function pointCount(s: Shape): number {
  if (s.kind !== "points") throw new Error("fuzzyAdjacency: input must be points");
  return s.n;
}

export const fuzzyAdjacencyOp: OpType = {
  name: "fuzzyAdjacency",
  label: "Fuzzy adjacency",
  describe: "Kernel-weighted membership μ_ij = exp(-d²/2σ²) (points -> n×n matrix).",
  inputs: [{ name: "points", kind: "points" }],
  outputs: [{ name: "membership", kind: "matrix", dtype: "f32" }],
  params: [
    { name: "sigma", type: "number", default: 2, min: 0.05, max: 40, step: 0.05, describe: "bandwidth σ (world units)" },
    { name: "radiusSigma", type: "number", default: 3, min: 1, max: 8, step: 0.5 },
  ],
  inferShapes(inputs) {
    const n = pointCount(inputs[0]!);
    return [{ kind: "matrix", rows: n, cols: n }];
  },
  async execute(_ctx, inputs, params) {
    const { xs, ys, n } = unpackPoints(inputs[0]!);
    const { membership } = await fuzzyAdjacencyGpu(xs, ys, {
      sigma: params.sigma as number,
      radiusSigma: params.radiusSigma as number,
    });
    return [{ shape: { kind: "matrix", rows: n, cols: n }, dtype: "f32", data: membership }];
  },
};
