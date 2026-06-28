// Tier-1 graph node wrapping `kthNeighborDistanceGpu` — each point's distance to
// its k-th nearest neighbour (the local density estimate ρ_i / local bandwidth).
// Output is an n×1 matrix, the per-point bandwidth that feeds fuzzyAdjacencyAdaptive.
import { kthNeighborDistanceGpu } from "../../spatial/kthNeighborDistance";
import { unpackPoints } from "../handle";
import type { Shape } from "../handle";
import type { OpType } from "../op";

function pointCount(s: Shape): number {
  if (s.kind !== "points") throw new Error("kthNeighborDistance: input must be points");
  return s.n;
}

export const kthNeighborDistanceOp: OpType = {
  name: "kthNeighborDistance",
  label: "k-th neighbour distance",
  describe: "Per-point distance to its k-th nearest neighbour (local bandwidth ρ).",
  inputs: [{ name: "points", kind: "points" }],
  outputs: [{ name: "rho", kind: "matrix", dtype: "f32" }],
  params: [{ name: "k", type: "int", default: 4, min: 1, max: 32, describe: "neighbour rank (1..32, < N)" }],
  inferShapes(inputs) {
    return [{ kind: "matrix", rows: pointCount(inputs[0]!), cols: 1 }];
  },
  async execute(_ctx, inputs, params) {
    const { xs, ys, n } = unpackPoints(inputs[0]!);
    const rho = await kthNeighborDistanceGpu(xs, ys, params.k as number);
    return [{ shape: { kind: "matrix", rows: n, cols: 1 }, dtype: "f32", data: rho }];
  },
  cpuGolden(inputs, params) {
    const { xs, ys, n } = unpackPoints(inputs[0]!);
    const k = params.k as number;
    const rho = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const ds: number[] = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        ds.push(Math.hypot(xs[j]! - xs[i]!, ys[j]! - ys[i]!));
      }
      ds.sort((a, b) => a - b);
      rho[i] = ds[Math.min(k, ds.length) - 1] ?? 0;
    }
    return [{ shape: { kind: "matrix", rows: n, cols: 1 }, dtype: "f32", data: rho }];
  },
};
