// Convert a fuzzy membership matrix μ_ij ∈ [0,1] to a filtration distance matrix so
// the persistence reducer can sweep it. Sweeping membership 1 -> 0 is the same as
// sweeping distance 0 -> up, so a strong edge (μ≈1) must map to a small distance:
//
//   oneMinusMu:  d_ij = 1 − μ_ij            (bounded [0,1], simple)
//   negLog:      d_ij = −log(max(μ_ij, ε))  (UMAP cross-entropy geometry; unbounded)
//
// The diagonal is forced to 0 (a vertex is at zero distance from itself), which the
// Vietoris–Rips reducer expects. CPU op — trivially memory-bound at the small N the
// reducer handles.
import type { Shape } from "../handle";
import type { OpType, Params } from "../op";

function squareN(s: Shape): number {
  if (s.kind !== "matrix" || s.rows !== s.cols) throw new Error("membershipToDistance: input must be a square matrix");
  return s.rows;
}

function convert(mu: ArrayLike<number>, n: number, params: Params): Float32Array {
  const mode = params.mode as string;
  const eps = (params.eps as number) || 1e-6;
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { out[i * n + j] = 0; continue; }
      const m = mu[i * n + j]!;
      out[i * n + j] = mode === "negLog" ? -Math.log(Math.max(m, eps)) : 1 - m;
    }
  }
  return out;
}

export const membershipToDistanceOp: OpType = {
  name: "membershipToDistance",
  label: "Membership -> distance",
  describe: "Map a fuzzy membership matrix to a filtration distance for persistence.",
  inputs: [{ name: "membership", kind: "matrix" }],
  outputs: [{ name: "distance", kind: "matrix", dtype: "f32" }],
  params: [
    { name: "mode", type: "enum", default: "oneMinusMu", options: ["oneMinusMu", "negLog"] },
    { name: "eps", type: "number", default: 1e-6, min: 1e-9, max: 0.1, step: 1e-6, describe: "floor for negLog" },
  ],
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  async execute(_ctx, inputs, params) {
    const n = squareN(inputs[0]!.shape);
    return [{ shape: inputs[0]!.shape, dtype: "f32", data: convert(inputs[0]!.data!, n, params) }];
  },
  cpuGolden(inputs, params) {
    const n = squareN(inputs[0]!.shape);
    return [{ shape: inputs[0]!.shape, dtype: "f32", data: convert(inputs[0]!.data!, n, params) }];
  },
};
