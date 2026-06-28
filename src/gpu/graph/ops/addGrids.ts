// Pointwise weighted sum of two grids: out = a*wa + b*wb. A cheap CPU op (the work
// is trivially memory-bound and small in the demos) — it shows the runtime handles
// multi-input nodes and ops that don't need the GPU at all, and gives the executor
// a genuine fork-join (diamond) to dedup over.
import type { Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("addGrids: inputs must be grids");
  return { width: s.width, height: s.height };
}

function combine(a: ArrayLike<number>, b: ArrayLike<number>, params: Params): Float32Array {
  const wa = (params.wa as number) ?? 1, wb = (params.wb as number) ?? 1;
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i]! * wa + b[i]! * wb;
  return out;
}

export const addGridsOp: OpType = {
  name: "addGrids",
  label: "Combine grids",
  describe: "Pointwise weighted sum of two same-shape grids (a·wa + b·wb).",
  inputs: [
    { name: "a", kind: "grid" },
    { name: "b", kind: "grid" },
  ],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "wa", type: "number", default: 1, min: -4, max: 4, step: 0.1 },
    { name: "wb", type: "number", default: 1, min: -4, max: 4, step: 0.1 },
  ],
  inferShapes(inputs) {
    const a = gridShape(inputs[0]!), b = gridShape(inputs[1]!);
    if (a.width !== b.width || a.height !== b.height) throw new Error("addGrids: grid shapes differ");
    return [inputs[0]!];
  },
  async execute(_ctx, inputs, params) {
    return [{ shape: inputs[0]!.shape, dtype: "f32", data: combine(inputs[0]!.data!, inputs[1]!.data!, params) }];
  },
  cpuGolden(inputs, params) {
    return [{ shape: inputs[0]!.shape, dtype: "f32", data: combine(inputs[0]!.data!, inputs[1]!.data!, params) }];
  },
};
