// Tier-1 graph node wrapping `getisOrdGpu` (grid -> Gi* z-score grid).
import { getisOrdGpu } from "../../spatial/getisOrd";
import type { Shape } from "../handle";
import type { OpType } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("getisOrd: input must be a grid");
  return { width: s.width, height: s.height };
}

export const getisOrdOp: OpType = {
  name: "getisOrd",
  label: "Getis-Ord Gi*",
  describe: "Local hotspot z-scores over a box neighbourhood (grid -> grid).",
  inputs: [{ name: "grid", kind: "grid" }],
  outputs: [{ name: "z", kind: "grid", dtype: "f32" }],
  params: [{ name: "radius", type: "int", default: 2, min: 1, max: 32, describe: "box neighbourhood radius (cells)" }],
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  async execute(_ctx, inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const { z } = await getisOrdGpu(inputs[0]!.data!, width, height, { radius: params.radius as number });
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data: z }];
  },
};
