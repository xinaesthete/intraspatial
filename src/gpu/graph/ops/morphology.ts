// Tier-2 graph node wrapping `morphologyResident` (grid -> grid, separable min/max).
//
// The first of the terrain-derivation primitives (docs/gap-analysis.md, "terrain" gap 1): an
// r=1 opening of the canopy mask stops building perimeters reading as woodland; an r≈25 opening
// of the last-return surface is a bare-earth estimate. Same resident shape as
// `convolveSeparableOp`; `cpuGolden` is the separable CPU reference in `spatial/morphology.ts`,
// which is itself checked bit-exact against a direct 2-D window in the GPU test.
import { type MorphOp, morphologyCpu, morphologyResident } from "../../spatial/morphology";
import type { Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("morphology: input must be a grid");
  return { width: s.width, height: s.height };
}

function opOf(params: Params): MorphOp {
  return params.op as MorphOp;
}

export const morphologyOp: OpType = {
  name: "morphology",
  label: "Morphology (min/max)",
  describe: "Erode, dilate, open or close a grid over a (2r+1)² square — separable local min/max.",
  inputs: [{ name: "grid", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "op", type: "enum", default: "open", options: ["erode", "dilate", "open", "close"] },
    { name: "radius", type: "int", default: 1, min: 1, max: 64 },
  ],
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const { width, height } = gridShape(inField.shape);
    const src = inField.buffer;
    if (!src) throw new Error("morphology: resident op received a non-resident input");
    const dst = await ctx.backend.lease(width * height * 4);
    await morphologyResident(src.buffer, dst.buffer, width, height, params.radius as number, opOf(params));
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const data = morphologyCpu(inputs[0]!.data!, width, height, params.radius as number, opOf(params));
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data }];
  },
};
