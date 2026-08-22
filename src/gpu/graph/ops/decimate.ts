// Tier-2 graph node wrapping `decimateResident` (grid -> smaller grid, block mean/min/max).
//
// Terrain-derivation primitive 2 (docs/gap-analysis.md): the first op that CHANGES GRID SIZE, so
// it is also the first op that must *derive* a placement rather than pass one through (ADR-0018).
//
// Placement convention. This repo's array space is **corner-indexed**: cell (i, j) occupies the
// array-space box [i, i+1) × [j, j+1) and `worldFromArray` maps array coordinates to world, so
// `origin` is the world position of the grid's outer corner and a cell's world *centre* is
// `origin + (i + ½)·axes[0] + (j + ½)·axes[1]` (the same convention `splatDensity`'s
// `gridWorldFromArray` constructs: cell (0,0) at bbox.min, one cell step = one cellSize). Output
// cell (I, J) covers input cells [I·f, I·f+f) × [J·f, J·f+f), i.e. array-space [I·f, (I+1)·f) —
// so the derived placement is
//
//     worldFromArray_out = worldFromArray_in · scale(f, f, 1)
//
// origin unchanged, the two in-plane axes multiplied by `factor`. The output cell's centre,
// `origin + (I+½)·f·axes[0] + …`, is exactly the centroid of its block's input-cell centres
// (checked in the test). A partial edge block keeps the nominal full-block footprint — its
// placement centre sits past the input edge — which is the honest geometry of a block that was
// reduced over fewer cells; the value itself is the mean/min/max of the cells present.
import type { Vec3 } from "../../../coords";
import { type DecimateMode, decimateCpu, decimatedSize, decimateResident } from "../../spatial/decimate";
import type { ResolvedPlacement, Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("decimate: input must be a grid");
  return { width: s.width, height: s.height };
}

function factorOf(params: Params): number {
  const f = params.factor as number;
  if (!Number.isInteger(f) || f < 2 || f > 64) throw new Error(`decimate: factor must be an integer in 2..64 (got ${f})`);
  return f;
}

function modeOf(params: Params): DecimateMode {
  return params.mode as DecimateMode;
}

/** `worldFromArray · scale(f, f, 1)`: the placement of a grid decimated by `f` (see header). */
export function decimatedPlacement(p: ResolvedPlacement, factor: number): ResolvedPlacement {
  const a = p.worldFromArray;
  const scale = (v: Vec3): Vec3 => [v[0] * factor, v[1] * factor, v[2] * factor];
  return { system: p.system, worldFromArray: { origin: a.origin, axes: [scale(a.axes[0]), scale(a.axes[1]), a.axes[2]] } };
}

export const decimateOp: OpType = {
  name: "decimate",
  label: "Decimate (block reduce)",
  describe: "Collapse every factor×factor block to one cell (mean/min/max); output is ceil(w/f)×ceil(h/f).",
  inputs: [{ name: "grid", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "factor", type: "int", default: 2, min: 2, max: 64 },
    { name: "mode", type: "enum", default: "mean", options: ["mean", "min", "max"] },
  ],
  inferShapes(inputs, params) {
    const { width, height } = gridShape(inputs[0]!);
    return [{ kind: "grid", ...decimatedSize(width, height, factorOf(params)) }];
  },
  inferPlacement(inputs, params) {
    const pl = inputs[0];
    return [pl === undefined ? undefined : decimatedPlacement(pl, factorOf(params))];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const { width, height } = gridShape(inField.shape);
    const src = inField.buffer;
    if (!src) throw new Error("decimate: resident op received a non-resident input");
    const factor = factorOf(params);
    const out = decimatedSize(width, height, factor);
    const dst = await ctx.backend.lease(out.width * out.height * 4);
    await decimateResident(src.buffer, dst.buffer, width, height, factor, modeOf(params));
    return [{ shape: { kind: "grid", ...out }, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const factor = factorOf(params);
    const data = decimateCpu(inputs[0]!.data!, width, height, factor, modeOf(params));
    return [{ shape: { kind: "grid", ...decimatedSize(width, height, factor) }, dtype: "f32", data }];
  },
};
