// Single complex-field Gray–Scott step (ADR-0004). The (U,V) pair is one complex
// signal: re = U, im = V. Carrying it as a single `complex` field collapses the
// two-feedback-node workaround into one feedback node and one edge, with the identical
// Gray–Scott kernel underneath. A complex grid packs interleaved [u0,v0,u1,v1,...], so
// deinterleave → step → repack.
//
// Kept in its own file (not in reactionDiffusion.ts) so the base RD op's module stays
// byte-identical to baseline and the always-eager registry doesn't pull `elementMath`
// into every GPU test fork — see registerElementOps in ./index and the ADR-0002/0003
// Dawn-on-Node fragility notes.
import { grayScottStepCpu, grayScottStepsGpu } from "../../sim/reactionDiffusion";
import type { GrayScottParams, GrayScottState } from "../../sim/reactionDiffusion";
import type { ElementType, FieldValue, Shape } from "../handle";
import type { OpType, ParamSpec, Params } from "../op";
import { extractLane, packComplex } from "../elementMath";

const COMPLEX: ElementType = { kind: "complex" };

const RD_PARAM_SPECS: ParamSpec[] = [
  { name: "steps", type: "int", default: 20, min: 1, max: 500, describe: "Euler steps per pull" },
  { name: "du", type: "number", default: 0.16, min: 0, max: 1, step: 0.01 },
  { name: "dv", type: "number", default: 0.08, min: 0, max: 1, step: 0.01 },
  { name: "feed", type: "number", default: 0.06, min: 0, max: 0.1, step: 0.001 },
  { name: "kill", type: "number", default: 0.062, min: 0, max: 0.1, step: 0.001 },
  { name: "dt", type: "number", default: 1, min: 0.1, max: 1.5, step: 0.1 },
];

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("reactionDiffusionComplex: state must be a grid");
  return { width: s.width, height: s.height };
}

function rdParams(params: Params): GrayScottParams {
  return {
    du: params.du as number,
    dv: params.dv as number,
    feed: params.feed as number,
    kill: params.kill as number,
    dt: params.dt as number,
  };
}

function complexStateOf(z: FieldValue): GrayScottState {
  const { width, height } = gridShape(z.shape);
  return { u: extractLane(COMPLEX, z.data!, 0), v: extractLane(COMPLEX, z.data!, 1), width, height };
}

function complexOutputs(next: GrayScottState): FieldValue[] {
  const shape: Shape = { kind: "grid", width: next.width, height: next.height };
  return [{ shape, dtype: "f32", element: COMPLEX, data: packComplex(next.u, next.v) }];
}

export const reactionDiffusionComplexOp: OpType = {
  name: "reactionDiffusionComplex",
  label: "Reaction–diffusion (complex)",
  describe: "Advance a Gray–Scott state carried as one complex field (re = U, im = V) by N steps.",
  inputs: [{ name: "state", kind: "grid" }],
  outputs: [{ name: "state", kind: "grid", dtype: "f32" }],
  params: RD_PARAM_SPECS,
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  inferElements(inputs) {
    if (inputs[0]!.kind !== "complex") {
      throw new Error("reactionDiffusionComplex: state must be a complex field (re = U, im = V)");
    }
    return [COMPLEX];
  },
  async execute(_ctx, inputs, params) {
    const next = await grayScottStepsGpu(complexStateOf(inputs[0]!), params.steps as number, rdParams(params));
    return complexOutputs(next);
  },
  cpuGolden(inputs, params) {
    let state = complexStateOf(inputs[0]!);
    const steps = params.steps as number;
    const p = rdParams(params);
    for (let s = 0; s < steps; s++) state = grayScottStepCpu(state, p);
    return complexOutputs(state);
  },
};
