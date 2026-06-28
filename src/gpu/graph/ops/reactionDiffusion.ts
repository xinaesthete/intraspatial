// Graph node for the Gray–Scott reaction–diffusion step — the runtime's first
// iterative op. Takes the two state grids (U, V) and returns the advanced pair after
// `steps` explicit-Euler steps. In a graph, feeding the outputs back into a new
// step node (or pulling repeatedly) drives the simulation forward; the executor's
// per-stage submits are exactly the per-step boundary the integrator needs.
import { grayScottStepCpu, grayScottStepsGpu } from "../../sim/reactionDiffusion";
import type { GrayScottParams, GrayScottState } from "../../sim/reactionDiffusion";
import type { FieldValue, Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("reactionDiffusionStep: inputs must be grids");
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

function stateOf(u: FieldValue, v: FieldValue): GrayScottState {
  const { width, height } = gridShape(u.shape);
  return { u: Float32Array.from(u.data!), v: Float32Array.from(v.data!), width, height };
}

function outputs(next: GrayScottState): FieldValue[] {
  const shape: Shape = { kind: "grid", width: next.width, height: next.height };
  return [
    { shape, dtype: "f32", data: next.u },
    { shape, dtype: "f32", data: next.v },
  ];
}

export const reactionDiffusionStepOp: OpType = {
  name: "reactionDiffusionStep",
  label: "Reaction–diffusion step",
  describe: "Advance a Gray–Scott (U,V) state by N explicit-Euler steps.",
  inputs: [
    { name: "u", kind: "grid" },
    { name: "v", kind: "grid" },
  ],
  outputs: [
    { name: "u", kind: "grid", dtype: "f32" },
    { name: "v", kind: "grid", dtype: "f32" },
  ],
  params: [
    { name: "steps", type: "int", default: 20, min: 1, max: 500, describe: "Euler steps per pull" },
    { name: "du", type: "number", default: 0.16, min: 0, max: 1, step: 0.01 },
    { name: "dv", type: "number", default: 0.08, min: 0, max: 1, step: 0.01 },
    { name: "feed", type: "number", default: 0.06, min: 0, max: 0.1, step: 0.001 },
    { name: "kill", type: "number", default: 0.062, min: 0, max: 0.1, step: 0.001 },
    { name: "dt", type: "number", default: 1, min: 0.1, max: 1.5, step: 0.1 },
  ],
  inferShapes(inputs) {
    return [inputs[0]!, inputs[0]!];
  },
  async execute(_ctx, inputs, params) {
    const next = await grayScottStepsGpu(stateOf(inputs[0]!, inputs[1]!), params.steps as number, rdParams(params));
    return outputs(next);
  },
  cpuGolden(inputs, params) {
    let state = stateOf(inputs[0]!, inputs[1]!);
    const steps = params.steps as number;
    const p = rdParams(params);
    for (let s = 0; s < steps; s++) state = grayScottStepCpu(state, p);
    return outputs(state);
  },
};
