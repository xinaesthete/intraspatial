// danceField op — the graph node wrapping the reinterpreted DANCERL swarm (see
// ../../sim/danceField.ts). Its `params` ARE the trait-space the Mutator breeds: each
// numeric strength is a NUMBER trait, each `*On` flag an ENABLE trait (see src/evo —
// `traitSpaceFromParams(DANCE_PARAM_SPECS)`). Nothing about genes/specimens is
// duplicated; the gene-space is *derived* from this declaration, so editing a param's
// range here re-shapes what can be bred.
//
// State convention (see danceField.ts "Field packing"): the swarm is a vec3 `points`
// field of n=2N rows — rows [0,N) positions, [N,2N) velocities — so one feedback node
// carries it across ticks, exactly like the reaction-diffusion complex loop. A second
// `swarm` output (scalar points, [x,y] pairs) gives the top-down scatter preview
// something legible while the dedicated 3D renderer (DancerView) is the real view.
import {
  danceStepsCpu,
  packSwarm,
  swarmXY,
  unpackSwarm,
  type DanceParams,
} from "../../sim/danceField";
import type { ElementType, FieldValue, Shape } from "../handle";
import type { OpType, ParamSpec, Params } from "../op";

const VEC3: ElementType = { kind: "vec", n: 3 };

// A strength trait: a [0,1] number with a step. The enable flag that pairs with it is
// a separate bool ParamSpec (→ an ENABLE trait).
const strength = (name: string, describe: string): ParamSpec => ({ name, type: "number", default: 0.5, min: 0, max: 1, step: 0.01, describe });
const enable = (name: string, def: boolean, describe: string): ParamSpec => ({ name, type: "bool", default: def, describe });

export const DANCE_PARAM_SPECS: ParamSpec[] = [
  strength("attract", "Containment toward the centre (DANCERL constraint box)"),
  enable("attractOn", true, "Enable containment"),
  strength("orbit", "Angular momentum about the centre (DANCERL orbit)"),
  enable("orbitOn", true, "Enable orbit"),
  strength("vortex", "Circular stirring about the y-axis (DANCERL circle)"),
  enable("vortexOn", false, "Enable vortex"),
  strength("solenoid", "Single-coil solenoid field about the y-axis (DANCERL solenoid)"),
  enable("solenoidOn", false, "Enable solenoid"),
  strength("swim", "Outward drift from the centre (DANCERL swim)"),
  enable("swimOn", false, "Enable swim"),
  strength("cohesion", "Pull toward the swarm centroid (DANCERL distance bond)"),
  enable("cohesionOn", true, "Enable cohesion"),
  strength("separation", "Radius repulsion between neighbours (DANCERL collision)"),
  enable("separationOn", true, "Enable separation"),
  { name: "sepRadius", type: "number", default: 1.5, min: 0.3, max: 4, step: 0.1, describe: "Separation neighbour radius (world units)" },
  { name: "damping", type: "number", default: 0.94, min: 0.8, max: 0.99, step: 0.005, describe: "Velocity retained per step (DANCERL LinDamp)" },
  { name: "speedLimit", type: "number", default: 1.5, min: 0.2, max: 4, step: 0.1, describe: "Maximum speed (world units/step)" },
  // steps & dt have no min/max ⇒ they map to FIXED traits (never bred); steps stays
  // adjustable in the composer, dt is a constant of the integrator.
  { name: "steps", type: "int", default: 2, describe: "Sub-steps advanced per tick" },
  { name: "dt", type: "number", default: 1, describe: "Integration step (fixed)" },
];

function pointsShape(s: Shape): { rows: number } {
  if (s.kind !== "points") throw new Error("danceField: state must be a points field");
  return { rows: s.n };
}

function danceParams(params: Params): Partial<DanceParams> {
  const num = (k: string) => params[k] as number;
  const flag = (k: string) => params[k] as boolean;
  return {
    attract: num("attract"), orbit: num("orbit"), vortex: num("vortex"), solenoid: num("solenoid"),
    swim: num("swim"), cohesion: num("cohesion"), separation: num("separation"),
    sepRadius: num("sepRadius"), damping: num("damping"), speedLimit: num("speedLimit"),
    attractOn: flag("attractOn"), orbitOn: flag("orbitOn"), vortexOn: flag("vortexOn"),
    solenoidOn: flag("solenoidOn"), swimOn: flag("swimOn"), cohesionOn: flag("cohesionOn"),
    separationOn: flag("separationOn"),
    dt: num("dt"),
  };
}

function outputs(rows2N: number, data: ArrayLike<number>, steps: number, p: Partial<DanceParams>): FieldValue[] {
  const next = danceStepsCpu(unpackSwarm(data, rows2N), steps, p);
  const stateOut: FieldValue = {
    shape: { kind: "points", n: next.n * 2 },
    dtype: "f32",
    element: VEC3,
    data: packSwarm(next),
  };
  const swarmOut: FieldValue = {
    shape: { kind: "points", n: next.n },
    dtype: "f32",
    data: swarmXY(next),
  };
  return [stateOut, swarmOut];
}

export const danceFieldOp: OpType = {
  name: "danceField",
  label: "Dance field (swarm)",
  describe: "Advance a 3D swarm under a superposition of DANCERL-style force influences.",
  category: "Simulation",
  help: {
    detail:
      "A reinterpretation of Andy Lomas's 1992 DANCERL controller for William Latham's films: " +
      "motion emerges from named force influences (containment, orbit, vortex, solenoid, swim, " +
      "cohesion, separation), each with a strength and an on/off — the traits a Mutator breeds.",
  },
  inputs: [{ name: "state", kind: "points" }],
  outputs: [
    { name: "state", kind: "points", dtype: "f32" },
    { name: "swarm", kind: "points", dtype: "f32" },
  ],
  params: DANCE_PARAM_SPECS,
  inferShapes(inputs) {
    const { rows } = pointsShape(inputs[0]!);
    return [
      { kind: "points", n: rows }, // state: n = 2N (pos ‖ vel)
      { kind: "points", n: rows >> 1 }, // swarm: n = N positions
    ];
  },
  inferElements(inputs) {
    if (inputs[0]!.kind !== "vec" || inputs[0]!.n !== 3) {
      throw new Error("danceField: state must be a vec3 points field (pos ‖ vel)");
    }
    return [VEC3, { kind: "scalar" }];
  },
  async execute(_ctx, inputs, params) {
    const { rows } = pointsShape(inputs[0]!.shape);
    return outputs(rows, inputs[0]!.data!, params.steps as number, danceParams(params));
  },
  cpuGolden(inputs, params) {
    const { rows } = pointsShape(inputs[0]!.shape);
    return outputs(rows, inputs[0]!.data!, params.steps as number, danceParams(params));
  },
};
