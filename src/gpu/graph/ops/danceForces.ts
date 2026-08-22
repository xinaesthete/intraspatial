// The dance force building blocks — composable op-graph nodes, each a thin wrapper over
// the pure force math in src/gpu/sim/forces.ts. A swarm is carried as ONE body field
// (points vec3, n = 5·N — blocks pos|vel|accel|angPos|angVel; see sim/body.ts); `bodyTap`
// exposes pos/vel as plain vec3 point fields so the force ops stay decoupled from that
// layout. Each force op outputs a per-agent acceleration field (points vec3, n=N); chain
// them through the element-pack `addFields` and feed the sum to `integrate`, which closes
// a single feedback loop.
//
// Every op's ParamSpec[] is the breedable trait set (src/evo). CPU Tier-1 for now (the
// GPU artefact mirrors this math in TSL); validated at the boundary, checked access, no `!`.

import { BODY_BLOCK_COUNT, type IntegrateParams, integrateBody, readBodyState, tapBlock, writeBodyState } from "../../sim/body";
import { figureAt, figureTargetVel, partnerIndex } from "../../sim/figures";
import {
  cohereForce,
  constrainForce,
  orbitForce,
  partnerOrbitForce,
  separateForce,
  solenoidForce,
  springForce,
  swimForce,
  vortexForce,
} from "../../sim/forces";
import { readVec3, unpack, type Vec3, vec3, writeVec3 } from "../../sim/vec3";
import type { ElementType, FieldValue, Shape } from "../handle";
import type { OpType, ParamSpec, Params } from "../op";
import { hasOp, registerOp } from "../registry";

const VEC3: ElementType = { kind: "vec", n: 3 };
const SCALAR: ElementType = { kind: "scalar" };
const ZERO: Vec3 = vec3(0, 0, 0);

function pointsN(shape: Shape, who: string): number {
  if (shape.kind !== "points") throw new Error(`${who}: expected a points field`);
  return shape.n;
}

function requireVec3(el: ElementType | undefined, who: string): void {
  if (!el || el.kind !== "vec" || el.n !== 3) throw new Error(`${who}: expected a vec3 points field`);
}

function requireData(v: FieldValue, who: string): Float32Array {
  const d = v.data;
  if (!d) throw new Error(`${who}: field has no data`);
  return d instanceof Float32Array ? d : Float32Array.from(d);
}

const num = (p: Params, k: string): number => {
  const v = p[k];
  if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`param "${k}" must be a finite number`);
  return v;
};

// ── Force-op factory ────────────────────────────────────────────────────────────────

interface ForceKernel {
  /** Force on agent `i` given the position buffer, optional velocity buffer, count, params. */
  (i: number, pos: Float32Array, vel: Float32Array | null, n: number, params: Params): Vec3;
}

interface ForceOpConfig {
  name: string;
  label: string;
  describe: string;
  /** Whether the op also takes a velocity input. */
  needsVel: boolean;
  params: ParamSpec[];
  kernel: ForceKernel;
}

function makeForceOp(cfg: ForceOpConfig): OpType {
  const inputs = cfg.needsVel
    ? [
        { name: "pos", kind: "points" as const },
        { name: "vel", kind: "points" as const },
      ]
    : [{ name: "pos", kind: "points" as const }];

  return {
    name: cfg.name,
    label: cfg.label,
    describe: cfg.describe,
    category: "Dance forces",
    inputs,
    outputs: [{ name: "force", kind: "points", dtype: "f32" }],
    params: cfg.params,
    inferShapes(inShapes) {
      return [{ kind: "points", n: pointsN(inShapes[0]!, cfg.name) }];
    },
    inferElements(inEls) {
      requireVec3(inEls[0], `${cfg.name}.pos`);
      if (cfg.needsVel) requireVec3(inEls[1], `${cfg.name}.vel`);
      return [VEC3];
    },
    async execute(_ctx, ins, params) {
      return computeForce(cfg, ins, params);
    },
    cpuGolden(ins, params) {
      return computeForce(cfg, ins, params);
    },
  };
}

function computeForce(cfg: ForceOpConfig, ins: FieldValue[], params: Params): FieldValue[] {
  const posV = ins[0];
  if (!posV) throw new Error(`${cfg.name}: missing pos input`);
  const n = pointsN(posV.shape, cfg.name);
  const pos = requireData(posV, `${cfg.name}.pos`);
  let vel: Float32Array | null = null;
  if (cfg.needsVel) {
    const velV = ins[1];
    if (!velV) throw new Error(`${cfg.name}: missing vel input`);
    vel = requireData(velV, `${cfg.name}.vel`);
  }
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) writeVec3(out, i, cfg.kernel(i, pos, vel, n, params));
  return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC3, data: out }];
}

const strengthSpec = (def = 0.5): ParamSpec => ({ name: "strength", type: "number", default: def, min: 0, max: 1, step: 0.01 });

// ── The forces ──────────────────────────────────────────────────────────────────────

const constrainOp = makeForceOp({
  name: "constrain",
  label: "Constrain",
  describe: "Containment toward the centre (DANCERL constraint box).",
  needsVel: false,
  params: [strengthSpec(0.5), { name: "power", type: "number", default: 3, min: 1, max: 4, step: 0.5 }],
  kernel: (i, pos, _v, _n, p) => constrainForce(readVec3(pos, i), num(p, "strength"), num(p, "power")),
});

const swimOp = makeForceOp({
  name: "swim",
  label: "Swim",
  describe: "Outward drift from the centre (DANCERL swim).",
  needsVel: false,
  params: [strengthSpec(0.3)],
  kernel: (i, pos, _v, _n, p) => swimForce(readVec3(pos, i), num(p, "strength")),
});

const vortexOp = makeForceOp({
  name: "vortex",
  label: "Vortex",
  describe: "Circular stirring about the y-axis (DANCERL circle).",
  needsVel: false,
  params: [strengthSpec(0.5)],
  kernel: (i, pos, _v, _n, p) => vortexForce(readVec3(pos, i), num(p, "strength")),
});

const solenoidOp = makeForceOp({
  name: "solenoid",
  label: "Solenoid",
  describe: "Single-coil solenoid field about the y-axis (DANCERL solenoid).",
  needsVel: false,
  params: [strengthSpec(0.4), { name: "coilRadius", type: "number", default: 2.5, min: 0.5, max: 6, step: 0.1 }],
  kernel: (i, pos, _v, _n, p) => solenoidForce(readVec3(pos, i), num(p, "strength"), num(p, "coilRadius")),
});

const orbitOp = makeForceOp({
  name: "orbit",
  label: "Orbit",
  describe: "Orbital acceleration about the centre (DANCERL orbit, (p×v)×p).",
  needsVel: true,
  params: [strengthSpec(0.5)],
  kernel: (i, pos, vel, _n, p) => orbitForce(readVec3(pos, i), readVec3(vel ?? pos, i), num(p, "strength")),
});

const cohereOp = makeForceOp({
  name: "cohere",
  label: "Cohere",
  describe: "Pull toward the centroid of neighbours within radius (DANCERL bond).",
  needsVel: false,
  params: [strengthSpec(0.5), { name: "radius", type: "number", default: 3, min: 0.3, max: 20, step: 0.1 }],
  kernel: (i, pos, _v, n, p) => cohereForce(i, pos, n, num(p, "strength"), num(p, "radius")),
});

const separateOp = makeForceOp({
  name: "separate",
  label: "Separate",
  describe: "Radius repulsion between near neighbours (DANCERL collision).",
  needsVel: false,
  params: [strengthSpec(0.5), { name: "radius", type: "number", default: 1.5, min: 0.2, max: 8, step: 0.1 }],
  kernel: (i, pos, _v, n, p) => separateForce(i, pos, n, num(p, "strength"), num(p, "radius")),
});

const springOp = makeForceOp({
  name: "spring",
  label: "Spring",
  describe: "Bonds pulling neighbours toward an equilibrium distance (DANCERL distance).",
  needsVel: false,
  params: [
    strengthSpec(0.4),
    { name: "restLength", type: "number", default: 2, min: 0.2, max: 10, step: 0.1 },
    { name: "radius", type: "number", default: 4, min: 0.3, max: 20, step: 0.1 },
  ],
  kernel: (i, pos, _v, n, p) => springForce(i, pos, n, num(p, "strength"), num(p, "restLength"), num(p, "radius")),
});

// ── bodyTap: expose pos/vel from the body field ──────────────────────────────────────

const bodyTapOp: OpType = {
  name: "bodyTap",
  label: "Body tap (pos/vel)",
  describe: "Extract the position and velocity fields from a swarm body, for the force ops to read.",
  category: "Dance forces",
  inputs: [{ name: "body", kind: "points" }],
  outputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "vel", kind: "points", dtype: "f32" },
  ],
  params: [],
  inferShapes(inShapes) {
    const n = pointsN(inShapes[0]!, "bodyTap") / BODY_BLOCK_COUNT;
    return [
      { kind: "points", n },
      { kind: "points", n },
    ];
  },
  inferElements(inEls) {
    requireVec3(inEls[0], "bodyTap.body");
    return [VEC3, VEC3];
  },
  async execute(_ctx, ins) {
    return tapBody(ins);
  },
  cpuGolden(ins) {
    return tapBody(ins);
  },
};

function tapBody(ins: FieldValue[]): FieldValue[] {
  const bodyV = ins[0];
  if (!bodyV) throw new Error("bodyTap: missing body input");
  const rows = pointsN(bodyV.shape, "bodyTap");
  const n = rows / BODY_BLOCK_COUNT;
  const data = requireData(bodyV, "bodyTap.body");
  return [
    { shape: { kind: "points", n }, dtype: "f32", element: VEC3, data: tapBlock(data, n, "pos") },
    { shape: { kind: "points", n }, dtype: "f32", element: VEC3, data: tapBlock(data, n, "vel") },
  ];
}

// ── integrate: advance the body by the summed force, close the loop ──────────────────

const INTEGRATE_PARAMS: ParamSpec[] = [
  { name: "timeFactor", type: "number", default: 0.2, min: 0.02, max: 1, step: 0.01, describe: "DANCERL timescale" },
  { name: "jerkLimit", type: "number", default: 0.05, min: 0.005, max: 0.5, step: 0.005 },
  { name: "linDamp", type: "number", default: 0.96, min: 0.8, max: 1, step: 0.005 },
  { name: "angDamp", type: "number", default: 0.9, min: 0.7, max: 1, step: 0.005 },
  { name: "speedLimit", type: "number", default: 1.2, min: 0.2, max: 4, step: 0.1 },
  { name: "face", type: "number", default: 0.3, min: 0, max: 2, step: 0.05, describe: "turn to face motion" },
  { name: "maxRadius", type: "number", default: 40, min: 5, max: 200, step: 1 },
  { name: "dt", type: "number", default: 1 }, // fixed
];

function integrateParams(p: Params): IntegrateParams {
  return {
    dt: num(p, "dt"),
    timeFactor: num(p, "timeFactor"),
    jerkLimit: num(p, "jerkLimit"),
    linDamp: num(p, "linDamp"),
    angDamp: num(p, "angDamp"),
    speedLimit: num(p, "speedLimit"),
    maxRadius: num(p, "maxRadius"),
    face: num(p, "face"),
  };
}

const integrateOp: OpType = {
  name: "integrate",
  label: "Integrate (rigid body)",
  describe: "Advance the swarm body by the summed force (jerk-limited, DANCERL timescale + rigid-body spin).",
  category: "Dance forces",
  inputs: [
    { name: "body", kind: "points" },
    { name: "force", kind: "points" },
  ],
  outputs: [
    { name: "body", kind: "points", dtype: "f32" },
    { name: "swarm", kind: "points", dtype: "f32" },
  ],
  params: INTEGRATE_PARAMS,
  inferShapes(inShapes) {
    const n = pointsN(inShapes[1]!, "integrate.force");
    return [
      { kind: "points", n: n * BODY_BLOCK_COUNT }, // body
      { kind: "points", n }, // swarm (xy scatter)
    ];
  },
  inferElements(inEls) {
    requireVec3(inEls[0], "integrate.body");
    requireVec3(inEls[1], "integrate.force");
    return [VEC3, SCALAR];
  },
  async execute(_ctx, ins, params) {
    return runIntegrate(ins, params);
  },
  cpuGolden(ins, params) {
    return runIntegrate(ins, params);
  },
};

function runIntegrate(ins: FieldValue[], params: Params): FieldValue[] {
  const bodyV = ins[0];
  const forceV = ins[1];
  if (!bodyV || !forceV) throw new Error("integrate: missing body or force input");
  const n = pointsN(forceV.shape, "integrate.force");
  const bodyData = requireData(bodyV, "integrate.body");
  const forceData = requireData(forceV, "integrate.force");
  const p = integrateParams(params);

  const nextBody = new Float32Array(n * 3 * BODY_BLOCK_COUNT);
  const swarm = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const state = readBodyState(bodyData, i, n);
    const force = readVec3(forceData, i);
    const next = integrateBody(state, force, ZERO, p);
    writeBodyState(nextBody, i, n, next);
    const [nx, ny] = unpack(next.pos);
    swarm[i * 2] = nx;
    swarm[i * 2 + 1] = ny;
  }
  return [
    { shape: { kind: "points", n: n * BODY_BLOCK_COUNT }, dtype: "f32", element: VEC3, data: nextBody },
    { shape: { kind: "points", n }, dtype: "f32", data: swarm },
  ];
}

// ── clock: a graph-native frame counter (drives the caller's figures) ────────────────

const clockOp: OpType = {
  name: "clock",
  label: "Clock",
  describe: "A frame counter: outputs prev + rate each tick (seed from a scalar, close the loop).",
  category: "Dance forces",
  inputs: [{ name: "prev", kind: "scalar" }],
  outputs: [{ name: "t", kind: "scalar", dtype: "f32" }],
  params: [{ name: "rate", type: "number", default: 1, min: 0, max: 8, step: 0.5 }],
  inferShapes() {
    return [{ kind: "scalar" }];
  },
  async execute(_ctx, ins, params) {
    return tickClock(ins, params);
  },
  cpuGolden(ins, params) {
    return tickClock(ins, params);
  },
};

function tickClock(ins: FieldValue[], params: Params): FieldValue[] {
  const prev = ins[0];
  const cur = prev?.data?.[0] ?? 0;
  return [{ shape: { kind: "scalar" }, dtype: "f32", data: new Float32Array([cur + num(params, "rate")]) }];
}

// ── caller: the Ceilidh choreographer (figures + partner progression) ────────────────

const CALLER_PARAMS: ParamSpec[] = [
  { name: "period", type: "int", default: 160, min: 30, max: 600, describe: "frames per figure" },
  { name: "seed", type: "int", default: 0, min: 0, max: 9, describe: "which figure sequence" },
  { name: "tightness", type: "number", default: 1, min: 0, max: 2, step: 0.05, describe: "how hard dancers scramble to the called state" },
  { name: "gain", type: "number", default: 0.08, min: 0, max: 0.3, step: 0.005 },
  { name: "speed", type: "number", default: 0.6, min: 0, max: 2, step: 0.05, describe: "target speed of motion" },
];

const callerOp: OpType = {
  name: "caller",
  label: "Caller (Ceilidh)",
  describe: "Drives each dancer toward the called figure's state of motion; couples advance through partners.",
  category: "Dance forces",
  inputs: [
    { name: "pos", kind: "points" },
    { name: "vel", kind: "points" },
    { name: "frame", kind: "scalar" },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: CALLER_PARAMS,
  inferShapes(inShapes) {
    return [{ kind: "points", n: pointsN(inShapes[0]!, "caller") }];
  },
  inferElements(inEls) {
    requireVec3(inEls[0], "caller.pos");
    requireVec3(inEls[1], "caller.vel");
    return [VEC3];
  },
  async execute(_ctx, ins, params) {
    return callForces(ins, params);
  },
  cpuGolden(ins, params) {
    return callForces(ins, params);
  },
};

function callForces(ins: FieldValue[], params: Params): FieldValue[] {
  const posV = ins[0];
  const velV = ins[1];
  const frameV = ins[2];
  if (!posV || !velV) throw new Error("caller: missing pos or vel input");
  const n = pointsN(posV.shape, "caller");
  const pos = requireData(posV, "caller.pos");
  const vel = requireData(velV, "caller.vel");
  const frame = frameV?.data?.[0] ?? 0;
  const period = num(params, "period");
  const seed = num(params, "seed");
  const tightness = num(params, "tightness");
  const gain = num(params, "gain");
  const speed = num(params, "speed");

  const { figure, figureIndex } = figureAt(frame, period, seed);
  const out = new Float32Array(n * 3);
  const k = gain * tightness;
  for (let i = 0; i < n; i++) {
    const p = readVec3(pos, i);
    const v = readVec3(vel, i);
    const partner = readVec3(pos, partnerIndex(i, figureIndex, n));
    const target = figureTargetVel(figure, p, i, partner, speed);
    const [tx, ty, tz] = unpack(target);
    const [vx, vy, vz] = unpack(v);
    // accel toward the called state of motion; the figure/partner change at each call
    // makes (target − vel) jump, and the integrator's jerk limit turns that into an
    // urgent-but-smooth scramble.
    writeVec3(out, i, [(tx - vx) * k, (ty - vy) * k, (tz - vz) * k]);
  }
  return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC3, data: out }];
}

// ── partnerOrbit: a standalone couple-swing force ────────────────────────────────────

const partnerOrbitOp = makeForceOp({
  name: "partnerOrbit",
  label: "Partner orbit",
  describe: "Each dancer swings around a partner (offset in the ring) — DANCERL pairwise orbit.",
  needsVel: true,
  params: [strengthSpec(0.5), { name: "offset", type: "int", default: 1, min: 1, max: 16, describe: "partner = (i + offset) mod N" }],
  kernel: (i, pos, vel, n, p) => {
    const off = Math.max(1, Math.round(num(p, "offset")));
    const partner = readVec3(pos, (i + off) % n);
    return partnerOrbitForce(readVec3(pos, i), readVec3(vel ?? pos, i), partner, num(p, "strength"));
  },
});

// ── Registration ─────────────────────────────────────────────────────────────────────

/** The dance force building blocks, in palette order. The individual ops are reached through
 *  this list (the playground's palette) or `registerForceOps` (the registry); they are not
 *  exported one by one. */
export const FORCE_OPS: OpType[] = [
  constrainOp,
  swimOp,
  vortexOp,
  solenoidOp,
  orbitOp,
  cohereOp,
  separateOp,
  springOp,
  partnerOrbitOp,
  callerOp,
  clockOp,
  bodyTapOp,
  integrateOp,
];

let registered = false;
/** Register the dance force building blocks (idempotent). Opt-in, like the element pack. */
export function registerForceOps(): void {
  if (registered) return;
  registered = true;
  for (const op of FORCE_OPS) if (!hasOp(op.name)) registerOp(op);
}
