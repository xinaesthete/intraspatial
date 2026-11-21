// The spring model as a graph: one op per term, state on feedback edges, resident on the device
// (docs/spring-model-as-ops.md). `springGraph` builds it from the same state, topology and params
// the fused `stepSprings` takes, including only the terms that are switched on. `stepSprings` is
// the reference the graph must equal.
import type { GpuBackend } from "#src/gpu/graph/backend";
import { type BundleSpec, bundleValue, extractOp, packOp } from "#src/gpu/graph/ops/bundleOps";
import { bondTableOf, VEC4, writeBondTable } from "#src/gpu/graph/ops/springs/bondTable";
import { hasOp, registerOp } from "#src/gpu/graph/registry";
import { VEC2 } from "#src/gpu/graph/ops/springs/common";
import { pairProgram } from "#src/gpu/graph/ops/springs/pairInterpreter";
import {
  downloadValue,
  type ElementType,
  type FieldValue,
  Graph,
  type GpuField,
  type Params,
  type SimState,
  simStateValue,
} from "../../graph/index";
import { MAX_SPECIES, type SpringParams } from "./params";
import type { SpringMatrices, SpringState } from "./state";
import { pairTermsActive } from "./step";
import type { SpringRules } from "./rules";
import type { Scenario, SpheroidScenario } from "./scenarios";
import type { SpringTopology } from "./topology";

/** `springForce`'s params for a `SpringParams`. The kind multipliers fold `springforce` and both
 *  of CSynth's per-kind terms into one number each. */
export function springForceParams(p: SpringParams): Params {
  const out: Params = {
    springforce: 1,
    springlen: p.springlen,
    springlenfix: p.springlenfix,
    springpow: p.springpow,
    powBaseDist: p.powBaseDist,
    backboneStrength: p.backboneStrength,
    springrate: p.springrate,
  };
  for (let kind = 0; kind < 8; kind++) out[`kind${kind}`] = p.springforce * p.roleforces[kind]! + p.roleforcesFix[kind]!;
  return out;
}

/** `springIntegrate`'s params for a `SpringParams`. */
export function springIntegrateParams(p: SpringParams): Params {
  return { damp: p.damp, damptype: p.damptype, springrate: p.springrate, springmaxvel: p.springmaxvel, gravity: p.gravity };
}

const pairRange = (p: SpringParams): Params => (p.pairRange === undefined ? {} : { pairRange: p.pairRange });

/** `pairDensity`'s params for a `SpringParams`. */
export function pairDensityParams(p: SpringParams): Params {
  return { ...pairRange(p), densityRange: p.densityRange, springrate: p.springrate };
}

/** `pushApart`'s params for a `SpringParams`. */
export function pushApartParams(p: SpringParams): Params {
  return {
    ...pairRange(p),
    pushapartlocalforce: p.pushapartlocalforce,
    nonBackboneLen: p.nonBackboneLen,
    backboneScale: p.backboneScale,
    pushapartdelta: p.pushapartdelta,
    speciesMask: p.pairSpeciesMask,
  };
}

/** `outward`'s params for a `SpringParams`. */
export function outwardParams(p: SpringParams): Params {
  const out: Params = { ...pairRange(p), protrusionRange: p.protrusionRange, protrusionFull: p.protrusionFull, tissueMask: p.tissueMask };
  for (let s = 0; s < MAX_SPECIES; s++) {
    out[`protrusion${s}`] = p.protrusion[s]!;
    out[`thrust${s}`] = p.thrust[s]!;
  }
  return out;
}

const any = (xs: readonly number[]) => xs.some((v) => v !== 0);
const perSpecies = (name: string, values: readonly number[]): Params =>
  Object.fromEntries(Array.from({ length: MAX_SPECIES }, (_, s) => [`${name}${s}`, values[s]!]));

/** Whether any torsion spring acts: an explicit one, or the backbone's. */
function torsionOn(p: SpringParams, t: SpringTopology): boolean {
  return p.backbonetorsionspringforce !== 0 || (p.torsionspringforce !== 0 && t.torsion.some((v, k) => k % 2 === 1 && v !== 0));
}

/** The keys of a spring graph's state, for `simStateValue`. */
export const SPRING_STATE = {
  pos: "springs:pos",
  prev: "springs:prev",
  species: "springs:species",
  radius: "springs:radius",
  bonds: "springs:bonds",
  step: "springs:step",
  /** In a packed graph: the one bundle that holds every part above. */
  packed: "springs:state",
  /** In a packed graph: the one bundle that holds every fixed input. */
  inputs: "springs:inputs",
} as const;

/** The parts of a spring model's state, in the order a packed state lays them out. */
export const STATE_PARTS = ["pos", "prev", "species", "radius", "link", "spring", "step"] as const;
export type StatePart = (typeof STATE_PARTS)[number];

/** A spring model's whole state as one bundle. Packed, its parts are ranges of one buffer. */
export const SPRING_STATE_BUNDLE: BundleSpec = {
  name: "springState",
  label: "Spring state",
  parts: [
    { name: "pos", kind: "points", describe: "Positions: xyz and density." },
    { name: "prev", kind: "points", describe: "Positions one step ago." },
    { name: "species", kind: "points", describe: "Species of each particle; 255 is an unused slot." },
    { name: "radius", kind: "points", describe: "Radius of each particle." },
    { name: "link", kind: "points", describe: "Bond table: partner and kind per slot." },
    { name: "spring", kind: "points", describe: "Bond table: rest length, strength and power per slot." },
    { name: "step", kind: "scalar", describe: "The step counter." },
  ],
};

/** What a part of a packed bundle holds. A bundle's shape says how many samples each part has
 *  but not what a sample is, and the ops downstream check that when the graph is built. */
interface PartType {
  readonly dtype: "f32" | "u32";
  readonly element?: ElementType;
}

/** Register a bundle's pack op and typed take-apart ops, once. */
function registerPacked(spec: BundleSpec, types: Readonly<Record<string, PartType>>): void {
  if (!hasOp(`${spec.name}.pack`)) registerOp(packOp(spec));
  for (const part of spec.parts) {
    const base = extractOp(spec, part.name);
    if (hasOp(base.name)) continue;
    const type = types[part.name] ?? { dtype: "f32" };
    registerOp({
      ...base,
      outputs: [{ name: part.name, kind: part.kind, dtype: type.dtype }],
      inferElements: () => [type.element ?? { kind: "scalar" }],
    });
  }
}
registerPacked(SPRING_STATE_BUNDLE, {
  pos: { dtype: "f32", element: VEC4 },
  prev: { dtype: "f32", element: VEC4 },
  species: { dtype: "u32" },
  radius: { dtype: "f32" },
  link: { dtype: "u32" },
  spring: { dtype: "f32", element: VEC4 },
  step: { dtype: "u32" },
});

/** How a build lays out what the model reads. */
interface BuildMode {
  /** State and fixed inputs each in one buffer, their parts ranges of it. */
  readonly packed: boolean;
  /** Packed build, second pass: the fixed inputs as one bundle, made from the first pass. */
  readonly inputs?: { readonly spec: BundleSpec; field(graph: Graph): GpuField };
  /** Packed build, first pass: collects the fixed inputs the model asks for. */
  readonly record?: Map<string, FieldValue>;
  /** Do the model's own pair terms in one interpreted pass. Needs a packed build. */
  readonly fuse?: boolean;
}

export interface SpringGraphOptions {
  /**
   * Keep the state in one buffer and the fixed inputs in another, each part a range of its
   * buffer. The ops are the same and so are the results; what changes is that a kernel doing the
   * work of several ops can bind everything in two bindings (WebGPU allows eight).
   */
  readonly packed?: boolean;
  /**
   * Fuse the model's own pair terms (push-apart, adhesion, outward, density) into one pass over
   * the pairs, run by an interpreter from a program held as data (`pairInterpreter`). Implies
   * `packed`. The separate ops remain the definition; the two must give the same result.
   */
  readonly fusePairs?: boolean;
}

export type { SpringRules } from "./rules";

export interface SpringGraph {
  readonly graph: Graph;
  /** Positions after the tick, constraints applied: `vec4` per particle (xyz, density). The sink
   *  to `advance`. */
  readonly pos: GpuField;
  readonly prev: GpuField;
  /** The force the integrator was given: every term, after mobility, with the random terms. */
  readonly force: GpuField;
  /** Each term's own force, by op name, for looking at one in isolation. */
  readonly terms: Readonly<Record<string, GpuField>>;
  /** Density and directional density; zero when no pair term is on. */
  readonly density: GpuField;
  /** Species, radius and the bond table after the tick's rules; the inputs when there are none. */
  readonly species: GpuField;
  readonly radius: GpuField;
  readonly bonds: GpuField;
  /** Whether the state and inputs are packed (`SpringGraphOptions.packed`). */
  readonly packed: boolean;
}

/** The build behind `springGraph`: see there. */
function buildSpringGraph(
  s: SpringState,
  t: SpringTopology,
  p: SpringParams,
  m: SpringMatrices | undefined,
  rules: SpringRules,
  mode: BuildMode,
): SpringGraph {
  const graph = new Graph();
  const n = s.n;
  const points = { kind: "points", n } as const;
  const table = { kind: "matrix", rows: MAX_SPECIES, cols: MAX_SPECIES } as const;
  const vec4Value = (data: Float32Array): FieldValue => ({ shape: points, dtype: "f32", element: VEC4, data: data.slice() });
  // Everything the model reads is state on a feedback edge. What a rule rewrites is fed its new
  // value; the rest is fed itself. Feeding a value to itself is what keeps a fixed input on the
  // device: it is uploaded for the first tick and the state then holds that buffer, where a
  // plain source would be uploaded again every tick.
  const held = (value: FieldValue, key: string): GpuField => {
    // First pass of a packed build: note what the model needs, so one buffer can hold it all.
    mode.record?.set(key, value);
    if (mode.inputs) return graph.op1(`${mode.inputs.spec.name}.${key}`, { bundle: mode.inputs.field(graph) });
    const fb = graph.feedback(graph.source(value, key), `springs:${key}`);
    fb.close(fb.state);
    return fb.state;
  };
  const changesWho = rules.division !== undefined || rules.leaders !== undefined;
  const changesBonds = rules.breakRule !== undefined || rules.formRule !== undefined;

  // The state, part by part. Unpacked, each part is its own feedback edge. Packed, they are
  // ranges of one buffer on one edge: each is taken out here and they are packed again below.
  const seeds: Record<StatePart, FieldValue> = {
    pos: vec4Value(s.pos),
    prev: vec4Value(s.prev),
    species: { shape: points, dtype: "u32", data: Uint32Array.from(s.species) },
    radius: { shape: points, dtype: "f32", data: Float32Array.from(s.radius) },
    link: bondTableOf(t).parts!.link!,
    spring: bondTableOf(t).parts!.spring!,
    step: { shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(s.step) },
  };
  const nextState: Partial<Record<StatePart, GpuField>> = {};
  let stateBundle: GpuField | undefined;
  let cells: Record<StatePart, GpuField>;
  let commit: () => void;
  if (mode.packed) {
    const seed: FieldValue = {
      ...bundleValue(
        SPRING_STATE_BUNDLE,
        STATE_PARTS.map((part) => seeds[part]),
      ),
      packed: true,
    };
    const fb = graph.feedback(graph.source(seed, "state"), SPRING_STATE.packed);
    stateBundle = fb.state;
    const take = (part: StatePart) => graph.op1(`${SPRING_STATE_BUNDLE.name}.${part}`, { bundle: fb.state });
    cells = {
      pos: take("pos"),
      prev: take("prev"),
      species: take("species"),
      radius: take("radius"),
      link: take("link"),
      spring: take("spring"),
      step: take("step"),
    };
    commit = () => {
      const parts = Object.fromEntries(STATE_PARTS.map((part) => [part, nextState[part] ?? cells[part]]));
      fb.close(graph.op1(`${SPRING_STATE_BUNDLE.name}.pack`, parts));
    };
  } else {
    const fbs = {
      pos: graph.feedback(graph.source(seeds.pos, "pos"), SPRING_STATE.pos),
      prev: graph.feedback(graph.source(seeds.prev, "prev"), SPRING_STATE.prev),
      species: graph.feedback(graph.source(seeds.species, "species"), SPRING_STATE.species),
      radius: graph.feedback(graph.source(seeds.radius, "radius"), SPRING_STATE.radius),
      bonds: graph.feedback(graph.source(bondTableOf(t), "bonds"), SPRING_STATE.bonds),
      step: graph.feedback(graph.source(seeds.step, "step"), SPRING_STATE.step),
    };
    const bondsNow = fbs.bonds.state;
    cells = {
      pos: fbs.pos.state,
      prev: fbs.prev.state,
      species: fbs.species.state,
      radius: fbs.radius.state,
      link: graph.op1("bondTable.link", { bundle: bondsNow }),
      spring: graph.op1("bondTable.spring", { bundle: bondsNow }),
      step: fbs.step.state,
    };
    commit = () => {
      fbs.pos.close(nextState.pos ?? cells.pos);
      fbs.prev.close(nextState.prev ?? cells.prev);
      fbs.species.close(nextState.species ?? cells.species);
      fbs.radius.close(nextState.radius ?? cells.radius);
      fbs.step.close(nextState.step ?? cells.step);
      const unchanged = nextState.link === undefined && nextState.spring === undefined;
      fbs.bonds.close(
        unchanged
          ? bondsNow
          : graph.op1("bondTable.bundle", { link: nextState.link ?? cells.link, spring: nextState.spring ?? cells.spring }),
      );
    };
  }
  const { pos, prev, species, radius } = cells;
  const bonds = graph.op1("bondTable.bundle", { link: cells.link, spring: cells.spring });
  const hasPull = t.pull.some((v, k) => k % 4 === 3 && v !== 0);
  const pull = hasPull ? held(vec4Value(t.pull), "pull") : undefined;
  const rate = { springrate: p.springrate };

  // ── Forces that mobility scales ──
  const terms: Record<string, GpuField> = {
    springForce: graph.op1("springForce", { pos, bonds }, springForceParams(p)),
  };
  if (torsionOn(p, t)) {
    const torsion = held({ shape: points, dtype: "f32", element: VEC2, data: t.torsion.slice() }, "torsion");
    terms.torsionForce = graph.op1(
      "torsionForce",
      { pos, species, torsion },
      {
        torsionspringforce: p.torsionspringforce,
        backbonetorsionspringforce: p.backbonetorsionspringforce,
        backbonetorsionspringangle: p.backbonetorsionspringangle,
        torsionMaxLen: p.torsionMaxLen,
      },
    );
  }
  if (pull && p.pullspringforce !== 0) {
    terms.pullForce = graph.op1(
      "pullForce",
      { pos, species, pull },
      { ...rate, pullspringforce: p.pullspringforce, pullspringpow: p.pullspringpow },
    );
  }
  // The model's own pair terms: one op each, or, fused, one interpreted pass over the pairs that
  // does all of them from a program held as data.
  const pushOn = p.pushapartlocalforce !== 0;
  const adhesionOn = any(p.repulsion) || any(p.adhesion);
  const outwardOn = any(p.protrusion) || any(p.thrust);
  const tables = adhesionOn
    ? {
        repulsion: held({ shape: table, dtype: "f32", data: Float32Array.from(p.repulsion) }, "repulsion"),
        adhesion: held({ shape: table, dtype: "f32", data: Float32Array.from(p.adhesion) }, "adhesion"),
      }
    : undefined;
  let fusedDensity: GpuField | undefined;
  if (mode.fuse && (pushOn || adhesionOn || outwardOn)) {
    const range = p.pairRange ?? Number.POSITIVE_INFINITY;
    const steps: Parameters<typeof pairProgram>[0][number][] = [];
    if (pushOn) {
      steps.push({
        op: "pushApart",
        values: {
          force: p.pushapartlocalforce,
          nonBackboneLen: p.nonBackboneLen,
          backboneScale: p.backboneScale,
          delta: p.pushapartdelta,
          pairRange: range,
          speciesMask: p.pairSpeciesMask,
        },
      });
    }
    if (adhesionOn) steps.push({ op: "adhesion", values: { range: p.adhesionRange, pairRange: range } });
    if (outwardOn) {
      steps.push({
        op: "outward",
        values: {
          range: p.protrusionRange,
          full: p.protrusionFull,
          pairRange: range,
          tissueMask: p.tissueMask,
          ...perSpecies("protrusion", p.protrusion),
          ...perSpecies("thrust", p.thrust),
        },
      });
    }
    steps.push({ op: "density", values: { pairRange: range, densityRange: p.densityRange } });
    const compiled = pairProgram(steps);
    held({ shape: { kind: "points", n: compiled.program.length }, dtype: "u32", data: compiled.program }, "program");
    held({ shape: { kind: "points", n: compiled.params.length }, dtype: "f32", data: compiled.params }, "programParams");
    // The first pass of a packed build only finds out what is needed; the node is made on the second.
    if (mode.inputs && stateBundle) {
      const [fused, crowd] = graph.op("pairInterpreter", { state: stateBundle, inputs: mode.inputs.field(graph) }, rate);
      if (!fused || !crowd) throw new Error("springGraph: pairInterpreter did not produce a force and a density");
      terms.pairInterpreter = fused;
      fusedDensity = crowd;
    }
  } else {
    if (pushOn) terms.pushApart = graph.op1("pushApart", { pos, species }, pushApartParams(p));
    if (tables)
      terms.adhesion = graph.op1("adhesion", { pos, species, radius, ...tables }, { ...pairRange(p), adhesionRange: p.adhesionRange });
    if (outwardOn) terms.outward = graph.op1("outward", { pos, species }, outwardParams(p));
  }

  // ── CSynth's other pair terms. Each reads the contact map to work out its own gate. ──
  const size = m?.size ?? 0;
  const square = (data: Float32Array | undefined, label: string) =>
    data
      ? held({ shape: { kind: "matrix", rows: size, cols: size }, dtype: "f32", data: data.slice() }, label)
      : held({ shape: { kind: "matrix", rows: 1, cols: 1 }, dtype: "f32", data: new Float32Array(1) }, label);
  const csynthOn =
    p.pushapartforce !== 0 ||
    p.pushapartDensityFactor !== 0 ||
    p.fractforce !== 0 ||
    p.backboneforce !== 0 ||
    p.boidalignment !== 0 ||
    (p.xyzforce !== 0 && m?.dist !== undefined) ||
    ((p.contactforcesc !== 0 || p.m_force !== 0) && m?.contact !== undefined);
  if (csynthOn) {
    const contact = square(m?.contact, "contact");
    const common: Params = {
      ...pairRange(p),
      speciesMask: p.pairSpeciesMask,
      matrixSize: size,
      contactsOn: p.contactforcesc !== 0 && m?.contact !== undefined,
      lorentzianOn: p.m_force !== 0 && m?.contact !== undefined,
      maxBackboneDist: p.maxBackboneDist,
      powBaseDist: p.powBaseDist,
      springpow: p.springpow,
    };
    const coupling = {
      contactforcesc: p.contactforcesc,
      contactthreshold: p.contactthreshold,
      backboneContactStrength: p.backboneContactStrength,
      wrongfade: p.wrongfade,
    };
    const pair = { pos, species, contact };
    if (p.pushapartDensityFactor !== 0) {
      terms.pushApartDensity = graph.op1("pushApartDensity", pair, {
        ...common,
        pushapartDensityFactor: p.pushapartDensityFactor,
        pushapartDensityPow: p.pushapartDensityPow,
      });
    }
    if (p.fractforce !== 0)
      terms.fractalForce = graph.op1("fractalForce", pair, { ...common, fractforce: p.fractforce, fractpow: p.fractpow });
    if (p.xyzforce !== 0 && m?.dist) {
      terms.distanceForce = graph.op1(
        "distanceForce",
        { ...pair, dist: square(m.dist, "dist") },
        { ...common, xyzforce: p.xyzforce, xyzpow: p.xyzpow, xyzMaxDist: p.xyzMaxDist },
      );
    }
    if (p.backboneforce !== 0)
      terms.backboneForce = graph.op1("backboneForce", pair, { ...common, backboneforce: p.backboneforce, backboneScale: p.backboneScale });
    if (common.contactsOn) {
      terms.contactForce = graph.op1("contactForce", pair, {
        ...common,
        ...coupling,
        pushapartforce: p.pushapartforce,
        pushapartpow: p.pushapartpow,
      });
    }
    if (p.pushapartforce !== 0) {
      terms.pushApartGlobal = graph.op1("pushApartGlobal", pair, {
        ...common,
        ...coupling,
        pushapartforce: p.pushapartforce,
        pushapartpow: p.pushapartpow,
        backboneScale: p.backboneScale,
        ignoreBackbone: p.ignoreBackbone,
      });
    }
    if (p.boidalignment !== 0) {
      terms.boids = graph.op1(
        "boids",
        { ...pair, prev },
        {
          ...common,
          ...rate,
          boidalignment: p.boidalignment,
          boidcohesion: p.boidcohesion,
          boidseparation: p.boidseparation,
          boidseparationpow: p.boidseparationpow,
          boidrad: p.boidrad,
        },
      );
    }
    if (common.lorentzianOn) {
      terms.lorentzianContact = graph.op1("lorentzianContact", pair, {
        ...common,
        backboneContactStrength: p.backboneContactStrength,
        m_force: p.m_force,
        m_k: p.m_k,
        m_alpha: p.m_alpha,
        m_c: p.m_c,
        representativeContact: p.representativeContact,
        maxv: p.maxv,
        patchval: p.patchval,
        patchwidth: p.patchwidth,
      });
    }
  }
  // The fused step measures density whenever a pair term's setting is on, and so does this.
  const pairs = pairTermsActive(p);
  if (p.visc !== 0) terms.viscosity = graph.op1("viscosity", { pos, prev, species }, { ...rate, visc: p.visc, viscpow: p.viscpow });
  if (p.modelSphereForce > 0) {
    terms.container = graph.op1(
      "container",
      { pos, species },
      { modelSphereForce: p.modelSphereForce, modelSphereRadius: p.modelSphereRadius },
    );
  }
  const add = (a: GpuField, b: GpuField) => graph.op1("addForces", { a, b });
  let force = Object.values(terms).reduce(add);
  if (p.mobility.some((v) => v !== 1)) force = graph.op1("mobility", { force, species }, perSpecies("mobility", p.mobility));

  // ── Random terms, added after mobility. They need the step as a value. ──
  const noise = p.noiseforce !== 0;
  const motile = any(p.motility);
  const ruled = changesWho || changesBonds;
  let stepAfter: GpuField | undefined;
  // A packed graph carries the step in the state bundle, so the counter has to move even when
  // nothing reads it: otherwise each pack copies the seed and a later read reports it unchanged.
  // Random terms read the step as it stands; rules read the value after this increment.
  if (mode.packed || noise || motile || ruled) {
    const fbStep = { state: cells.step };
    stepAfter = graph.op1("tick", { step: cells.step });
    nextState.step = stepAfter;
    const random = { seed: p.seed, dims: p.dims };
    if (noise) {
      terms.noiseForce = graph.op1(
        "noiseForce",
        { species, step: fbStep.state },
        { ...random, noiseforce: p.noiseforce, noiseprob: p.noiseprob },
      );
      force = add(force, terms.noiseForce);
    }
    if (motile) {
      terms.motilityForce = graph.op1(
        "motilityForce",
        { species, step: fbStep.state },
        { ...random, ...perSpecies("motility", p.motility) },
      );
      force = add(force, terms.motilityForce);
    }
  }

  const density = pairs
    ? (fusedDensity ?? graph.op1("pairDensity", { pos, prev, species }, pairDensityParams(p)))
    : held({ shape: points, dtype: "f32", element: VEC2, data: new Float32Array(n * 2) }, "density");
  const [moved, nextPrev] = graph.op("springIntegrate", { pos, prev, force, density, species }, springIntegrateParams(p));
  if (!moved || !nextPrev) throw new Error("springGraph: springIntegrate did not produce pos and prev");

  // ── Constraints, in the fused step's order ──
  let next = moved;
  if (t.rod.some((v, k) => k % 4 === 0 && v >= 0)) {
    next = graph.op1(
      "rodConstraint",
      { pos: next, before: pos, species, rod: held(vec4Value(t.rod), "rod") },
      { springlen: p.springlen, springlenfix: p.springlenfix },
    );
  }
  if (p.springCentreDamp !== 1) {
    const [centreX, centreY, centreZ] = p.springCentre;
    next = graph.op1("centreConstraint", { pos: next, species }, { springCentreDamp: p.springCentreDamp, centreX, centreY, centreZ });
  }
  if (t.fix.some((v, k) => k % 4 === 0 && v === 1))
    next = graph.op1("fixConstraint", { pos: next, species, fix: held(vec4Value(t.fix), "fix") });
  if (pull && p.pullfixdamp !== 0) next = graph.op1("pullFix", { pos: next, species, pull }, { pullfixdamp: p.pullfixdamp });

  // ── Rules, after the step, as the host scenario orders them. They read the step's number as
  // it is after the step, and the new positions. ──
  let nextPrevOut = nextPrev;
  let speciesOut = species;
  let radiusOut = radius;
  let bondsOut = bonds;
  if (stepAfter) {
    const step = stepAfter;
    const lengths = { springlen: p.springlen, springlenfix: p.springlenfix, seed: p.seed };
    if (rules.breakRule) {
      const r = rules.breakRule;
      bondsOut = graph.op1(
        "bondBreak",
        { pos: next, species, bonds: bondsOut, step },
        {
          ...lengths,
          kinds: r.kinds,
          maxStrain: r.maxStrain ?? Number.POSITIVE_INFINITY,
          tensionOnly: r.tensionOnly ?? false,
          tau: r.tau ?? Number.POSITIVE_INFINITY,
        },
      );
    }
    if (rules.formRule) {
      const r = rules.formRule;
      const kindOf = held(
        { shape: { kind: "matrix", rows: r.speciesCount, cols: r.speciesCount }, dtype: "f32", data: Float32Array.from(r.kindOf) },
        "kindOf",
      );
      const strengths = Object.fromEntries(Array.from({ length: 8 }, (_, kind) => [`strength${kind}`, r.strength[kind] ?? 1]));
      bondsOut = graph.op1(
        "bondForm",
        { pos: next, species, bonds: bondsOut, kindOf, step },
        {
          ...lengths,
          ...strengths,
          range: r.range,
          prob: r.prob ?? 1,
          restCurrent: r.rest === "current",
          rest: r.rest === "current" ? 1 : r.rest,
        },
      );
    }
    if (rules.division) {
      const r = rules.division;
      const wants = graph.op1(
        "divisionDraw",
        { pos: next, species, step },
        { species: r.species, tau: r.tau, maxDensity: r.maxDensity, seed: p.seed },
      );
      const plan = graph.op1("divisionPlan", { wants, species });
      next = graph.op1("divideMove", { pos: next, plan, step }, { offset: r.offset, dims: p.dims, seed: p.seed });
      nextPrevOut = graph.op1("restAfterDivision", { prev: nextPrev, pos: next, plan });
      // Division knows nothing of what the particles carry: each field is passed through.
      speciesOut = graph.op1("inheritField", { field: species, plan });
      radiusOut = graph.op1("inheritField", { field: radius, plan });
    }
    if (rules.leaders) {
      const r = rules.leaders;
      const every = Math.max(1, rules.leaderEvery ?? 1);
      const kinds = { from: r.from, to: r.to };
      const status = graph.op1(
        "leaderStatus",
        { pos: next, species: speciesOut, step },
        {
          ...kinds,
          tau: Math.max(1, r.tau / every),
          maxNeighbours: r.maxNeighbours,
          revertNeighbours: r.revertNeighbours,
          every,
          seed: p.seed,
        },
      );
      speciesOut = graph.op1("leaderApply", { pos: next, species: speciesOut, status }, { ...kinds, spacing: r.spacing });
    }
  }

  nextState.pos = next;
  nextState.prev = nextPrevOut;
  nextState.species = speciesOut;
  nextState.radius = radiusOut;
  if (bondsOut !== bonds) {
    nextState.link = graph.op1("bondTable.link", { bundle: bondsOut });
    nextState.spring = graph.op1("bondTable.spring", { bundle: bondsOut });
  }
  commit();
  return {
    graph,
    pos: next,
    prev: nextPrevOut,
    force,
    terms,
    density,
    species: speciesOut,
    radius: radiusOut,
    bonds: bondsOut,
    packed: mode.packed,
  };
}

/**
 * A graph that advances `s` by one step per tick under the terms `p` and `t` switch on. The
 * state is copied in as the seed; nothing here writes back to `s`. `m` holds the distance and
 * contact matrices CSynth's targets read, when the model has them. `rules` adds bond turnover,
 * division and leader selection after the step.
 */
export function springGraph(
  s: SpringState,
  t: SpringTopology,
  p: SpringParams,
  m?: SpringMatrices,
  rules: SpringRules = {},
  opts: SpringGraphOptions = {},
): SpringGraph {
  const fuse = opts.fusePairs === true;
  if (!opts.packed && !fuse) return buildSpringGraph(s, t, p, m, rules, { packed: false });
  // Which fixed inputs a model reads depends on which terms are on. Build once to find out,
  // then again with those inputs as one packed bundle.
  const record = new Map<string, FieldValue>();
  buildSpringGraph(s, t, p, m, rules, { packed: true, record, fuse });
  if (record.size === 0) return buildSpringGraph(s, t, p, m, rules, { packed: true, fuse });
  const names = [...record.keys()];
  const spec: BundleSpec = {
    name: `springInputs[${names.join(",")}]`,
    label: "Spring inputs",
    parts: names.map((name) => ({ name, kind: record.get(name)!.shape.kind })),
  };
  registerPacked(
    spec,
    Object.fromEntries(
      names.map((name) => [
        name,
        { dtype: record.get(name)!.dtype === "u32" ? "u32" : "f32", element: record.get(name)!.element } satisfies PartType,
      ]),
    ),
  );
  const value: FieldValue = {
    ...bundleValue(
      spec,
      names.map((name) => record.get(name)!),
    ),
    packed: true,
  };
  let held: GpuField | undefined;
  const field = (graph: Graph): GpuField => {
    if (!held) {
      // Fed to itself: uploaded for the first tick, on the device from then on.
      const fb = graph.feedback(graph.source(value, "inputs"), SPRING_STATE.inputs);
      fb.close(fb.state);
      held = fb.state;
    }
    return held;
  };
  return buildSpringGraph(s, t, p, m, rules, { packed: true, inputs: { spec, field }, fuse });
}

/** A scenario as a graph: its forces, and whatever rules it applies after each step, with its
 *  settings as they are now. The whole tick runs on the device. Settings are read once, here: a
 *  scenario that is reconfigured needs a new graph (`createSpringSim` does that). */
export function scenarioGraph(sc: Scenario, opts: SpringGraphOptions = {}): SpringGraph {
  return springGraph(sc.state, sc.topology, sc.params, sc.matrices, sc.graphRules?.() ?? {}, opts);
}

/** `scenarioGraph`, by its first name. */
export function spheroidGraph(sc: SpheroidScenario): SpringGraph {
  return scenarioGraph(sc);
}

/**
 * Bring a spring graph's state back into host objects: positions, species, radius, the step and
 * the bond table, whichever of them the graph holds as state. The one download a resident run
 * needs, for a readout or a test. `s` and `t` must be the size the graph was built for.
 */
export async function readSpringState(state: SimState, s: SpringState, t: SpringTopology, backend?: GpuBackend): Promise<void> {
  const fetch = async (key: string) => {
    const v = simStateValue(state, key);
    return v ? await downloadValue(v, backend) : undefined;
  };
  // A packed graph holds every part in one bundle; an unpacked one holds each under its own key.
  const packed = await fetch(SPRING_STATE.packed);
  const bonds = packed ? undefined : await fetch(SPRING_STATE.bonds);
  const part = async (name: StatePart, key: string) => (packed ? packed.parts?.[name]?.data : (await fetch(key))?.data);
  const pos = await part("pos", SPRING_STATE.pos);
  const prev = await part("prev", SPRING_STATE.prev);
  const species = await part("species", SPRING_STATE.species);
  const radius = await part("radius", SPRING_STATE.radius);
  const step = await part("step", SPRING_STATE.step);
  if (pos) s.pos.set(pos);
  if (prev) s.prev.set(prev);
  if (species) s.species.set(species);
  if (radius) s.radius.set(radius);
  if (step) s.step = step[0]!;
  const link = packed ? packed.parts?.link?.data : bonds?.parts?.link?.data;
  const spring = packed ? packed.parts?.spring?.data : bonds?.parts?.spring?.data;
  if (link && spring) writeBondTable(t, link, spring);
}
