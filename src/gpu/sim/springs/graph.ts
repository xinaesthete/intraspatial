// The spring model as a graph: one op per term, state on feedback edges, resident on the device
// (docs/spring-model-as-ops.md). `springGraph` builds it from the same state, topology and params
// the fused `stepSprings` takes, including only the terms that are switched on. `stepSprings` is
// the reference the graph must equal.
import type { GpuBackend } from "#src/gpu/graph/backend";
import { bondTableOf, VEC4, writeBondTable } from "#src/gpu/graph/ops/springs/bondTable";
import { VEC2 } from "#src/gpu/graph/ops/springs/common";
import { downloadValue, type FieldValue, Graph, type GpuField, type Params, type SimState, simStateValue } from "../../graph/index";
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
} as const;

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
}

/**
 * A graph that advances `s` by one step per tick under the terms `p` and `t` switch on. The
 * state is copied in as the seed; nothing here writes back to `s`. `m` holds the distance and
 * contact matrices CSynth's targets read, when the model has them. `rules` adds bond turnover,
 * division and leader selection after the step; what they change (the bond table, species,
 * radius) is then state on feedback edges too.
 */
export function springGraph(s: SpringState, t: SpringTopology, p: SpringParams, m?: SpringMatrices, rules: SpringRules = {}): SpringGraph {
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
    const fb = graph.feedback(graph.source(value, key), `springs:${key}`);
    fb.close(fb.state);
    return fb.state;
  };
  const changesWho = rules.division !== undefined || rules.leaders !== undefined;
  const changesBonds = rules.breakRule !== undefined || rules.formRule !== undefined;
  const fbSpecies = graph.feedback(
    graph.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species"),
    SPRING_STATE.species,
  );
  const species = fbSpecies.state;
  const fbRadius = graph.feedback(
    graph.source({ shape: points, dtype: "f32", data: Float32Array.from(s.radius) }, "radius"),
    SPRING_STATE.radius,
  );
  const radius = fbRadius.state;
  const fbBonds = graph.feedback(graph.source(bondTableOf(t), "bonds"), SPRING_STATE.bonds);
  const bonds = fbBonds.state;
  const fbPos = graph.feedback(graph.source(vec4Value(s.pos), "pos"), SPRING_STATE.pos);
  const fbPrev = graph.feedback(graph.source(vec4Value(s.prev), "prev"), SPRING_STATE.prev);
  const pos = fbPos.state;
  const prev = fbPrev.state;
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
  if (p.pushapartlocalforce !== 0) terms.pushApart = graph.op1("pushApart", { pos, species }, pushApartParams(p));
  if (any(p.repulsion) || any(p.adhesion)) {
    const repulsion = held({ shape: table, dtype: "f32", data: Float32Array.from(p.repulsion) }, "repulsion");
    const adhesion = held({ shape: table, dtype: "f32", data: Float32Array.from(p.adhesion) }, "adhesion");
    terms.adhesion = graph.op1(
      "adhesion",
      { pos, species, radius, repulsion, adhesion },
      { ...pairRange(p), adhesionRange: p.adhesionRange },
    );
  }
  if (any(p.protrusion) || any(p.thrust)) terms.outward = graph.op1("outward", { pos, species }, outwardParams(p));

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
  if (noise || motile || ruled) {
    const fbStep = graph.feedback(
      graph.source({ shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(s.step) }, "step"),
      SPRING_STATE.step,
    );
    stepAfter = graph.op1("tick", { step: fbStep.state });
    fbStep.close(stepAfter);
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
    ? graph.op1("pairDensity", { pos, prev, species }, pairDensityParams(p))
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

  fbPos.close(next);
  fbPrev.close(nextPrevOut);
  fbSpecies.close(speciesOut);
  fbRadius.close(radiusOut);
  fbBonds.close(bondsOut);
  return { graph, pos: next, prev: nextPrevOut, force, terms, density, species: speciesOut, radius: radiusOut, bonds: bondsOut };
}

/** A scenario as a graph: its forces, and whatever rules it applies after each step, with its
 *  settings as they are now. The whole tick runs on the device. Settings are read once, here: a
 *  scenario that is reconfigured needs a new graph (`createSpringSim` does that). */
export function scenarioGraph(sc: Scenario): SpringGraph {
  return springGraph(sc.state, sc.topology, sc.params, sc.matrices, sc.graphRules?.() ?? {});
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
  const pos = await fetch(SPRING_STATE.pos);
  const prev = await fetch(SPRING_STATE.prev);
  if (pos?.data) s.pos.set(pos.data);
  if (prev?.data) s.prev.set(prev.data);
  const species = await fetch(SPRING_STATE.species);
  if (species?.data) s.species.set(species.data);
  const radius = await fetch(SPRING_STATE.radius);
  if (radius?.data) s.radius.set(radius.data);
  const step = await fetch(SPRING_STATE.step);
  if (step?.data) s.step = step.data[0]!;
  const bonds = await fetch(SPRING_STATE.bonds);
  const link = bonds?.parts?.link?.data;
  const spring = bonds?.parts?.spring?.data;
  if (link && spring) writeBondTable(t, link, spring);
}
