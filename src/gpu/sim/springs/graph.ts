// The spring model as a graph: one op per term, state on feedback edges, resident on the device
// (docs/spring-model-as-ops.md). `springGraph` builds it from the same state, topology and params
// the fused `stepSprings` takes, including only the terms that are switched on. `stepSprings` is
// the reference the graph must equal.
import { bondTableOf, VEC4 } from "#src/gpu/graph/ops/springs/bondTable";
import { VEC2 } from "#src/gpu/graph/ops/springs/common";
import { Graph, type GpuField, type Params } from "../../graph/index";
import { MAX_SPECIES, type SpringParams } from "./params";
import type { SpringMatrices, SpringState } from "./state";
import { pairTermsActive } from "./step";
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
}

/**
 * A graph that advances `s` by one step per tick under the terms `p` and `t` switch on. The
 * state is copied in as the seed; nothing here writes back to `s`. `m` holds the distance and
 * contact matrices CSynth's targets read, when the model has them.
 */
export function springGraph(s: SpringState, t: SpringTopology, p: SpringParams, m?: SpringMatrices): SpringGraph {
  const graph = new Graph();
  const n = s.n;
  const points = { kind: "points", n } as const;
  const table = { kind: "matrix", rows: MAX_SPECIES, cols: MAX_SPECIES } as const;
  const vec4Source = (data: Float32Array, label: string) =>
    graph.source({ shape: points, dtype: "f32", element: VEC4, data: data.slice() }, label);
  const species = graph.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species");
  const fbPos = graph.feedback(vec4Source(s.pos, "pos"), "springs:pos");
  const fbPrev = graph.feedback(vec4Source(s.prev, "prev"), "springs:prev");
  const pos = fbPos.state;
  const prev = fbPrev.state;
  const hasPull = t.pull.some((v, k) => k % 4 === 3 && v !== 0);
  const pull = hasPull ? vec4Source(t.pull, "pull") : undefined;
  const rate = { springrate: p.springrate };

  // ── Forces that mobility scales ──
  const terms: Record<string, GpuField> = {
    springForce: graph.op1("springForce", { pos, bonds: graph.source(bondTableOf(t), "bonds") }, springForceParams(p)),
  };
  if (torsionOn(p, t)) {
    const torsion = graph.source({ shape: points, dtype: "f32", element: VEC2, data: t.torsion.slice() }, "torsion");
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
    const radius = graph.source({ shape: points, dtype: "f32", data: Float32Array.from(s.radius) }, "radius");
    const repulsion = graph.source({ shape: table, dtype: "f32", data: Float32Array.from(p.repulsion) }, "repulsion");
    const adhesion = graph.source({ shape: table, dtype: "f32", data: Float32Array.from(p.adhesion) }, "adhesion");
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
      ? graph.source({ shape: { kind: "matrix", rows: size, cols: size }, dtype: "f32", data: data.slice() }, label)
      : graph.source({ shape: { kind: "matrix", rows: 1, cols: 1 }, dtype: "f32", data: new Float32Array(1) }, label);
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
  if (noise || motile) {
    const fbStep = graph.feedback(
      graph.source({ shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(s.step) }, "step"),
      "springs:step",
    );
    fbStep.close(graph.op1("tick", { step: fbStep.state }));
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
    : graph.source({ shape: points, dtype: "f32", element: VEC2, data: new Float32Array(n * 2) }, "density");
  const [moved, nextPrev] = graph.op("springIntegrate", { pos, prev, force, density, species }, springIntegrateParams(p));
  if (!moved || !nextPrev) throw new Error("springGraph: springIntegrate did not produce pos and prev");

  // ── Constraints, in the fused step's order ──
  let next = moved;
  if (t.rod.some((v, k) => k % 4 === 0 && v >= 0)) {
    next = graph.op1(
      "rodConstraint",
      { pos: next, before: pos, species, rod: vec4Source(t.rod, "rod") },
      { springlen: p.springlen, springlenfix: p.springlenfix },
    );
  }
  if (p.springCentreDamp !== 1) {
    const [centreX, centreY, centreZ] = p.springCentre;
    next = graph.op1("centreConstraint", { pos: next, species }, { springCentreDamp: p.springCentreDamp, centreX, centreY, centreZ });
  }
  if (t.fix.some((v, k) => k % 4 === 0 && v === 1))
    next = graph.op1("fixConstraint", { pos: next, species, fix: vec4Source(t.fix, "fix") });
  if (pull && p.pullfixdamp !== 0) next = graph.op1("pullFix", { pos: next, species, pull }, { pullfixdamp: p.pullfixdamp });

  fbPos.close(next);
  fbPrev.close(nextPrev);
  return { graph, pos: next, prev: nextPrev, force, terms, density };
}
