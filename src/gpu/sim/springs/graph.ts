// The spring model as a graph: one op per term, state on feedback edges, resident on the device
// (docs/spring-model-as-ops.md). `springGraph` builds it from the same state, topology and params
// the fused `stepSprings` takes, including only the terms that are switched on, and refuses
// params that switch on a term with no op yet. `stepSprings` is the reference the graph must equal.
import { bondTableOf, VEC4 } from "#src/gpu/graph/ops/springs/bondTable";
import { VEC2 } from "#src/gpu/graph/ops/springs/common";
import { Graph, type GpuField, type Params } from "../../graph/index";
import { MAX_SPECIES, type SpringParams } from "./params";
import type { SpringState } from "./state";
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

/** Terms of the fused step that `p` and `t` switch on and that are not graph ops yet. */
export function termsWithoutOps(p: SpringParams, t: SpringTopology): string[] {
  const missing: string[] = [];
  const on = (name: string, active: boolean) => {
    if (active) missing.push(name);
  };
  on("global push-apart (pushapartforce)", p.pushapartforce !== 0);
  on("density push-apart (pushapartDensityFactor)", p.pushapartDensityFactor !== 0);
  on("fractal force (fractforce)", p.fractforce !== 0);
  on("distance targets (xyzforce)", p.xyzforce !== 0);
  on("backbone force (backboneforce)", p.backboneforce !== 0);
  on("contacts (contactforcesc)", p.contactforcesc !== 0);
  on("Lorentzian contacts (m_force)", p.m_force !== 0);
  on("boids (boidalignment)", p.boidalignment !== 0);
  on("viscosity (visc)", p.visc !== 0);
  on("container (modelSphereForce)", p.modelSphereForce > 0);
  on(
    "mobility",
    p.mobility.some((v) => v !== 1),
  );
  on("noise (noiseforce)", p.noiseforce !== 0);
  on("motility", any(p.motility));
  on("centring (springCentreDamp)", p.springCentreDamp !== 1);
  on("pull springs", p.pullspringforce !== 0 || p.pullfixdamp !== 0);
  on("backbone torsion", p.backbonetorsionspringforce !== 0);
  on("torsion springs", p.torsionspringforce !== 0 && t.torsion.some((v, k) => k % 2 === 1 && v !== 0));
  on(
    "rods",
    t.rod.some((v, k) => k % 4 === 0 && v >= 0),
  );
  on(
    "fixed particles",
    t.fix.some((v, k) => k % 4 === 0 && v === 1),
  );
  return missing;
}

export interface SpringGraph {
  readonly graph: Graph;
  /** Positions after the tick: `vec4` per particle (xyz, density). The sink to `advance`. */
  readonly pos: GpuField;
  readonly prev: GpuField;
  /** The summed force of the tick. */
  readonly force: GpuField;
  /** Each term's own force, by op name, for looking at one in isolation. */
  readonly terms: Readonly<Record<string, GpuField>>;
  /** Density and directional density; zero when no pair term is on. */
  readonly density: GpuField;
}

/**
 * A graph that advances `s` by one step per tick under the terms `p` switches on. The state is
 * copied in as the seed; nothing here writes back to `s`. Throws if `p` or `t` needs a term that
 * is not an op yet.
 */
export function springGraph(s: SpringState, t: SpringTopology, p: SpringParams): SpringGraph {
  const missing = termsWithoutOps(p, t);
  if (missing.length > 0) throw new Error(`springGraph: no graph op yet for: ${missing.join("; ")}`);

  const graph = new Graph();
  const n = s.n;
  const points = { kind: "points", n } as const;
  const table = { kind: "matrix", rows: MAX_SPECIES, cols: MAX_SPECIES } as const;
  const pos0 = graph.source({ shape: points, dtype: "f32", element: VEC4, data: s.pos.slice() }, "pos");
  const prev0 = graph.source({ shape: points, dtype: "f32", element: VEC4, data: s.prev.slice() }, "prev");
  const species = graph.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species");
  const fbPos = graph.feedback(pos0, "springs:pos");
  const fbPrev = graph.feedback(prev0, "springs:prev");
  const pos = fbPos.state;

  const terms: Record<string, GpuField> = {
    springForce: graph.op1("springForce", { pos, bonds: graph.source(bondTableOf(t), "bonds") }, springForceParams(p)),
  };
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

  // The fused step measures density only when a pair term is on, and so does this.
  const pairs = Object.keys(terms).length > 1;
  const density = pairs
    ? graph.op1("pairDensity", { pos, prev: fbPrev.state, species }, pairDensityParams(p))
    : graph.source({ shape: points, dtype: "f32", element: VEC2, data: new Float32Array(n * 2) }, "density");

  const force = Object.values(terms).reduce((a, b) => graph.op1("addForces", { a, b }));
  const [next, nextPrev] = graph.op("springIntegrate", { pos, prev: fbPrev.state, force, density, species }, springIntegrateParams(p));
  if (!next || !nextPrev) throw new Error("springGraph: springIntegrate did not produce pos and prev");
  fbPos.close(next);
  fbPrev.close(nextPrev);
  return { graph, pos: next, prev: nextPrev, force, terms, density };
}
