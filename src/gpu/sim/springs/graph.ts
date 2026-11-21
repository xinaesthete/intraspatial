// The spring model as a graph: one op per term, state on feedback edges, resident on the device
// (docs/spring-model-as-ops.md). This builds the graph for the terms that exist as ops so far:
// bonded springs and the integrator. `stepSprings` is the reference the graph must equal for the
// same terms.
import { bondTableOf, VEC4 } from "#src/gpu/graph/ops/springs/bondTable";
import { Graph, type GpuField, type Params } from "../../graph/index";
import type { SpringParams } from "./params";
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

export interface SpringGraph {
  readonly graph: Graph;
  /** Positions after the tick: `vec4` per particle (xyz, density). The sink to `advance`. */
  readonly pos: GpuField;
  readonly prev: GpuField;
  /** The summed force of the tick. */
  readonly force: GpuField;
}

/**
 * A graph that advances `s` by one step per tick under the bonded springs of `t`. The state is
 * copied in as the seed; nothing here writes back to `s`.
 */
export function springGraph(s: SpringState, t: SpringTopology, p: SpringParams): SpringGraph {
  const graph = new Graph();
  const shape = { kind: "points", n: s.n } as const;
  const pos0 = graph.source({ shape, dtype: "f32", element: VEC4, data: s.pos.slice() }, "pos");
  const prev0 = graph.source({ shape, dtype: "f32", element: VEC4, data: s.prev.slice() }, "prev");
  const bonds = graph.source(bondTableOf(t), "bonds");
  const fbPos = graph.feedback(pos0, "springs:pos");
  const fbPrev = graph.feedback(prev0, "springs:prev");
  const force = graph.op1("springForce", { pos: fbPos.state, bonds }, springForceParams(p));
  const [pos, prev] = graph.op("springIntegrate", { pos: fbPos.state, prev: fbPrev.state, force }, springIntegrateParams(p));
  if (!pos || !prev) throw new Error("springGraph: springIntegrate did not produce pos and prev");
  fbPos.close(pos);
  fbPrev.close(prev);
  return { graph, pos, prev, force };
}
