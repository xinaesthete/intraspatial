// Lazy-pull executor for the operation graph.
//
// `pull(graph, field)` resolves the transitive dependencies of the requested
// field, topologically orders them, and executes each node once — caching its
// outputs so a node feeding two consumers runs a single time. This is the
// correctness-first baseline of docs/gpu-resource-sync.md: nodes run in dependency
// order, each op owns its own submits internally (Tier-1), and every node is gated
// by a sanity check that falls back to its CPU golden (invariant 5).
//
// Phase 0 carries host `FieldValue`s along edges (Tier-1, boundary-granularity);
// the IR, pull semantics, topo order, dedup, and validate/fallback are all proven
// here. GPU-resident interiors (Tier-2) and param-level memoisation layer on top
// without changing this contract.
import type { FieldValue, GpuField } from "./handle";
import type { ExecCtx, OpType } from "./op";
import { allFinite } from "./op";
import type { GraphNode } from "./graph";
import { Graph } from "./graph";
import { getOp } from "./registry";
import { nodeBackend } from "./backend.node";

export interface PullOptions {
  /** Which backend to run native ops on. Defaults to the Node (Dawn) backend. */
  ctx?: ExecCtx;
  /** "gpu" (default) runs each op's `execute`; "cpu" forces every op through its
   *  `cpuGolden` — an independent reference the GPU path must match. */
  mode?: "gpu" | "cpu";
  /** Receives a value for every produced port (`"nodeId:port"`). Lets callers
   *  inspect intermediates without re-pulling. */
  onValue?: (key: string, value: FieldValue) => void;
}

const key = (nodeId: string, port: string) => `${nodeId}:${port}`;

/** Topologically order the transitive dependencies of `roots` (producer-first). */
function topoOrder(graph: Graph, roots: string[]): GraphNode[] {
  const order: GraphNode[] = [];
  const state = new Map<string, 0 | 1>(); // 0 = visiting, 1 = done
  const visit = (id: string) => {
    const s = state.get(id);
    if (s === 1) return;
    if (s === 0) throw new Error(`executor: cycle through node "${id}"`);
    state.set(id, 0);
    const node = graph.getNode(id);
    for (const ref of Object.values(node.inputs)) visit(ref.node);
    state.set(id, 1);
    order.push(node);
  };
  for (const r of roots) visit(r);
  return order;
}

async function runNode(
  op: OpType,
  ctx: ExecCtx,
  mode: "gpu" | "cpu",
  inputs: FieldValue[],
  params: Record<string, unknown>,
): Promise<FieldValue[]> {
  if (mode === "cpu") {
    if (!op.cpuGolden) throw new Error(`executor: op "${op.name}" has no cpuGolden (cpu mode)`);
    return op.cpuGolden(inputs, params);
  }
  const out = await op.execute(ctx, inputs, params);
  const ok = op.sanity ? op.sanity(out) : allFinite(out);
  if (!ok && op.cpuGolden) return op.cpuGolden(inputs, params); // invariant 5: validate + fall back
  return out;
}

/** Execute the graph as needed to produce `field`, returning its value. */
export async function pull(graph: Graph, field: GpuField, opts: PullOptions = {}): Promise<FieldValue> {
  const ctx = opts.ctx ?? { backend: nodeBackend };
  const mode = opts.mode ?? "gpu";
  const cache = new Map<string, FieldValue>();

  for (const node of topoOrder(graph, [field.producer])) {
    if (node.op === "source") {
      if (!node.source) throw new Error(`executor: source node "${node.id}" has no value`);
      const v = node.source;
      cache.set(key(node.id, "out"), v);
      opts.onValue?.(key(node.id, "out"), v);
      continue;
    }
    const op = getOp(node.op);
    const inputs = op.inputs.map((spec) => {
      const ref = node.inputs[spec.name]!;
      const v = cache.get(key(ref.node, ref.port));
      if (!v) throw new Error(`executor: input "${spec.name}" of "${node.id}" not computed`);
      return v;
    });
    const outs = await runNode(op, ctx, mode, inputs, node.params);
    op.outputs.forEach((o, i) => {
      const v = outs[i]!;
      cache.set(key(node.id, o.name), v);
      opts.onValue?.(key(node.id, o.name), v);
    });
  }

  const out = cache.get(key(field.producer, field.outPort));
  if (!out) throw new Error(`executor: requested field not produced`);
  return out;
}

/** Convenience: pull a numeric field and return its host data. */
export async function pullData(graph: Graph, field: GpuField, opts?: PullOptions): Promise<Float32Array> {
  const v = await pull(graph, field, opts);
  if (!v.data) throw new Error("executor: pulled field has no numeric data");
  return v.data instanceof Float32Array ? v.data : Float32Array.from(v.data);
}
