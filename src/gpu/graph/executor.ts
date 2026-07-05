// Executor for the operation graph.
//
// `pull(graph, field)` resolves the transitive dependencies of the requested field,
// topologically orders them, and executes each node once (deduped), with per-node
// validate→CPU-fallback (docs/gpu-resource-sync.md, invariant 5) and optional
// content-addressed memoisation.
//
// Feedback / time. The graph is a DAG *per tick*. A `feedback` (delay) node outputs
// the PREVIOUS tick's value (seeded by `init`), so it acts as a source within a tick
// and the edge wired to its `next` input is a deferred write committed after the
// tick — this is what cuts the cycle (a unit delay, z⁻¹). `advance(graph, field,
// {steps, state})` runs that loop; `pull` is one tick from a fresh (init) state.

import { nodeBackend } from "./backend.node";
import type { GraphNode } from "./graph";
import { Graph } from "./graph";
import type { FieldValue, GpuField } from "./handle";
import type { GraphMemo } from "./memo";
import { hashSource, hashString, stableJSON } from "./memo";
import type { ExecCtx, OpType } from "./op";
import { allFinite } from "./op";
import { getOp } from "./registry";
import { FieldRing } from "./ringBuffer";

export interface PullOptions {
  /** Which backend to run native ops on. Defaults to the Node (Dawn) backend. */
  ctx?: ExecCtx;
  /** "gpu" (default) runs each op's `execute`; "cpu" forces every op through its
   *  `cpuGolden` — an independent reference the GPU path must match. */
  mode?: "gpu" | "cpu";
  /** Receives a value for every produced port (`"nodeId:port"`). */
  onValue?: (key: string, value: FieldValue) => void;
  /** A persistent, content-addressed memo (from `createMemo`). Ignored for graphs
   *  containing feedback (their values change each tick). */
  cache?: GraphMemo;
}

/** Per-node state persisted across ticks, keyed by stableKey: a `feedback` node stores its
 *  last value (a `FieldValue`); a `delay` node stores a `FieldRing` history. */
export type SimState = Map<string, FieldValue | FieldRing>;
export const createSimState = (): SimState => new Map();

/** Total resident bytes held in a sim state (feedback values + delay/history rings). */
export function simStateBytes(state: SimState): number {
  let total = 0;
  for (const v of state.values()) total += v instanceof FieldRing ? v.byteLength : v.data ? v.data.byteLength : 0;
  return total;
}

const key = (nodeId: string, port: string) => `${nodeId}:${port}`;
const storeKey = (node: GraphNode) => node.stableKey ?? node.id;

/** Topologically order the deps of `roots`, producer-first. Feedback nodes only
 *  depend on their `init` input — the `next` back-edge is cut, so the loop is a DAG
 *  per tick. Any remaining cycle (one not through a delay) is a real error. */
function topoOrder(graph: Graph, roots: string[]): GraphNode[] {
  const order: GraphNode[] = [];
  const state = new Map<string, 0 | 1>(); // 0 = visiting, 1 = done
  const visit = (id: string) => {
    const s = state.get(id);
    if (s === 1) return;
    if (s === 0) throw new Error(`executor: cycle through node "${id}" (a cycle must pass through a feedback node)`);
    state.set(id, 0);
    const node = graph.getNode(id);
    // feedback (z⁻¹) and delay (z⁻ᵏ) only depend on `init` within a tick — the `next`
    // back-edge is a deferred write, so the loop is a DAG per tick.
    const deps = node.op === "feedback" || node.op === "delay" ? (node.inputs.init ? [node.inputs.init] : []) : Object.values(node.inputs);
    for (const ref of deps) visit(ref.node);
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

interface TickOptions {
  ctx: ExecCtx;
  mode: "gpu" | "cpu";
  memo?: GraphMemo;
  store: SimState;
  onValue?: (key: string, value: FieldValue) => void;
}

/** Run one tick: execute the cut-DAG, then commit feedback `next` values into the
 *  store for the following tick. Returns every produced value by "nodeId:port". */
async function runTick(graph: Graph, field: GpuField, o: TickOptions): Promise<Map<string, FieldValue>> {
  const order = topoOrder(graph, [field.producer]);
  // Content-memo is unsound across ticks for stateful graphs (values change each tick),
  // so disable it whenever a feedback or delay node is in play.
  const memo = order.some((n) => n.op === "feedback" || n.op === "delay") ? undefined : o.memo;

  const pulled = new Map<string, FieldValue>();
  const contentKey = new Map<string, string>();
  const feedbackNodes: GraphNode[] = [];
  const delayNodes: GraphNode[] = [];
  const emit = (nodeId: string, port: string, v: FieldValue) => {
    pulled.set(key(nodeId, port), v);
    o.onValue?.(key(nodeId, port), v);
  };

  for (const node of order) {
    if (node.op === "source") {
      if (!node.source) throw new Error(`executor: source node "${node.id}" has no value`);
      const v = node.source;
      if (memo) {
        const ck = hashSource(v);
        contentKey.set(node.id, ck);
        memo.set(`${ck}#out`, v);
      }
      emit(node.id, "out", v);
      continue;
    }

    if (node.op === "feedback") {
      const sk = storeKey(node);
      const stored = o.store.get(sk);
      let cur = stored instanceof FieldRing ? undefined : stored;
      if (cur === undefined) {
        // first tick (or post-reset): seed from `init`.
        const initRef = node.inputs.init;
        if (!initRef) throw new Error(`executor: feedback node "${node.id}" has no init`);
        cur = pulled.get(key(initRef.node, initRef.port));
        if (!cur) throw new Error(`executor: feedback init of "${node.id}" not computed`);
        o.store.set(sk, cur);
      }
      feedbackNodes.push(node);
      emit(node.id, "state", cur);
      continue;
    }

    if (node.op === "delay") {
      // z⁻ᵏ: emit the value from `depth` ticks ago (a FieldRing of capacity `depth`),
      // seeded by `init` until enough history has accrued. `next` is pushed at commit.
      const sk = storeKey(node);
      const depth = Math.max(1, Math.round(Number(node.params.depth ?? 1)));
      const initRef = node.inputs.init;
      if (!initRef) throw new Error(`executor: delay node "${node.id}" has no init`);
      const initVal = pulled.get(key(initRef.node, initRef.port));
      if (!initVal) throw new Error(`executor: delay init of "${node.id}" not computed`);
      const stored = o.store.get(sk);
      let ring = stored instanceof FieldRing ? stored : undefined;
      if (!ring) {
        ring = new FieldRing(initVal.shape, depth, initVal.element, initVal.dtype);
        o.store.set(sk, ring);
      }
      const out = ring.frames >= depth ? ring.frame(depth - 1) : initVal;
      delayNodes.push(node);
      emit(node.id, "out", out);
      continue;
    }

    const op = getOp(node.op);
    let ck: string | undefined;
    if (memo) {
      const inputKeys = op.inputs.map((spec) => {
        const ref = node.inputs[spec.name]!;
        return `${contentKey.get(ref.node) ?? ref.node}#${ref.port}`;
      });
      ck = hashString(`${node.op}|${stableJSON(node.params)}|${inputKeys.join(",")}`);
      contentKey.set(node.id, ck);
      const hits = op.outputs.map((out) => memo.get(`${ck}#${out.name}`));
      if (hits.every((h) => h !== undefined)) {
        op.outputs.forEach((out, i) => emit(node.id, out.name, hits[i]!));
        continue;
      }
    }

    const inputs = op.inputs.map((spec) => {
      const ref = node.inputs[spec.name]!;
      const v = pulled.get(key(ref.node, ref.port));
      if (!v) throw new Error(`executor: input "${spec.name}" of "${node.id}" not computed`);
      return v;
    });
    const outs = await runNode(op, o.ctx, o.mode, inputs, node.params);
    op.outputs.forEach((out, i) => {
      const v = outs[i]!;
      // Stamp the build-time-inferred basis (ADR-0006) so it propagates through ops
      // that don't set it themselves (e.g. editing wavelet coefficients then idwt).
      const b = node.outBases?.[i];
      if (b && v.basis === undefined) v.basis = b;
      if (memo && ck) memo.set(`${ck}#${out.name}`, v);
      emit(node.id, out.name, v);
    });
  }

  // Commit: store each feedback node's fed-back value for the next tick.
  for (const node of feedbackNodes) {
    const nextRef = node.inputs.next;
    if (nextRef) {
      const v = pulled.get(key(nextRef.node, nextRef.port));
      if (v) o.store.set(storeKey(node), v);
    }
  }
  // Commit: push each delay node's fed value into its history ring.
  for (const node of delayNodes) {
    const nextRef = node.inputs.next;
    if (nextRef) {
      const v = pulled.get(key(nextRef.node, nextRef.port));
      const ring = o.store.get(storeKey(node));
      if (v && ring instanceof FieldRing) ring.push(v);
    }
  }

  return pulled;
}

/** Execute the graph to produce `field`, returning its value (one tick from a fresh
 *  feedback state, i.e. seeds at `init`). For a stepped simulation use `advance`. */
export async function pull(graph: Graph, field: GpuField, opts: PullOptions = {}): Promise<FieldValue> {
  const pulled = await runTick(graph, field, {
    ctx: opts.ctx ?? { backend: nodeBackend },
    mode: opts.mode ?? "gpu",
    memo: opts.cache,
    store: createSimState(),
    onValue: opts.onValue,
  });
  const out = pulled.get(key(field.producer, field.outPort));
  if (!out) throw new Error(`executor: requested field not produced`);
  return out;
}

export interface AdvanceOptions extends PullOptions {
  /** Number of ticks to run (default 1). */
  steps?: number;
  /** Persistent feedback state across calls (from `createSimState`). */
  state?: SimState;
  /** Clear the feedback state before running (re-seed from `init`). */
  reset?: boolean;
  /** Called after each tick with the sink value. */
  onFrame?: (frame: number, value: FieldValue) => void;
}

/** Advance a feedback graph `steps` ticks, returning the sink at the final tick. */
export async function advance(graph: Graph, field: GpuField, opts: AdvanceOptions = {}): Promise<FieldValue> {
  const store = opts.state ?? createSimState();
  if (opts.reset) store.clear();
  const steps = Math.max(1, opts.steps ?? 1);
  const ctx = opts.ctx ?? { backend: nodeBackend };
  const mode = opts.mode ?? "gpu";

  let last: FieldValue | undefined;
  for (let t = 0; t < steps; t++) {
    const pulled = await runTick(graph, field, { ctx, mode, store, onValue: opts.onValue });
    last = pulled.get(key(field.producer, field.outPort));
    if (!last) throw new Error("advance: sink not produced");
    opts.onFrame?.(t, last);
  }
  return last!;
}

/** Convenience: pull a numeric field and return its host data. */
export async function pullData(graph: Graph, field: GpuField, opts?: PullOptions): Promise<Float32Array> {
  const v = await pull(graph, field, opts);
  if (!v.data) throw new Error("executor: pulled field has no numeric data");
  return v.data instanceof Float32Array ? v.data : Float32Array.from(v.data);
}
