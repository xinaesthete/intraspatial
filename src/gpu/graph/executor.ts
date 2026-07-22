// Executor for the operation graph.
//
// `pull(graph, field)` resolves the transitive dependencies of the requested field,
// topologically orders them, and executes each node once (deduped), with optional
// content-addressed memoisation. A failing op is a fail-state, not a fallback
// (docs/gpu-resource-sync.md, invariant 5 as revised by ADR-0017): errors propagate.
//
// Feedback / time. The graph is a DAG *per tick*. A `feedback` (delay) node outputs
// the PREVIOUS tick's value (seeded by `init`), so it acts as a source within a tick
// and the edge wired to its `next` input is a deferred write committed after the
// tick — this is what cuts the cycle (a unit delay, z⁻¹). `advance(graph, field,
// {steps, state})` runs that loop; `pull` is one tick from a fresh (init) state.

import type { GpuBackend } from "./backend";
import { nodeBackend } from "./backend.node";
import type { Graph, GraphNode } from "./graph";
import type { FieldValue, GpuField, ResidentBuffer } from "./handle";
import type { GraphMemo } from "./memo";
import { hashSource, hashString, stableJSON } from "./memo";
import type { ExecCtx, OpType } from "./op";
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
  /** Observe every executor-inserted host↔GPU transfer (ADR-0017). The `resident?` bridge can
   *  mask a regression — an op that quietly fell back to a host round-trip still produces the
   *  right numbers — so this makes the materialisations visible by node id. */
  onBridge?: (key: string, direction: "download" | "upload") => void;
}

/** Per-node state persisted across ticks, keyed by stableKey: a `feedback` node stores its
 *  last value (a `FieldValue`); a `delay` node stores a `FieldRing` history. */
export type SimState = Map<string, FieldValue | FieldRing>;
export const createSimState = (): SimState => new Map();

/** Return every GPU lease a sim state holds and empty it.
 *
 *  A resident `feedback` state owns a pooled buffer between ticks (ADR-0017 stage 3), so simply
 *  dropping the state would strand it: the pool never destroys buffers, so a stranded lease is
 *  never reused and the footprint grows by one buffer per discarded simulation. Call this when
 *  resetting or disposing of a state. `advance({reset: true})` does it for you. */
export function disposeSimState(state: SimState, backend: GpuBackend): void {
  for (const v of state.values()) {
    if (!(v instanceof FieldRing) && v.buffer) backend.release(v.buffer);
  }
  state.clear();
}

/** Total resident bytes held in a sim state (feedback values + delay/history rings). Counts a
 *  GPU-resident value by its buffer's logical size — a resident feedback state occupies bytes
 *  just as a host one does, only on the device. */
export function simStateBytes(state: SimState): number {
  let total = 0;
  for (const v of state.values()) {
    total += v instanceof FieldRing ? v.byteLength : (v.data?.byteLength ?? v.buffer?.byteLength ?? 0);
  }
  return total;
}

const key = (nodeId: string, port: string) => `${nodeId}:${port}`;
const storeKey = (node: GraphNode) => node.stableKey ?? node.id;

/** Within a tick, `feedback`/`delay` only *read* their `init` input; `next` is a deferred write
 *  committed after the tick. Both the topological order and the liveness refcount must cut the
 *  same edge, or the DAG and the accounting disagree. */
const readsWithinTick = (node: GraphNode, port: string) => (node.op === "feedback" || node.op === "delay" ? port === "init" : true);

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
    const deps = Object.entries(node.inputs)
      .filter(([port]) => readsWithinTick(node, port))
      .map(([, ref]) => ref);
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
  // No validate-and-fall-back (gpu-resource-sync invariant 5, revised 2026-07-13). An op that
  // cannot run on the GPU is a fail-state: whatever `execute` throws propagates, and the
  // executor does not silently substitute a CPU result. The scan this used to do —
  // `op.sanity ?? allFinite` over every element of every output — is incompatible with
  // invariant 4, because reading every element forces a download on every pull. `cpuGolden`
  // remains the test-time reference oracle and the `mode: "cpu"` implementation; it is not a
  // production recovery path.
  return op.execute(ctx, inputs, params);
}

interface TickOptions {
  ctx: ExecCtx;
  mode: "gpu" | "cpu";
  memo?: GraphMemo;
  store: SimState;
  onValue?: (key: string, value: FieldValue) => void;
  /** Download the sink into host `data` before returning (see invariant 4's sink rule).
   *  `pull`/`advance` do; `pullResident` does not. */
  materialiseSink: boolean;
  /** Called for every executor-inserted representation change, with the port key and the
   *  direction. A resident op silently falling back to a host round-trip looks correct, so the
   *  bridge needs to be observable (ADR-0017, "the `resident?` bridge can mask regressions"). */
  onBridge?: (key: string, direction: "download" | "upload") => void;
}

/** Bytes → f32 count. The bridge is f32-only for now: `readbackF32` converts element-wise
 *  rather than reinterpreting bits, so an i32/u32 resident value would come back mangled.
 *  Guarded explicitly rather than left to produce silently wrong numbers. */
function residentF32Count(v: FieldValue): number {
  if (v.dtype !== "f32") {
    throw new Error(`executor: resident bridging is f32-only; got dtype "${v.dtype}" (ADR-0017 stage 1)`);
  }
  return (v.buffer?.byteLength ?? 0) / 4;
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

  // --- Tier-2 liveness bookkeeping (ADR-0017, invariant 3) ---
  //
  // `owned` holds the leases this tick is responsible for returning: buffers produced by a
  // resident op, plus any the bridge leased to upload a host value. Buffers arriving from
  // elsewhere (a memo hit, a feedback store) are deliberately absent — the tick did not lease
  // them, so it must not release them.
  const owned = new Map<string, ResidentBuffer>();

  // Consumers still to run for each produced port. Counting the same edges the topological
  // order walks (cut `next` back-edges excluded) means a port hits zero exactly when its last
  // reader within the tick has finished.
  const remaining = new Map<string, number>();
  for (const n of order) {
    for (const [port, ref] of Object.entries(n.inputs)) {
      if (!readsWithinTick(n, port)) continue;
      const k = key(ref.node, ref.port);
      remaining.set(k, (remaining.get(k) ?? 0) + 1);
    }
  }

  // Values that outlive the tick and must never be released into the pool:
  //   - the sink, which is what the caller receives;
  //   - anything fed back, because it is committed into the sim state for the next tick.
  const pinned = new Set<string>([key(field.producer, field.outPort)]);
  for (const n of order) {
    if (n.op !== "feedback" && n.op !== "delay") continue;
    const nx = n.inputs.next;
    if (nx) pinned.add(key(nx.node, nx.port));
  }

  // Ports whose lease has been handed to the sim state (a feedback node's stored value). The
  // tick allocated them but no longer owns them: the store releases them, one tick later.
  const adopted = new Set<string>();

  const releaseIfDead = (k: string) => {
    if (pinned.has(k) || adopted.has(k)) return;
    const res = owned.get(k);
    if (!res) return;
    owned.delete(k);
    o.ctx.backend.release(res);
    // Drop the handle so a stale reader fails loudly instead of silently reading a buffer the
    // pool has already handed to someone else.
    const v = pulled.get(k);
    if (v) pulled.set(k, { ...v, buffer: undefined });
  };

  /** Record that a consumer of `k` has finished; recycle the value once its last one has. */
  const consume = (k: string) => {
    const left = (remaining.get(k) ?? 0) - 1;
    remaining.set(k, left);
    if (left <= 0) releaseIfDead(k);
  };

  /** Move a lease from one port key to another, when a node emits its input unchanged (a
   *  `delay` passing `init` straight through while its history fills). Without this the lease
   *  is stranded: the input key's refcount drops to zero and releases a buffer the output key
   *  is still handing to downstream consumers. */
  const transferOwnership = (from: string, to: string) => {
    const res = owned.get(from);
    if (!res) return;
    owned.delete(from);
    owned.set(to, res);
  };

  /** Ensure the value at `k` has host `data`, downloading it if resident-only. */
  const hostAt = async (k: string): Promise<FieldValue> => {
    const v = pulled.get(k);
    if (!v) throw new Error(`executor (unexpected): no value at ${k}`);
    if (v.data || v.payload !== undefined || !v.buffer) return v;
    o.onBridge?.(k, "download");
    const data = await o.ctx.backend.readbackF32(v.buffer.buffer, residentF32Count(v));
    const bridged: FieldValue = { ...v, data };
    pulled.set(k, bridged); // cache: a second consumer reuses the download
    return bridged;
  };

  /** Ensure the value at `k` is GPU-resident, uploading it if host-only. */
  const residentAt = async (k: string): Promise<FieldValue> => {
    const v = pulled.get(k);
    if (!v) throw new Error(`executor (unexpected): no value at ${k}`);
    if (v.buffer) return v;
    if (!v.data) throw new Error(`executor: cannot make value at ${k} resident — it has no data`);
    o.onBridge?.(k, "upload");
    const res = await o.ctx.backend.upload(v.data);
    const bridged: FieldValue = { ...v, buffer: res };
    // Replace rather than mutate: `v` may be a source node's own value, which outlives the
    // tick, and attaching a pooled lease to it would let the refcounter recycle a buffer the
    // graph still points at.
    pulled.set(k, bridged);
    owned.set(k, res); // the tick leased it, so the tick returns it
    return bridged;
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
      const initRef = node.inputs.init;
      if (!initRef) throw new Error(`executor: feedback node "${node.id}" has no init`);
      const initKey = key(initRef.node, initRef.port);
      const stored = o.store.get(sk);
      let cur = stored instanceof FieldRing ? undefined : stored;
      if (cur === undefined) {
        // first tick (or post-reset): seed from `init`.
        cur = pulled.get(initKey);
        if (!cur) throw new Error(`executor: feedback init of "${node.id}" not computed`);
        o.store.set(sk, cur);
        // The store keeps this value past the end of the tick, so its lease must not be
        // recycled here — the store releases it at the next commit (the ping-pong swap).
        if (cur.buffer) adopted.add(initKey);
      }
      // `init` is evaluated every tick but only *used* on the seeding one; recycle it on the
      // ticks where it goes unread, or a resident init leaks one buffer per tick.
      consume(initKey);
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
      const initKey = key(initRef.node, initRef.port);
      const initVal = pulled.get(initKey);
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
      // While the history is still filling, `out` IS `init` — the same value under two keys.
      // Hand the lease to the output key so downstream consumers govern its lifetime; releasing
      // it on the input key would recycle a buffer they are about to read.
      if (out === initVal) transferOwnership(initKey, key(node.id, "out"));
      consume(initKey);
      continue;
    }

    const op = getOp(node.op);
    let ck: string | undefined;
    if (memo) {
      const inputKeys = op.inputs.map((spec) => {
        const ref = node.inputs[spec.name];
        if (!ref) throw new Error(`executor (unexpected): No input '${spec.name}' on ${node.id}`);
        return `${contentKey.get(ref.node) ?? ref.node}#${ref.port}`;
      });
      ck = hashString(`${node.op}|${stableJSON(node.params)}|${inputKeys.join(",")}`);
      contentKey.set(node.id, ck);
      const hits = op.outputs.map((out) => memo.get(`${ck}#${out.name}`));
      if (hits.every((h) => h !== undefined)) {
        op.outputs.forEach((out, i) => {
          const hit = hits[i];
          if (hit !== undefined) emit(node.id, out.name, hit);
        });
        continue;
      }
    }

    const inputKeys = op.inputs.map((spec) => {
      const ref = node.inputs[spec.name];
      if (!ref) throw new Error(`executor (unexpected): No input '${spec.name}' on ${node.id}`);
      const k = key(ref.node, ref.port);
      if (!pulled.has(k)) throw new Error(`executor: input "${spec.name}" of "${node.id}" not computed`);
      return k;
    });

    // The bridge (ADR-0017). A resident op is handed GPU-resident inputs; every other op — and
    // anything running in `cpu` mode — is handed host data. Either side may need a transfer,
    // and this is the only place one is inserted, which is what makes invariant 4 measurable:
    // once both ends of an edge are resident, no transfer happens at all.
    const wantResident = o.mode === "gpu" && op.resident === true;
    const inputs: FieldValue[] = [];
    for (const k of inputKeys) inputs.push(wantResident ? await residentAt(k) : await hostAt(k));

    const outs = await runNode(op, o.ctx, o.mode, inputs, node.params);
    op.outputs.forEach((out, i) => {
      const v = outs[i];
      if (v === undefined) throw new Error(`executor (unexpected): no output for ${i}`);
      // Stamp the build-time-inferred basis (ADR-0006) so it propagates through ops
      // that don't set it themselves (e.g. editing wavelet coefficients then idwt).
      const b = node.outBases?.[i];
      if (b && v.basis === undefined) v.basis = b;
      if (memo && ck) memo.set(`${ck}#${out.name}`, v);
      // A memoised value outlives the tick, so the tick must not reclaim its buffer. Not
      // owning it pins it — correct, but it means resident values accumulate in the memo until
      // eviction releases them (ADR-0017 leaves that coupling to a later stage).
      else if (v.buffer) owned.set(key(node.id, out.name), v.buffer);
      emit(node.id, out.name, v);
    });

    // This node has consumed its inputs; any whose last reader that was can go back to the pool.
    for (const k of inputKeys) consume(k);
  }

  // Commit: store each feedback node's fed-back value for the next tick.
  //
  // RESIDENT FEEDBACK = PING-PONG (invariant 1). When the fed-back value is GPU-resident, the
  // store adopts its lease and hands the *outgoing* state's buffer straight back to the pool.
  // Because the pool is a free list, next tick's producer is then handed that very buffer: two
  // buffers alternate for the lifetime of the simulation, which is exactly HsPf's
  // `[src, dst] = [dst, src]` expressed through the lease API rather than by hand. The steady
  // state is constant in tick count — no per-tick allocation, and no download anywhere in the
  // loop, which is what invariant 1 asks for and what Tier-1 could not provide.
  for (const node of feedbackNodes) {
    const nextRef = node.inputs.next;
    if (!nextRef) continue;
    const k = key(nextRef.node, nextRef.port);
    const v = pulled.get(k);
    if (!v) continue;
    const sk = storeKey(node);
    const prev = o.store.get(sk);
    o.store.set(sk, v);
    if (v.buffer) {
      adopted.add(k); // the store owns this lease now, not the tick
      owned.delete(k);
    }
    // Return the state we just superseded. Guarded against a self-loop, where the incoming and
    // outgoing values are backed by the same buffer and releasing would be a double free.
    if (prev && !(prev instanceof FieldRing) && prev.buffer && prev.buffer !== v.buffer) {
      o.ctx.backend.release(prev.buffer);
    }
  }
  // Commit: push each delay node's fed value into its history ring. The ring is host-backed
  // (a `delay(k)` keeps k interpolatable frames), so a resident value is downloaded here —
  // deliberate and explicit, not an accident of the bridge. ADR-0017 stage 3 leaves the
  // ring-of-leases generalisation open; until then `delay` is a host sink.
  for (const node of delayNodes) {
    const nextRef = node.inputs.next;
    if (!nextRef) continue;
    const ring = o.store.get(storeKey(node));
    if (!(ring instanceof FieldRing)) continue;
    const k = key(nextRef.node, nextRef.port);
    ring.push(await hostAt(k));
    // The ring copied the bytes into its own host store, so the lease is dead the moment the
    // push returns. Unlike `feedback`, nothing adopts it: a `FieldRing` holds host frames, so a
    // `delay` keeps no GPU buffers and has nothing to ping-pong. Generalising the ring to a ring
    // of leases is left open by ADR-0017 — ping-pong is the depth-1 case, and a depth-k history
    // needs k+1 rotating buffers plus an on-device path for `sample`'s interpolation, which is a
    // larger change than stage 3 calls for. Until then `delay` is an explicit host boundary.
    pinned.delete(k);
    releaseIfDead(k);
  }

  // Invariant 4's sink rule: the download happens here, once, because the *host* consumes the
  // value — not because an interior edge needed it. A render sink asks for `pullResident` and
  // this never runs, which is why the target for a render-terminated graph is zero downloads.
  if (o.materialiseSink) {
    const sinkKey = key(field.producer, field.outPort);
    await hostAt(sinkKey);
    // The host now holds the bytes, so the device copy is dead and its lease goes back. Without
    // this the sink is the one value pinned forever and the pool grows by one buffer per pull.
    // `pullResident` deliberately skips this: there the *caller* owns the buffer.
    pinned.delete(sinkKey);
    releaseIfDead(sinkKey);
  }

  // Return every lease the tick still owns. Nodes release as their last reader completes; this
  // catches values nothing consumed. Pinned ports (the sink, anything fed back) are excluded.
  for (const k of [...owned.keys()]) releaseIfDead(k);

  return pulled;
}

/** Execute the graph to produce `field`, returning its value (one tick from a fresh
 *  feedback state, i.e. seeds at `init`). For a stepped simulation use `advance`.
 *
 *  The returned value is host-materialised — this is a *host-consuming* sink. To keep the
 *  result on the device (a render/display consumer), use `pullResident`. */
export async function pull(graph: Graph, field: GpuField, opts: PullOptions = {}): Promise<FieldValue> {
  return runOnce(graph, field, opts, true);
}

/** Execute the graph to produce `field` and return it **without downloading** (ADR-0017 §5).
 *
 *  For consumers that read the value on-device — a render pass binding the buffer directly, as
 *  HsPf's field renderer already does. Only a subset of sinks are host-consuming, so the
 *  honest reading of invariant 4 is "download only where the host actually consumes the value",
 *  and a render-terminated graph should perform **zero** downloads, not one.
 *
 *  The returned value carries `buffer` when the producing op is resident, and `data` when it is
 *  not — a Tier-1 op still computes on the host, and this does not upload it just to look
 *  resident. Check which you got.
 *
 *  OWNERSHIP: when a `buffer` comes back, the **caller** owns that lease and should hand it to
 *  `backend.release` when done, or the pool grows by one buffer per pull. `pull` has no such
 *  requirement — it downloads and returns the lease itself. */
export async function pullResident(graph: Graph, field: GpuField, opts: PullOptions = {}): Promise<FieldValue> {
  return runOnce(graph, field, opts, false);
}

async function runOnce(graph: Graph, field: GpuField, opts: PullOptions, materialiseSink: boolean): Promise<FieldValue> {
  const pulled = await runTick(graph, field, {
    ctx: opts.ctx ?? { backend: nodeBackend },
    mode: opts.mode ?? "gpu",
    memo: opts.cache,
    store: createSimState(),
    onValue: opts.onValue,
    onBridge: opts.onBridge,
    materialiseSink,
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
  const ctx = opts.ctx ?? { backend: nodeBackend };
  // Dispose rather than clear: a resident feedback state holds a pooled lease, and dropping the
  // map would strand it (the pool never destroys, so a stranded buffer is never reused).
  if (opts.reset) disposeSimState(store, ctx.backend);
  const steps = Math.max(1, opts.steps ?? 1);
  const mode = opts.mode ?? "gpu";

  let last: FieldValue | undefined;
  for (let t = 0; t < steps; t++) {
    const pulled = await runTick(graph, field, {
      ctx,
      mode,
      store,
      onValue: opts.onValue,
      onBridge: opts.onBridge,
      materialiseSink: true,
    });
    last = pulled.get(key(field.producer, field.outPort));
    if (!last) throw new Error("advance: sink not produced");
    opts.onFrame?.(t, last);
  }
  if (last === undefined) throw new Error(`executor (unexpected): no value output for field ${field.id}`);
  return last;
}

/** Convenience: pull a numeric field and return its host data. */
export async function pullData(graph: Graph, field: GpuField, opts?: PullOptions): Promise<Float32Array> {
  const v = await pull(graph, field, opts);
  if (!v.data) throw new Error("executor: pulled field has no numeric data");
  return v.data instanceof Float32Array ? v.data : Float32Array.from(v.data);
}
