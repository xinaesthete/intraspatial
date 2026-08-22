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
import type { FieldValue, GpuField, ResidentPayload } from "./handle";
import { isResidentTexture, payloadsOf } from "./handle";
import type { GraphMemo } from "./memo";
import { hashSource, hashString, stableJSON } from "./memo";
import type { ExecCtx, OpType } from "./op";
import { getOp } from "./registry";
import { FieldRing } from "./ringBuffer";
import { textureToBuffer } from "./textureBridge";

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
  onBridge?: (key: string, direction: "download" | "upload" | "detexture") => void;
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
    if (v instanceof FieldRing) continue;
    if (v.buffer) backend.release(v.buffer);
    if (v.texture) backend.releaseTexture(v.texture);
  }
  state.clear();
}

/** Total resident bytes held in a sim state (feedback values + delay/history rings). Counts a
 *  GPU-resident value by its buffer's logical size — a resident feedback state occupies bytes
 *  just as a host one does, only on the device. */
export function simStateBytes(state: SimState): number {
  let total = 0;
  for (const v of state.values()) {
    if (v instanceof FieldRing) {
      total += v.byteLength;
    } else {
      const tex = v.texture ? v.texture.width * v.texture.height * 4 : 0;
      total += v.data?.byteLength ?? v.buffer?.byteLength ?? tex;
    }
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
  onBridge?: (key: string, direction: "download" | "upload" | "detexture") => void;
}

/** Bytes → f32 count, for the TEXTURE path only: a texture's readback goes through its render
 *  format, so it stays f32. Resident *buffers* of any dtype are handled by `downloadBuffer`. */
function residentF32Count(v: FieldValue): number {
  if (v.dtype !== "f32") {
    throw new Error(`executor: texture bridging is f32-only; got dtype "${v.dtype}" (ADR-0017 stage 1)`);
  }
  if (v.buffer) return v.buffer.byteLength / 4;
  // A texture-resident value's logical count is its extent — it has no byteLength of its own.
  return v.texture ? v.texture.width * v.texture.height : 0;
}

/** Strip the resident handles from a value whose leases have gone back to the pool, keeping
 *  whatever host representation it has. A stale reader then fails loudly instead of reading a
 *  buffer the pool has already handed to someone else — but a bundle keeps its `parts`, because
 *  after a host pull those parts carry the downloaded `data` and are the answer. */
function withoutResidentHandles(v: FieldValue): FieldValue {
  const parts = v.parts
    ? Object.fromEntries(Object.entries(v.parts).map(([name, part]) => [name, withoutResidentHandles(part)]))
    : undefined;
  return { ...v, buffer: undefined, texture: undefined, ...(parts ? { parts } : {}) };
}

/** Download a resident BUFFER as the typed array its dtype names.
 *
 *  f32 keeps the cached-wrapper `readbackF32` path (proven, and the common case); u32/i32 go
 *  through the raw byte readback, because decoding their words as floats would mangle them —
 *  `readBackBytes` in `device.ts` says why that is a separate path rather than a second wrapper. */
async function downloadBuffer(ctx: ExecCtx, v: FieldValue): Promise<Float32Array | Uint32Array | Int32Array> {
  const buf = v.buffer!;
  if (v.dtype === "f32") return ctx.backend.readbackF32(buf.buffer, buf.byteLength / 4);
  const bytes = await ctx.backend.readbackBytes(buf.buffer, buf.byteLength);
  return v.dtype === "u32" ? new Uint32Array(bytes) : new Int32Array(bytes);
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
  // `owned` holds the leases this tick is responsible for returning: payloads produced by a
  // resident op, plus any the bridge leased to upload a host value or to de-texture one. Payloads
  // arriving from elsewhere (a memo hit, a feedback store) are deliberately absent — the tick did
  // not lease them, so it must not release them.
  //
  // A LIST per key, not a single payload: once a texture-resident value is bridged for a buffer
  // consumer, that one port legitimately owns both, and dropping either would strand it.
  const owned = new Map<string, ResidentPayload[]>();
  const own = (k: string, p: ResidentPayload) => {
    const list = owned.get(k);
    if (list) list.push(p);
    else owned.set(k, [p]);
  };
  // --- Borrowed payloads (ADR-0023) ---
  //
  // An extract op hands back a bundle part's buffer AS IS — no copy, because copying `items` to
  // look at it defeats the bundle. The tick therefore does not own that payload, and the lender
  // must outlive the borrower: `lenders` records `borrowerKey -> the input keys it borrowed
  // from`, and a port with a live borrower is not released until the borrower itself dies.
  const lenders = new Map<string, Set<string>>();
  const borrowers = new Map<string, Set<string>>();
  const borrow = (borrower: string, lender: string) => {
    if (borrower === lender) return;
    let ls = lenders.get(borrower);
    if (!ls) lenders.set(borrower, (ls = new Set()));
    ls.add(lender);
    let bs = borrowers.get(lender);
    if (!bs) borrowers.set(lender, (bs = new Set()));
    bs.add(borrower);
  };
  /** A port is dead once its own consumers have run AND nothing is still borrowing from it. */
  const isDead = (k: string) => (remaining.get(k) ?? 0) <= 0 && (borrowers.get(k)?.size ?? 0) === 0;

  const releasePayload = (p: ResidentPayload) => {
    if (isResidentTexture(p)) o.ctx.backend.releaseTexture(p);
    else o.ctx.backend.release(p);
  };

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
    if (!isDead(k)) return;
    const list = owned.get(k);
    // A borrower owns nothing, but its death still frees its lenders, so keep going.
    if (list) {
      owned.delete(k);
      for (const p of list) releasePayload(p);
      // Drop the handles so a stale reader fails loudly instead of silently reading a resource the
      // pool has already handed to someone else.
      const v = pulled.get(k);
      if (v) pulled.set(k, withoutResidentHandles(v));
    }
    // This port is finished with whatever it borrowed; a lender whose last borrower just went
    // may now be releasable itself.
    const ls = lenders.get(k);
    if (!ls) return;
    lenders.delete(k);
    for (const lender of ls) {
      borrowers.get(lender)?.delete(k);
      releaseIfDead(lender);
    }
  };

  /** Record that a consumer of `k` has finished; recycle the value once its last one has. */
  const consume = (k: string) => {
    const left = (remaining.get(k) ?? 0) - 1;
    remaining.set(k, left);
    if (left <= 0) releaseIfDead(k);
  };

  /** A port that no node in this tick reads (a sibling output, or a borrower whose consumers have
   *  all run) still has to be swept, or its lease never returns. */
  const sweep = (k: string) => {
    if ((remaining.get(k) ?? 0) <= 0) releaseIfDead(k);
  };

  /** Move a lease from one port key to another, when a node emits its input unchanged (a
   *  `delay` passing `init` straight through while its history fills). Without this the lease
   *  is stranded: the input key's refcount drops to zero and releases a buffer the output key
   *  is still handing to downstream consumers. */
  const transferOwnership = (from: string, to: string) => {
    const list = owned.get(from);
    if (!list) return;
    owned.delete(from);
    for (const p of list) own(to, p);
  };

  /** Download one bundle part (ADR-0023 decision 3), by dtype: the index's u32 parts come back
   *  as `Uint32Array`, not as floats decoded from their bits. Parts are values, not ports, so they
   *  do not go through `hostAt`'s per-key cache. A texture-resident part still cannot cross, and
   *  names itself rather than leaving the bundle host-side with a hole in it. */
  const hostPart = async (v: FieldValue, label: string): Promise<FieldValue> => {
    if (v.parts) {
      const parts: Record<string, FieldValue> = {};
      for (const [name, part] of Object.entries(v.parts)) parts[name] = await hostPart(part, `${label}.${name}`);
      return { ...v, parts };
    }
    if (v.data || v.payload !== undefined || (!v.buffer && !v.texture)) return v;
    if (v.texture)
      throw new Error(`executor: bundle part "${label}" is texture-resident, which the bundle bridge does not handle (ADR-0023)`);
    return { ...v, data: await downloadBuffer(o.ctx, v) };
  };

  /** Ensure the value at `k` has host `data`, downloading it if resident-only. */
  const hostAt = async (k: string): Promise<FieldValue> => {
    const v = pulled.get(k);
    if (!v) throw new Error(`executor (unexpected): no value at ${k}`);
    if (v.parts) {
      o.onBridge?.(k, "download");
      const bridged = await hostPart(v, k);
      pulled.set(k, bridged);
      return bridged;
    }
    if (v.data || v.payload !== undefined) return v;
    if (!v.buffer && !v.texture) return v;
    // A texture cannot be read back directly by the f32 path; adapt first, then download once.
    const res = v.buffer ? v : await residentAt(k);
    o.onBridge?.(k, "download");
    const data = res.buffer ? await downloadBuffer(o.ctx, res) : await o.ctx.backend.readbackF32(res.buffer!.buffer, residentF32Count(res));
    const bridged: FieldValue = { ...res, data };
    pulled.set(k, bridged); // cache: a second consumer reuses the download
    return bridged;
  };

  /** Upload one bundle part, returning the leases taken so the caller can own them under the
   *  bundle's port key (ADR-0023). A part that is already resident, or carries only a payload,
   *  is passed through untouched. */
  const residentPart = async (v: FieldValue, took: ResidentPayload[]): Promise<FieldValue> => {
    if (v.parts) {
      const parts: Record<string, FieldValue> = {};
      for (const [name, part] of Object.entries(v.parts)) parts[name] = await residentPart(part, took);
      return { ...v, parts };
    }
    if (v.buffer || v.texture || v.payload !== undefined || !v.data) return v;
    const res = await o.ctx.backend.upload(v.data);
    took.push(res);
    return { ...v, buffer: res };
  };

  /** True when every part of a bundle already lives on the device. */
  const bundleIsResident = (v: FieldValue): boolean =>
    payloadsOf(v).length > 0 && !Object.values(v.parts ?? {}).some((p) => p.data && !p.buffer);

  /** Upload whichever parts of a bundle are still host-side, and own the leases that took. */
  const residentBundle = async (k: string, v: FieldValue): Promise<FieldValue> => {
    if (bundleIsResident(v)) return v;
    o.onBridge?.(k, "upload");
    const took: ResidentPayload[] = [];
    const bridged = await residentPart(v, took);
    pulled.set(k, bridged);
    for (const p of took) own(k, p);
    return bridged;
  };

  /** Ensure the value at `k` is GPU-resident, uploading it if host-only. */
  const residentAt = async (k: string): Promise<FieldValue> => {
    const v = pulled.get(k);
    if (!v) throw new Error(`executor (unexpected): no value at ${k}`);
    if (v.parts) return await residentBundle(k, v);
    if (v.buffer) return v;
    // Texture-resident, but this consumer binds a storage buffer. Adapt on-device — the value stays
    // GPU-resident throughout, so this is a copy, never a round trip. Paid only because a buffer
    // consumer exists: a render→render edge never reaches here.
    if (v.texture) {
      o.onBridge?.(k, "detexture");
      const res = await o.ctx.backend.lease(v.texture.width * v.texture.height * 4);
      await textureToBuffer(v.texture, res.buffer);
      const bridged: FieldValue = { ...v, buffer: res };
      pulled.set(k, bridged);
      own(k, res); // the tick leased it, so the tick returns it — alongside the texture
      return bridged;
    }
    if (!v.data) throw new Error(`executor: cannot make value at ${k} resident — it has no data`);
    o.onBridge?.(k, "upload");
    const res = await o.ctx.backend.upload(v.data);
    const bridged: FieldValue = { ...v, buffer: res };
    // Replace rather than mutate: `v` may be a source node's own value, which outlives the
    // tick, and attaching a pooled lease to it would let the refcounter recycle a buffer the
    // graph still points at.
    pulled.set(k, bridged);
    own(k, res); // the tick leased it, so the tick returns it
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

    // Which input port each payload came in on, so an output that hands one straight back is
    // recognised as a borrow rather than double-owned (ADR-0023).
    const lenderOf = new Map<ResidentPayload, string>();
    inputs.forEach((v, i) => {
      for (const p of payloadsOf(v)) if (!lenderOf.has(p)) lenderOf.set(p, inputKeys[i]!);
    });

    const outs = await runNode(op, o.ctx, o.mode, inputs, node.params);
    op.outputs.forEach((out, i) => {
      const v = outs[i];
      if (v === undefined) throw new Error(`executor (unexpected): no output for ${i}`);
      // Stamp the build-time-inferred basis (ADR-0006) so it propagates through ops
      // that don't set it themselves (e.g. editing wavelet coefficients then idwt).
      const b = node.outBases?.[i];
      if (b && v.basis === undefined) v.basis = b;
      // Axes/role/placement propagate the same way (ADR-0015 known gap + ADR-0018): stamp the
      // inferred facet only when the op left it unset, so a constructing op (splat, centroid
      // extract) that sets its own placement wins, while a generic op carries it through.
      const ax = node.outAxes?.[i];
      if (ax !== undefined && v.axes === undefined) v.axes = ax;
      const rl = node.outRoles?.[i];
      if (rl !== undefined && v.role === undefined) v.role = rl;
      const pl = node.outPlacements?.[i];
      if (pl !== undefined && v.placement === undefined) v.placement = pl;
      if (memo && ck) memo.set(`${ck}#${out.name}`, v);
      // A memoised value outlives the tick, so the tick must not reclaim its buffer. Not
      // owning it pins it — correct, but it means resident values accumulate in the memo until
      // eviction releases them (ADR-0017 leaves that coupling to a later stage).
      else {
        // ADR-0023: a payload the op did not create — it handed back one reachable from an input,
        // as an extract op does — is BORROWED, not owned. Owning it would return a lease to the
        // pool while the lender still points at it; the borrow edge instead keeps the lender
        // alive until this port dies.
        const outKey = key(node.id, out.name);
        for (const p of payloadsOf(v)) {
          const lender = lenderOf.get(p);
          if (lender) borrow(outKey, lender);
          else own(outKey, p);
        }
      }
      emit(node.id, out.name, v);
    });

    // This node has consumed its inputs; any whose last reader that was can go back to the pool.
    for (const k of inputKeys) consume(k);
  }

  /** A `pullResident` sink that BORROWED its payload (an extracted bundle part) is the one place
   *  a lender must be dismantled rather than kept whole: the caller is handed one buffer and owns
   *  it, so the lender's OTHER payloads — the index's `start` when you pulled its `items` — would
   *  otherwise never come back. Release those, and let the borrowed one travel to the caller. */
  const detachBorrowedSink = (k: string) => {
    const ls = lenders.get(k);
    if (!ls) return;
    const sinkValue = pulled.get(k);
    const keep = new Set<ResidentPayload>(sinkValue ? payloadsOf(sinkValue) : []);
    lenders.delete(k);
    for (const lender of ls) {
      borrowers.get(lender)?.delete(k);
      const list = owned.get(lender);
      if (!list) continue;
      owned.delete(lender);
      for (const p of list) if (!keep.has(p)) releasePayload(p);
    }
  };

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
    if (v.buffer || v.texture) {
      adopted.add(k); // the store owns this lease now, not the tick
      owned.delete(k);
    }
    // Return the state we just superseded. Guarded against a self-loop, where the incoming and
    // outgoing values are backed by the same resource and releasing would be a double free.
    if (prev && !(prev instanceof FieldRing)) {
      if (prev.buffer && prev.buffer !== v.buffer) o.ctx.backend.release(prev.buffer);
      if (prev.texture && prev.texture !== v.texture) o.ctx.backend.releaseTexture(prev.texture);
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

  // The sink is pinned, so nothing above released it; if it borrowed, split its lender now.
  if (!o.materialiseSink) detachBorrowedSink(key(field.producer, field.outPort));

  // Return every lease the tick still owns. Nodes release as their last reader completes; this
  // catches values nothing consumed. Pinned ports (the sink, anything fed back) are excluded.
  // Borrowers first: a borrower owns nothing, but until it dies its lender cannot be released.
  for (const k of [...lenders.keys()]) releaseIfDead(k);
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
