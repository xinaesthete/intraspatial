# ADR-0017 — Tier-2 resident buffer edges (implementing gpu-resource-sync invariants 1/3/4)

Status: **draft / proposed** (2026-07-13)

## Context

This ADR proposes **nothing new**. `docs/gpu-resource-sync.md` already specifies resident edges
as a requirement the executor *must* hold — invariant 4, verbatim:

> **Boundary-only transfer.** `upload` at sources, `download` at sinks; interior edges stay
> on-GPU. Download is the slow + Dawn-on-Node-fragile op (readback ceiling ~512² there) —
> minimise it and isolate it to sinks.

The executor violates it on **every edge**, and now we have the number.
[`readbackBudget.gpu.test.ts`](../../src/gpu/graph/readbackBudget.gpu.test.ts) measures real
downloads (`mapAsync(READ)`) across chains of increasing length:

```
1 op:  1 download,   6.0 KB
2 ops: 2 downloads,  8.3 KB
3 ops: 3 downloads, 10.5 KB
4 ops: 4 downloads, 12.8 KB
```

**Downloads == op count.** The byte deltas confirm the mechanism exactly: +2.25 KB per added op
= one 24×24 f32 grid (2304 B). Every interior edge downloads its whole field and re-uploads it.
Under invariant 4 all four rows should read `1 download`.

**Why it went unnoticed for so long.** The repo's validation discipline is **per-op** (every node
has a `cpuGolden` the GPU must match); invariant 4 is a **graph-level** property. No test could
fail because an interior edge round-tripped — every op passes its golden either way. The
invariant was unenforceable by construction. [`instrument.ts`](../../src/gpu/graph/instrument.ts)
(patching `GPUBuffer.prototype.mapAsync`, the universal download chokepoint, with no op changes)
plus the budget ratchet is the first thing in the repo that *can* fail on a host round-trip.

**Scale.** At 24×24 this is invisible. A 1024×1024 4-channel f32 tile is **16 MiB**; a 5-op chain
becomes ~80 MiB down + 80 MiB back up where the invariant requires 16 MiB once — and Dawn-on-Node's
~512² readback ceiling means a real tile cannot traverse this path intact at all. For an
interactive brush (ADR-0015's wand) that cost lands *per drag event*.

**There is a working reference implementation — outside the graph.** HsPf keeps state GPU-resident
across steps via ping-pong (`src/gpu/sim/hspf/kernel.ts:350` "No readback — the state stays
resident") and its renderer binds that buffer directly (`playground/src/hspf/fieldRenderer.ts:182`).
That is invariants 1 + 4 implemented correctly, by hand, in a module that deliberately bypasses the
op-graph (main's ADR-0011 decision 1). The satellites (`hspf/*`, `implicitGpu.ts`, `sweptGpu.ts`)
are not a rejection of the architecture — they are *compliance with the spec the graph fails to
provide*. Each one added raises the eventual retrofit cost.

## Decision

Implement invariants **1 (ping-pong for read-modify-write)**, **3 (pool reuse by liveness)** and
**4 (boundary-only transfer)** in the executor, following the **additive-facet discipline** that
landed `element`/`basis`/`axes`/`role`: absent facet ⇒ today's behaviour, so **every existing op
keeps working unchanged**.

### 1. `FieldValue` gains a resident representation

```ts
interface ResidentBuffer {
  buffer: GPUBuffer;
  byteLength: number;
  lease: LeaseToken;        // pool identity for invariant-3 liveness
}

interface FieldValue {
  /* shape, dtype, element?, basis?, axes?, role? unchanged */
  data?: Float32Array | Int32Array | Uint32Array;  // host (Tier-1)
  buffer?: ResidentBuffer;                          // GPU  (Tier-2)  <-- new
  payload?: unknown;
}
```

Invariant: at least one of `data` / `buffer` / `payload` is present. A value may transiently carry
both (immediately after a sink download).

### 2. Ops opt in; the executor bridges

```ts
interface OpType {
  /** Handles `FieldValue`s carrying `buffer` instead of `data`. Absent ⇒ host-only (Tier-1). */
  resident?: boolean;
}
```

The executor materialises host `data` before invoking a non-resident op, and uploads before a
resident op that received a host value. **This is what makes the migration incremental**: nothing
breaks on day one, and each op converted removes downloads measurably from the budget test.

### 3. Pool leases with DAG liveness (invariant 3) — return, never destroy

`GpuBackend` grows the lease API its own header already anticipates ("Tier-2 resident ops will
grow the pool API (lease/return by liveness) on top"):

```ts
lease(byteLength: number, usage: GPUBufferUsageFlags): ResidentBuffer;
release(b: ResidentBuffer): void;   // returns to pool — NEVER buffer.destroy()
```

The executor already has `topoOrder` and each node's `inputs`, so it can refcount consumers per
produced port and release at zero. **`release` returns the buffer to the pool and must never call
`destroy()`** — mid-process buffer destruction segfaults Dawn-on-Node (ADR-0002/0003, and the
existing pools' comments).

### 4. Resident feedback = ping-pong (invariant 1)

A resident `feedback`/`delay` node swaps two leased buffers instead of storing a host array —
precisely HsPf's `[src, dst] = [dst, src]`. `SimState` holds `ResidentBuffer`s; `FieldRing`
generalises to a ring of leases. **HsPf is the reference spec, not a node to absorb** — main's
ADR-0011 is right that wrapping it as one giant graph node buys nothing.

### 5. Download only at *host-consuming* sinks (invariant 4)

`pull()`/`pullData()` download the requested field at the end; interior edges stay resident. Add
`pullResident()` returning the handle **without** downloading, for render/display consumers.

**Only a subset of sinks download at all.** A render/canvas sink consumes the buffer on-device and
entails **no download, ever** — as HsPf's renderer already demonstrates
(`fieldRenderer.ts:182`). So the invariant is better read as *"download only where the host
actually consumes the value"*, and the target for a render-terminated graph is **zero** downloads,
not one. The budget test's target is therefore per-sink-kind: `1` for a `pullData` sink, `0` for a
render sink.

### 6. Submits stay per-stage (invariant 2)

Per-dependent-stage submit remains the correctness baseline; batching/pipelining independent
branches is an **optimisation for later**, and the per-stage path stays the golden executor any
batched variant must match on every fixture (the doc records a silent wrong-`ll0` from getting
this wrong).

## Invariant 5 is amended: GPU failure is a fail-state, not a CPU fallback

**Invariant 5 as implemented is incompatible with invariant 4.** `runNode` runs
`op.sanity ?? allFinite` over **every element of every output** on every pull
(`executor.ts:87-90`, `op.ts:102-110`) — which *requires a download*. Validate-and-fall-back is
fundamentally Tier-1-shaped.

The resolution is to **amend invariant 5, not work around it.** `gpu-resource-sync.md` (and ADRs
echoing it) overstate the runtime CPU-fallback discipline; this project's focus is GPU compute.

- **`cpuGolden` is a test-time reference oracle, not a production fallback.** Full GPU-vs-CPU
  comparison happens in tests, where an explicit download is free and correct.
- **An op that cannot run on the GPU is a fail-state.** The executor throws; it does not silently
  degrade to a CPU path. CPU fallback is **not guaranteed by the system**. This removes the
  `sanity → cpuGolden` branch from `runNode` entirely, which is also what makes invariant 4
  reachable.
- **`mode: "cpu"`** stays as an explicit, opt-in execution mode for testing and reference — it is a
  deliberate request, not an automatic recovery path.

Consequence: `docs/gpu-resource-sync.md` invariant 5 should be revised to say this, and ops keep
`cpuGolden` purely as a test oracle (still valuable — it is what makes the kernels trustworthy).

## Staging (each stage tightens the budget ratchet)

- **Stage 0 — done.** `instrument.ts` + `readbackBudget.gpu.test.ts`; baseline `[1,2,3,4]`.
- **Stage 1 — substrate, no op conversions.** `ResidentBuffer`, lease/release + DAG liveness,
  `resident?` opt-in, executor bridge. Budget unchanged; full suite still green.
- **Stage 2 — pilot, linear chain.** Convert `convolveSeparable` + `threshold` to resident. The
  3-op chain drops **3 → 1** downloads — a directly measured win on the existing test.
- **Stage 3 — resident feedback.** The HsPf-shaped case: ping-pong leases through a `feedback`
  node, validated against HsPf's behaviour.
- **Stage 4 — resident render op.** ADR-0014's `surface(camera) → {position, normal, valid}` +
  `material` contract as a graph op consuming a resident field; no-download display.
- **Stage 5 — retrofit satellites** (`sweptGpu`, `implicitGpu`, HsPf's kernel) onto resident edges.

Stages 1–2 are the load-bearing ones; 3–5 are unblocked consequences.

## Why

- **It's owed, not proposed.** The design, rationale, lifetime discipline and open questions are
  already written in `gpu-resource-sync.md`; this ADR schedules the implementation.
- **It unblocks four directions at once** — rendering-as-ops (ADR-0009), geometry no-download
  (ADR-0010-geometry, which names Tier-2 explicitly), real-data/large-tile (ADR-0015), and
  scale-equivariance/streaming (ADR-0004) all wait on this single capability.
- **It stops the drift.** The graph currently earns nothing for heavy GPU work, so heavy GPU work
  goes around it. Resident edges are what make the graph worth using — and every additional
  satellite makes the retrofit dearer.
- **De-risked, not research.** HsPf and `sweptGpu` prove resident buffers work in this codebase on
  Dawn; what's missing is carrying them *across a graph edge*.

## Consequences / open questions

- **The `resident?` bridge can mask regressions** — an op silently falling back to host download
  looks correct. Mitigation: the budget ratchet, plus a debug mode that logs every executor-inserted
  materialisation with its node id.
- **Buffer usage flags — resolved: split the pool by mappability.** WebGPU *forbids* combining
  mappable and non-mappable usage (`MAP_READ` may combine with nothing but `COPY_DST`; `MAP_WRITE`
  with nothing but `COPY_SRC`), so `STORAGE | MAP_READ` is invalid — there is no universal buffer to
  over-provision toward. Therefore: a **resident class** (`STORAGE | COPY_SRC | COPY_DST`, plus
  `VERTEX` where geometry needs it) that is pooled and aliased, where over-provisioning is
  essentially free (the flags are mostly a placement hint and these live in device-local memory
  regardless); and a **staging class** (`MAP_READ | COPY_DST`) created short-lived at the download
  boundary, never pooled — which is what TypeGPU's `.read()` already does internally. The cost to
  avoid is placement, not flag count: mappable buffers sit in host-visible memory, so keeping them
  out of the resident pool is the whole point.
- **`hashSource`'s per-byte hashing must move to identity/`version` keying** (ADR-0015 §Scale) — a
  resident value has no host bytes to hash, so this is forced, not optional.
- **Aliasing depth — start with online/first-fit.** (NB: this is an *allocator* policy and is
  **orthogonal to evaluation strategy** — the executor remains lazily demand-pulled either way.
  "First-fit" describes which buffer a running node is handed, not which nodes run.) "Reuse only
  after last reader" is an *online* refcount + free-list. True **interval-graph aliasing** is
  *offline*: every value's lifetime is a known interval, so optimal assignment is interval-graph
  colouring (exactly solvable; minimum buffers = peak concurrent liveness). Laziness does **not**
  preclude this — `topoOrder` resolves the whole demanded subgraph *before any node runs*, so the
  pull boundary is precisely where the full schedule becomes available to plan over. The offline
  version buys an optimal high-water mark, no size fragmentation, and — most usefully — an
  **a-priori budget** ("this graph needs 3 buffers / 48 MB") that ADR-0008's byte-ceiling
  degrade-to-fit solver could consume. But first-fit is likely sufficient for a long time: memoised
  values and feedback/delay state are **pinned** (refcount never drops, so they cannot be aliased at
  all), and since buffers are never destroyed and the pool persists across pulls, the steady-state
  footprint converges to peak liveness anyway. Revisit when big-tile memory pressure is real or the
  ADR-0008 ceiling wants the a-priori number.
- **Residency along the volatility gradient (scoped, deliberately not built).** The motivating case:
  an expensive, rarely-changing upstream (a tile splat, a per-gene SAT) feeds — indirectly —
  volatile nodes near the sink (a tolerance slider). Can the stable node's buffer be released?
  The tension is three-way: **pinning** every memoised value grows VRAM with the cache (the memo is
  a 512-entry LRU — fatal at tile sizes), **releasing** forces an expensive recompute, and
  **downloading** trades VRAM for a deliberate invariant-4 exception.
  *Most of the mechanism is already implied by this ADR*: two policies over one `release()`
  primitive — (i) refcount liveness for **within-pull** values, which handles the volatile end
  automatically (short intervals, immediate recycle), and (ii) **memo eviction triggering
  `release()`** for **across-pull** values, since the LRU already encodes "least valuable to keep".
  Precedent exists in-repo: `TileCache` already fires `onEvict` → dispose of the GPU payload.
  The genuinely *new* tiers — and the over-engineering risk — are: **spill** (on eviction, download
  to host instead of dropping, giving three tiers: resident / host-spilled / absent-recompute) and
  **cost-aware eviction** (evict by recompute-cost ÷ bytes rather than recency, so expensive-large
  values spill while cheap ones drop).
  **Recommendation:** build only (ii), the memo-eviction→release coupling — small, and it prevents
  unbounded VRAM growth. Defer spill and cost-aware policy until measured. Observability first, per
  the invariant-4 precedent: the memo's content key already yields a **free volatility signal** (how
  often does a node's key change across pulls?), so measure the gradient before designing policy
  around it.
- **Open (from the doc, still open):** how far intra-encoder ordering can be trusted before a submit
  is forced — needs a small Dawn/WebKit behaviour matrix, since we have been surprised once.
- **Cancellation/interruption is unmodelled — across *all* ADRs, not just this one.** `pull`/`advance`
  take no `AbortSignal` and there is no interruption path. Two things force it: executor
  **pipelining** of independent branches (which `gpu-resource-sync.md` §Throughput already calls
  for, so it arrives *with* Tier-2, not after), and **interactive re-pull supersession** — a brush
  drag or param slider generates pulls that obsolete their predecessors, which is exactly the
  problem sd.js currently solves by hand with abort signals and signature-staleness checks.
  Falsifiable check to build *before* the machinery: **nodes executed after an abort == 0**.
- **Graph-level streaming is unmodelled.** ADR-0004 states the *intent* ("large pulls fuse and
  stream", dense-vs-lazy) and ADR-0008 streams **bytes** (progressive decode at the datasource);
  nothing streams **computation**. Missing: a chunked/unbounded source, the **chunk-as-clock**
  accumulator (an additive `splatDensity` into a resident grid *is* a fold, and the graph already
  models folds via `feedback` — the clock is the chunk index rather than a sim tick), and an op
  classification for what can be shown progressively: **(1) foldable state** (splat, counts,
  moments); **(2) cheap derivations of that state** (`convolveSeparable`, `getisOrd`, `threshold`,
  `emptySpace` — recomputable from the accumulator at any point, so displayable immediately);
  **(3) neighbourhood-complete** (`kthNeighborDistance`/CkNN, `fuzzyAdjacency`, `vietorisRips`,
  `anni` — need the full point set, or at least a spatial halo). Note Tier-1 actively **penalises**
  progressive streaming in proportion to how progressive it is: every chunk round-trips the whole
  accumulator, so transfers scale with chunk count. Falsifiable checks: **peak resident bytes
  constant in N chunks**, and **zero downloads** when the sink is a render.
- **Cross-tile / halo completeness is one problem wearing two hats.** A wand region growing across a
  tile boundary and a category-(3) spatial statistic over a tiled point stream are both frontier
  propagation over asynchronously-arriving neighbours. Neither is modelled; recorded here so it
  isn't rediscovered.
- **Browser backend parity — verify empirically, don't design for it speculatively.**
  `backend.browser.ts` will need the same lease API, but the honest test is a substantial compute
  task on a substantial data sample; treat parity as something to *measure* when such a workload
  exists rather than a constraint to pre-engineer around.

## References

- **`docs/gpu-resource-sync.md`** — invariants 1–5 and the runner spec ("topological order,
  ping-pong allocation, barrier/submit insertion, pool reuse by liveness, single download at
  sinks"); this ADR implements 1/3/4 and resolves the 4↔5 conflict.
- Measurement: [`instrument.ts`](../../src/gpu/graph/instrument.ts),
  [`readbackBudget.gpu.test.ts`](../../src/gpu/graph/readbackBudget.gpu.test.ts).
- Reference implementations (outside the graph): `src/gpu/sim/hspf/kernel.ts:350`,
  `playground/src/hspf/fieldRenderer.ts:182`, `src/geometry/sweptGpu.ts`.
- Blocked-on-this: ADR-0004 (scale-equivariance, dense-vs-lazy), ADR-0009 (rendering-as-ops),
  ADR-0010-geometry (no-download render), ADR-0014 (render contract), ADR-0015 (real-tile scale).
- Constraints: ADR-0002/0003 (Dawn-on-Node teardown; never `destroy()` mid-process).
