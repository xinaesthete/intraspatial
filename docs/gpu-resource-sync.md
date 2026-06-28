# Resource synchronisation for in-GPU operation graphs

Status: **design note** (2026-06-28)

The toolbox is heading toward *chaining primitives on the GPU* — the `GpuField`
handle + shared runner in [`gpu-primitives-toolbox.md`](gpu-primitives-toolbox.md),
e.g. `upload → fdwt → threshold → idwt → download`, and the spatial front's
`splat → window → z-score`. Once that becomes a real **graph of operations** sharing
pooled resources, the thing that decides whether it is correct is *how we sequence
reads and writes to shared resources*. This note fixes a model for that, because every
sync bug we have hit so far is the same bug wearing a different hat, and ad-hoc fixes
will not survive a graph.

## The one failure mode

> A mutable resource is read by a consumer that is not correctly ordered after its
> writer (or written by a producer not ordered after its last reader).

Three instances, all real, all from this project:

1. **Cross-pass RAW within one command encoder (DWT).** A buffer bound read-only in
   one pass and mutable in another *within a single encoder* did **not** get the
   write→read barrier we assumed — the forward DWT's `ll0` was wrong until we split
   into per-level submits. WebGPU does not promise a barrier across every
   read/write transition you encode; it promises far less.
2. **Pooled-buffer reuse race.** Reusing a pooled buffer for the next op while the
   previous op's readback/consumer might still need it corrupts data. We dodged it in
   the demo's GPU field path with **single-flight + copy-on-read**, but a global lock
   is a sledgehammer, not a model.
3. **Shared mutable resource, deferred read (Safari canvas).** Both demo panels drew
   through one shared offscreen canvas, then `drawImage`'d it to two targets in one
   frame. Safari defers canvas rasterisation, so both `drawImage`s sampled the
   *last* write — the "hard" panel rendered the "fuzzy" field. Same shape as (1),
   on the CPU, exposed by a backend with different timing.

(3) is the tell: this is **not** a GPU-only or a Safari-only problem. It is a
sequencing-of-shared-mutable-state problem, and backends differ in how forgiving their
*implicit* ordering is. A model that only works because one backend happens to flush
synchronously is not a model.

## The model: a DAG of ops over resources

Represent a chained computation explicitly:

- **Resources** — `GpuField`s (pooled buffer + shape/dtype) and textures.
- **Operations** — nodes that declare, up front, the resources they **read** and the
  resources they **write** (`reads: Field[]`, `writes: Field[]`). A node body is "a
  WGSL template + a layout + dispatch sizes" (the runner already wants this).
- **Edges** — derived, never hand-declared: an edge `A → B` exists iff `B` reads a
  resource `A` writes (RAW), `B` writes one `A` reads (WAR), or both write one (WAW).

Ordering, barriers, pooling, and download points are all **derived from this graph**,
not decided per call site. That is the whole point: you declare *data flow*, the
executor enforces *sequencing*.

## Invariants the executor must hold

1. **Single writer per resource per submit.** Within one submitted batch, a resource
   has at most one writer, and every reader of it is ordered after that writer by an
   explicit barrier or a submit boundary. If an op needs to both read and write a
   logically-shared field, it gets **two physical buffers** (ping-pong), never one.
   — Prevents (1) and (3).
2. **Barriers/boundaries are explicit and derived, never assumed.** Do not rely on
   WebGPU inserting a barrier because you "obviously" wrote then read. Where the API
   guarantees ordering (separate compute passes over the same storage buffer in one
   encoder *do* serialise on most backends, but we burned on the readonly↔mutable
   case), encode the dependency anyway; where it does not, insert a **submit
   boundary**. Default to per-dependent-stage submits; batching is an optimisation
   (below), not the baseline.
3. **Pool reuse respects liveness.** The pool may hand a buffer to a new op only once
   its **last reader in the DAG has completed**. Compute lifetimes from the graph;
   two fields may alias the same physical buffer **only if their lifetimes do not
   overlap**. When in doubt, copy. — Generalises (2)'s single-flight to "reuse by
   liveness, not by luck".
4. **Boundary-only transfer.** `upload` at sources, `download` at sinks; interior
   edges stay on-GPU. Download is the slow + Dawn-on-Node-fragile op (readback ceiling
   ~512² there) — minimise it and isolate it to sinks.
5. **Validate + fall back per node.** Each node has a CPU golden (bit-exact for int,
   bounded error for float). In production, a cheap output sanity check (finite,
   plausible range) gates a CPU fallback — the pattern already used for the demo's GPU
   field. Backends are **non-uniform** (Dawn vs WebKit/Metal differ in strictness and
   timing); never trust a result blindly.

## Throughput: batch only what's independent

A global lock (serialise everything) is *correct* and is the right **default** while
the graph is small. But it throws away the win of a graph. The executor should:

- **Serialise along dependency edges**, and **pipeline independent branches.** Two ops
  with no path between them may run in the same submit / overlap. The DAG already
  tells you which.
- Treat **submit granularity as a tuning knob, not a correctness lever.** Per-stage
  submits are the safe baseline (correctness); coalescing stages into one encoder is
  an optimisation that must preserve every derived edge. We have direct evidence that
  getting this wrong is silent: batching all DWT levels into one encoder produced a
  *wrong* `ll0`, no error raised. So: coalesce only across edges the API actually
  barriers for, and keep the per-stage path as the validated reference to diff
  against.

## What this asks of `GpuField` / the runner

The [planned](gpu-primitives-toolbox.md) `GpuField` + shared runner is the right
substrate; this note just says the runner should own **scheduling**, not only
dispatch:

- `GpuField` carries identity + shape + dtype + a **liveness/version** the scheduler
  can reason about (not just a raw buffer).
- The runner records nodes (`reads`/`writes`) into a graph and **executes the graph**:
  topological order, ping-pong allocation for read-modify-write, barrier/submit
  insertion per the invariants, pool reuse by liveness, single download at sinks.
- Keep the per-stage-submit path as the **golden executor**; the batched/pipelined
  executor must produce identical results on every fixture or it is wrong.

## Open questions (decide when building it)

- **Storage barriers vs submit boundaries** — how far can we lean on intra-encoder
  ordering before a submit is forced? Needs a small matrix of WebGPU behaviours across
  Dawn + WebKit, since we have already been surprised once.
- **Aliasing analysis depth** — start with "reuse a buffer only after its last reader"
  (simple, conservative); a true interval/graph-colouring aliasing pass is a later
  optimisation.
- **Where the graph is authored** — implicitly (each `*Gpu` call records a node and
  returns a lazy `GpuField`, executed at `download`) vs an explicit builder. Implicit
  is the nicer surface and matches the chaining ergonomics goal.

See also [`gpu-primitives-toolbox.md`](gpu-primitives-toolbox.md) (the `GpuField`
chaining vision + constraints), [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md),
and the Dawn-on-Node teardown/readback notes in
[`dwt-gpu-and-high-bit-depth.md`](dwt-gpu-and-high-bit-depth.md).
