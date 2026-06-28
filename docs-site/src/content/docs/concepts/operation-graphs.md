---
title: Operation graphs
description: A lazy-pull DAG runtime that chains GPU primitives without round-tripping to the CPU between stages.
---

The primitives compose. Today a pipeline like *splat → window → z-score* runs each
stage as its own call and copies the intermediate back to the CPU in between. That is
correct but wasteful: the interesting data never needs to leave the GPU until the very
end. The **operation-graph runtime** (`src/gpu/graph/`) makes the chain explicit and
runs it as a single dependency-ordered computation.

## The model

You build a **graph** of **operations** over **resources**, then **pull** a result:

- **Resources** are lazy handles (`GpuField`) carrying a shape (`grid`, `points`,
  `matrix`, …) and dtype — not data. They name *what* a node produces, not a buffer.
- **Operations** are nodes with typed input/output ports and parameters. Each declares
  the resources it reads and writes; the registry lists them so a UI can offer them.
- **Edges** are *derived* from those reads/writes — you wire ports, the runtime infers
  the dependency graph.
- **Pull** asks for one sink. The executor resolves its transitive dependencies,
  orders them topologically, runs each once (a shared producer is not recomputed), and
  returns the value.

This is the data-flow you declare; the executor owns the *sequencing*. See the repo
design note `docs/gpu-resource-sync.md` for the five invariants it enforces —
single-writer-per-submit, derived barriers, pool reuse by liveness, boundary-only
transfer (upload at sources, download only at the pulled sink), and per-node
validate-or-fall-back-to-CPU.

## Two tiers of node

- **Tier 1** wraps an existing `*Gpu` primitive as-is: it downloads its inputs, runs
  the legacy call, and uploads the result. Every primitive becomes a node immediately
  — correct, but it still round-trips on interior edges.
- **Tier 2** keeps the buffers resident on the GPU across a hot edge, so only the
  sources upload and only the pulled sink downloads. Promoted per pipeline, with the
  Tier-1 path kept as the bit-exact golden reference.

## Authoring

Programmatically the same graph is a fluent builder:

```ts
import { Graph, pullData } from "tgpu-htj2k/gpu/graph";

const g = new Graph();
const pts = g.points(xs, ys);
const dens = g.op1("splatDensity", { points: pts }, { width: 64, height: 64, sigma: 2 });
const z = g.op1("getisOrd", { grid: dens }, { radius: 2 });
const hotspots = await pullData(g, z);   // runs splat → getisOrd, downloads once
```

The same explicit graph is what the visual **composer** edits — a node per op, an edge
per dependency, parameter controls generated from each op's declared params. Because
the graph is serialisable, a canvas and a code builder are two front doors to one
runtime.

## Feedback & time

A pure pull-DAG can't express a simulation (`state_{t+1} = f(state_t)`). The graph
stays acyclic by separating **space** (the dataflow) from **time** (the iteration): a
**feedback (unit-delay) node** outputs the *previous* tick's value (seeded by `init`),
and the edge into its `next` input is a deferred write committed after the tick. Within
a tick the feedback node is a source, so the loop is a DAG per tick — the z⁻¹ break
used by synchronous dataflow languages, Houdini's solver, and GPU ping-pong textures. A
cycle that doesn't pass through a feedback node is rejected.

`pull` is one tick from the seed; `advance(field, { steps, state })` runs the loop with
state persisted across ticks. The composer adds Play / Step / Reset transport and
animates the sink — the reaction-diffusion example is exactly this: a Gray–Scott seed
feeding two delay nodes whose state drives the step, whose outputs feed back.

**Status:** ✅ runtime + Tier-1 ops + fuzzy-TDA chain + feedback/simulation · `src/gpu/graph/`
