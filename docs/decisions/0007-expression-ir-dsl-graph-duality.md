# ADR-0007 — Render-trait mappings as one expression IR, dual with DSL & graph

Status: **proposed** (2026-07-03)

## Decision

Visual traits (mesh geometry, PBR material, colour) will be driven by mappings from simulation
**channels** (per-instance sim state) expressed in a single **typed expression IR**. That IR has three
interchangeable **surfaces** — **DSL text**, the **playground node graph**, and a **serialised form**
— and two compile **backends** — **TSL (GPU)** and **CPU JS (golden reference)**. The IR is the source
of truth; DSL⇄IR is parse/print, graph⇄IR is extract/layout, so moving between text and nodes is a
round-trip through the IR, and a **DSL node inside the graph** is a graph node whose body parses to a
sub-IR. Numeric literals carry `ParamSpec` metadata so the Mutator breeds them.

Full rationale and layer design: [`../render-traits-and-expression-dsl.md`](../render-traits-and-expression-dsl.md).

### Amendment (2026-07-07, via ADR-0010) — GPU backend is TGSL, not TSL

This ADR names the two compile backends as **TSL (GPU)** and **CPU JS**. The GPU backend is
corrected to **TGSL / TypeGPU** (`"use gpu"` kernels, ADR-0003). At the time of writing, the
near-term render work was hand-authored TSL on the three.js side, so TSL was named as the GPU
target; since then ADR-0009 has demoted three.js/TSL to a thin presentation shell, and the whole
`src/` core is TGSL. TSL survives only as presentation mirrors in `docs-site/src/lib/*Tsl.ts` and
is being retired. Read every "IR → TSL" reference below as **IR → TGSL**. ADR-0010 (procedural
geometry-ops) is the first concrete consumer of this IR and builds its GPU backend as TGSL from
the start.

## Context & provenance

The user asked for render traits mappable from sim params "or indeed somewhat arbitrary expressions",
globally and per instance, plus procedural superegg geometry, procedural PBR texturing, and okLab/okLCH
colour. Offered a menu (declarative-only / TSL-native-only / hybrid / mini-DSL), the user redirected:
the playground already provides visual dataflow editing, and what's actually wanted is to **go between
a DSL and that graph representation**, with **a DSL node in the playground** too — i.e. not a choice
between text and nodes but a **duality** over a shared representation. That is the VEX⇄VOP / MaterialX
model, and it lines up with keeping the CPU golden and GPU kernel honest against each other (as
`forces.ts`/`dancerGpu` already do by hand) — here generated from one IR rather than written twice.

The user also asked to **plan this now but make near-term, less-generic rendering changes first**,
written "with reference to how it will be possible to adapt". So this ADR records direction; the
immediate work is hand-authored TSL that anticipates the IR (starting with okLab).

## Consequences

- **New:** an expression IR (typed DAG), a DSL grammar (parse/print), graph⇄IR mapping + a DSL node
  type, and IR→TSL / IR→CPU emitters. Sequenced behind concrete demo work, not up front.
- **Reuse, don't reinvent:** element/colour types extend the field type model (ADR-0004); the dual
  CPU+GPU emission is the existing `use-gpu` TGSL discipline (ADR-0003) generated instead of manual;
  breeding rides the existing `ParamSpec[]`/Mutator surface (constants now, DAG structure later).
- **Near-term, non-blocking:** okLab/okLCH module (CPU+TSL, golden-tested) → channel bridge (retire
  the per-frame CPU `setColorAt` upload) → superegg geometry → procedural PBR. Each names its channels
  and targets explicitly so the later lift into the registry/IR is refactoring, not rewriting.
- **Risk:** scope. Mitigated by the escape hatch — a raw-TSL node covers anything the IR doesn't yet
  model, so the IR can grow to fit the traits actually used rather than aiming for TSL completeness.
