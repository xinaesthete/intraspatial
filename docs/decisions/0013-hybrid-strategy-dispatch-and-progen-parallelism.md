# ADR-0013 — Hybrid strategy dispatch & Progen-scale parallelism

Status: **proposed / exploratory** (2026-07-16)

## Context

The implicit (SDF/CSG) kind now lowers three ways from one IR:

- **grid dual-contouring** (`tessellateSdf`) — smooth / volume-derived surfaces; staircases planes.
- **plane-native BSP** (`evaluateBrep` → `mergeCoplanar`) — exact meshed CSG for the *polyhedral*
  subset; clean lines, per-face provenance, octree-localised (ADR-0010, this repo).
- **raymarch** (`sdScene` sphere-tracing) — exact per-pixel / image-space CSG; sharp edges *and*
  curves, no mesh, composited against real meshes through the depth buffer.

This ADR records how those compose **at scale** and how they handle **ornament that breaks the BSP**
(curved cupolas, spires, tracery, fluting) sitting inside planar architecture. It is design intent
captured mid-exploration — not a post-hoc rationalisation and not yet implemented — so the reasoning,
including a decision we reconsidered, is on the record before code exists (per the repo's
agentic-provenance-honesty stance).

The concrete driver is **Progen** (the sibling procedural-architecture lineage, `aaprogen` prior
art), where each building is a **separate CSG model**.

## Decision

### 1. Two nesting units: the building and the mass

Progen's separate-model-per-building gives a two-level hierarchy, and almost every concern below
lands on one level or the other:

- **Building** — no geometry is shared *between* buildings, so meshing is embarrassingly parallel:
  N independent `evaluateBrep` calls whose results **concatenate** (no cross-building merge, because
  there is no shared boundary to reconcile). The building is therefore the **worker payload** —
  coarse, deterministic, no shared state — the ideal fan-out unit.
- **Mass** — regions *within* a building that share walls (the L/T junctions). This is where the
  octree localisation already built earns its keep. It stays **inside one worker**.

The octree is the *intra-building* accelerator, not the parallelism boundary. The parallelism
boundary is the building, and it needs no spatial reasoning at all — just fan-out and concatenate.

### 2. Per-building raymarch, not per-scene

Whole-scene raymarch evaluates a `min` over *every* building on *every* ray. Instead, give each
building a **proxy box** whose fragments run **only that building's** small `sdScene`. Per-fragment
cost becomes one building, not the city.

This reuses the **structure/value split** (`implicit.ts`, ADR-0007): identical building *models* at
different placements share **one pipeline** with a per-instance `P[]` uniform buffer (the uniform
codegen path, not `bakeConstants`). Progen's separate-models-per-building maps onto *one pipeline per
distinct model, instanced across placements*.

The residual cost is **screen-space overdraw**: overlapping proxy boxes each march where they
overlap, and a ray that *misses* its building still pays for the march before discarding. This is why
raymarch stays the tool for the ornamented **few**, and meshing remains the workhorse for the bulk —
confirming the earlier intuition that a fully-raymarched city would not scale.

### 3. Hybrid strategy dispatch through the octree

The octree already computes, per leaf, *which masses are live there*. Extend that from "which masses"
to "which **op-kinds**": a leaf's strategy becomes `f(live op-kinds)` — pure-planar leaf → BSP; a leaf
touching a `sphere`/`smoothUnion`/other non-planar op → an alternate strategy. Subdivision gains a new
job: **subdivide to isolate** the non-planar detail into its own small leaf, so the surrounding planar
bulk stays BSP and the expensive strategy is confined.

### 4. The seam asymmetry — the constraint that shapes everything

"Switch to grid *or* raymarch per region" hides a real difference in **seam cost**:

- **BSP ↔ raymarch** composites through the **depth buffer** (the hybrid pass already does this). Two
  representations meet and mutually occlude; nobody agrees on vertices. The seam is *free and exact*.
- **BSP ↔ grid-DC** is a **topological** seam: two meshes with *different vertices* on the shared
  plane → cracks / T-junctions. Neither method is watertight across a method boundary.

Therefore: **ornament that breaks the BSP routes to raymarch** in the hybrid (a depth seam), and
**grid-DC stays a per-*object* choice** — a whole data-derived / volume mesh meshed by one method —
*not* a per-leaf tile abutting BSP architecture. Every hybrid seam is then a depth seam.

### 5. Tags and analytic recognition are layers, not alternatives

- **Recognition — free, for correctness.** The IR already distinguishes `sphere`/`smoothUnion` as
  node kinds; it is the exact predicate `evaluateBrep` currently *throws* on. The evolution is to stop
  throwing and instead **segregate**: partition a model into a planar sub-IR (BSP) and detail sub-IRs
  (raymarch) by op-kind. This reuses the octree directly and is the concrete first step.
- **Tags — policy, on top.** A tag does not say "this is non-planar" (recognition knows that); it
  *overrides* — "this fluting is technically 200 planar facets, but raymarch it, it's cheaper as
  detail" — or carries LOD / importance the analysis cannot infer.

### 6. Lighting the raymarched surface (real PBR, positional lights) is viable

The first cut (`raymarchMain.ts`) shades the meshed house and the raymarched growth with **one shared
hand-shade node**, so they match. That was pragmatic, but an early note here wrongly implied real PBR
with positional lights on the raymarched surface was only possible because the scene lights are
directional. It isn't a constraint: the sphere-trace computes the **exact world-space hit position**,
which is exactly what positional lighting needs. The hand-shade was a shortcut, not a necessity.

Two routes to the real thing, both feeding `pHit` + gradient normal + material params into three's node
lighting rather than approximating it: a **custom lighting node** (run the physical lighting model over
the scene `lights([...])` with our geometric inputs — same BRDF, lights, and tone-mapping as the mesh,
so they match by construction), or a **deferred G-buffer** (mesh and raymarch both write
position/normal/albedo; one lighting pass covers all — more infrastructure, the clean answer at scale).
**Shadows** are not a raymarch-special difficulty — *receiving* is a shadow-map sample at `pHit`, and
*casting* is the general contract that **procedural geometry contributes a depth pass for shadow
cameras**: a `customDepthMaterial` (directional/spot) and `customDistanceMaterial` (point) that
reproduce the surface from an arbitrary camera. The one contract covers **horn** (the depth material
shares the vertex transform), **terrain** (already done for the height-field in the psychogeo prior
art), and the **raymarch** (the depth material shares the march). What keeps shadows attached to
geometry is the repo's usual discipline — author the surface *once* and let the lit, depth, and
distance materials all consume it (one definition, several lowerings, as with CPU-golden ==
GPU-kernel). Expressing that contract generalises shadow-casting across every procedural kind at once.

### 7. Rust relevance tracks the algorithm (not an immediate priority)

The earlier "rule out Rust" (see [ADR-0010](0010-procedural-geometry-composable-ops.md) discussion)
was predicated on **grid-based** meshing, which is GPU-friendly and wants no native code. The original
reason to *consider* Rust was the assumption we would use a **CSGLib-style** algorithm, which would
benefit from it. So the moment we went back from grid to the BSP approach, the Rust question was
implicitly **re-opened** — not reversed on a whim, but tracking the algorithm choice.

The BSP mesher is the honest candidate: pure, numeric, irregular, already the sanctioned **non-GPU
island** (per the WebGPU-first principle it is the explicit CPU exception). Rust/wasm would buy
throughput and better worker ergonomics (shared memory, no GC pauses across a fan-out). The cost is a
**third lowering** to keep in agreement with the CPU-TS golden and the GPU path. The mitigant is
already in place: the **differential tests are language-agnostic oracles** (`brep == SDF-golden`,
`octree == global`), so a future port is checkable against them. **Not an immediate priority.**

## Consequences

- **First concrete step** toward the hybrid is *segregate-don't-throw* (§5): the octree partitions a
  model into planar (BSP) and detail (raymarch) sub-IRs by op-kind. It is a modest refactor of code
  that exists, and it is the foundation everything else in §3–§4 sits on.
- **Suggested sequencing:** (1) per-building fan-out for meshing (nearly free — N `evaluateBrep`s +
  concat); (2) per-building raymarch proxies with the instanced `P[]` path; (3) segregate / dispatch.
- **Deliberately not decided here:** the subdivision *criterion* for isolating ornament; how tags are
  represented in the IR; the LOD story; whether workers are web-workers now and wasm/Rust later.
- **Coherence with existing ADRs:** the parallel unit (building) and the localisation unit (mass) sit
  under [ADR-0010](0010-procedural-geometry-composable-ops.md); provenance per face survives all three
  strategies and feeds [ADR-0012](0012-geometry-provenance-and-pick-to-feature-editing.md);
  strategy-dispatch is [ADR-0009](0009-rendering-as-ops.md)'s "rendering as ops" made spatial.
