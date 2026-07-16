# ADR-0012 — Geometry provenance & pick-to-feature editing

Status: **proposed / exploratory** (2026-07-15)

## Context

ADR-0010 repeatedly asserts that a Geometry is **"inspectable, serialisable, breedable, and
round-trippable"** and that a pulled surface should let a user *edit the procedural feature that
produced it*. But it never designs the loop that makes this true: **pick a rendered surface →
identify the op/feature that produced it → edit (or breed) its parameters → re-evaluate.** That
loop — "interactively edit procedural geometry with provenance-tracking of features to edit" — is
the specific capability that made the prior-art tools good, and it is the cross-cutting piece both
the built **swept** kind and the designed-for **implicit** kind need. This ADR designs it.

### Prior art (reference, re-derived — never ported)

Two C#/XNA prior-art codebases (`~/code/gold/aaquickhouse` `CSGlib`, and
`~/code/gold/aaprogen` `LevelSketcher2`) both solved this with the **same** mechanism, and both
converged on decisions this repo already shares (immutable nodes; operator-overloaded booleans;
buy-not-build 2D clipping). We take the *pattern*, not the code (consistent with ADR-0010's stance
on `horn.ts`). The mechanism, in three layers:

1. **`IHasProvenance` — an immutable back-pointer.** `CSG.cs:2950`: every node carries
   `object Provenance`, and `SetProvenance` **returns a new object** ("setting provenance must
   create a new object", `CSG.cs:1020`) so the tag survives boolean rewrites unchanged. A poly is
   derived from a plane derived from a mass component; the chain is preserved.
2. **`SourceProvenance<T>()` — a generic chain-walk.** `BuildingDraw.cs:29` walks the back-pointers
   until it finds the requested feature type. Rendering writes provenance **per triangle**
   (`crm.SetProvenance(i, …)`, `CSGFramework.cs:118`); a ray hit reads it back
   (`csgmesh.GetProvenance(f*3).SourceProvenance<BuildingFeature>()`, `RayCast.cs:352`).
3. **The target is an editable *feature*, not the geometry.** `BuildingFeature` (`BuildingFeature.cs:32`)
   exists "to identify a subpart of a building… for gene freeze, interactive manipulation" and
   carries a **stable key** (`ProfileKey`, `BuildingFeature.cs:67`) so a selection re-attaches across
   re-evaluation. Its parametric metadata is the `[Mutatable]` attribute — `Min/Max/Delta/Default,
   FeatureGroup, Tags` (`Mutatable.cs:18`) — which **already maps almost 1:1** onto our
   [`ParamSpec`](../../src/gpu/graph/op.ts) (`name`, `min/max/step/default`, `tags`).

## Decision

Add a **provenance / feature-identity layer over the existing geometry IR** — not a new object graph,
but a small set of conventions that (a) name a location in the IR, (b) survive lowering into the GPU
buffers as a picking channel, and (c) resolve a pick back to the editable `ParamSpec` surface. Six
points.

1. **Provenance is an *address into the IR*, not a stored back-pointer.** Our IR is already what
   `IHasProvenance` had to fake: **immutable** (every op returns a new node — `Swept.with`/`push`,
   `Structured.composeWith`) and a **typed DAG** (ADR-0007). So the provenance token is a stable,
   serialisable **address** computed from the DAG — e.g. `transforms[2]` (the third transform),
   `branch#1/instance` (a structural op), `prim#3` (an SDF leaf). No per-node mutable field, no
   rewrite-preservation problem: the address *is* the structure. A pre-order structural path
   serialises trivially and survives the ADR-0007 graph⇄IR round-trip, so a selection persists across
   a save (settled, not an open question).

2. **The "feature" is a *selection over `ParamSpec`s*, not a new class.** We do **not** re-derive
   `BuildingFeature`. Resolving a pick yields the IR node(s) at an address and their `specs()` —
   which is already the breeding surface (`src/evo`) and already carries `name`/`tags` grouping.
   Provenance is therefore the **inverse of `paramVector()`/`specs()`**: the map from a rendered
   pixel back to the gene(s) that control it. Edit and breed act on the same `ParamSpec` a
   feature-group tag already selects.

3. **Provenance survives lowering as a picking channel**, kind-specific in the buffer but uniform at
   resolve:
   - **structured** (cleanest, immediate): the instanced draw already has an instance index; map
     `instance → { structural-op address, instance-index }`. Pick an arm of a whorl → the `branch`
     op + which arm. Zero new geometry cost — the index already exists.
   - **swept**: the tessellation vertex grid already encodes `(s, θ)` (`gridSampleAngles`); a picked
     triangle decodes to an `(s, θ)` band. Note that **op-level spatial attribution is not a
     meaningful goal here** — every transform in the stack acts pointwise over the *whole* surface,
     so "this fragment came from `bend`" has no honest answer. Two richer-than-coarse capabilities
     replace it, and they are distinct:
     - **Evaluation inspection** (cheap, needs nothing new): pick → `(s, θ)` → **replay every
       param-expression at that point** (`evalExpr` already exists) to read out what each ramp/taper
       *evaluated to* here (twist = 252°, scale = 0.6, …). This answers "why is this point where it
       is", not "which feature is this" — the inverse of `paramVector()` extended from *which gene*
       to *what value the gene took at this fragment*.
     - **Feature regions** (deferred): named `(s, θ)` bands from the Profile sub-grammar
       (ribbon/arrowhead — ADR-0010) — the only thing that gives a swept surface genuine *feature*
       identity, as opposed to evaluation read-out.
   - **implicit** (designed-for): write the **argmin primitive id** of the SDF min/max tree — the
     direct analog of legacy triangle→primitive. It resolves identically for a **raymarch hit**
     (write the id at the hit fragment) and a **surface-net triangle** (per-triangle source prim),
     which is exactly what lets the ADR-0009 hybrid raster+raymarch seam resolve picks uniformly.

4. **One `resolve(pickId) → { address, node, specs }`, kind- and render-mode-agnostic** — the
   re-derivation of `SourceProvenance<T>`. A raster-triangle pick and a raymarch-fragment pick enter
   the same resolver; the caller gets an address and the editable `ParamSpec` set, never a
   kind-specific object.

5. **Stable identity across edits (the `ProfileKey` problem), stated honestly.** A *param* edit
   (change a bend angle) must not move an address — addresses are structural, so this holds by
   construction. A *structural* edit or a **breed** that adds/removes ops **can invalidate** an
   address that no longer exists. We do not pretend otherwise: `resolve` returns a
   `stale`/`not-found` result rather than silently snapping to a neighbour, and a UI re-attaches by
   re-picking. (Legacy bought stability with a hand-maintained `ProfileKey`; we prefer an explicit
   stale signal over a fragile global key.)

6. **Addresses must be deterministic and memo-safe.** Provenance ids are derived purely from IR
   structure (a pre-order DAG walk), so they are identical across a memoised re-execution
   (`src/gpu/graph`) — a geometry pulled twice yields the same picking channel. No `Date.now()`/
   counter state leaks into an address.

### Scope line (first slice)

**In:** the **structured** per-instance path end-to-end — a `provenanceAddress` for structural ops,
the instance→address map emitted alongside `instanceMatrices()`, and `resolve(instanceId)` returning
the `branch`/`stack` op address + its `specs()`. Plus the `Address` type and the uniform `resolve`
surface, designed so swept-bands and implicit-argmin slot in additively.

**Out (designed-for, not built):** the swept `(s,θ)`-band feature attribution (waits on the Profile
sub-grammar), the implicit argmin-id channel (waits on the implicit kind, ADR-0010 deferred), a
CPU-side barycentric picker vs a GPU picking-buffer (either satisfies `resolve`), and any breeding-UI.

## Why

- **It cashes ADR-0010's promise.** "Inspectable/breedable" becomes a concrete pick→gene loop instead
  of an assertion.
- **It reuses, it doesn't add.** The feature is `ParamSpec` + `tags` (already there); the identity is
  the IR address (already immutable + DAG); the resolver is one function. Nothing parallel to the
  op-graph is introduced.
- **It unifies the hybrid render seam.** Because `resolve` is render-mode-agnostic, the ADR-0009
  raster+raymarch depth-composite gets *one* picking story — a rastered swept triangle and a
  raymarched implicit hit both resolve to an editable feature.
- **It re-derives the prior art faithfully.** The immutable-tag/chain-walk/stable-key trio maps
  cleanly (immutability→IR, chain-walk→`resolve`, stable-key→structural address + explicit stale),
  keeping the character that made those tools good while shedding XNA/`BuildingFeature`/app coupling.

## Consequences / open questions

- **"Feature" granularity for a pointwise swept surface is genuinely coarse until the Profile
  sub-grammar lands.** This is a real limitation, not a slice cut: on a plain horn *every* transform
  is active at every point, so there is no honest per-op spatial attribution (see decision 3). We
  ship the structured/implicit *spatial* granularity first, plus swept **evaluation inspection**
  (which needs nothing new), and are explicit that swept *feature* identity waits on the Profile
  sub-grammar. This is accepted, with a roadmap — not an open question.
- **Data-derived geometry resolves differently, by design.** A geometry extracted from a **sampled
  volume** (isosurface of an image field — anticipated, and the same surface-extraction pass as the
  implicit kind) has no primitive tree, so its provenance is not an argmin-primitive-id but a
  **field-space address** (which voxel / data region / channel a fragment came from). The `resolve`
  surface is deliberately kind-agnostic so this slots in additively; the *token* differs (primitive
  id vs field coordinate), the *resolver shape* does not. Designed-for, not built.
- **Picking buffer (decided) travels with depth.** `resolve` is served by a **GPU picking pass** that
  writes the id channel; it is a companion to the **depth** the render seam always produces (depth is
  needed independently for world-position reconstruction — e.g. orbit-pivot under the cursor — and is
  a separate pass's concern). Picking-id + depth are co-produced G-buffer-style outputs; a CPU
  barycentric fallback also satisfies `resolve` where no picking pass has run.
- **Breeding churn.** Heavy structural mutation churns addresses. The explicit `stale` result is the
  honest floor; a future *structural* stable key (opt-in, per-op) could reduce re-picking if it earns
  its place — deferred rather than baked in.

## References

- ADR-0010 (procedural geometry — the "inspectable/breedable" promise this cashes), ADR-0007
  (expression-IR ⇄ DSL ⇄ graph duality — address serialisation rides its round-trip), ADR-0009
  (rendering as ops — the hybrid raster+raymarch seam `resolve` unifies), ADR-0003 (`"use gpu"` TGSL
  kernels — the lowering the picking channel travels through).
- `src/geometry/{swept,structured,expr}.ts` (the immutable IR + `specs()`/`paramVector()` this
  inverts), `src/gpu/graph/op.ts` (`ParamSpec` — the feature/edit target), `src/evo` (the Mutator
  that breeds the resolved `ParamSpec`s), `src/geometry/CONTEXT.md` (ubiquitous language — extend
  with *Provenance address*, *Feature (selection)*, *Resolve*).
- Prior art (reference, re-derived only): `aaquickhouse` `CSGlib/CSG.cs` (`IHasProvenance`),
  `BuildingFeature.cs` / `BuildingDraw.cs` (`SourceProvenance<T>`, the feature target, `ProfileKey`),
  `Mutatable.cs` (the `[Mutatable]` attribute ≈ `ParamSpec`); `LevelSketcher2` `IEditorObject.cs`
  (the parallel `PDirty` dirty-reevaluation loop).
```
