# Field-derived geometry: wand contours to a lofted Swept

Status: **design note** (2026-07-23)

> Written as ADR-0021 and **demoted the same day** — see
> [`decisions/README.md`](decisions/README.md) for why direction-setting documents are notes rather
> than decision records here.
>
> Beyond that general reason, this one has a specific successor worth stating plainly: **TDA on cells
> is likely a better feature-finder than a magic wand** — more reproducible, less operator-dependent,
> and already half-built (`src/spatial/{persistence,sublevelsetPersistence}.ts`, the fuzzy-adjacency
> ops in `src/gpu/spatial`). The wand was designed against a store that contains *only images*; once
> fuller SpatialData objects with cells are available, the feature-finding half of this note should be
> re-derived from that starting point rather than implemented as written. What survives either way is
> everything downstream of the contour: the array-space storage, the human-authored correspondence,
> the `Swept` extension, and the oblique-cut correction.

## Context

ADR-0012 designed the pick → feature → edit loop and, in its Consequences, named a third case it
deliberately did not build:

> "A geometry extracted from a **sampled volume** … has no primitive tree, so its provenance is not
> an argmin-primitive-id but a **field-space address** (which voxel / data region / channel a fragment
> came from). The `resolve` surface is deliberately kind-agnostic so this slots in additively; the
> *token* differs, the *resolver shape* does not. Designed-for, not built."

This ADR builds that leg, on the serial-section stack of the [scene note](serial-section-alignment-and-multi-viewport.md): **magic-wand a feature on a
section, correspond it across sections by hand, and loft the result into a tube.** ADR-0015 likewise
anticipated the destination — the wand mask is precisely the `label` field whose polarity and
invariants it landed (`FieldRole`/`LabelMeta`, `src/gpu/graph/handle.ts:108`).

### What the geometry is, and is not

The target features are **glomeruli** (~150–250 µm), not literal renal tubules. That is forced by the
data: sections are 100 µm apart, and a renal tubule is ~30–60 µm across — between two sections it can
move sideways further than its own diameter, branch, or leave the block. **There is no reliable
automatic correspondence at this sampling**, and pretending otherwise would produce confident,
scientifically misleading anatomy. A glomerulus spans 2–3 sections and *is* trackable.

So correspondence is a **human judgement, recorded as data**. Automatic suggestion may later pre-fill
links; it is never the source of truth.

### The geometric subtlety that shapes the representation

A contour drawn on a section is an **oblique cut, not a cross-section**. A structure running at angle
θ to the section normal cuts an ellipse elongated by 1/cos θ along its direction of travel. Feeding
those rings in as true cross-sections systematically inflates the tube wherever it runs obliquely —
which is everywhere interesting. Correcting for this needs a local centreline direction *and* a
profile richer than a radius.

And a tubule may **turn and re-enter the same section**. So the centreline is a general 3-D polyline,
possibly locally tangent to a section plane, and the chain order is the user's, not z order.

## Decision

### 1. The wand is a pipeline of pure stages, in `src/`, CPU first

```
patch ──▶ distanceField ──▶ threshold ──▶ seededSelect ──▶ contour ──▶ linkRings ──▶ rings
         (comparator,      (tolerance)   (seed,           (marching   (order +
          reference)                      maxRadius)       squares)    simplify)
```

Each stage is a pure function with a fixed contract, headless and tested, living in `src/` (a new
`src/imaging/` — these are image-grid operations, whereas `src/spatial` is point/shape analysis).
Implementations are CPU now; each stage ports to a `"use gpu"` kernel independently under a
CPU/GPU parity test.

The decomposition is built **now**, before the GPU port, because a monolithic CPU flood-fill "to be
GPU-ified later" never ports — the seams are not there.

Three of the four stages are embarrassingly parallel; the interesting one is `seededSelect`. Its
parallel form is iterative label propagation ("dilate within the mask"), which normally needs a
change-detection readback per iteration — **except that `maxRadius` bounds the iteration count in
advance**, so the convergence readback disappears entirely, and a jump-flood variant (stride 2ᵏ)
reduces it to O(log r) passes. `linkRings` is genuinely serial but operates on a few thousand
segments: one small readback for the whole wand, comfortably inside ADR-0017's discipline.

So `maxRadius` is not only a leak guard — it is what makes the GPU form clean. It earns its place
twice, which is why it is a first-class parameter rather than a safety valve.

**Parameters, and why each exists:**

- **`comparator` + `reference`** — the metric and the picked sample it is measured against
  (the [stain-space note](stain-space-and-stack-transparency.md)'s registry). Not RGB similarity: the
  wand must be stable across six differently-stained slides, because that is where correspondence lives.
- **`tolerance`** — a threshold on the distance field. Set by **click-and-drag** (drag distance sets
  it live, with the stain-space note's isoline visualising it), with a slider as fallback.
- **`maxRadius`** — flood fill *will* leak: a glomerulus connects to surrounding stroma through
  continuous tissue, and one weak boundary pixel lets the fill escape across the section. A radius
  cap converts a catastrophic failure into a visibly-clipped one the user can see and correct.
- **morphological close-then-open**, small kernel — kills single-pixel bridges and speckle before
  contouring.

Connectivity is **4-connected foreground, 8-connected background** — the standard pairing that avoids
the topological paradox where a diagonal chain is both connected and not.

The wand runs at a **fixed pyramid level** (level 2 ≈ 1.096 µm/px, where a 200 µm glomerulus is
~180 px), on a bounded patch fetched around the seed, in a worker. Fixed rather than
currently-displayed, because a result that depends on zoom is a bug generator; and bounded-patch
rather than resident-tile, because a result that depends on where the cameras have been looking is
worse. The decoded patch is cached by `(section, level, box)` so dragging the tolerance re-runs only
the fill.

**The mask lands as a real `label` field** — `FieldRole { kind: "label", labels: LabelMeta }`, already
in `src/gpu/graph/handle.ts`, with ADR-0015's invariants (integer dtype, 0 = background, no channel
axis, nearest-only resampling). This is the case ADR-0015 wrote it for.

### 2. A contour is one connected region, stored in its section's array space

```ts
interface Contour {
  id: string;
  element: string;            // the section it was drawn on
  params: WandParams;         // seed (array coords), comparator, tolerance, level, morphology
  rings: Ring[];              // outer CCW, holes CW — level-0 pixel coordinates
}
```

- **One connected region, holes allowed.** That is what a wand result is; multipolygons would be a
  different (and later) gesture.
- **Array space, not world or `aligned` µm.** A contour is drawn on a section whose alignment the user
  will keep editing. Stored in its element's array coordinates it inherits that element's
  transformation chain, so it moves with its section for free in *every* coordinate system. Stored in
  world µm it would silently detach from its section the moment that section was corrected — after
  the work was done. This matches SpatialData: a `shapes` element with the same transformation chain
  as its parent image.
- **Both `params` and `rings` are persisted.** They answer different needs and neither substitutes.
  `params` is the ADR-0012 *feature* — the editable `ParamSpec` surface a picked tubule resolves back
  to, and what makes the selection re-evaluable (drag tolerance, watch the contour breathe). `rings`
  is what the loft consumes, what exports as a `shapes` polygon, and what survives a change to the
  stain matrix or the wand implementation. Without the rings, reopening a document after a pipeline
  change would silently yield different anatomy.
- **Rings are in level-0 coordinates** even though the wand ran at level 2, so they are
  level-independent.

Export target: a `shapes` element for the vector rings; `labels` remains available for the raster
mask, with the caveat that a full-resolution label raster per section is expensive
(14165 × 18155) and is not the default.

### 3. A tubule is an ordered chain of contour references

```ts
interface Tubule { id: string; name: string; stations: ContourRef[] }
```

- **Ordered by the user, not by z** — a tubule may turn and re-enter the same section, so z order is
  not even well-defined as a sort key.
- **Authoring gesture:** create a tubule, it becomes active, and each subsequent wand click
  **appends** its contour to the chain. Step section with `[` / `]`, wand, repeat; the 3-D viewport
  shows the loft updating live. Escape ends the chain. The common case — walk the stack, click the
  same glomerulus six times — is six clicks and five keystrokes, and correspondence is recorded *at
  the moment of judgement* rather than reconstructed later in a linking UI.
- **Insertion is positional.** Adding a missed station between two existing ones inserts there; with
  re-visiting allowed, "there" cannot be inferred from z, so explicit insert-before/after exists, with
  append as the default when the new station is beyond both ends.
- **Re-wanding a section the tubule already has replaces that station**, and **warns when ambiguous**
  (i.e. when the tubule has more than one station on that section) rather than guessing.
- `< 2` stations renders nothing; 2 renders a straight tube; ≥ 3 a spline. The loft recomputes on
  every change — tens of stations and a few hundred θ-samples, so no incremental machinery.

### 4. `Swept` is **extended**, not joined by a new kind

`src/geometry/swept.ts` today sweeps a superellipse profile along a straight +Z axis, warped by a
closed-form `bend`/`twist`/`scale` stack. A wandering, possibly self-revisiting 3-D centreline cannot
be fitted to bend-warps of a straight axis in any robust way.

Rather than add a fourth Geometry-kind, `Swept` gains two optional facets:

- **a sampled path** — an explicit centreline (spline through station centroids) replacing the
  straight +Z axis, with parallel-transport frames;
- **a sampled profile** — `r(s, θ)` from a data table, replacing the analytic superellipse.

`src/geometry/CONTEXT.md` already defines Profile as a function of **both** `(s, θ)` that may "morph
along the sweep", explicitly anticipating richer profiles; and "a Profile swept along the sweep
coordinate under a Transform-stack" stays literally true — we are only saying the sweep coordinate may
follow a measured path. So: one kind, one tessellator, one provenance story, and the existing
`bend`/`twist`/`scale` transforms compose **on top of** a measured tubule, so a data-derived form
remains procedurally deformable and breedable.

**The oblique-cut correction is part of the loft, not a refinement.** Before a ring is used as a
profile, estimate the local tangent from neighbouring stations and project the ring onto the plane
perpendicular to it. It is a dozen lines and it is the difference between a measurement and a cartoon.
This is also the specific reason the profile must be sampled `r(s,θ)` rather than centroid+radius: a
radius cannot express the correction.

**Holes are persisted but ignored by the first loft.** Lofting with holes needs correspondence between
hole rings across stations, which is a second correspondence problem. Flagged where relevant rather
than silently dropped.

### 5. Provenance: the field-space address, ADR-0012's third leg

A picked tubule fragment resolves through the **same** `resolve(pickId) → { address, node, specs }`
surface as a structured instance or an implicit argmin — only the token differs. Here the token names
a location in the *producing chain*: `tubule#k / station#j → contour → (element, level, seed)`.
Resolving yields the contour's `WandParams` as the editable `ParamSpec` set, so picking a bulge in the
3-D tube takes you to the wand parameters that produced it.

ADR-0012's rules carry over unchanged: addresses are structural and deterministic (no counters, no
`Date.now()`, memo-safe), a param edit never moves an address, and a structural edit that invalidates
one returns an explicit **`stale`** rather than snapping to a neighbour.

### Scope line (this pass)

**In:** the five wand stages in `src/imaging/` with CPU implementations and tests; the mask as a
`label` field; `Contour` (params + rings, array space, holes allowed) and `Tubule` (ordered stations)
in the `SceneDocument`; click-and-drag tolerance; the `Swept` sampled-path + sampled-profile
extension with the oblique correction; `resolve` over field-space addresses.

**Out (designed-for, not built):** GPU ports of the stages (each independently, under parity tests);
holes in the loft; automatic correspondence suggestion; multipolygon wand results; `labels` raster
export; writing any of it back to the store (sd.js has no write path).

**Deferred, with the mechanism already identified — persistence-informed auto-tolerance.** The wand's
selection at tolerance `t` *is* the connected component of the sublevel set `{d ≤ t}` containing the
seed, and `src/spatial/sublevelsetPersistence.ts` already computes cubical H0/H1 with birth cells. The
seed component's **death value** is the threshold at which it merges into a neighbouring structure, so
"just below the merge" is a principled default *and* precisely what stops the fill leaking through one
weak boundary pixel. It would also make the drag legible by annotating the tolerance track with the
merge points. Not built this pass: the existing implementation was written for KDE-scale grids and a
2048² patch is 4 M cells, so its cost must be measured first (a 512² patch — ample for a glomerulus at
level 2 — is 262 k cells, or persistence can run downsampled purely to choose `t`).
**This is why `distanceField` is its own stage**: persistence and the stain-space note's render mode both consume
that field, and the factoring only exists if it is known up front.

## Why

- **It cashes ADR-0012's third leg on a real application** rather than a synthetic one, and does it
  through the resolver that already exists.
- **It is honest about the sampling.** Human-authored correspondence is not a simplification here; at
  100 µm spacing it is the only defensible source of truth, and encoding it as data makes the
  judgement reviewable.
- **It reuses rather than adds** — the label polarity (ADR-0015), the comparator registry (the [stain-space note](stain-space-and-stack-transparency.md)),
  `ParamSpec`s and `resolve` (ADR-0012), `Swept`'s tessellator and transform stack (ADR-0010). The
  only genuinely new vocabulary is Contour / Station / Tubule.
- **The op decomposition is the deliverable**, not just the wand. Five composable stages over fields
  are the first concrete instance of the user-extensible visualisation/analysis pipeline the op-graph
  is for.

## Consequences / open questions

- **`Swept` GPU evaluation needs new plumbing.** `sweptShaderWgsl`/`sweptGpu.ts` emit expressions over
  constants; a sampled path and profile need a storage buffer. CPU `toMesh` works immediately, so the
  first tubules render without it, but the GPU path is real work and is not free.
- **Swept feature granularity remains coarse** (ADR-0012's standing limitation): every transform in
  the stack acts pointwise over the whole surface. Data-derived stations give genuine spatial
  granularity along `s` — arguably the first honest *feature regions* a Swept has had — but the
  analytic transforms above them still do not.
- **Duplicated unmix maths.** The wand's CPU path re-implements the stain-space note's GPU unmix. Deliberate and
  pinned by a parity test, but it is a real invariant to maintain.
- **`maxRadius` is in pixels at a fixed level.** It should probably be in µm, which means it depends on
  the element's scale — a small thing that will be wrong once before it is right.
- **Correspondence is unvalidated.** Nothing in this design checks that the user linked the *same*
  structure across sections. That is intentional (it is their judgement) but it means a tubule carries
  no confidence measure. A later ADR could add one — e.g. centroid displacement per station against
  the local trend — as a flag, never as a veto.
- **Persistence performance is unmeasured**, as noted above. If it does not hold at patch scale the
  auto-tolerance is a downsampled heuristic rather than exact, which is fine but should be stated in
  the UI rather than implied.

## References

- **[in-repo]** ADR-0012 (provenance / pick-to-feature — this is its field-space-address leg),
  ADR-0010 (procedural geometry as composable ops — the `Swept` kind being extended), ADR-0015
  (`FieldRole`/`LabelMeta` — the wand mask's home; the coordinate systems contours ride),
  the [scene note](serial-section-alignment-and-multi-viewport.md) (the scene, the sections, and the `pick()` that seeds the wand), the
  [stain-space note](stain-space-and-stack-transparency.md) (derived stain
  channels, the comparator registry, the `distance` render mode and isoline), ADR-0014 (procedural
  geometry render contract — depth/picking the tubule participates in), ADR-0017 (readback
  discipline — one readback per wand), ADR-0003 (`"use gpu"` kernels — the per-stage port target).
- **[code]** `src/geometry/swept.ts` (`Profile`, the Transform-stack, `toMesh`, `specs()`),
  `src/geometry/CONTEXT.md` (the ubiquitous language to extend with *Contour*, *Station*, *Tubule*),
  `src/gpu/graph/handle.ts:108` (`FieldRole`, `LabelMeta`, `assertLabelInvariants`),
  `src/spatial/sublevelsetPersistence.ts` (the deferred auto-tolerance mechanism),
  `src/gpu/graph/op.ts` (`ParamSpec` — the feature/edit target).
- **[data]** `1113PMDC1_IgAN_slices_htj2k_q0.002.sdata.zarr` — the six sections, 100 µm apart, whose
  spacing is the reason correspondence is human.
