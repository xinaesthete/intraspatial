# ADR-0018 — Field domains: extent, placement, and resolution as carried facets

Status: **draft / proposed** (2026-07-21) — written for reaction. **Decision 2 (resolution is
derived, never stored) is agreed** (2026-07-21); the rest is still open, in particular the
build-time-vs-run-time placement question and whether points carry a full placement.

## Context

A grid field in this repo does not know where it is. `Shape` is
[`{ kind: "grid", width, height }`](../../src/gpu/graph/handle.ts) — pure sample counts — and
`FieldValue` carries no world extent, no coordinate system, and no cell size. Grep for
`placement`/`worldFrom`/`bbox`/`extent` in `handle.ts` and nothing comes back.

That absence is now doing visible damage.

**1. `bbox` is a ghost parameter.** `splatDensity` takes a world box, but `bbox` is declared on no
op's `params` list. It reaches the op only through the pass-through merge in `Graph.op()`, so the
palette cannot surface it, nothing validates it, and it cannot be given a default. It *cannot* be
declared either: `ParamType` is `"number" | "int" | "enum" | "bool"` — there is no vector type.
(The wand prototype hit the same wall; ADR-0015 records it.)

**2. The box is discarded at the output.** `splatDensity` consumes a world box, rasterises, and
emits a bare `{kind:"grid",width,height}`. The world→cell relationship that the box defined exists
only for the duration of one `execute` call. Nothing downstream can recover the cell size.

**3. So ops in one chain mix unit systems with nothing to reconcile them:**

| op | param | units |
|---|---|---|
| `splatDensity` | `sigma` | "bandwidth (world units)" |
| `fuzzyAdjacency` | `sigma` | "bandwidth σ (world units)" |
| `getisOrd` | `radius` | "box neighbourhood radius (cells)" |
| `convolveSeparable` | `radius` | cells — **undocumented** |

`splat(σ=world) → convolve(r=cells) → getisOrd(r=cells)` is a graph in which the user is silently
switching coordinate systems mid-chain. Nobody can answer "what does radius 2 mean in microns?"

**4. Two competing default policies already exist, and one already caused a bug.**
`splatDensityGpu` pads the derived box by `sigma · radiusSigma`; the op's `resolveBbox` overrides
that with a fixed 12% fractional margin, and its comment records why: the kernel-dependent pad grew
the world box as fast as the kernel, so *"raising sigma zoomed out instead of blurring"*. The
workaround is sound; the fact that two layers disagreed about what a default box means is the
symptom.

**5. Tier-2 forced the issue.** Implementing ADR-0017, `splatDensity`'s resident path must **throw**
when its points are GPU-resident and no explicit `bbox` was given, because the default is *derived
from the point values* and reading them back is exactly the transfer residency exists to remove.
That is the sharpest statement of the problem: **a domain inferred from data couples every op that
needs it to the host.** A declared domain does not.

**6. The model is already half-decided, in two places, and unimplemented in both.**

- **ADR-0004** decided `Shape` becomes `Domain` — the discrete sample structure, grown to 2-or-3
  dimensions. Still `Shape`, still 2D.
- **ADR-0015** decided the world model: `ResolvedPlacement { system: string; worldFromArray: Affine3 }`,
  **consumed** from sd.js rather than re-derived (sd.js owns the transform algebra). It scoped out
  *"ops that actually apply a placement"* and shipped no field-level facet. It also already writes
  the target shape of a resident field: `{ domain, element, axes, role, placement }`.

`bbox` is not a missing feature. It is a **weaker, redundant, per-op encoding of what
`placement` was already designed to carry** — the axis-aligned scale+translate special case of an
affine, smuggled through as an undeclared param and then dropped.

## Decision (proposed)

### 1. Do not invent a domain algebra. Land the two facets already decided.

Follow the additive-facet discipline that landed `element`/`basis`/`axes`/`role`: **absent facet ⇒
today's behaviour**, so every existing op keeps working.

```ts
interface FieldValue {
  /* shape/dtype/element/basis/axes/role/data/buffer unchanged */
  /** Where this field's samples sit in world space (ADR-0015). Absent ⇒ array space:
   *  the field is unitless and cell-indexed, exactly as every field is today. */
  placement?: ResolvedPlacement;
}
```

`ResolvedPlacement` is ADR-0015's, unchanged and still **consumed**, not composed here.

### 2. Resolution is derived, never stored. **(agreed 2026-07-21)**

There is no third facet. Given a domain (sample counts) and a placement (affine to world), cell size
falls out:

```
cellSize = |worldFromArray · ê| per axis        // world units per sample step
extent   = worldFromArray applied to the domain's corners
```

This matters because it identifies the **invariant across a multiscale pyramid**: levels differ in
`domain` (sample counts) and therefore in `cellSize`, while `extent` is unchanged. That is precisely
the relation ADR-0008's level selection needs and cannot currently express, and the one
ADR-0004's scale-equivariance is about — a filter with a world-unit σ must produce the same result
at any level.

**What being agreed forecloses.** No `cellSize`, `resolution`, `pixelSize` or `bbox` field may be
added to `FieldValue`, `GpuField`, `Shape`/`Domain`, or a `Multiscale` level descriptor — each would
be a second source of truth that can disagree with `domain × placement`, and the disagreement is
silent (a resampling op that updates one and not the other produces a field that is wrong about
where it is). Where a hot path wants the number, it is a **derived accessor** over the two facets,
not a stored field. Corollary: a resample op changes `domain` and the scale part of `placement`
together, in one place; if that ever needs to be two writes, this decision is being violated.

### 3. `bbox` becomes a placement constructor at sources, not a free parameter.

A source op that rasterises into a grid (`splatDensity`) is *choosing* a placement: a world region
plus a sample count. That choice should be explicit and should be **recorded on the output**, so
`bbox` stops being an input that vanishes. Downstream:

- **Pointwise and neighbourhood ops pass placement through unchanged** — a convolve does not move
  the grid. This is the same "pass through the first input's facet" default that `basis` uses.
- **Resampling ops compose a scale** — the only ops that may change it.
- **Binary ops require agreement** — adding two grids in different systems is a build-time error,
  which is the check `addGrids` cannot make today.

### 4. Declared, not derived — with an explicit escape hatch.

The default should be a **declared** domain. Where a data-derived box is genuinely wanted, it should
be an explicit op (`boundsOf(points) → domain`), computable on the GPU as a min/max reduction, not an
implicit host-side fallback buried in a source op. That removes the Tier-2 throw in `splatDensity`
by making the host dependency a visible node rather than a hidden one.

### 5. Ops declare the units of their spatial params.

`ParamSpec` gains an optional `units?: "cells" | "world"`. Absent ⇒ `cells` (today's behaviour for
every integer radius). This is the smallest change that lets the UI say "2 cells = 0.8 µm" and lets
a world-unit param be converted against the field's placement. It is also the prerequisite for
scale-equivariance being *testable*: the same world-unit filter at two resolutions should agree.

## Consequences / open questions

- **`ParamType` needs a vector kind** before a placement or box can be a first-class, UI-editable
  param. Known gap, recorded in ADR-0015, still open. Until then a placement can only be constructed
  programmatically — which is fine for the loader path (sd.js hands one over) and blocking for the
  composer.
- **Points fields are the awkward case.** Point coordinates are *already* world values
  (`splatDensity` takes world `xs`/`ys`), so a points field's placement is an identity affine that
  exists only to name a system. Is that worth carrying, or should points instead declare
  `system: string` alone? Leaning toward carrying the full placement for uniformity with ADR-0015's
  "one coordinate model for raster + label + points + geometry", but it is genuinely redundant.
- **Does a grid's placement belong on `GpuField` (build time) or `FieldValue` (run time), or both?**
  `basis` is stamped at build time and propagated onto values by the executor; placement plausibly
  wants the same treatment so binary-op agreement is a *build-time* check. Not settled here.
- **Migration.** Every existing op is placement-absent and stays correct (array space). The real
  work is the loader path (sd.js already resolves a placement — ADR-0015's `placements[]`), and
  `splatDensity`, which is the only op that currently pretends to know about world space.
- **Interaction with ADR-0017's `hashSource`.** A placement is part of a value's identity: the same
  bytes at a different extent are a different field. Whatever keying replaces byte-hashing must
  include it.
- **Not addressed here: 3D.** ADR-0004 owns growing `Domain` to 3D, and ADR-0015 explicitly deferred
  `z`/`t` axes. This ADR is written so that nothing in it assumes 2D, but it does not do that work.
- **Deliberately not proposed: a transform algebra.** ADR-0015's ownership boundary stands — sd.js
  composes transforms, this repo consumes the collapsed matrix. If this ADR ever seems to want
  `compose(a, b)`, that is a signal the boundary is being violated.

## References

- **ADR-0004** — `Shape` → `Domain`, element ⊥ axes ⊥ domain, scale-equivariance.
- **ADR-0015** — `ResolvedPlacement`, the sd.js ownership boundary, the resident-field target shape
  `{ domain, element, axes, role, placement }`, the vector-`ParamType` gap.
- **ADR-0008** — view-driven multiscale; the level-selection relation this makes expressible.
- **ADR-0017** — Tier-2 resident edges; the `residentBbox` throw that forced this.
- Code: [`handle.ts`](../../src/gpu/graph/handle.ts) (`Shape`, `FieldValue`),
  [`ops/splatDensity.ts`](../../src/gpu/graph/ops/splatDensity.ts) (`resolveBbox`, `residentBbox`),
  [`op.ts`](../../src/gpu/graph/op.ts) (`ParamType`, `ParamSpec`).
