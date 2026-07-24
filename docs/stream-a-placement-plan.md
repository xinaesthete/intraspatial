# Stream A — placement facet + builder threading (implementation plan)

Implements the two decisions resolved in [ADR-0018](decisions/0018-field-domains-placement-and-resolution.md)
(2026-07-23) on top of [ADR-0015 §3](decisions/0015-channel-axis-labels-coordinate-systems.md) and
[ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md). This is an *implementation
plan for accepted decisions*, not a new decision — no ADR.

**One-line goal.** Give the op-graph a coordinate model: a field knows where it is, ops that
combine fields check they agree, and the resolved transform reaches execute — so a real spatial
tile can enter the graph without its placement falling on the floor at the `Loader` seam.

## Why this is the prerequisite for stream B (the SpatialData→ops bridge)

A bridge built before the facet exists would drop the sd.js-resolved placement (nowhere to put it)
and re-create ADR-0018's ghost-`bbox` problem one layer up. So A lands first; B binds onto it.

## The core mechanic (reuse, don't invent)

Placement propagates **exactly like `basis`**, which already works end-to-end:

1. `graph.op()` computes `outBases = def.inferBasis(inBases, params)`, stores it on the `GraphNode`,
   and stamps it onto the returned handle via `makeField` ([graph.ts:162](../src/gpu/graph/graph.ts)).
2. The executor stamps `node.outBases[i]` onto the produced `FieldValue` when the op left it unset
   ([executor.ts:365](../src/gpu/graph/executor.ts)).

`axes` and `role` were *never* wired into this path (the ADR-0015 known gap), and placement is new.
All three are the **same mechanical change**, so land them together in one pass rather than three.

## Scope

**In:** `placement` facet on `GpuField`/`FieldValue`; `inferPlacement`/`inferAxes`/`inferRole` hooks;
builder + executor threading for all three, across every node-creation path; binary-op agreement
check; `splatDensity` as a placement-constructing source; `hashSource` keying folds in placement;
loader `Multiscale.worldFromArray → placements[]` (one `global`); a per-`(table, region)`
centroid-extract source (the dominant real case); `ParamSpec.units`.

**Out — declared-for, not built (leave seams, don't foreclose):**
- **Tiled points-*element* apron-splat-cache path** (future revision, see "Forward-compat" below).
  Points *elements* are spatial elements carrying their own transforms; sd.js support lands soon and
  processing them well is critical — but not this pass.
- Multi-system `placements[]` beyond one `global`; section-stack alignment.
- **Vector `ParamType`** — needed before a placement/bbox is a UI-editable param; the loader path
  (sd.js hands one over) works without it. Tracked in ADR-0015; not in scope.
- 3D `z`/`t` (ADR-0004 owns growing `Domain`); the real sd.js points loader; the Table/instance join
  beyond carrying the linkage as provenance.

## Ordered slices (each a reviewable commit/PR)

### Slice 0 — types + shared location
- **Decide where `ResolvedPlacement` lives.** It is `{ system: string; worldFromArray: Affine3 }`
  (ADR-0015). `Affine3` is in `src/datasource/math.ts`; `handle.ts` currently imports nothing from
  `datasource`, and `datasource` depends on `handle` (not vice-versa). **Recommendation:** extract
  `Affine3` (+ the small vec/affine helpers placement needs) into a leaf module both import — e.g.
  `src/coords.ts` — rather than a type-only back-import that inverts the layer direction. Minimal
  alternative: `import type { Affine3 }` into `handle.ts` (no runtime cycle, `math.ts` is pure), but
  prefer the leaf.
- Add `placement?: ResolvedPlacement` to `GpuField` and `FieldValue` (absent ⇒ **array space**).
- Helpers: `placementOf(v)` (absent ⇒ `undefined`; **do not** default to identity — array-space and
  placed-at-identity-in-system-S are distinct states); `systemsAgree(a, b)` (both absent ⇒ ok; both
  present ⇒ `system`-name equality; exactly one present ⇒ error — placed and unplaced can't combine).

### Slice 1 — builder + executor threading (axes + role + placement together)
- Extend `OpType` with `inferPlacement?`, `inferAxes?`, `inferRole?` (mirror the `inferBasis`
  signature and the "absent ⇒ pass through first input" default).
- In `op()`: gather `inAxes`/`inRoles`/`inPlacements` from input handles; compute the three `out*`
  arrays via the hooks; **persist them on the `GraphNode`** (alongside `outBases`); pass to
  `makeField`.
- Extend `makeField` and `GraphNode` to carry the three.
- Thread the same through `source()`/`points()`/`grid()`/`feedback()`/`delay()` — they currently
  carry only `element`+`basis`, so a source's `axes`/`role`/`placement` are lost at build today.
- Executor: stamp `node.outPlacements[i]` (and `outAxes`/`outRoles`) onto the produced value **when
  unset** — the `basis` line at executor.ts:365, generalised. An op that sets placement itself
  (e.g. `splatDensity`) must win, exactly as basis does.
- Default `inferPlacement` = pass through `inputs[0]` — the "a convolve does not move the grid"
  default (pointwise + neighbourhood ops are all correct under it).

### Slice 2 — agreement + first constructing source
- Binary ops (`addGrids`, `fieldArithmetic`) get an `inferPlacement` that asserts `systemsAgree`
  then passes through — the build-time "reject add across systems" check, sitting beside the
  width/height check `addGrids.inferShapes` already makes ([addGrids.ts:37](../src/gpu/graph/ops/addGrids.ts)).
- `splatDensity` becomes a **source that constructs a placement**: it splats points *in their own
  system* and stamps that system's `ResolvedPlacement` (from bbox + resolution) onto the output grid,
  so the world→cell relation is recorded on the output instead of vanishing. The `bbox` param stays
  the region selector until vector-`ParamType` lands (note it; don't fix here).
- `hashSource` / source identity keying folds `placement` in (same bytes at a different extent ⇒ a
  different field). Unblocked now because placement is on the build-time handle + node.

### Slice 3 — table-centroid source (the dominant real case; on-ramp to B)
- A per-`(table, region)` centroid-extract source: given a table and a **resolved** per-region
  transform, yield a points `FieldValue` with a full `ResolvedPlacement` and the
  `region/region_key/instance_key` linkage as provenance. The transform *resolution* (annotation
  walk + policy — the five `point_transform` modes in MDV's `_resolve_regions_for_table` /
  `_choose_point_transform`) is host/loader-side, above the `Loader`, per the sd.js ownership
  boundary; the op receives the resolved matrix. **One field per (table, region) — never a merged
  cloud** (that is how we avoid MDV's `_concat_spatial_tables` multi-modal-merge bug).
- Carry native coords + matrix; apply on the GPU. **Do not** host-bake transformed coordinates like
  MDV's `_transform_table_coordinates` — that is its documented stopgap for not being able to apply
  transforms at view time, and rewriting a million centroids on the host is the Tier-2-hostile
  transfer residency exists to remove.
- Loader: `Multiscale.worldFromArray → placements: ResolvedPlacement[]` (one `global` this pass) so a
  loaded raster tile carries a placement into the graph.

### Slice 4 — param units (small; enables the equivariance test)
- `ParamSpec.units?: "cells" | "world"` (absent ⇒ `cells`, today's behaviour). Lets the UI say
  "2 cells = 0.8 µm" and lets a world-unit param convert against the field's placement. Prerequisite
  for scale-equivariance being *testable*.

## Forward-compat seam: tiled points elements (do not foreclose)

A points *element* (future revision) is a spatial element with its own transform and, depending on
indexing, **natural tiles**. The intended path: when a spatial tile loads, splat that tile **with an
apron** into a cached raster. Preserve these now so that path is "call it per tile," not a redesign:

- **The facet needs no change for it.** A points field's `placement` is the same `ResolvedPlacement`
  whether resolved from a table-association (slice 3) or read off a points-element's own transform.
  Only the *resolution* differs, and that is loader-side.
- **Keep `splatDensity`'s placement construction tile-agnostic** — it constructs from a bbox +
  resolution over whatever point subset it is given, so a tile's point subset with a tile bbox
  (= tile placement) already works; the tiled path becomes "invoke per tile," and the apron is a
  placement-aware halo (relates to the windows-vs-quadrats halo question). Tiling + caching itself is
  ADR-0008 datasource territory, not this pass.

## Testing (heed the GPU-suite hazards)

- CPU-golden for each `inferPlacement` default and the agreement error (build-time throw).
- Placement round-trip: source-with-placement → passthrough ops → `pull`; placement survives
  (mirror the basis/backend-parity tests).
- `hashSource`: identical bytes, different placement ⇒ **different** memo key (identity, not
  collision) — and unchanged ⇒ hit.
- `splatDensity` placement: a **paired elementwise** check, aggregated into one failure (per-element
  `expect` loops kill the Dawn fork — see the GPU-test-assertion-loops note); compare against the
  **host GPU path**, not `cpuGolden`, and use a non-256-aligned grid width (24, not 64) so row
  padding is actually exercised (the splat de-pad integer-division lesson).
- After slice 4: scale-equivariance — a world-σ filter agrees across two resolutions.

## Watch-items

- **array-space (placement absent) ≠ identity placement.** Absent means unitless/cell-indexed;
  identity means "already in system S." Collapsing them would claim every bare test grid lives in
  `global`. Keep `placementOf` returning `undefined` for absent.
- **Executor stamps only when unset** — a constructing op (splat, centroid-extract) must set its own
  placement and win.
- **Layer direction** — don't let `handle.ts` back-import `datasource`; use the leaf (slice 0).
- **Vector-`ParamType` gap** blocks UI-editable placement; loader/programmatic path is fine. Not a
  blocker for A; a blocker for the composer surfacing it.
