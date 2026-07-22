# ADR-0015 — Channel axis, label polarity, and coordinate systems (NGFF/spatialdata-grounded)

Status: **draft / proposed** (2026-07-12)

## Context

To run the compute graph on **real** multichannel image data (first target: the Xenium
morphology image — 4 channels — through the sd.js loader, the sibling of the already-landed
H&E tile, ADR-0010-loader), the value lattice needs three things it does not carry today.
The gap surfaced concretely while spiking a magic-wand `regionGrow` op (`prototype/wand/`):
a multichannel image only fit by abusing `ElementType.vec3`, and a selection mask had nowhere
honest to land.

This ADR is **not** a new axis architecture. **ADR-0004 already decided** the model:
`Field = { domain ⊥ element ⊥ axes ⊥ dtype }`, with `TensorAxis = {name, length}` an *open,
runtime-length* axis **deliberately distinct** from the closed `ElementType` algebra
(scalar/complex/vec/quaternion). The element-type facet **landed**; the `axes` facet is still
**proposed** (ADR-0004 §"Implementation status"). This ADR **lands the `axes` facet for the
channel case**, and adds the two facets ADR-0004 does not cover: an **intensity/label polarity**
and **coordinate systems**. Everything here is grounded in the NGFF (OME-Zarr 0.4/0.5) and
spatialdata models, which the sd.js packages already implement (`packages/core/src/schemas`,
`.../transformations`, `.../models`) and which this repo's `src/datasource` does **not** yet
(it is intensity-only, fixed-arity, single-world — `src/datasource/types.ts`).

### Ownership boundary — sd.js owns axis/transform *reasoning* (user steer, 2026-07-12)

> "Where possible, sd.js should own the reasoning about how to handle axes/transformations."

sd.js already has the full model (NGFF axis parsing, `omero` channel descriptors, the
`Transform` classes → `Matrix4`, named coordinate systems, `mapSpatialValuesToXYZ`). tgpu-htj2k
must **not re-implement** any of it. The seam is already clean: `src/datasource` exposes a pure
`Loader` (`getChunk(id) → Tile`) + `Multiscale` metadata (`src/datasource/types.ts`), and the
**sd.js-backed adapter lives playground-side** (`playground/src/datasource/spatialDataLoader.ts`).
So NGFF interpretation, channel-metadata parsing, and transform composition all happen on the
**sd.js side of the `Loader`**; tgpu-htj2k **consumes resolved values** across it. This turns
several would-be decisions here into boundary mappings rather than re-modelled types (see Forks).

### The Xenium-4ch trap (why this matters even at n≤4)

4 channels fit `vec4`'s lane count exactly — and that is the trap. Morphology channels
(DAPI, boundary, interior, …) are an **open axis of bulk samples with no per-sample algebra**,
not a geometric 4-vector with dot/cross. Xenium *protein* reaches ~1–40 channels; multiplexed
IF, tens. The channel axis is genuinely open; it must not squat in the algebra slot.

## Decision

Three additions, each an optional facet (absent ⇒ today's behaviour, so existing ops are
untouched — the ADR-0004 discipline).

### 1. Channel is an enriched `TensorAxis` (lands ADR-0004's `axes`, NGFF-grounded)

Keep space in `Domain` (ADR-0004: `grid.size = [w,h] | [w,h,d]`) and channel/time/gene in
`axes` — consistent with how sd.js handles transforms (spatial transforms project out c/t via
`mapSpatialValuesToXYZ`, `packages/core/.../transformations.ts`). Enrich `TensorAxis` with NGFF
axis semantics:

```ts
type AxisType = "channel" | "time" | "gene" | "custom";  // NGFF: space|time|channel|custom (space ⇒ Domain)

interface TensorAxis {
  name: string;                 // "c", "t", "gene"
  type: AxisType;
  length: number;               // OPEN, runtime — NO per-element algebra (ADR-0004)
  unit?: string;                // UDUNITS-2 (time); channels usually none
  entries?: ChannelEntry[];     // index-aligned per-channel metadata (omero.channels)
}

interface ChannelEntry {        // NGFF omero.channels[i]
  label?: string;               // marker/stain name — the semantic identity of a channel
  color?: string;               // hex, e.g. "0000FF"
  window?: { min: number; max: number; start: number; end: number };
  active?: boolean;
}
```

A 4-channel morphology image is `domain: grid[w,h], element: scalar, axes: [{name:"c",
type:"channel", length:4, entries:[…]}]` — **not** `element: vec4`. `entries` are
**sd.js-resolved** (parsed from `omero.channels` on the sd.js side of the `Loader`) and consumed,
never parsed here.

#### Open axes carry metric linear algebra (**amends ADR-0004**)

ADR-0004 said an open axis has "no intrinsic arithmetic" and only *axis-parametric* ops
(reduce/select/contract). **That is too strong, and segmentation is the counter-example** — the
wand's own `dist2` is a channel-weighted Euclidean distance over the channel axis, already
load-bearing. An open axis is a **metric vector space of runtime dimension**. The correct split
is by op *family*, not "algebra vs none":

- **Metric linear-algebra family — available on any vector dimension (open axis *or* fixed `vec`
  element).** Dot/contraction, norm, distance (Euclidean/Minkowski/cosine/Mahalanobis), normalize,
  projection/matmul, and the metric-aware **n-ary cross** (determinant construction, orthogonal to
  its inputs, `|·|` = Gram determinant — Ref [cross]). `dot` over a `vec3` and over a `channel[3]`
  are the *same* operation; interleaved-lanes vs planar-axis is a lowering detail, so the family is
  defined over "the contraction dimension" and lowered per representation.
- **Closed fixed-arity structured algebras — element-only, rejected on an open axis.** Complex
  ×/exp/FFT, quaternion ×/slerp: meaningful only at their specific small arity. The build-time
  check narrows from "reject all element algebra on an axis" to "reject just the closed structured
  algebras"; the metric family is *provided* on axes.

**The whole metric family is parameterised by a metric** (Ref [cross] develops the cross product
for a general metric matrix): identity ⇒ ordinary dot/Euclidean; **diagonal** ⇒ the wand's
`channelWeights` (the diagonal case of a general metric — not an ad-hoc knob); **full** ⇒ a channel
covariance ⇒ Mahalanobis. Dot, distance, and cross are thus one metric-parameterised family over
the channel axis. Plain axis-parametric ops (`selectAxis(c)`, `reduceAxis`, `compositeChannels`)
remain alongside it.

### 2. Intensity vs label is a field **polarity** (new; NGFF `image-label`)

A label/selection image is *not* a per-sample element (correcting the `prototype/wand/` note):
it is a whole-field polarity with structural constraints.

```ts
type FieldRole =
  | { kind: "intensity" }                         // default; real/complex; MAY carry a channel axis
  | { kind: "label"; labels: LabelMeta };         // integer, 0 = background; NO channel axis

interface LabelMeta {
  source?: string;                                 // ref to parent intensity field (NGFF image-label.source)
  colors?: Array<{ value: number; rgba: [number, number, number, number] }>;
  properties?: Array<{ value: number; [k: string]: unknown }>;   // value → property bag
  resample: "nearest";                             // INVARIANT: never linear — averaging fabricates ids
  instanceKey?: string;                            // link to a Table by instance id (region/region_key/instance_key)
}
```

Build-time invariants: a `label` field's `dtype` is integer (`u32`/`i32`), it has **no**
`channel` axis, and any multiscale/resample path must honour `resample: "nearest"`. The wand's
mask lands here honestly: `role: label`, `dtype: u32`, one region id.

### 3. Coordinate systems: **consume** sd.js's resolution, don't re-model it

Per the ownership boundary, tgpu-htj2k does **not** introduce a `Transform` algebra
(scale/translation/affine/rotation/mapAxis/sequence) — that is sd.js's, already built
(`packages/core/.../transformations.ts`). ADR-0008's `worldFromArray: Affine3` stays the shape of
what we hold: a **resolved placement handed across the `Loader`**, generalised only enough to name
its target system.

```ts
// What sd.js resolves and hands us across the Loader boundary — NOT a transform algebra we compose.
interface ResolvedPlacement {
  system: string;          // target coordinate-system name (default "global")
  worldFromArray: Affine3; // sd.js-composed matrix (Sequence/Affine/Rotation already collapsed)
}
```

`Multiscale.worldFromArray: Affine3` becomes `placements: ResolvedPlacement[]` (the single-world
case is `[{system:"global", worldFromArray}]`) — **populate one system this pass**. Composition,
axis-ordering, and c/t projection stay on the sd.js side; this repo receives the collapsed matrix.
The facet is still **cross-cutting** — raster, label, points, and *geometry* (ADR-0010) all carry a
`ResolvedPlacement`, so the 3D section-stack alignment (the tubule pipeline) plugs in by asking
sd.js for the per-section matrices, not by composing transforms here.

### Scope line (this pass)

**In:** enrich `TensorAxis` (type/unit/entries, sd.js-resolved) and land the **channel** axis for
2D grids; the **label** `FieldRole` with its invariants; the wand mask lands as a real label field;
`ResolvedPlacement` **consumed** from sd.js (one `global` system this pass). Exercised end-to-end
on the real 4-channel Xenium morphology tile via the sd.js loader.

**Out (declared-for, not built):** `z`/`t` axes and 3D domains (ADR-0004 already owns 3D
`Domain`); ops that actually *apply* a placement; multi-system placement and section-stack
alignment (obtained by asking sd.js for the per-section matrices, not composed here); channel
compositing/LUT ops beyond `selectAxis`; the Table/instance join. Seams the types admit but
nothing yet populates.

## Why

- **No new architecture, honest grounding.** Reuses ADR-0004's decided domain⊥element⊥axes split
  and its "absent facet ⇒ untouched" additivity; specialises the open axis to NGFF's channel and
  adds only what ADR-0004 lacked, each traced to a primary-source metadata shape.
- **The Xenium-4ch trap is defused by construction** — channels can't reach the algebra slot.
- **Label as polarity, not element** — carries the value→properties map, source link, and the
  nearest-only resample invariant that a per-sample element could not express.
- **One coordinate model for raster + label + points + geometry** — unifies with spatialdata and
  with ADR-0010's geometry (a mesh also lives in a coordinate system), instead of a per-facet world.

## Scale / Tier-2 (the reason to do this on real data now)

The small-data assumptions are **structural, not the hash alone**: Tier-1 ops move host arrays
in and read the full result back (`.read()`), `cpuGolden`/`allFinite` scans every output element,
and `hashSource` re-hashes every byte on **every** pull (executor.ts:122–126) — all O(data) per op,
fine at 40×28, ruinous at a real tile. ADR-0009/0010 already target **Tier-2 resident-buffer
edges** as the exit. This ADR's job is to make the *type* carry what Tier-2 needs:

- **Source keying moves off byte-hashing** to identity/`version` (`GpuField.version` exists for
  exactly this) for large inputs — a channel tile is hashed once, not per re-derive.
- **Readback becomes axis-sliced** — pull one channel / one label region, not the whole stack.
- A resident field is `{ domain, element, axes, role, placement }` **without host `data`** — a
  GPU buffer handle instead (the Tier-2 `FieldValue` variant handle.ts already anticipates).

Landing the channel/label types is the prerequisite for exercising the real tile; the Tier-2
edge is the immediate follow-on ADR that the real tile's size will force and prioritise.

## Open questions (react to these)

The sd.js-ownership steer **settles two of the original four** — recorded here as resolved so the
reasoning is visible; the remaining two are the only ones needing a call.

- **Fork A — channel axis vs unified NGFF list. → SETTLED by the steer.** sd.js owns axis
  interpretation, so the NGFF unified list is resolved on its side of the `Loader`; tgpu-htj2k
  keeps the compute-convenient `Domain` (space) + `axes` (c/t) split (ADR-0004), populated from
  sd.js's resolved axes. Not re-modelled here.
- **Fork C — named space axes / units. → MOSTLY SETTLED by the steer.** Names/units come resolved
  from sd.js; tgpu-htj2k holds a thin `spaceAxes: {name,unit}[]` sidecar (index-aligned to the
  `[w,h]` tuple) populated at the boundary, so ops keep indexing positionally. Only residual call:
  whether even that sidecar is worth carrying vs reading names off the `ResolvedPlacement.system`.
  (Recommend: carry the sidecar.)
- **Fork B — where the *consumed* channel metadata lives** (still ours). On `TensorAxis.entries`
  (index-aligned, chosen here, mirrors `omero.channels`) vs a field-level sidecar. (Recommend: on
  the axis.)
- **Fork D — label multiscale** (still ours; the datasource is in this repo). The
  `resample:"nearest"` invariant must reach the floor-halving pyramid path (`multiscale.ts`),
  which currently assumes intensity. In scope now, or deferred with 3D? (Recommend: enforce the
  invariant in the type now; wire the pyramid path when labels first get multiscaled.)
- **Fork E — cross-product representation on an open axis** (new; from the metric-algebra
  amendment). Dot/cosine/distance are unambiguous binary ops in any dimension, but the generalised
  cross is **(n−1)-ary** (length-L axis ⇒ L−1 inputs → 1, determinant form, Ref [cross]) — not a
  binary `a×b`. Adopt the n-ary determinant form, or a wedge/bivector `a∧b`, or both? Not a blocker
  (dot/distance land first). (Recommend: n-ary determinant form, metric-aware, matching Ref [cross];
  revisit wedge if a bivector consumer appears.)

## Implementation status

- **Facets landed (2026-07-12), additive.** `handle.ts` gains `TensorAxis`/`AxisType`/`ChannelEntry`
  (fork B: channel metadata on the axis), `FieldRole`/`LabelMeta` + `INTENSITY` (label polarity),
  optional `axes?`/`role?` on `GpuField`+`FieldValue` (absent ⇒ today's behaviour, every existing op
  untouched), and helpers `axesProduct`/`channelAxis`/`roleOf`/`assertLabelInvariants` (fork D
  invariants). Data-layout convention recorded: element lanes interleaved, open axes **planar**
  (zarr/xarray `(c,y,x)`). `tsc --noEmit` clean; no existing op changed.
- **Validated on the real model (2026-07-12).** The `prototype/wand/` spike now runs a 4-channel
  synthetic image as `axes:[{c,channel,4}]` (element scalar, **not** `vec4`) and emits the mask as
  a real `label(u32)` field through the actual `Graph`/executor/memo. Provenance↔memo behaviour
  holds unchanged (edit ⇒ 1 fill, unchanged ⇒ 0). #1 and #2 are now resolved *in the model*.
- **Known gap for the full landing (not the spike):** `graph.ts` `makeField`/`op()` thread
  `element`/`basis` onto the returned `GpuField` handle but **not** `axes`/`role`, so *build-time*
  inference can't yet see them — the spike works only because ops read `axes` off the resolved
  `FieldValue` at *execute* time. Threading `axes`/`role` through the builder (parallel to
  `inferElements`/`inferBasis`, e.g. an `inferAxes`/`inferRole`) is the first real-code follow-on,
  needed before an op can validate on a channel axis or reject a label at build time.
- **Known gap — vector params have no `ParamType`.** Surfaced by the spike: `channelWeights` is a
  `number[]`, but `ParamType` is only `number | int | enum | bool`, so it rides through `Params`
  untyped. That works at runtime but is **invisible to the React-Flow palette / any UI**, so a
  channel-weighted op cannot expose its weights as editable. A vector/array `ParamType` (or a
  per-axis-index param family) is needed before channel weights become user-facing. Orthogonal to
  the axis/label facets themselves; recorded here so it isn't rediscovered.
- **Still proposed:** `ResolvedPlacement` consumption at the `Loader` seam; the datasource
  `Multiscale.worldFromArray → placements[]` change; wiring the `nearest` invariant into the
  floor-halving pyramid; the real 4-channel Xenium tile end-to-end.

## References

- **ADR-0004** (field type model: domain⊥element⊥axes; this ADR *lands* its `axes` facet),
  **ADR-0008** (single `worldFromArray` — generalised here), **ADR-0009/0010** (Tier-1→Tier-2
  resident edges; geometry also lives in coordinate systems), **ADR-0010-loader** (sd.js source).
- **[cross]** "Generalization of the cross product" (arXiv:2206.13809): n-ary cross via a
  determinant construction, extended to M vectors in N dimensions and to a **general metric
  matrix** — the basis for the metric-parameterised open-axis linear-algebra family (amends
  ADR-0004's "no arithmetic on open axes").
- NGFF OME-Zarr 0.4/0.5 schemas (axes, `omero`, `image-label`) — vendored at
  `SpatialData.ts/.../packages/core/src/schemas/{image,label}.json`; spatialdata design doc
  (Image/Labels dims, coordinate systems, transforms).
- sd.js prior art: `packages/core/src/{schemas,transformations,models}`,
  `packages/zarrextra/src/omeZarr.ts`, `packages/avivatorish/src/layerChannelState.ts`.
- `prototype/wand/` (the spike + NOTES.md that surfaced the gap), `src/gpu/graph/handle.ts`
  (the lattice being extended).
