# Stain-space channels, transparency ordering, and the comparator registry

Status: **design note** (2026-07-23)

> Written as ADR-0020 and **demoted the same day**. The ADR audit
> ([`decisions/README.md`](decisions/README.md)) found that every decision record written *about
> direction* rather than about work in flight has gone unimplemented — giving speculation the form of
> a decision is what turns the log into a source of guilt rather than a map. Nothing below is
> withdrawn; it is simply not being built yet, and a design note is the honest home for that. It
> becomes an ADR the day someone starts implementing it.
>
> Two parts are load-bearing sooner than the rest and may be extracted early: the
> **transparency-ordering change with coarse-tile culling** (§3), which is a prerequisite for *any*
> transparent stack, and the **comparator registry** (§5), which is useful well beyond H&E.

## Context

The [scene note](serial-section-alignment-and-multi-viewport.md) stacks six H&E serial sections in one 3-D scene. Two things follow immediately, and they
turn out to be the same problem.

**Transparency.** An H&E slide is mostly white background. Six opaque near-parallel planes stacked
100 µm apart are an opaque brick — the free-3D view shows the top section and nothing else, and the
nadir lightbox cannot compare a section against its neighbour. Making white transparent is not a
nicety here; without it the multi-viewport shell has nothing useful to show.

**Feature selection.** The wand (see the [wand note](wand-contours-and-lofted-geometry.md)) has to
pick "the same glomerulus" on six differently-stained slides. Inter-section stain variation is the most notorious artefact in serial histology, and a
selection metric computed on raw RGB will chase it. So the wand needs a colour representation in
which "which stain" is separable from "how much stain and how dark this slide came out".

Both are answered by leaving RGB.

### What the material does today

`playground/src/datasource/tileChannelMaterial.ts` is a TSL node material: it samples a 4-plane tile
texture, windows each channel by `uLo`/`uHi`, gates by `uVis`, tints and sums, and returns
`vec4(rgb, uOpacity)` — a **constant per-image alpha**. `ChannelComposite` holds the uniform nodes so
one `update()` moves every resident tile at once.

### What the depth/blend model does today

From the 1b co-registration work: the first image is an opaque base (depth test + write on, so
intra-image LOD resolves via a geometric `LEVEL_Z_BIAS` of 0.1 array units per level), and later
images are overlays drawn `depthTest = false` in **add order** (`renderOrder = layer index`). That
model is correct for two co-registered modalities of one slide. It is wrong for a stack you orbit:
add order does not change with the camera, so the stack composites back-to-front from one side and
front-to-back from the other.

### One constraint from the slice-0 spike

Replacing `colorNode` on a mounted node material is a **no-op** — the compiled graph is not rebuilt.
Every live change must therefore travel through **uniforms**. This is not a limitation to work
around; it is the shape the design already wanted, and it decides §1 below.

## Decision

### 1. Store unmixed optical density **in place of** RGB

Per tile, on arrival: convert to optical density, unmix through a 3×3 stain matrix, and store the
three resulting components — **hematoxylin, eosin, residual** — in the same `rgb8` texture that holds
RGB today.

```
OD    = -log10(max(I, ε))          // per channel, I in (0,1]
stain = M⁻¹ · OD                   // M = column-wise stain vectors (Ruifrok–Johnston)
alpha = saturate((|OD| - t0) / (t1 - t0))
```

The properties that make this worth doing:

- **It costs no memory.** Three components in, three out. The stain matrix is invertible, so **RGB is
  exactly recoverable by re-mixing in the material** — a true H&E view remains available as a render
  mode, not as a second copy.
- **It gives the wand what it needs**, on the GPU, in the same texture the renderer draws — rather
  than a second implementation of the same maths for the selection path.
- **The stain matrix is a `mat3` uniform.** Changing it re-runs the kernel over resident tiles
  (cheap; they are already on the GPU) and never rebuilds a node graph — which the spike proved is
  the only thing that works.
- **It is opt-out.** An identity matrix reproduces today's behaviour exactly, so the path can land
  before the stain vectors are tuned.

Alpha comes from total OD, so no fourth channel is needed. Thresholds `t0`/`t1` are **per image** —
staining varies slide to slide, and a global threshold would force the user to compromise across six
sections.

The per-image channel panel addresses the derived channels directly: its three channels become
H / E / residual with their own contrast limits and tints, and "true H&E" becomes a preset that
re-mixes.

### 2. It runs as a per-tile compute pass, op-shaped, outside the pull-graph

Three levels of commitment were considered:

| | |
|---|---|
| **A** — TSL only | Cheapest; leaves the wand with no resident derived data and forces the maths to exist twice for the wrong reason. |
| **B** — a `"use gpu"` TGSL kernel on tile arrival, outside `src/gpu/graph` | **Chosen.** The material becomes pure presentation (sample, window, tint, threshold) per ADR-0009; the kernel is a normal pure op with `ParamSpec`s, testable headless. |
| **C** — tiles as nodes in the pull-graph | The real ADR-0009 endgame; requires solving streaming residency inside the graph, which is a project, not a step. |

B is a way-station and is labelled as one. The honest cost: it inserts a compute dispatch into a
tile-arrival path that currently goes straight from decode to `texture.needsUpdate`, and it must not
stall streaming.

A CPU twin of the unmix exists for the wand (the [wand note](wand-contours-and-lofted-geometry.md)), pinned by a parity test. That duplication is
the established pattern in this repo — `applyTransform` in `swept.ts` is explicitly "the CPU image of
its WGSL codegen", and `implicit`/`implicitGpu`, `swept`/`sweptGpu`, `backendParity.gpu.test.ts` all
work this way — not an accident to be cleaned up.

### 3. Ordering: use three's transparent sort; stop using depth for LOD

Once every section is transparent, three changes land together:

1. **All tiles: `transparent = true`, `depthTest = true`, `depthWrite = false`, and stop forcing
   `renderOrder`.** three already sorts transparent objects back-to-front by projected depth, and
   each tile is its own mesh — so that *is* per-tile ordering, for free, re-sorted per `render()`
   call and therefore correct per viewport. Keeping `depthTest` on (unlike today's overlays) is what
   lets opaque tubule meshes correctly occlude sections behind them.
2. **Coarse-tile culling replaces `LEVEL_Z_BIAS`.** The geometric z-bias relies on the depth *test*
   to hide a coarse tile under its finer replacement; with `depthWrite` off, a coarse tile and its
   resident children both blend — double density, worst during streaming. `TileRenderer` must instead
   hide a tile once every child covering it is resident. That is exact, deletes the z-bias hack, and
   cuts overdraw.
   This is a **prerequisite** of per-pixel alpha, not a follow-up: the artefact is subtle enough to be
   mistaken for real staining variation, which in this application is dangerous.
3. **Plane-splitting stays an escape hatch, documented not built.** Per-tile sorting is exact while
   tiles do not interpenetrate; two sections tilted into each other break it. `src/geometry/bsp.ts`
   (plane-native BSP) is the exact fix, but it costs a rebuild on every gizmo drag — precisely when
   you are rotating sections into each other. Revisit only if it bites during real alignment.

**Not chosen: order-independent transparency.** Weighted-blended OIT removes sorting and handles
intersections, but it approximates the blend — and in H&E the colour *is* the measurement.

### 4. Onion-skinning is the alignment aid

With per-pixel alpha the nadir viewport becomes a lightbox, and the natural registration aid is
onion-skinning: active section at full strength, immediate neighbours ghosted in contrasting tints,
the rest hidden. It reuses per-image opacity, blend mode, and the new alpha — nearly free.

This is the aid that answers the actual question ("is *this* section aligned to its neighbour"). The
world grid of the [scene note](serial-section-alignment-and-multi-viewport.md) tells you where the axes are, which is a different and lesser question; it
stays as a simple toggle.

### 5. Comparators are an **open registry** of CPU/WGSL twins

A selection metric and its on-screen visualisation must agree, or the picture lies about the
selection. So a comparator has to exist on both sides:

```ts
interface Comparator {
  name: string;
  polarity: "distance" | "similarity";      // lower-is-closer vs higher-is-closer
  evalCpu(a: Float32Array, b: Float32Array): number;
  emitWgsl(ctx: UniformCtx): string;
  specs(): ParamSpec[];
}
```

Registered, not enumerated — a bring-your-own comparator is an ordinary registration rather than a
core edit, which is the direction of "a library where users bring extensions that hook into
rendering". Carrying `ParamSpec`s makes comparators editable and breedable like everything else
(ADR-0012). Defined over an **N-vector sample**, not over H/E specifically, so they apply to any
tensor-valued field — consistent with ADR-0004's element ⊥ axes split.

Initial set: `l2` (optionally weighted), `l1`, `linf`, `dot`, `cosine`, `angular`, `mahalanobis`.

**`angular` is not optional.** Colour-deconvolution stain vectors *are* unit directions in OD space:
direction encodes *which* stain, magnitude encodes *how much*. Angular distance therefore selects
"hematoxylin-like regardless of staining intensity" — exactly the invariance needed when the same
structure is picked across six differently-stained slides, which is where correspondence lives.
Euclidean distance in OD conflates stain identity with concentration and will select the dark parts
of the wrong structure.

**`polarity` is a required field, not a convention.** `dot` and `cosine` are similarities; `l2` and
`angular` are distances. Without an explicit flag every threshold comparison has to know which kind
it holds, and getting it wrong inverts a selection silently.

The closed-union alternative was rejected for extensibility; the ADR-0007 expression IR is the
eventual authoring language for a comparator, at which point registry entries can be *generated*
rather than hand-written — the interface above is unchanged by that.

### 6. `distance` is a per-viewport render mode

Render modes: `colour` (re-mixed true H&E) / `channels` (H/E/residual tints, today's behaviour) /
`distance`. In `distance`, `d(sample, reference)` is mapped through a colormap with the current
tolerance drawn as an **isoline**; the reference comes from the same `pick()` result, so hovering
previews a distance field before any seed is committed.

Per-viewport rather than global: the nadir pane can show the distance field while the free-3D pane
still shows true colour. A render mode rather than an overlay: it replaces the colour mapping rather
than compositing over it.

This makes the drag-to-set-tolerance gesture legible — you see the continuous field and watch the
isoline sweep the structure — and it is the first instance of a general class (a scalar derived from
a sample, colour-mapped, thresholded) that the op-graph should eventually own.

### Scope line (this pass)

**In:** OD + stain `mat3` uniform with identity default; unmixed storage in place of RGB; per-image
alpha thresholds; the transparency-ordering change **including coarse-tile culling**; onion-skin; the
comparator registry with CPU/WGSL twins and the `distance` render mode.

**Out (designed-for, not built):** pull-graph integration of the tile pipeline (level C); tuned
per-store stain matrices (identity default, manual entry first); plane-splitting; OIT;
comparators authored as ADR-0007 expressions.

## Why

- **It makes the transparency requirement pay for the selection requirement.** The same change that
  lets you see through a slide is the one that lets the wand ignore how darkly it was stained.
- **It costs no memory and loses no information.** Unmixing in place is invertible; nothing is
  traded away for the derived channels.
- **It removes a latent bug rather than adding a feature.** `LEVEL_Z_BIAS` + depth-test LOD already
  depends on view direction; the coarse-tile culling that transparency forces is simply the correct
  mechanism.
- **Uniform-driven change is the only thing that works**, per the spike — so making the stain matrix
  and thresholds uniforms is both the right design and the only viable one.

## Consequences / open questions

- **Coarse-tile culling has blast radius.** `TileRenderer` is shared with `dualView` and
  `spatialvolume`. The change is a strict improvement (exactness + less overdraw) but it is not
  contained to the new page, and it interacts with the scene note's eviction: a culled-but-resident coarse
  tile is the zoom-out fallback and must not be evicted first.
- **Stain vectors are unmeasured.** Ruifrok–Johnston's published H&E vectors are a starting point;
  real slides differ. Identity default plus manual entry is the honest first step; estimating the
  matrix from the data (e.g. Macenko) is a later op — and a natural op-graph citizen.
- **The residual channel is a diagnostic, not a stain.** It absorbs everything the two-stain model
  cannot explain. Exposing it in the channel panel is deliberate — a large residual means the matrix
  is wrong — but it should not be read as biology.
- **A dispatch now sits in the tile-arrival path.** If it stalls streaming, the fallback is to unmix
  lazily per level or to keep RGB and unmix in the material (level A) for the render while the wand
  uses its CPU twin — worse, but contained.
- **Comparator parity is a test burden.** Every registered comparator needs a CPU/GPU parity test or
  the render mode and the selection can silently disagree — which is exactly the failure the twin
  design exists to prevent.

## References

- **[in-repo]** the [scene note](serial-section-alignment-and-multi-viewport.md) (the scene this serves), the
  [wand note](wand-contours-and-lofted-geometry.md) (the wand that consumes the derived
  channels and the comparators), ADR-0009 (rendering as ops — B is a way-station to C), ADR-0015
  (channel axis, `omero` channel entries, label polarity), ADR-0004 (element ⊥ axes — why comparators
  are defined over N-vectors), ADR-0007 (expression IR — the eventual comparator authoring language),
  ADR-0003 (`"use gpu"` TGSL kernels), ADR-0012 (`ParamSpec`s as the edit/breed surface).
- **[code]** `playground/src/datasource/tileChannelMaterial.ts` (the uniforms this extends),
  `playground/src/datasource/tileRenderer.ts` (`LEVEL_Z_BIAS`, `setDepth`, `setRenderOrder`),
  `src/geometry/bsp.ts` (the plane-splitting escape hatch), `src/color/oklab.ts` (the colourimetry
  home this joins), `playground/src/R3fSpike.tsx` (the uniform-vs-graph finding).
- **[method]** Ruifrok & Johnston, colour deconvolution for immunohistochemical stains (the OD
  unmixing model); Macenko et al., stain-vector estimation (the later automatic path).
