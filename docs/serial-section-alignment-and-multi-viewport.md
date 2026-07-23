# Serial-section alignment: coordinate systems, scene document, and multi-viewport residency

Status: **design note** (2026-07-23)

> Written as ADR-0019 and **demoted the same day**, by the rule in
> [`decisions/README.md`](decisions/README.md): an ADR is for work in flight, and this is not. The
> serial-section editor sits behind the package surface, the viewer-layer promotion, the
> SpatialData→ops bridge, and the deck.gl interleaving spike. "Parts of it will be reused" is not the
> test — applying the rule to the ADR that prompted writing the rule is the case that decides whether
> the rule is real.
>
> Numbers 0019–0021 are all retired rather than reused, so commit history keeps pointing at something
> real. The next ADR is 0022.
>
> **Two sections outlive the application they were written for**, and are the reason to keep reading
> this note even if the serial-section editor is never built:
>
> - **§4, the camera state model.** `{pivot, orientation, distance}` with constraints as projections
>   rather than accumulator clamps. It is API-agnostic, which makes it the natural shared camera
>   representation for three ⇄ deck interleaving
>   ([packaging note](packaging-and-consumers.md)).
> - **§7, the measured R3F + WebGPU findings.** Including the `WebGPURenderer` scissor/viewport Y-flip
>   and the fact that TSL node graphs do not hot-swap. These were *measured*, and this is now their
>   durable home — the spike page that produced them is disposable.
>
> §6 (residency over the union of viewport selections) and §5 (one `pick()`) are the other parts most
> likely to be lifted when the viewer layer is promoted out of `playground/`.

## Context

The multi-image scene editor (`playground/src/datasource/multiImageScene.ts`, ADR-0010-loader slice
1b) puts N SpatialData images in one WebGPU scene with a per-image gizmo. It was built for
*co-registration* — two modalities of the same slide. The next application is different in kind:
**serial sections of one specimen, hand-aligned into a 3-D stack**, so that features can be picked
per section and lofted into geometry (the [wand note](wand-contours-and-lofted-geometry.md)).

### The data this is designed against

`1113PMDC1_IgAN_slices_htj2k_q0.002.sdata.zarr` — six images `IgAN1..IgAN6`, each
`[1,3,1,14165,18155]` uint8 RGB, 6 pyramid levels, 1024² chunks, `experimental.openjph_htj2k`.
Each carries `sequence[scale 0.274 µm/px (x,y), translation z]` to `global`.

Three properties of that store drive most of what follows:

- **Section spacing is 100 µm and in-plane sampling is 0.274 µm** — a ~365:1 anisotropy. The stack is
  a slab ≈ 4974 × 3881 × 500 µm.
- **The z spacing is fabricated.** It came from a conversion parameter (`z_translation_um`), not from
  the specimen. Users are *expected* to edit it, so nothing may treat it as ground truth.
- **Name order is not z order** — IgAN6=0, IgAN4=100, IgAN2=200, IgAN5=300, IgAN3=400, IgAN1=500 µm.
  Any code that sorts by element name is wrong and will look plausible while being wrong.

### What the current scene cannot do

- **One camera.** `MultiImageScene` owns a canvas, a `PerspectiveCamera`, an `OrbitControls`, and one
  `setAnimationLoop`. The alignment task needs at least two simultaneous views (down-z to judge
  in-plane registration, free-3D to judge the stack).
- **Unbounded residency.** It calls bare `select()`, never `selectWithinBudget`, and `TileRenderer`
  has no byte accounting. Level 0 is **772 MB per section**, 4.6 GB for six. One camera over
  near-coplanar images is self-limiting; a nadir overview *plus* a zoomed 3-D view is not.
- **Single-camera assumptions in shared code.** `select()` (`src/datasource/select.ts:72`) is
  perspective-only — it builds frustum planes from `fovY`/`aspect` and measures `nearestDepth` along
  the camera forward. `LoadScheduler` prioritises purely by that `nearestDepth`, which is meaningless
  across cameras.
- **Nowhere to put the work.** The editable transform lives only in `getState()`. Close the tab and
  an afternoon of hand alignment is gone.

### Ownership boundary (inherited from ADR-0015)

> "Where possible, sd.js should own the reasoning about how to handle axes/transformations."

ADR-0015 already names this exact application — *"the 3D section-stack alignment (the tubule
pipeline) plugs in by asking sd.js for the per-section matrices, not by composing transforms here"* —
and defines `ResolvedPlacement { system, worldFromArray }` as what crosses the `Loader`. This ADR
stays inside that boundary, with one honest exception recorded under Consequences.

## Decision

### 1. The user's alignment is a **new named coordinate system**, not an edit of `global`

Each element keeps its shipped transform to `global` and gains a second transformation to a named
system — `aligned` by default. Multiple alignment attempts are simply multiple named systems, which
is what SpatialData's multi-coordinate-system model is *for*. The viewer gets a coordinate-system
selector; nothing ever mutates the store's own placement.

The correction is persisted as a **delta**, not an absolute:

```ts
interface ImageAlignment {
  element: string;             // e.g. "IgAN3"
  system: string;              // "aligned"
  baseFingerprint: string;     // hash of the element's stored coordinateTransformations + L0 shape
  correction: NgffAffine;      // full 4×4 over (z,y,x), appended to the stored sequence
}
```

Three sub-decisions, each with a reason that is not aesthetic:

- **Delta, not absolute.** The correction *is* the scientific artefact — it is the registration the
  human performed. An absolute matrix dissolves it into an opaque affine, and makes "reset this
  section" unrecoverable.
- **A full affine, not rigid+scale.** Sections stretch and shear during cutting and mounting. The UI
  may offer only translate/rotate/scale at first; the *format* must not inherit that limit, because
  widening a persisted schema later is far more expensive than writing it wide now.
- **`baseFingerprint` is mandatory.** A delta is meaningless against a base it was not authored on,
  and we have already established that the z spacing will be re-converted. Without a fingerprint a
  re-converted store makes every saved correction silently wrong, in the most dangerous way: slightly
  and plausibly. On mismatch we **warn and offer to discard** — never silently apply, never silently
  drop.

### 2. Two persisted objects, split by failure mode

- **`SceneDocument`** — the data: per-image `ImageAlignment`, channel/stain settings, and — if the
  [wand note](wand-contours-and-lofted-geometry.md) is ever built — contours and tubules.
  **Speaks NGFF natively**: transformations are serialised in the
  `coordinateTransformations` vocabulary the store already uses, contours as polygon geometry. Strict
  validation; a parse failure is surfaced, never swallowed.
- **`Workspace`** — the UI: mosaic layout, per-viewport camera pose and mode, grid/onion toggles,
  active tool. Lenient; a bad or absent workspace falls back to a default layout and never blocks
  loading a document.

Both keyed by store URL (`tgpu:scene:<url>` / `tgpu:workspace:<url>`) so opening a different store
cannot inherit someone else's alignment.

Not one object with two subtrees: the failure modes are opposite. A workspace that fails to parse
*should* be discarded; a document that fails to parse must never be. That is not expressible cleanly
in one blob.

**`SceneDocument` is the SpatialData form from day one**, not a private shape to be translated later.
sd.js has no write support, so "write to SpatialData" is currently a *transport* gap — which is a far
better thing to be blocked on than an impedance mismatch discovered at the moment of writing.

### 3. `Viewport` becomes a first-class object; the scene is shared

Everything currently singular in `MultiImageScene` — camera, controls, aspect, the `dirty` flag, and
`pickAt`'s canvas-rect assumption — moves into a `Viewport`. The images, tiles, and resident set stay
shared.

- **Two kinds:** `nadir` (locked down-z, pan + zoom only) and `free3d` (full trackball). No
  orthographic projection: `select()` is perspective-only, and a nadir-locked perspective camera is
  visually equivalent here for zero engine change. No side-slice view either — a cross-section
  through six 1-voxel-thick sections 100 µm apart is six lines.
- **react-mosaic hosts viewports only.** Panels stay fixed chrome; a mosaic that can close the panel
  containing the button that reopens things is a support burden for no gain. Uncapped, deliberately:
  we want to see how it behaves.
- **Global:** active section, active coordinate system, per-image channel/stain settings, tubules.
  **Per-viewport:** kind, camera pose, render mode, grid toggle, onion-skin *expression*
  (`all` / `onion` / `solo`).

The active section is global on purpose. Two viewports exist to show *the same thing two ways*; if
the active section could differ per pane, then "which section does my wand click land on" would
depend on where the pointer is — and a mis-attributed contour is the worst failure this application
has, because it is anatomy silently assigned to the wrong z.

### 4. Camera state is `{pivot, orientation, distance}` — no angular chart

The interaction design is ported from psychogeo's `MapCameraControls`: cursor-anchored zoom,
pick-the-feature-under-the-cursor as the orbit pivot, smooth zoom, pan inertia, the two-finger
mapping, distance-relative sensitivity. That is the part worth having.

The **state representation is rebuilt**. psychogeo stores `(bearing, pitch, distance)` about a target
and round-trips through it every drag (`readSphericalFromCamera` → mutate → `applySphericalToCamera`).
That chart is singular at nadir: the bearing is undefined when the view direction is parallel to
world-up, and `lookAt` has no defined right-vector — hence `maxPitch = π/2 − 0.02` (you can never
quite look straight down) and the bearing-based fallback in `pitchAxis`. Separately,
`orbitDragDelta` clamps elevation and writes the clamp *back into the accumulator*
(`this.orbitEl = elev − elev0`), which is why constrained drags feel sticky and stop being reversible.

For us the singularity is not an edge case — **the nadir viewport sits exactly on it**, and a
microscope slide has no north, so the roll-free-relative-to-north constraint that earns its keep on
terrain is mostly a liability.

So: `position = pivot + orientation · (0,0,distance)`. Drags compose incremental quaternions about
the current view-right and a chosen up-axis. Constraints become **projections of the orientation into
an allowed set, applied after the fact** — so a clamp cannot corrupt an accumulator; the sticky-drag
bug becomes structurally impossible rather than carefully avoided. For a no-roll feel,
re-orthogonalise against world-up each frame **except** within ε of world-up, where the previous
right-vector is retained: keep the last good frame rather than forbidding the pose. `free3d` drops the
up-constraint entirely. Viewport kind is then a *mode over one state*, which it is not in the
spherical form.

This is a re-derivation in this repo, not shared code: psychogeo carries `EastNorth`/WGS adapters we
do not want, and its sensitivity reference distances are tuned for terrain, not ~5000 µm spans. The
quaternion fix is portable back to psychogeo as separate work.

### 5. One pick path

```ts
pick(viewport, px, py) → { worldPoint, imageId, arrayXY, geometryHit? } | null
```

`worldPoint` serves the camera pivot; `imageId + arrayXY` serves the wand (the [wand note](wand-contours-and-lofted-geometry.md)), image
selection, and the hover readout; `geometryHit` serves ADR-0012 `resolve`. One raycast, one
world→array inverse (the effective matrix already exists), one place that understands
viewport-local coordinates. Camera work can land first using only `worldPoint`.

In the nadir viewport the ray hits all six sections; **the pick is disambiguated by the active
section**, not by depth order. The result still reports every intersected image, so a click in
`free3d` can change the active section by proximity — but in `nadir` the default is "the click
applies to the active section", full stop.

Implementation starts as a CPU raycast against the existing per-image `pickPlane`s. ADR-0012 already
decided that picking-id travels with depth, and ADR-0014 makes depth part of the render contract, so
a depth-buffer unprojection later gives a pivot on *anything drawn* — including raymarched and
implicit geometry — without per-kind raycast code. The signature above is the seam for that swap.

### 6. Residency is scene-level, over the **union** of viewport selections

- **One byte budget for the scene**, enforced over the union — not per viewport. `selectWithinBudget`
  generalises directly: raise a global minimum-level floor until the union fits.
- **Per-viewport quality weight, focused viewport favoured.** Degrading every viewport equally when
  the budget bites is wrong — you are working in one of them. This is an explicit weight in the union
  computation, not an emergent property.
- **Viewport-aware load priority:** `min` over viewports of `depth × weight`, so the focused
  viewport's tiles arrive first while the others still make progress.
- **Eviction** gains byte accounting, which `TileRenderer` does not have today.

The union computation is pure (selections in, per-image selection out) and therefore belongs in
`src/datasource` beside `select.test.ts`, not in the playground.

### 7. The rendering shell is R3F + drei `<View>`, with one shim

Settled empirically by the slice-0 spike (`playground/src/R3fSpike.tsx`), measured on three r185 /
R3F 9.6.1 / drei 10.7.7:

- Async `WebGPURenderer` through R3F's `gl` prop yields a real WebGPU backend. Spread R3F's
  `defaultProps` — they carry `alpha: true`.
- drei `<View>` scissors two panes with two cameras over one canvas, and its **scissored mid-frame
  `gl.clear(true, true)` does not wipe the frame** — the main risk going in, given that a mid-frame
  `clearDepth()` is known to.
- **The shim:** `WebGPURenderer.setViewport`/`setScissor` are **top-left** origin while
  `WebGLRenderer` — and therefore `<View>` — passes **bottom-left**. Measured: a pane at CSS rect
  `(12, 90, 683×799)` on a 1400×900 canvas gets a correct `y = 11` and is drawn 11 px from the *top*.
  `y' = height − y − h` is the entire fix.
- **The HMR wart is gone** — two Fast-Refresh edits, canvas alive, renderer constructed once. That
  wart is why `spatialSceneMain.tsx` avoids `StrictMode` and why every edit to `SpatialScene.tsx`
  currently costs a hard reload.
- **One `Object3D` must not be shared across `<View>`s.** It appears to work, but a scene walk shows
  it parented at the view-scene *root*, outside either view's group — an accident of double
  `<primitive>` attachment that defeats per-viewport visibility (which onion-skin needs). So **N
  viewports means N lightweight meshes per tile**, sharing one geometry/material/texture. GPU memory
  lives in the textures, so the 4.6 GB figure is unaffected; the cost is CPU-side bookkeeping in
  `TileRenderer`.

### Scope line (this pass)

**In:** the new page (`spatialscene.html` freezes as the simpler co-registration demo); `Viewport` +
react-mosaic with `nadir`/`free3d`; the quaternion camera with the psychogeo interaction set;
`pick()` + hover readout; the union budget, per-viewport weighting, and eviction; the toggleable
world grid; `SceneDocument`/`Workspace` persistence with the `aligned` system, delta affine, and
fingerprint; active section; the IgAN store as the default.

**Out (designed-for, not built):** writing back to the store (sd.js has no write path); non-rigid /
per-section warping beyond a single affine; orthographic `select()`; the depth-buffer pick.

## Why

- **It puts the alignment where SpatialData already has a place for it.** A second named coordinate
  system is the model's own answer to "the same data, placed two ways"; inventing a private transform
  store would have been a parallel mechanism for something already designed.
- **The union budget is the difference between a demo and a tool.** Uncapped viewports over 4.6 GB of
  level 0 is not a caveat you can document your way out of; it is the first thing a user does.
- **The camera rebuild removes a class of bug rather than a bug.** The nadir singularity and the
  sticky clamp are both consequences of the chart, not mistakes in the code that uses it.
- **The spike replaced a coin-flip with a measurement**, including two findings (the Y-flip, the
  `<View>` scene ownership) that would each have cost a day mid-slice.

## Consequences / open questions

- **We author one affine, which brushes the ADR-0015 ownership boundary.** sd.js owns transform
  *reasoning* — parsing, composition, axis projection — and we still consume `worldFromArray` across
  the `Loader`. But authoring a *new* coordinate system means producing an affine here. We take that
  narrowly: we construct the correction and hand it over to be composed; we do **not** build a
  `Transform` algebra. If sd.js grows an authoring API, this collapses into it.
- **`ResolvedPlacement` has not landed.** ADR-0015 §3 proposed `Multiscale.placements:
  ResolvedPlacement[]` and the code still carries a single `worldFromArray`. A coordinate-system
  selector wants the plural form. This ADR does not land it either; it is the natural next
  `src/datasource` change and the selector is a one-system special case until then.
- **Eviction fights coarse-tile retention.** A culled-but-resident coarse tile is exactly what you
  want to keep (it is the fallback when you zoom out) and exactly what a naive evictor drops first.
  This interacts with the [stain-space note](stain-space-and-stack-transparency.md)'s coarse-tile culling and is the fiddliest part of the budget work.
- **Procedural tile deformation would break raycast picking.** Warping tiles — which is what
  non-rigid section registration eventually wants — invalidates the flat `pickPlane` raycast. The
  `pick()` signature is the seam; the implementation would move to depth-unprojection or a GPU
  picking pass, which is ADR-0012's answer anyway. It also interacts badly with plane-splitting
  (the [stain-space note](stain-space-and-stack-transparency.md)).
- **Uncapped viewports are a deliberate experiment.** Each costs a scene traversal and a render pass
  and widens the union. If the honest answer turns out to be "cap at 4", the budget work is what will
  tell us.
- **`select()` stays perspective-only.** A true orthographic nadir view would need a new branch in
  tested engine code. Locked-nadir perspective is the deliberate trade.

## References

- **[in-repo]** ADR-0008 (view-driven multiscale datasource — `select`, the camera as a pull-time
  input), ADR-0010-loader (sd.js as the `Loader` source), ADR-0015 (coordinate systems, the sd.js
  ownership boundary, `ResolvedPlacement`), ADR-0012 (pick → feature; picking travels with depth),
  ADR-0014 (depth in the render contract), ADR-0009 (three.js demoted to presentation), ADR-0017
  (readback discipline).
- **[code]** `src/datasource/select.ts` (perspective-only selection, `selectWithinBudget`),
  `playground/src/datasource/{multiImageScene,tileRenderer,loadScheduler}.ts`,
  `playground/src/R3fSpike.tsx` (the slice-0 measurements quoted in §7).
- **[prior art, re-derived not ported]** `psychogeo` `src/camera/{MapCameraControls,viewState}.ts` —
  the interaction design; its spherical state and clamp handling are explicitly not carried over.
- **[data]** `1113PMDC1_IgAN_slices_htj2k_q0.002.sdata.zarr` (six sections; the numbers in Context).
