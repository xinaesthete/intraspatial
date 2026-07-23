# Packaging and consumer seams

Status: **design note** (2026-07-23) — promote to an ADR when the first package is actually cut.

The goal is to be able to use this work from other codebases: psychogeo/TerraCognita (terrain, GIS
raster), MDV + spatialdata.js (interactive spatial analysis), assorted experimental graphics and
architectural work. Today none of that is possible — the repo publishes nothing, and the most
reusable code is inside a private demo app.

## What the audit found (the good news first)

- **`src/` is entirely three.js-free.** Verified: zero `from "three"` outside `playground/`. The
  renderer-agnostic boundary ADR-0008 asserted actually holds.
- **`src/datasource/index.ts` already says** *"Self-contained: no dependency on the dancer/evo/sim
  code, so it can graduate."* The intent predates this note.
- **`src/gpu/spatial` has ten GPU-tested ops** and `src/geometry` a working procedural-geometry
  catalogue. There is real substance to publish, not just scaffolding.

The blockers are small and specific:

1. **`webgpu` (the Node-only Dawn addon) is a direct dependency of the root package.** A browser
   consumer would pull it. It must become `optionalDependencies` + peer, resolved through the
   existing `src/gpu/device.ts` seam (which already prefers `navigator.gpu`).
2. **No `exports` field, no build step, version `0.0.0`.**
3. **The reusable *viewer* layer is 6 381 lines trapped in `playground/`** — `TileRenderer`,
   `ChannelComposite`, the sd.js loaders, the volume renderers, camera work. This is precisely what
   MDV and psychogeo would consume, and it is private only because it was written inside a demo.

## The proposal: fewest packages that respect the hard boundaries

A package boundary should exist where a **dependency would otherwise be forced on consumers who do
not want it**. By that test there are three, so:

| Package | Contains | Heavy deps |
|---|---|---|
| **`@intraspatial/core`** | all of `src/`, subpath exports: `/datasource`, `/spatial`, `/geometry`, `/graph`, `/color`, `/evo` | typegpu (peer); **no** three, deck, or Dawn |
| **`@intraspatial/viewer-three`** | the three.js layer promoted out of `playground/`: tile renderer, channel composite, camera controls, `Viewport` shell | three (peer) |
| **`@intraspatial/viewer-deck`** | deck.gl interop — layers over the same datasource/ops, sharing one `GPUDevice` with `viewer-three` | deck.gl, luma.gl (peer) |

Not six packages, one per `src/` subdirectory: every extra package is release overhead, and
subpath exports give the same import ergonomics without it. Split `core` further only when a real
consumer's dependency footprint demands it.

`@intraspatial` and the bare `intraspatial` are both free on npm (checked 2026-07-23).

### A likely fourth: the codec

`rust/htj2k-core` + the DWT kernels are the repo's origin and have no relationship to spatial
analysis. `@intraspatial/htj2k` — or keeping it under its own name entirely, as `openjph-wasm`
already is — is a reasonable split, but it is not on the critical path and can wait until someone
wants the codec without the toolbox.

## The deck.gl seam

deck.gl is expected to be central to the data-vis stack, and it has deliberately not been used here
yet: it is a fiddly dependency tree and less pleasant to prototype against than three.js. Both of
those are arguments for the seam rather than against the dependency — consumers who want deck opt in,
and nobody else pays.

**The target is shared-framebuffer integration** — one `GPUDevice`, deck layers and three geometry
interleaved with a common depth buffer — not two stacked canvases with synchronised cameras. Facts
checked 2026-07-23:

**Both device-sharing primitives already exist.**

- **luma.gl:** `luma.attachDevice(handle: WebGL2RenderingContext | GPUDevice | null, {adapters,
  ...deviceProps})`, documented explicitly as the way to *"interleave rendering with other GPU
  libraries"*. A `GPUDevice` is an accepted handle.
- **three.js r185:** `WebGPUBackend.init()` reads `parameters.device` — *"create the device if it is
  not passed with parameters"* (`WebGPUBackend.js:209`) — and `parameters.context` for an
  externally-configured canvas context (`:336`). It also only destroys the device if it created it
  (`:2903`). So `new WebGPURenderer({ device, context })` is supported today.

So the hard part is **not** device sharing. It is deck's side of the render pass.

**The real gaps, from deck's own WebGPU docs:**

- **WebGPU support is "still a work in progress and is not production ready"**, landing layer by
  layer. Ported: `ScatterplotLayer`, `PointCloudLayer`, `PathLayer`, `LineLayer`, `IconLayer`.
  Everything else — `PolygonLayer`, `GeoJsonLayer`, `TextLayer`, `BitmapLayer`, `ArcLayer`, all
  aggregation and geo layers — is WebGL-only.
- **Picking is skipped entirely on WebGPU**, "including hover and click picking paths".
- **All `@deck.gl/extensions` are WebGL-only** (GLSL injection, GLSL-only shader modules, extra
  render/picking passes).
- **"No current base map integration path"** supporting WebGPU interleaving or transparent overlays —
  i.e. the deck-on-Mapbox interleaving pattern has no WebGPU equivalent yet. This is the specific
  thing shared-framebuffer integration needs.

**Two of those are unusually favourable for us.** The ported layer set is very nearly exactly what
spatial-data work wants: cells as `ScatterplotLayer`/`PointCloudLayer`, boundaries as `PathLayer`.
And the missing piece we would feel first — picking — is something this repo is building anyway
(the [scene note](serial-section-alignment-and-multi-viewport.md) §5's `pick()`), over data we own rather than over deck's layer state.

**The concrete asks, if we contribute upstream.** Shared-framebuffer interleaving needs deck to,
on WebGPU:

1. render into a caller-supplied colour **and depth** attachment rather than owning the canvas
   context (the WebGPU analogue of the base-map interleaving path);
2. expose `loadOp` control so a deck pass does not clear what three already drew;
3. agree a depth-texture format and sample count with the host renderer;
4. agree a projection/depth convention, so the two sides' clip-space output is comparable — this is
   the substantive one, since deck's `project` module carries its own coordinate-system machinery;
5. (eventually) WebGPU picking, or a documented way to opt out and supply picking from the host.

Items 1–3 are mechanical; 4 is a design conversation; 5 may not block us at all.

**The spike, in order:**

1. One `GPUDevice`, `luma.attachDevice` on it, three's `WebGPURenderer({ device })` on the same one —
   confirm both render at all without fighting over the canvas context.
2. Interleave in one pass: can deck be persuaded to render into a texture view we own, with
   `loadOp: 'load'` and our depth attachment? This is where the gap is expected, and where a patch
   would go.
3. Depth agreement: put a `ScatterplotLayer` and a three mesh at known depths and check occlusion is
   correct from both sides.
4. Then the useful thing — a SpatialData element as a deck layer coloured by a `src/gpu/spatial` op
   output.

Sources for the above (checked 2026-07-23; deck's WebGPU surface moves quickly, so re-check before
relying on any of it): [deck.gl WebGPU developer guide](https://deck.gl/docs/developer-guide/webgpu),
[luma.gl `luma.attachDevice`](https://luma.gl/docs/api-reference/core/luma),
[luma.gl WebGPU adapter](https://luma.gl/docs/api-reference/webgpu); three.js
`src/renderers/webgpu/WebGPUBackend.js` at r185.

## Sequencing

1. **Make `core` publishable.** Move Dawn to optional/peer, add `exports` + a build, version it.
   Cheap, because `src/` is already clean.
2. **Promote the viewer layer out of `playground/`** into `viewer-three`. This is the single
   highest-value structural move: it is what every downstream target needs, and the playground
   becomes a consumer of the package rather than its owner — which also proves the surface is real.
3. **Bridge SpatialData elements → `src/gpu/spatial` ops**, on a real 2-D store. This is the thing
   that makes the whole enterprise believable, and the audit says it is a bridge, not a build.
4. **The deck interleaving spike.**

Steps 1–2 of the deck spike (shared device, then interleaving) are cheap and independent of the rest,
and they are the ones that would surface a gap worth raising upstream. Given an imminent deck.gl
developer summit — where this work is to be presented alongside sd.js/MDV — there is a good argument
for running them *early and out of order*, purely to arrive with a measured result rather than a
question. A one-day answer to "does three-on-WebGPU interleave with deck-on-WebGPU, and exactly where
does it stop?" is worth more in that room than a finished package.

## Open questions

- **Release mechanics.** Changesets vs manual versioning; whether the packages version in lockstep.
  Unresolved and deliberately deferred until there are two packages to co-ordinate.
- **Build tooling.** The repo currently ships no built artefacts at all. TS 7 native emit vs a
  bundler (tsdown/unbuild) is an open call; the deciding factor is likely the `"use gpu"` TGSL
  transform, which today runs through `unplugin-typegpu` in the *consumer's* Vite config — so either
  the package ships pre-transformed kernels or it documents the plugin as a requirement. **This is
  the least understood part of publishing and should be resolved early**, because it constrains
  everything else.
- **Name.** `IntraSpatial`. One honest caveat: "intra-" conventionally reads as *within a single*
  discipline, which is close to the opposite of "usable across histology, GIS, architecture, and
  graphics". The intended reading — one substrate living *inside* each of many disciplines rather
  than bridging two — is coherent, but it will need explaining more than once.
