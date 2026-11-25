# WebGPU views: what exists and what to start from

Design note, 2026-11-21. Not an ADR.

**Recommendation.** Start new views on deck.gl (WebGPU), with layers that bind the compute's own
buffers. Keep three.js for raymarched surfaces and volumes. Use raw WebGPU only for a full-screen
pass with no camera.

| Kind of view | Base | Why |
|---|---|---|
| 2D spatial (tissue, fields, points) | deck, `SpatialDeckView` | Already the stack; sd.js images underneath. |
| 3D particle scenes | deck, `OrbitView` + `@intraspatial/deck` layers | One device with the compute, same layers as the 2D map. See the spike below. |
| Data beside simulation | one Deck, several `views` | The only way to link a map and a 3D scene on one device today. |
| Iso-surface of a sampled 3D field (metaballs, 3D density) | deck, `IsoSurfaceLayer` | Ray-marched in a deck layer over a voxel buffer; shares depth with the other layers. |
| Raymarched analytic SDF, image volumes | three.js `WebGPURenderer` + TSL | Needs `depthNode`, lights and shadows, 3D textures; deck has none of these. |
| Full-screen field, no camera | raw WebGPU | `hspf/fieldRenderer.ts` is 200 lines; a framework adds nothing. |

## What exists

| Stack | View | Draws | Device | Buffers cross by | Reusable |
|---|---|---|---|---|---|
| three + TSL | dancer (`docs-site/.../dancer/renderer.ts`) | Instanced swept creatures; every vertex from `vertexIndex` + storage buffers | three's, or injected (`WebGPURenderer({ device })`); sim built on `renderer.backend.device` | Sim writes three-owned `StorageBufferAttribute`s, fetched with `backend.get(attr).buffer` (exist only after a first render) | `gpuStateBuffers`, `gpuTrailBuffer`, the injected-device grid |
| three + TSL | `raymarchMain.ts` | BSP mesh + sphere-traced SDF on proxy boxes, one depth buffer | three's own | None (WGSL injected with `wgslFn`) | `shade()` node; `depthNode` compositing |
| three | `geometryMain.ts` | CPU-meshed implicit | three's own | None (CPU arrays) | Little |
| three + TSL | `spatialVolumeMain.ts`, `datasource/naiveVolumeRenderer.ts` | OME-Zarr volume bricks | three's; passed to `handle.volume({ device })` | Brick `GPUTexture` wrapped as `THREE.ExternalTexture` | `externalTexture3D` |
| three | `datasource/dualView.ts` | Main camera + inset, two scissor viewports | three's; `adoptDevice(r.backend.device)` | Bricks computed on three's device | The two-pass viewport code |
| three | `SpatialScene.tsx`, `R3fSpike.tsx` | Multi-image scene; R3F with two drei `<View>`s | three's own | None | `WebGpuScissorFlip` (pinned to r185) |
| deck 9.4 | `deck/SpatialDeckView.tsx` | sd.js image + page layers, 2D | deck creates; `useSharedDevice` adopts it | n/a | The view, `useSharedDevice`, `useWorldPick` |
| deck 9.4 | `modes/GramModesLayer.ts` | Mode map from resident rasters | deck's, adopted | Wraps our `GPUBuffer`: `device.createBuffer({ handle, byteLength, usage })` | The zero-copy pattern |
| deck 9.4 | `wand/BoundaryLayer.ts` | Region outline | deck's, adopted | Same | Same |
| deck 9.4 | `packages/deck` `DensityLayer` | KDE fill and contours | deck's, passed to the kernels (no adoption) | Layer owns the field buffer the splat writes | Published layer; per-device kernel state |
| deck 9.4 | `packages/deck` `ParticleLayer`, `BondLayer` (new) | Sphere impostors and bonds, 2D or 3D | deck's, passed to the simulation | Bind the simulation's position, attribute and slot buffers | See the spike below |
| deck 9.4 | `packages/deck` `TubeLayer`, `SurfaceLayer`, `TraceLayer` (new) | A chain as an extruded tube, or with a `profile` (`cartoonProfile`) as a protein cartoon: ribbons and arrows, eased between profiles; a mesh over particles; the sheet a chain sweeps through time. Used by `foldsynth.html` | deck's, passed to the simulation | Bind the position buffer and a `PositionHistory` ring filled by buffer copies | Any ordered particles; any index list |
| deck 9.4 | `packages/deck` `IsoSurfaceLayer` (new) | The surface where a 3D field crosses a level (metaballs, 3D density), ray-marched per pixel with depth written. Used by `foldsynth.html` ("Blobs") and `density3d.html` (measured transcripts, one field per gene) | deck's, passed to the kernel | Binds a voxel buffer that `splatVolume` (`src/gpu/spatial`) fills from the simulation's positions | Any f32 voxel buffer. A mask would be a second buffer read in the march; a mesh for export is not made |
| raw | `docs-site/src/lib/hornMutator.ts` | 3×3 grid of horns, viewport + scissor per cell | Its own `requestDevice` | Compute output bound as vertex buffers | `syncSize` |
| raw | `hspf/fieldRenderer.ts` | Full-screen field with palette and contours | Library's `getDevice()` | Binds `sim.currentStateBuffer()` | The whole file as a template |

Every stack already draws from the compute's device with no readback. They differ in who creates
the device and who owns the buffers.

| | Who creates the device | Who owns shared buffers |
|---|---|---|
| three | three, or the caller | three (attributes) or us (external textures) |
| deck | deck only | us; luma wraps a handle |
| raw | us | us |

## Constraints already hit

| Constraint | Where | Consequence |
|---|---|---|
| luma 9.4 cannot attach a `GPUDevice` (`WebGPUAdapter.attach` throws "not implemented") | `packages/deck` `sharedDeviceAdapter`, `deck/useGlobalDevice.ts` | Worked around: an adapter whose `create()` wraps a device it is given. The page makes one device, the library adopts it, and every Deck draws on it. `springs.html` (3D), `foldsynth.html` and `density3d.html` use this; the `useSharedDevice` pages (`tissuesim`, cell and transcript modes) have not been moved. |
| `adoptDevice` takes one device per page | `src/gpu/device.ts` | One Deck per page on that path, and no GPU work before deck is up. |
| Deck never destroys a finalised Deck's device | `device.ts` (`perDevice`) | A remount brings a second device. Kernels on `deviceScoped` and layers that pass their device explicitly survive it; `adoptDevice` pages reload. |
| A layer must not read a buffer the next compute overwrites | `GramModesLayer` | Snapshot per result, or (simulation) ask for the buffer at draw time. |
| deck WebGPU: no picking, no extensions, few stock layers ported | `packaging-and-consumers.md` | Custom WGSL layers only; pointer → world by hand (`useWorldPick`). |
| three viewport/scissor origin is top-left; drei passes bottom-left | `dualView.ts`, `R3fSpike.tsx` | Monkey-patch tied to r185. |
| three: mid-frame depth clears unreliable on WebGPU | `multiImageScene.ts` | Overlays drawn depth-test-off in add order. |
| three buffers exist only after a first render | dancer | Force a `renderAsync` before building the simulation. |
| `ExternalTexture` binds as 2D unless `isData3DTexture` is set; three never frees its source | `naiveVolumeRenderer.ts` | Silent black volume; manual ownership. |
| No view handles `device.lost` | all | Open. |

`hspf/fieldRenderer.ts` caches its pipeline in one module-level promise, not per device. It is safe
only while the page has one device.

## Why deck first

1. The 2D pages are on it, and a simulation over spatial data must share their view and device.
2. Its buffer model suits a simulation: we own the buffers and hand deck a handle. On three the
   simulation has to write into buffers three allocated.
3. Layers written against it are the ones `@intraspatial/deck` publishes for MDV.

What it costs: no lights, shadows or post-processing, and anything beyond a flat quad is our own
WGSL. That is why raymarching stays on three.

## Relation to the proposed ADRs and packages

| Document | Relation |
|---|---|
| ADR-0009 (rendering as ops, proposed) | Unchanged. A `Raymarch` op outputs colour + depth; deck could present that as a textured layer, which would remove the last reason for three in volume views. Not attempted here. |
| ADR-0014 (surface program, proposed) | A particle layer is a small case of it: `surface` is the impostor (position, normal, valid), `material` is the palette lookup. The layer's shader keeps those two parts separate so it can move to that contract. |
| `viewer-deck` (planned) | `@intraspatial/deck` is already this package in practice. Suggest dropping the separate name. |
| `viewer-three` (planned) | Still wanted for volumes and SDF. A shared framebuffer between deck and three is blocked on luma attaching a device; until then they are separate canvases. |

## The deck spike on the spring model

Built: `ParticleLayer` and `BondLayer` in `packages/deck/src/` (inputs are GPU buffers, strides and
a palette); the "Branching in 3D" setup of `playground/springs.html` on `OrbitView`; and
`playground/tissuesim.html`, one Deck with a 2D map of measured cells beside a 3D view of a
simulation seeded from them.

| Question | Answer | How checked |
|---|---|---|
| Does `OrbitView` work on deck 9.4's WebGPU path with depth-correct custom layers? | Yes. deck's WebGPU pass has a depth buffer (`depthWriteEnabled`, `less-equal` by default) and a layer may write `frag_depth`. | Spheres intersect correctly while orbiting, in Chrome (Dawn/Metal). |
| Can one Deck host a 2D spatial view and a 3D orbit view? | Yes: two `views`, per-view `initialViewState`, `layerFilter` by id prefix, each with its own controller. Click → world coordinate works per view. | `tissuesim.html` on a Xenium store (162,254 cells). |
| Frame cost | 1.1 ms of GPU work to draw 10,716 particles (the 3D setup's size; cells drawn, gel hidden); 1.8 ms with the gel's beads and 171,456 bond-slot line instances. Display stayed at 120 frames/s. | `onSubmittedWorkDone` after each frame, simulation paused, pane visible. |
| Per-frame readback | None for drawing. The spheroid runner still downloads every 10 steps, because cells divide on the host. | `hostCurrent: false`; test counts the syncs. |

With the simulation running at 4 steps per frame the same readout shows about 33 ms: that is the
step kernels queued behind the frame, not drawing.

| deck could not do, that the dancer's three.js path does | Matters for |
|---|---|
| Take a device we made, so several canvases share one (the dancer's breeding strip) | Grids of small views. One Deck with many `views` covers it only inside one canvas. |
| Lit materials, shadows, post-processing from a scene graph | Presentation-quality 3D. The impostor's lighting is three lines of WGSL. |
| Thick lines and antialiased edges without extra work | Bonds are 1-pixel lines; sphere edges are hard. |

Nothing fundamental blocked. Three things to know:

| Wrinkle | Handling |
|---|---|
| A layer that writes `frag_depth` must give its `Model` a depth format at construction; deck supplies one only at first draw, after a failed pipeline. | `ParticleLayer` reads it from the canvas framebuffer. |
| deck redraws only when props change, and cannot see a buffer changing. | `_animate` on the Deck; layers take the position buffer as a function and ask at every draw. |
| Sphere depth is exact in an orthographic view and approximate in perspective (the silhouette is the centre plane's). | Accepted for particles; a ray-sphere test would fix it. |

Two device paths exist. `useGlobalDevice` makes one device, the library adopts it and every Deck
on the page draws on it through `sharedDeviceAdapter`; the springs, FoldSynth and 3D density pages use it, and
new views should. `useSharedDevice` lets deck create the device and adopts that; `tissuesim` and the
older 2D pages still use it.

Not checked: browsers other than Chrome; the made-up-cells fallback of `tissuesim` with no store
(it is the initial state while a store opens, but was not exercised with the store absent);
`device.lost`; more than about 11,000 particles.
