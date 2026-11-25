# Cameras: psychogeo's against deck.gl's

Design note, 2026-11-21. Not an ADR. From a read of `~/code/gis/psychogeo/src/camera/` and deck.gl
9.4.0's controllers; nothing was run. Line references are to those two sources.

**Finding.** psychogeo's camera is better at the one thing that matters most in a 3D scene: it
rotates and zooms about the point under the cursor. deck's `OrbitController` rotates only about a
fixed target. deck is better at everything around that (keyboard, transitions, inertia on all
gestures, orthographic, several views). The useful move is to bring psychogeo's pivot and tilt clamp
to deck as an `OrbitController` option, not to carry psychogeo's controller across.

| | psychogeo `MapCameraControls` | deck `OrbitController` | deck `MapController` / `TerrainController` |
|---|---|---|---|
| State | target, bearing, pitch, distance | target, rotationX, rotationOrbit, zoom | lon/lat, zoom, bearing, pitch |
| Rotate about | the picked point under the cursor (`MapCameraControls.ts:636`) | the fixed target only (`orbit-controller.js:86`) | `rotationPivot`: centre, 2D or 3D (`map-controller.js:507`) |
| Zoom | dolly toward the picked point (`:903`) | scale, keeping the cursor's point fixed (`:163`) | mercator zoom |
| Tilt limit | exact: never past straight down, never under the pivot's ground (`orbitClamp.ts:26`) | clamp on rotationX | clamp on pitch |
| Depth under cursor | injected callback; app does a 1×1 float readback | `unproject3D` → `pickPosition`, **null on WebGPU** (`deck.js:545`) | same; Terrain also follows centre altitude |
| Inertia | pan only | pan, rotate, pinch (off by default) | same |
| Touch | pan, pitch, pinch, twist, all about the picked midpoint | pinch, rotate, multi-finger pan | same |
| Keyboard, transitions, fly-to | none | yes | yes |
| Orthographic, several views | no | yes | yes |
| Terrain following | no | no | yes, smoothed (`terrain-controller.js:84`) |
| Up axis, roll | Z up, no roll | Y or Z, no roll | Z up, no roll |
| Size, tests | about 1,450 lines; only the clamp is tested | – | – |

## What is ours and worth having

| Piece | Where | Portable? |
|---|---|---|
| Rotate about the cursor's point | `MapCameraControls.ts:636-706` | Yes: deck already has it for maps; `OrbitState` lacks it. |
| Tilt clamp that holds with an off-centre pivot | `orbitClamp.ts:26-81`, 9 tests | Yes: about 40 lines of vector algebra once off `THREE.Vector3`. |
| Two-finger pitch and twist about the pinch midpoint | `:801-851` | Possible; deck's gesture handling differs. |
| True dolly | `:903-925` | No: `OrbitViewport` bakes zoom into the view matrix. Same picture for a perspective orbit, so not needed. |

psychogeo's weaknesses to leave behind: the view state drops the target's height, tuning lives in
module-level globals, near and far are fixed, and the controller is bound to a mutable three
`PerspectiveCamera`.

## Upstream contribution

A `rotationPivot` option for `OrbitController`, mirroring `MapController`'s. `rotateStart` keeps
the unprojected cursor point; `rotate` sets the new angles and then moves `target` so that point
stays on its pixel, which `OrbitViewport.panByPosition` can already solve. Roughly 30 lines beside
the existing `map-controller.js:152-194`, plus the clamp as the option's tilt limit.

One blocker for us, and a second contribution: on WebGPU deck's controllers get no depth, because
`_pickPositionForController` returns null unless picking is synchronous and WebGPU picking is
async only (`deck.js:545-549`, `:587-597`). A controller option to supply the 3D point
(`unproject3D` is already an internal hook, `orbit-controller.js:49`) would fix that for everyone.

## For our 3D pages

| Page | Camera today | With the pivot controller |
|---|---|---|
| `springs` 3D, `foldsynth`, `tissuesim` | deck `OrbitView`, default controller | Drop-in: a controller subclass. Depth from our own buffers (nearest particle to the cursor ray) since deck cannot pick on WebGPU. |
| `raymarch`, `geometry`, `spatialvolume`, `datasource`, `spatialscene`, `r3fspike` | three `OrbitControls` | Needs the same maths behind three's camera: a second binding, or leave as is. |
| `cellmodes-legacy` terrain | hand-rolled orbit | Goes away when the terrain moves to deck. |

Suggested order: (1) an `OrbitController` subclass in `@intraspatial/deck` with cursor pivot and
the clamp, fed by a caller-supplied depth function; (2) use it on the three deck 3D pages; (3) open
the upstream PR from it once it has been used; (4) decide about the three.js pages after that.

Not checked: deck's `FlyToInterpolator` with `OrbitView`; `TerrainExtension` and `TerrainLayer`;
whether psychogeo's tiles use a local origin; sign conventions for mapping bearing and pitch to
`rotationOrbit` and `rotationX`.
