# @intraspatial/deck

## 0.2.0

### Minor Changes

- 258176d: Chain views on deck's WebGPU path.
  
  - `TubeLayer`: an ordered chain of particles as an extruded tube, or with a `profile` as a protein-style cartoon (ribbons and arrows), eased between profiles. `cartoonProfile` builds the profile from a list of `ChainStructure` spans; `tubeProfile` exports the profile packing.
  - `SurfaceLayer`: a lit mesh over particles from an index list.
  - `TraceLayer`: the paths particles took, or the sheet a chain sweeps through time, from a `PositionHistory` ring of position snapshots filled by device copies (`traceSlot` finds a snapshot by age).
  - `bondFaces` (spring model): the triangles closed by bonds, for `SurfaceLayer`.
- 647a3e0: Particle views on deck's WebGPU path, drawing straight from a simulation's buffers.
  
  - `ParticleLayer`: spheres (impostors that write depth) from a buffer of positions, coloured by category through a palette. Species and radius come from a per-particle attribute buffer or from separate `radii`; a buffer may be given as a function, asked for at every draw, so a simulation's ping-pong buffers can be bound.
  - `BondLayer`: a line for each bond in a slot table (`K` `vec4u` slots per particle), coloured by bond kind.
  - `categoricalColors`, `hueRamp`, `packPalette`, `syncPalette`: palettes as the layers read them. `CATEGORY_WGSL` and `BOND_WGSL` are the shader pieces they share.
  - `SpringsGpu` exposes `attrs` and `slots` for a renderer to bind (`ATTR_FLOATS`, `ATTR_SPECIES`, `ATTR_RADIUS` give the layout), and `uploadTopology` sends a host edit of the bond table without touching positions. `createSpheroidGpuRunner` takes `hostCurrent: false` to skip the download after each advance, and returns its `gpu` and a `synced` count.
- cd8c95d: `PivotOrbitController`: deck's orbit controller, turning about a point the page picks instead of the view's target.
  
  - `getPivot(pointer, viewport)` returns the world point a drag turns about (usually the thing under the pointer); null falls back to deck's own behaviour. `getAnchor` does the same for zoom and pan, so the point under the pointer stays under it at its own depth.
  - `nearestPoint(positions, pointer, viewport, options)` finds the point nearest the pointer on screen in a packed position array, nearest the camera on ties.
  - `anchorUnder(positions, pointer, viewport, options)` gives the world point under the pointer at the nearest point's depth, which is what `getAnchor` should return. Points out of view (behind the camera, or past the near or far plane) are never picked.
- 21a7133: A 3D density field from points, and a deck layer that draws its surfaces.
  
  - `intraspatial/gpu/spatial`: `createSplatVolume` / `splatVolumeResident` fill a voxel buffer on the GPU with the kernel density of a point buffer (`(1 - d²/R²)²` within radius R). Small clouds are summed directly; from 2,048 points the points are first bucketed into the 3D grid index and each voxel reads only the 27 cells around it, with the same result. An optional `weights` buffer gives each point its own weight (a cell's expression of a gene, or 0 to leave a point out).
  - The 3D grid index accepts point buffers with more than three floats per point (`stride >= 3`).
  - `@intraspatial/deck`: `IsoSurfaceLayer` draws the surface where a voxel buffer crosses a level, by marching a ray per pixel. It writes depth, so it intersects other layers correctly. `sharedDeviceAdapter` lets Deck draw on a `GPUDevice` that already exists (luma 9.4 cannot attach one), so the field can be computed and drawn on one device. `@luma.gl/webgpu` is now a peer dependency.
- b8b6d9e: `TcmLayer`: a topographical correlation map between two cell types (`weightsA`, `weightsB` over one point set), drawn as a diverging field on deck's WebGPU device. Needs `float32-blendable`; without it the layer reports an error and draws nothing.

### Patch Changes

- Updated dependencies [129f106]
- Updated dependencies [258176d]
- Updated dependencies [129f106]
- Updated dependencies [f3154a8]
- Updated dependencies [2238112]
- Updated dependencies [647a3e0]
- Updated dependencies [b56e125]
- Updated dependencies [dd97e2b]
- Updated dependencies [3731ac1]
- Updated dependencies [1d3cf02]
- Updated dependencies [21a7133]
- Updated dependencies [7e7e10b]
- Updated dependencies [b8b6d9e]
- Updated dependencies [b5bf639]
  - intraspatial@0.4.0

## 0.1.3

### Patch Changes

- Updated dependencies [35fd282]
  - intraspatial@0.3.0

## 0.1.2

### Patch Changes

- Updated dependencies [b9cf5ae]
- Updated dependencies [b9cf5ae]
- Updated dependencies [3a20a06]
- Updated dependencies [15635f9]
  - intraspatial@0.2.0

## 0.1.1

### Patch Changes

- 5069626: `DensityLayer`: replacing or removing a layer while its splat is in flight no longer destroys buffers that splat still uses ("[Buffer] used in submit while destroyed"). Buffers a splat holds are retired at finalize and freed when it settles, and a finalized layer submits no further work. Its GPU buffers are labelled `<layer id>:field`, `:max`, `:points` and `:ramp`, so a device error names the layer.
- Updated dependencies [d78900d]
- Updated dependencies [5069626]
  - intraspatial@0.1.1
