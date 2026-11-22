---
"intraspatial": minor
"@intraspatial/deck": minor
---

A 3D density field from points, and a deck layer that draws its surfaces.

- `intraspatial/gpu/spatial`: `createSplatVolume` / `splatVolumeResident` fill a voxel buffer on the GPU with the kernel density of a point buffer (`(1 - d²/R²)²` within radius R). Small clouds are summed directly; from 2,048 points the points are first bucketed into the 3D grid index and each voxel reads only the 27 cells around it, with the same result. An optional `weights` buffer gives each point its own weight (a cell's expression of a gene, or 0 to leave a point out).
- The 3D grid index accepts point buffers with more than three floats per point (`stride >= 3`).
- `@intraspatial/deck`: `IsoSurfaceLayer` draws the surface where a voxel buffer crosses a level, by marching a ray per pixel. It writes depth, so it intersects other layers correctly. `sharedDeviceAdapter` lets Deck draw on a `GPUDevice` that already exists (luma 9.4 cannot attach one), so the field can be computed and drawn on one device. `@luma.gl/webgpu` is now a peer dependency.
