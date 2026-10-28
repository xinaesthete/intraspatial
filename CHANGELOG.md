# intraspatial

## 0.1.2

### Patch Changes

- 3a20a06: Points tiles can now come from a multiscale element's sampled levels:
  
  - `PointsTile` gains an optional `sampleFraction`, the share of the element's points a sampled tile was read from. Absent means every point, so existing tiles are unchanged.
  - `pointsGridAtLevel(bounds, totalRows, level)` builds the grid at one quadtree level, so a loader can cut bigger tiles for a bigger window. `pointsGrid` now delegates to it.

## 0.1.1

### Patch Changes

- d78900d: Every GPU object `intraspatial` creates now carries a name, so a WebGPU validation error can be traced to the kernel that caused it instead of reporting `[Buffer (unlabeled)]`.
  
  - Raw WebGPU objects carry a `<kernel>:<role>` label (`gramTerrain:depth`, `dancerGpu:step`, `scan:u32:addOffsets`, …). This covers buffers, textures, samplers, shader modules, layouts, pipelines, bind groups, command encoders and passes.
  - TypeGPU buffers and bind-group layouts are `$name`d the same way (`knn:outIdx`, `knn:indexed`), and TypeGPU bind groups take their layout's name. Previously they showed bare variable names such as `params` or `layout`.
  - A device created by `getDevice()`, and its queue, are labelled `intraspatial`.
  - `rawBindGroup` takes an optional trailing `label`.
- 5069626: The density-splat and field-max kernels label their GPU buffers and textures (`splatDensity:*`, `splatDensityAtomic:*`, `fieldMax:uniforms`), so validation errors name them.
