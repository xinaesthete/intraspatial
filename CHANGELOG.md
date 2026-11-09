# intraspatial

## 0.2.0

### Minor Changes

- b9cf5ae: The cross-PCF can now be measured on an ROI of any shape, not only a rectangle. Pass the region as a `domain`. Without one, nothing changes.
  
  - New `Domain` API in `intraspatial/spatial`:
    - `rectDomain(bbox)` and `maskDomain(bbox, width, height, data)` build a domain. A mask is a raster with row 0 at the top and non-zero meaning inside. `maskDomain` rejects a raster whose length doesn't match its shape.
    - `polygonDomain(rings)` takes a drawn shape, such as an MDV gate or a SpatialData `shapes` annotation, as GeoJSON-style rings. Nested rings are holes (even–odd), and orientation and a repeated closing vertex don't matter. The edge correction clips the polygon in closed form, so unlike a mask it has no staircase error.
    - `domainArea`, `domainContains`, `domainAnnulusAreasInto` and `inverseAnnulusAreas` work on every kind.
  - `crossPCF`, `crossPCFMatrixBinned` and `crossPCFBinMembership` take an optional `domain`. It sets the area used for ρ_B (unless `roiArea` is given), the edge-correction clip when `edgeCorrected` is set, and which cells count.
    - Cells outside the domain are dropped from both populations, and results report how many in `outside`.
    - The clipped area on a mask is exact for the mask's pixels. The remaining error comes from the pixel staircase along the boundary, and costs at most 0.1% on `g` at 1–2 µm pixels. The bounding-box clip reads 1–5% low on a tilted or round ROI.
  - Masks can come from the data:
    - `maskWhere(field, keep, { derivedFrom })` thresholds any `ScalarField`-convention raster (a density field, a mode projection) into a mask.
    - `similarityField` and `pixelZ` compute the Gram wand's "places like this one" distance on the CPU, so it can be thresholded into a mask the same way.
    - `whiteningMatrix` and `standardise` now live in `intraspatial/spatial` too; `gramTerrain` still exports them.
  - Masks built from the data carry `derivedFrom`, and `crossPCFMatrixBinned` returns `confounded`, which flags pairs whose two types both built the mask. Measured on independently clustered types, such pairs read g 14–23% low against a truth of 1. A mask that never saw the neighbour type B leaves g_AB unbiased. One that left out only the anchor type A still read 2–4% high, and `confounded` does not flag that case.
  - `crossPCFGpu` and `crossPCFBootstrap` throw when given a `domain`. They don't support it yet, and before they would have ignored it silently.

### Patch Changes

- b9cf5ae: `gramMatrixGpu` now throws when its points need a storage binding larger than the device allows, instead of returning a blank Gram. Expression channels each carry every cell, so a whole Xenium table (397k cells × 32 genes, 152 MB) passes WebGPU's default 128 MiB. The error names the size and the limit.
  
  It checks before packing the points, so the error no longer costs the packed copy first. Its pooled buffers also stop doubling at the device's `maxBufferSize`, where a larger request would have given an invalid buffer and a blank Gram.
- 3a20a06: Points tiles can now come from a multiscale element's sampled levels:
  
  - `PointsTile` gains an optional `sampleFraction`, the share of the element's points a sampled tile was read from. Absent means every point, so existing tiles are unchanged.
  - `pointsGridAtLevel(bounds, totalRows, level)` builds the grid at one quadtree level, so a loader can cut bigger tiles for a bigger window. `pointsGrid` now delegates to it.
- 15635f9: `probeChannels` (`intraspatial/gpu/spatial`) reads between pixels and from half-precision maps:
  
  - Fractional `col, row` are interpolated bilinearly, pixel centres at whole numbers, instead of rounded to the nearest pixel.
  - f16 rasters are read; they were previously refused.
  - A fourth argument, `rasters`, reads a copy of the result's rasters in place of `res.resident.buffer`, which the next compute overwrites.
  - `standardise` accepts any `ArrayLike<number>` for its mean and sd.

## 0.1.1

### Patch Changes

- d78900d: Every GPU object `intraspatial` creates now carries a name, so a WebGPU validation error can be traced to the kernel that caused it instead of reporting `[Buffer (unlabeled)]`.
  
  - Raw WebGPU objects carry a `<kernel>:<role>` label (`gramTerrain:depth`, `dancerGpu:step`, `scan:u32:addOffsets`, …). This covers buffers, textures, samplers, shader modules, layouts, pipelines, bind groups, command encoders and passes.
  - TypeGPU buffers and bind-group layouts are `$name`d the same way (`knn:outIdx`, `knn:indexed`), and TypeGPU bind groups take their layout's name. Previously they showed bare variable names such as `params` or `layout`.
  - A device created by `getDevice()`, and its queue, are labelled `intraspatial`.
  - `rawBindGroup` takes an optional trailing `label`.
- 5069626: The density-splat and field-max kernels label their GPU buffers and textures (`splatDensity:*`, `splatDensityAtomic:*`, `fieldMax:uniforms`), so validation errors name them.
