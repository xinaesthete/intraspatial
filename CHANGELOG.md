# intraspatial

## 0.4.0

### Minor Changes

- 129f106: Gene modules for whole-transcriptome points, and sampled points weighted to stand for what was left out.
  
  - `geneEmbedding(xs, ys, codes, nFeatures, { binSize, … })` in `intraspatial/spatial` is a PCA gene embedding of a points sample. It bins the sample, normalises and logs each bin, standardises each gene, and runs a sparse randomized PCA. It keeps only the components above the Marchenko–Pastur noise edge.
  - `geneModules(embedding, k)` partitions the spatially structured genes into `k` disjoint modules with e-means. `binSizeFor` sizes bins for ~750 molecules each.
  - `emeans` and `lloyd` take `seeding: "kmeans++"`. The default is still random seeding, as in the reference implementation.
  - `PointsTile.featureFractions` records per-gene sample fractions from a subset read. `pointFraction(tile, code)` reads a molecule's fraction.
  - **Behaviour change:** `transcriptChannels` now weights each molecule by 1 / its sample fraction (`sampleFraction`, or `featureFractions` for its gene). Channel densities from sampled tiles used to read low by that fraction. They are now in molecules per unit area at any level. `g` and the correlation modes of uniformly sampled tiles are unchanged, because a per-channel scale cancels in both.
- 258176d: Chain views on deck's WebGPU path.
  
  - `TubeLayer`: an ordered chain of particles as an extruded tube, or with a `profile` as a protein-style cartoon (ribbons and arrows), eased between profiles. `cartoonProfile` builds the profile from a list of `ChainStructure` spans; `tubeProfile` exports the profile packing.
  - `SurfaceLayer`: a lit mesh over particles from an index list.
  - `TraceLayer`: the paths particles took, or the sheet a chain sweeps through time, from a `PositionHistory` ring of position snapshots filled by device copies (`traceSlot` finds a snapshot by age).
  - `bondFaces` (spring model): the triangles closed by bonds, for `SurfaceLayer`.
- 129f106: k-means by e-means (Bessa et al. 2026, arXiv:2611.09501) in `intraspatial/spatial`. This is the CPU golden for a later GPU port.
  
  - `emeans(data, n, d, k, opts)` runs mini-batch k-means with exponential-moving-average centroid updates. The batch size and step come from the published scaling laws, so the only inputs are Lloyd's. It also has a cosine step schedule, dead-cluster revival by splitting, and a finalisation epoch. It is deterministic per `seed`. It returns centroids, labels, per-cluster counts, the MSE and the `schedule` it ran.
  - `lloyd(data, n, d, k, opts)` is the baseline. `emeansSchedule`, `emeansBatch` and `emeansAlpha0` expose the scaling laws. `assignNearest` is the nearest-centroid pass.
  - Revival is off by default for d ≤ 3, where the authors found that it degrades results.
- f3154a8: e-means can now fit a centroid table once and assign points to it later, with per-point weights and masks (`intraspatial/spatial`).
  
  - `emeansFit(data, n, d, k, opts)` returns a frozen `CentroidTable` (`k`, `d`, `centroids`) with the weighted cluster `mass`, the `schedule` and `splits`. `assignCentroids(table, data, n, opts)` labels any set of points against it, such as one viewport tile at a time. `emeans` is now the two together, and its result is itself a table.
  - `weights` (finite, ≥ 0) and `mask` (a `Domain` plus each point's position) are accepted by `emeansFit`, `emeans`, `lloyd` and `assignCentroids`. A point of weight 0 or outside the mask is left out of the fit exactly as if removed. It is never a seed, and assignment labels it −1 and leaves it out of the MSE. Weights are rescaled to mean 1, and the scaling laws take n = the number of points kept.
  - Results gain `mass` (weight per cluster) and the MSE is weight-averaged. Both are unchanged for unweighted input. `lloyd` now re-seeds only clusters of weight exactly 0.
  - `channelBins(tile, bins, channels)` counts a points tile's channel molecules into a finer quadtree level of bins, weighting each molecule by 1/`pointFraction`. `binComposition` turns those counts into Hellinger descriptors and molecule weights, for clustering bins into niches.
  - `pooledBins(own, neighbours, bins, radius)` sums each bin's counts over the `(2·radius + 1)²` bins around it, reaching into the neighbouring tiles' bins. Pooling tile by tile equals pooling the whole element at once.
- 2238112: Region statistics for the gram modes and the wand (R2 of the region/raster statistics plan).
  
  - `floodAreaByTolerance(d, width, height, seed)` in `intraspatial/spatial` gives the wand patch's area at every tolerance from one bottleneck (minimax) flood. The area is exactly the pixel count `connectedRegion` fills at that tolerance. `floodAreaAt` reads the area at one tolerance, `floodJoinTolerance` gives the least tolerance at which a pixel joins, `floodLeaks` marks single-step basin joins, and `bottleneckCosts` and `wandSeedPixel` are the pieces it is built from.
  - `modeHistograms(res, params, rasters)` in `intraspatial/gpu/spatial` histograms the three painted modes over a Gram window with the `histogram` op, and returns per-mode 1st/99th-percentile half-ranges. `modeProjection` reads the mode planes back.
  - `modeParams` / `paintGramModes` take `ranges`, a per-mode half-range that replaces `saturate·σ_k`.
  - `probeChannels` and `wandDistance` take an optional `device`, the one the Gram result lives on, as `modeHistograms` does. Their pipelines and buffers are now kept per device. `probeChannels` serialises its own calls on each device, so concurrent callers no longer share its output buffer.
- 647a3e0: Particle views on deck's WebGPU path, drawing straight from a simulation's buffers.
  
  - `ParticleLayer`: spheres (impostors that write depth) from a buffer of positions, coloured by category through a palette. Species and radius come from a per-particle attribute buffer or from separate `radii`; a buffer may be given as a function, asked for at every draw, so a simulation's ping-pong buffers can be bound.
  - `BondLayer`: a line for each bond in a slot table (`K` `vec4u` slots per particle), coloured by bond kind.
  - `categoricalColors`, `hueRamp`, `packPalette`, `syncPalette`: palettes as the layers read them. `CATEGORY_WGSL` and `BOND_WGSL` are the shader pieces they share.
  - `SpringsGpu` exposes `attrs` and `slots` for a renderer to bind (`ATTR_FLOATS`, `ATTR_SPECIES`, `ATTR_RADIUS` give the layout), and `uploadTopology` sends a host edit of the bond table without touching positions. `createSpheroidGpuRunner` takes `hostCurrent: false` to skip the download after each advance, and returns its `gpu` and a `synced` count.
- b56e125: Patch grid (R4): `patchGrid` in `intraspatial/spatial` lays square patches over a support mask with SIMBA's per-component layout and keeps those passing minimum-support rules (a mask fraction, a point count), matching SIMBA's patch tables exactly. Masks are row runs, so a 1 µm/px canvas can be streamed. `patchPolygons` hands the kept patches to zonal statistics (`zonalStatsTiled`, polygon k = patch k); `patchLayers` and `patchLabelImage` give non-overlapping u32 label rasters in the same convention. `pnpm patchgrid:parity` checks the grid through zonal statistics against SIMBA's per-patch features. A new `patchSupport` graph op (`patchSupportGpu` / `patchSupportResident` in `intraspatial/gpu/spatial`) counts a resident mask's support pixels per rectangle with one prefix scan.
- dd97e2b: PCF params: typed columns, optional `bbox`, and `edgeCorrected` on by default for a domain.
  
  - `LabelledCells` takes `Coords` (so `Float32Array`/`Float64Array`) for `xs`/`ys` and `Uint8Array`/`Uint16Array`/`Int32Array` for `typeId`, read in place rather than copied.
  - `bbox` is optional on `PcfParams` and `PcfMatrixBinnedParams` when `domain` is given (`PcfRegion`).
  - **Behaviour change:** `edgeCorrected` now defaults to `true` when a `domain` is given, since a drawn gate's boundary is long relative to its area and the uncorrected estimate reads low. A bare `bbox` still defaults to `false`. Pass `edgeCorrected: false` to restore the old result for a domain.
  - Row filter: `crossPCFMatrixBinned` takes `include`, and `crossPCF` / `crossPCFBinMembership` / `crossPCFBootstrap` take `includeA` / `includeB`. Each is a per-cell mask (nonzero keeps the cell, index-aligned, length-checked), so callers apply their filters and regions without copying columns. Excluded cells are not counted in `outside`.
- 3731ac1: Add `normaliseRings(shape, fillRule)` and a `fillRule` option on `polygonDomain` (#58). Crossing or overlapping rings, such as a self-intersecting lasso, are rewritten as simple rings covering the same region under `"evenodd"` (a figure-eight becomes two rings touching at the crossing; a loop drawn twice is outside) or `"nonzero"` (a point is inside unless the rings' windings round it sum to zero, so same-way overlaps merge, while one ring each way over the same area cancels to outside). Without `fillRule`, `polygonDomain` stays strict. Bad rings throw the same errors as before; a region that is empty under the rule throws too.
  
  Also fixes the strict path: a hole whose every vertex touches its outer ring was sometimes read as an island; two nearly collinear edges lying end to end could be reported as crossing through rounding; and ring areas far from the origin lost precision (a unit square at 2²⁷ read as zero area).
- 21a7133: A 3D density field from points, and a deck layer that draws its surfaces.
  
  - `intraspatial/gpu/spatial`: `createSplatVolume` / `splatVolumeResident` fill a voxel buffer on the GPU with the kernel density of a point buffer (`(1 - d²/R²)²` within radius R). Small clouds are summed directly; from 2,048 points the points are first bucketed into the 3D grid index and each voxel reads only the 27 cells around it, with the same result. An optional `weights` buffer gives each point its own weight (a cell's expression of a gene, or 0 to leave a point out).
  - The 3D grid index accepts point buffers with more than three floats per point (`stride >= 3`).
  - `@intraspatial/deck`: `IsoSurfaceLayer` draws the surface where a voxel buffer crosses a level, by marching a ray per pixel. It writes depth, so it intersects other layers correctly. `sharedDeviceAdapter` lets Deck draw on a `GPUDevice` that already exists (luma 9.4 cannot attach one), so the field can be computed and drawn on one device. `@luma.gl/webgpu` is now a peer dependency.
- 7e7e10b: A spring model in `intraspatial/gpu/sim` (the `springs` namespace): a port of CSynth's spring dynamics with bond turnover, adhesion and growth.
  
  - `stepSprings` is one step on the CPU. It has CSynth's bonded springs, rods, fixes and pulls, and its pair forces (pushapart, contact and distance matrices, the Lorentzian contact model, boids), under CSynth's parameter names.
  - `setTorsion` and `backbonetorsionspringforce` add torsion springs, ported from FoldSynth: they hold the dihedral angle of four consecutive particles.
  - `applyLines`, `lineCells`, `helixLine` and `sheetLines` turn lines drawn on the particle × particle matrix into springs, after FoldSynth's vector tools.
  - `breakBonds` and `formBonds` break bonds by strain or at random and re-form them between mutual nearest neighbours. Turnover is what makes a bonded network viscoelastic.
  - Per-species `repulsion`, `adhesion`, `mobility` and `motility`, and `divideCells` for growth.
  - `createSpringsGpu` runs the step, break and form as compute kernels that match the CPU.
  - `projectedShape`, `bondReadout` and `phaseDiagram` measure outline circularity, bond stress, and behaviour over a grid of two parameters.
  - `chromatinScenario`, `chainScenario`, `spheroidScenario` and `relaxationScenario` are ready-made setups.
  - Spheroid and relaxation scenarios have `config` and `configure(change)`: stiffness, yield, healing, adhesion, motility and division all change on a running scenario without rebuilding it.
  - The spheroid runs in 3D (`dims: 3`) with `createSpheroidGpuRunner` driving it on the GPU. Leader cells (`leaderThrust`, `leaderSpacing`, `leaderGrip`, `selectLeaders`) make it branch; `matrixDisorder`, `adhesionRange`, `cellRepulsion`, `protrusion` and `yieldInCompression` are new settings.
  - `shollProfile` counts branches and `shape3d` measures reach, compactness and anisotropy.
- b8b6d9e: `TcmRenderer` (`intraspatial/gpu/spatial/tcmRender`): the TCM render on any device, owning its outputs, so several TCMs coexist. Renders are synchronous, so `destroy()` is always safe.
  
  - `renderResident` takes A and B as per-point weights over device buffers (one point set can serve both), separate Γ and mark grids, and optional caller-owned buffers for Γ and max |Γ|.
  - `renderTcm`, `computeTcmRender`, `renderKernelDensity` and `kernelDensityGpu` are unchanged apart from a new `device` option.
- b5bf639: Add zonal statistics (R3): `rasterisePolygons` and `zonalHistogram` graph ops, a tile-by-tile `ZonalAccumulator`, and `zonalStatsTiled` over a pyramid level. Polygons (overlapping ones split into layers by `polygonLayers`) or a label raster give one histogram row per region × channel, in R1's layout. From it, `zonalTable` derives per-region count, mean, median, IQR, percentiles and support/valid fractions. Per-channel support rasters and a `"positive"` valid rule cover the TNBC background conventions. An exact second pass (`zonalExactPlan` / `zonalExactQuantiles`) gives numpy-exact quantiles. R1's bin rule is now shared as `HIST_BIN_WGSL`.

### Patch Changes

- 1d3cf02: `polygonDomain` now throws on rings that cross themselves or each other (including collinear overlaps), naming the ring and edge indices. Previously `domainArea` (winding number) and `domainContains` (even–odd) disagreed on such rings, silently biasing ρ_B, edge correction and g. Touching at a vertex is still accepted. Callers passing hand-drawn lassos should repair self-crossing rings upstream before constructing the domain.

## 0.3.0

### Minor Changes

- 35fd282: Op parameters can now say what they are measured in, and how their range should be traversed. Specs that don't use the new fields behave exactly as before.
  
  - `ParamSpec` gains three optional fields:
    - `warp?: "lin" | "exp" | "amp" | "db" | number` sets how a position in [0, 1] (a slider, a controller, a bred trait) maps onto [min, max]. With `"exp"`, each step multiplies the value by the same factor. That suits scale-like params such as bandwidths, radii, times and frequencies, where 0.1–40 should not spend most of its travel above 20. A number gives a curve. The names follow SuperCollider's, so specs written there carry over.
    - `delta?: number` is how far one mutation typically moves the param, in its own units.
    - `units?: string` is what the value is measured in. A plain name such as `"Hz"` or `"s"` is a label. A relative unit such as `"cells"` or `"world"` means something only in context and can be converted by a resolver (see below). Absent means undeclared.
  - New in `intraspatial/graph`:
    - `warpMap` and `warpUnmap` convert between positions and values for any warp.
    - `UnitResolver`, `resolveUnit` and `convertUnit` provide a pluggable way to resolve relative units. Two units convert into each other when they resolve to the same base unit, so other domains can supply their own resolvers.
    - `spatialUnits` resolves `"cells"` and `"world"` against a field's placement and the dataset's µm-per-unit scale, so a UI can say "2 cells = 0.8 µm" (ADR-0018 decision 5, now amended).
    - The spatial params of `convolveSeparable`, `morphology`, `getisOrd`, the structure-tensor op, `splatDensity`, `maskedDensity` and `fuzzyAdjacency` now declare their units. Ops don't convert units yet.
  - In `intraspatial/evo`, breeding follows each param's warp and delta:
    - A param with `warp: "exp"` over 50–3000 reads 387 (the geometric mean) at the midpoint of its trait, and mutation moves it by even ratios.
    - `mutate` scales each param's step by its `delta`. A param without one mutates as before.
    - An exp warp over a range that touches or crosses zero, or a curve too steep to evaluate, falls back to linear with a console warning. A `delta` on an enum is ignored with a warning, since its options have no distance between them.

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
