---
"intraspatial": minor
---

Add zonal statistics (R3): `rasterisePolygons` and `zonalHistogram` graph ops, a tile-by-tile `ZonalAccumulator`, and `zonalStatsTiled` over a pyramid level. Polygons (overlapping ones split into layers by `polygonLayers`) or a label raster give one histogram row per region × channel, in R1's layout. From it, `zonalTable` derives per-region count, mean, median, IQR, percentiles and support/valid fractions. Per-channel support rasters and a `"positive"` valid rule cover the TNBC background conventions. An exact second pass (`zonalExactPlan` / `zonalExactQuantiles`) gives numpy-exact quantiles. R1's bin rule is now shared as `HIST_BIN_WGSL`.
