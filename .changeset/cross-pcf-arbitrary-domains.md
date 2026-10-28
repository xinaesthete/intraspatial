---
"intraspatial": minor
---

The cross-PCF can now be measured on an ROI of any shape, not only a rectangle. Pass the region as a `domain`. Without one, nothing changes.

- New `Domain` API in `intraspatial/spatial`:
  - `rectDomain(bbox)` and `maskDomain(bbox, width, height, data)` build a domain. A mask is a raster with row 0 at the top and non-zero meaning inside. `maskDomain` rejects a raster whose length doesn't match its shape.
  - `domainArea`, `domainContains`, `domainAnnulusAreasInto` and `inverseAnnulusAreas` work on either kind.
- `crossPCF`, `crossPCFMatrixBinned` and `crossPCFBinMembership` take an optional `domain`. It sets the area used for ρ_B (unless `roiArea` is given), the edge-correction clip when `edgeCorrected` is set, and which cells count.
  - Cells outside the domain are dropped from both populations, and results report how many in `outside`.
  - The clipped area on a mask is exact for the mask's pixels. The remaining error comes from the pixel staircase along the boundary, and costs at most 0.1% on `g` at 1–2 µm pixels. The bounding-box clip reads 1–5% low on a tilted or round ROI.
- Masks can come from the data:
  - `maskWhere(field, keep, { derivedFrom })` thresholds any `ScalarField`-convention raster (a density field, a mode projection) into a mask.
  - `similarityField` and `pixelZ` compute the Gram wand's "places like this one" distance on the CPU, so it can be thresholded into a mask the same way.
  - `whiteningMatrix` and `standardise` now live in `intraspatial/spatial` too; `gramTerrain` still exports them.
- Masks built from the data carry `derivedFrom`, and `crossPCFMatrixBinned` returns `confounded`, which flags pairs whose two types both built the mask. Measured on independently clustered types, such pairs read g 14–23% low against a truth of 1. A mask that never saw one type of the pair leaves that pair unbiased.
- `crossPCFGpu` and `crossPCFBootstrap` throw when given a `domain`. They don't support it yet, and before they would have ignored it silently.
