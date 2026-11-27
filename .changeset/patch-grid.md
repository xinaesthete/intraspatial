---
"intraspatial": minor
---

Patch grid (R4): `patchGrid` in `intraspatial/spatial` lays square patches over a support mask with SIMBA's per-component layout and keeps those passing minimum-support rules (a mask fraction, a point count), matching SIMBA's patch tables exactly. Masks are row runs, so a 1 µm/px canvas can be streamed. `patchLayers` and `patchLabelImage` split overlapping patches into non-overlapping u32 label images for zonal statistics. A new `patchSupport` graph op (`patchSupportGpu` / `patchSupportResident` in `intraspatial/gpu/spatial`) counts a resident mask's support pixels per rectangle with one prefix scan.
