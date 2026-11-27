---
"intraspatial": minor
---

Region statistics for the gram modes and the wand (R2 of the region/raster statistics plan).

- `floodAreaByTolerance(d, width, height, seed)` in `intraspatial/spatial` gives the wand patch's area at every tolerance from one bottleneck (minimax) flood. The area is exactly the pixel count `connectedRegion` fills at that tolerance. `floodAreaAt` reads the area at one tolerance, `floodJoinTolerance` gives the least tolerance at which a pixel joins, `floodLeaks` marks single-step basin joins, and `bottleneckCosts` and `wandSeedPixel` are the pieces it is built from.
- `modeHistograms(res, params, rasters)` in `intraspatial/gpu/spatial` histograms the three painted modes over a Gram window with the `histogram` op, and returns per-mode 1st/99th-percentile half-ranges. `modeProjection` reads the mode planes back.
- `modeParams` / `paintGramModes` take `ranges`, a per-mode half-range that replaces `saturate·σ_k`.
- `probeChannels` and `wandDistance` take an optional `device`, the one the Gram result lives on, as `modeHistograms` does. Their pipelines and buffers are now kept per device. `probeChannels` serialises its own calls on each device, so concurrent callers no longer share its output buffer.
