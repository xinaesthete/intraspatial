---
"intraspatial": patch
---

`gramMatrixGpu` now throws when its points need a storage binding larger than the device allows, instead of returning a blank Gram. Expression channels each carry every cell, so a whole Xenium table (397k cells × 32 genes, 152 MB) passes WebGPU's default 128 MiB. The error names the size and the limit.

It checks before packing the points, so the error no longer costs the packed copy first. Its pooled buffers also stop doubling at the device's `maxBufferSize`, where a larger request would have given an invalid buffer and a blank Gram.
