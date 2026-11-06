---
"intraspatial": patch
---

`gramMatrixGpu` now throws when its points need a storage binding larger than the device allows, instead of returning a blank Gram. Expression channels each carry every cell, so a whole Xenium table (397k cells × 32 genes, 152 MB) passes WebGPU's default 128 MiB. The error names the size and the limit.
