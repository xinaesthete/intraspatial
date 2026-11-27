---
"intraspatial": minor
---

Anchor screen with a within-sample permutation null (R5 of the region/raster statistics plan).

- `anchorScreen({ anchor, features, samples }, { permutations, minFinite, seed })` in `intraspatial/spatial` gives, per sample and feature, Spearman ρ between the anchor and the feature, a two-sided empirical p from shuffles of the anchor, and the number of finite pairs. It follows SIMBA's continuous-anchor scorer rule for rule (average ranks, per-feature finite pairs, minimum support, constant features, shuffling only finite anchors). `features` is a `Map` of columns, so R3's `zonalTable(...).columns` feeds it directly. `prepareAnchorBlock`, `anchorNullExceedances` and `anchorEmpiricalP` are the per-sample pieces.
- `anchorScreenGpu` in `intraspatial/gpu/spatial` runs the null's dot products on the GPU. By default it uses the CPU's own permutations, so p equals the CPU's; `shuffle: "feistel"` computes them on the device instead.
- The Feistel permutation behind `quadratCorrelationGpu` is now a shared WGSL snippet, `FEISTEL_WGSL` in `feistelWgsl.ts`, with a host twin `feistelPermuteCpu`. `quadratCorrelationGpu`'s output is unchanged.
