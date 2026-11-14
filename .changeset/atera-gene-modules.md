---
"intraspatial": minor
---

Gene modules for whole-transcriptome points, and sampled points weighted to stand for what was left out.

- `geneEmbedding(xs, ys, codes, nFeatures, { binSize, … })` in `intraspatial/spatial` is a PCA gene embedding of a points sample. It bins the sample, normalises and logs each bin, standardises each gene, and runs a sparse randomized PCA. It keeps only the components above the Marchenko–Pastur noise edge.
- `geneModules(embedding, k)` partitions the spatially structured genes into `k` disjoint modules with e-means. `binSizeFor` sizes bins for ~750 molecules each.
- `emeans` and `lloyd` take `seeding: "kmeans++"`. The default is still random seeding, as in the reference implementation.
- `PointsTile.featureFractions` records per-gene sample fractions from a subset read. `pointFraction(tile, code)` reads a molecule's fraction.
- **Behaviour change:** `transcriptChannels` now weights each molecule by 1 / its sample fraction (`sampleFraction`, or `featureFractions` for its gene). Channel densities from sampled tiles used to read low by that fraction. They are now in molecules per unit area at any level. `g` and the correlation modes of uniformly sampled tiles are unchanged, because a per-channel scale cancels in both.
