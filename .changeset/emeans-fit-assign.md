---
"intraspatial": minor
---

e-means can now fit a centroid table once and assign points to it later, with per-point weights and masks (`intraspatial/spatial`).

- `emeansFit(data, n, d, k, opts)` returns a frozen `CentroidTable` (`k`, `d`, `centroids`) with the weighted cluster `mass`, the `schedule` and `splits`. `assignCentroids(table, data, n, opts)` labels any set of points against it, such as one viewport tile at a time. `emeans` is now the two together, and its result is itself a table.
- `weights` (finite, ≥ 0) and `mask` (a `Domain` plus each point's position) are accepted by `emeansFit`, `emeans`, `lloyd` and `assignCentroids`. A point of weight 0 or outside the mask is left out of the fit exactly as if removed. It is never a seed, and assignment labels it −1 and leaves it out of the MSE. Weights are rescaled to mean 1, and the scaling laws take n = the number of points kept.
- Results gain `mass` (weight per cluster) and the MSE is weight-averaged. Both are unchanged for unweighted input. `lloyd` now re-seeds only clusters of weight exactly 0.
- `channelBins(tile, bins, channels)` counts a points tile's channel molecules into a finer quadtree level of bins, weighting each molecule by 1/`pointFraction`. `binComposition` turns those counts into Hellinger descriptors and molecule weights, for clustering bins into niches.
- `pooledBins(own, neighbours, bins, radius)` sums each bin's counts over the `(2·radius + 1)²` bins around it, reaching into the neighbouring tiles' bins. Pooling tile by tile equals pooling the whole element at once.
