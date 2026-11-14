# k-means by e-means

Status: **design note** (2026-11-14). **Slice 1 is built**: the CPU golden `src/spatial/emeans.ts`
(`emeans`, `lloyd`, `emeansSchedule`, `assignNearest`), exported from `intraspatial/spatial`. Slices
2–5 are not.

Source: Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze & Jégou, *e-means: outperforming k-means (in
one-tenth the epochs)*, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501) (Meta FAIR, 2026). The
blog and playground are at <https://e-means.github.io>, and the library is `faiss/contrib/e_means`.
The playground's JavaScript is MIT (Copyright (c) Meta Platforms, Inc. and affiliates), and our port
follows it.

## What it is

Mini-batch k-means in which every centroid is a ratio `c_j = S_j / N_j` of a running sum and count,
updated as an exponential moving average:

    S_j ← (1−α)·S_j + α·s_j        N_j ← (1−α)·N_j + α·n_j

Here `s_j` and `n_j` are the sum and count of the batch points assigned to `j`. Lloyd is the
special case `B = n, α = 1`. The authors' sweep over `(B, α)` puts that corner well outside the
low-loss band. The other parts of the method:

- **Scaling laws** set B and α₀ from `(n, k, T)`, so the inputs are the same as Lloyd's:
  `B = 0.00439·k^0.135·n^0.876·T^0.813` (the "quality" law, rounded to a power of two, at most
  n/2; the "default" law is 8× larger) and `α₀ = min(0.824·k^−0.462·B^0.900·n^−0.450·T^−0.106, 1)`.
- **A cosine schedule** anneals α from α₀ to 0.1·α₀.
- **Dead-cluster revival** runs every 5 steps. A cluster whose EMA count is below 0.1× the mean is
  re-seeded by splitting a live cluster drawn in proportion to its count.
- **Finalisation.** The last epoch accumulates the exact sum and count of each cluster's
  assignments, and the centroids end on those means.

The headline claim is a ~10× saving. e-means beats Lloyd's 25-epoch test loss in 1.2–3 epochs, at
k = 8k–65k on embedding datasets of 1M–100M points. This is 10–45× faster on an H100 and 8–15× faster
on CPU.

## Why it fits intraspatial

Our clustering sizes are smaller than the paper's, so the speed headline is not the main reason.
These are:

1. **It escapes local minima at small k.** On the authors' toys (k = 5–10, blobs of uneven size),
   Lloyd stays stuck on 15–46% of seeds and e-means on 0–4%. Mini-batch noise moves centroids out
   of arrangements that Lloyd's monotone descent cannot leave. **Cellular-neighbourhood
   ("microenvironment") clustering** is in this regime. It is the k-hop composition + k-means item
   in [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md), with k ≈ 5–20 on
   per-cell type-count vectors. A bad local minimum there merges two niches or leaves one out,
   which is a wrong biological answer and not only a higher loss.
2. **No hyperparameters beyond Lloyd's.** Ours is a catalogue of self-describing ops. An op that
   needs a batch size and learning rate tuned per dataset would be a poor entry. The scaling laws
   remove that.
3. **It fits the tiled, lazy model.** Fitting needs only a sample, and points-zarr's coarse
   levels are one. Assignment is per point against a frozen table, so it runs tile by tile as the
   viewport loads. Nothing has to be resident at full resolution (see "Applications").
4. **The GPU kernels are ones we already have**, as listed under slice 4 below.

## What we measured (slice 1, CPU golden)

The setup is the blog's "five blobs on a line, light middle blob", with k = 5 and 30 seeds at
n = 20 000. e-means ran its default 5 epochs and Lloyd ran 25 iterations. A run counts as stuck when
its MSE is more than 1% above the best run:

| data                                   | e-means stuck | Lloyd stuck |
| -------------------------------------- | ------------- | ----------- |
| 5 blobs on a line, tails overlapping   | 0/30          | 6/30        |
| 3×3 grid, heavier corners, k = 10      | 1/30          | 12/30       |
| 12 blobs in 16-D, uneven, k = 12       | 7/30          | 13/30       |
| 5 blobs on a line, **8σ apart**        | 16/30         | 16/30       |

This is also about 4× less assignment work (6 passes against 26). On the host, e-means measured
2.7–4.3× faster in the same runs.

**Caveats found:**

- **The escape needs overlap.** When blobs are far enough apart that no point falls between them,
  e-means is trapped exactly as often as Lloyd (last row). An EMA centroid stays in the convex hull
  of the points assigned to it, so it cannot cross empty space. Real composition vectors overlap
  heavily, so this case should be rare in use. Still, it is not a guarantee.
- **The advantage shrinks at small n.** At n = 4000 the grid case is 5/12 against 6/12. The laws
  were fit on n ≥ 10⁵. The line case still separates there, and the unit test pins it.
- **Revival hurts in very low dimension**, as the authors note. Their playground turns it off for
  its 2-D sets. Our default is off for d ≤ 3 (clustering raw coordinates) and 0.1 otherwise. That
  threshold is our reading of their note, not a measured cut-off.
- **The split is multiplicative** (`v·(1 ± 10⁻⁴)`, as in the reference). A centroid that is exactly
  zero in every coordinate splits into two identical copies. The copy then loses every tie and dies
  again. This cannot happen with a non-zero centroid.

## Applications

There is no committed application yet. e-means is in the toolbox because it is a better default
than Lloyd. The notes below record how it would be used in the settings we expect, so that the API
fits them.

Every use has the same shape: compute a descriptor per entity, standardise it, fit e-means, then
assign. The output is a label per entity (a column, or a raster) plus a centroid table. The
descriptor is usually a *windowed local statistic* (what is around this point), which the repo
already computes. So "neighbourhood descriptor → e-means" is one op that serves cells, transcripts
and pixels alike.

### The operating model: fit on a sample, assign lazily per tile

The data is zarr: points-zarr through sd.js (experimental) for transcripts, and zarr / OME-Zarr
pyramids for imagery and environmental grids. Work is tiled and driven lazily by the viewport.
Two consequences:

- **Never fit on what is in view.** Clusters fitted on the viewport would change as you pan. The
  same structure would get different labels in different places. So fit **once**, on a sample of
  the whole element. Then **assign** each tile as the viewport loads it, against the frozen
  centroid table. Assignment is per point and deterministic, so labels are stable across pans
  and zooms, and two tiles always agree where they meet. (Descriptors near a tile edge need an
  apron, as `transcriptChannels.ts` already handles.)
- **The sample is often free.** In points-zarr the coarser levels are *uniform random samples*
  of the element. `openPointsSource` serves them with `sampleFraction`. A coarse level is
  therefore an unbiased training set of a size we choose. The scaling laws then take n = the
  sample size, and the fit costs a few passes over that sample, not over 10⁹ points. Image
  pyramids need care here. A **subsampled** level is fine (IMC's is, per
  `covid-spatialdata-migration.md`). A **mean-pooled** level is not: its pixels are blends of
  classes, which pulls centroids towards mixtures. For mean-pooled pyramids, sample level-0
  pixels from random tiles instead.
- **Shuffle across tiles.** This caveat is specific to the EMA. The update forgets old batches at
  rate α. Fed tile after tile, the centroids would drift towards the region read last. Load the
  training sample (or a spread of tiles) first, then batch it in shuffled order. Several slides
  are the same problem one level up: batches must mix slides.

### Whole-transcriptome points (Atera)

Atera holds more than 10⁹ transcripts on a small sample, with whole-transcriptome panels of about
18k genes. A dense per-entity gene vector (72 KB in f32) is out of the question, so the descriptor
must be **reduced as it is built**. Give each gene a fixed embedding row `E[g]` (d′ ≈ 32–64,
from PCA on a pseudo-bulk sample or from gene modules). The neighbourhood descriptor is then
Σ over nearby transcripts of `E[gene]`: a d′-channel splat, which is exactly the
projection of the neighbourhood's count vector. The cost is that it is linear. A `log1p` on
counts does not commute with the projection, so normalise per gene (scale the rows of `E`)
rather than per entity after the fact. This descriptor serves both uses:

- **Segmentation-free domains** at k ≈ 10–30 (the Points2Regions / SSAM approach, which is
  mini-batch k-means at heart);
- **large-k sketching**, where a k ≈ 4096 codebook summarises the slide, and downstream statistics
  run on centroids weighted by their counts. This is the paper's own regime.

### Several slides

- **One centroid table, every slide.** Fit on a sample pooled across slides, then assign each
  slide against the frozen table. Every slide gets the same niche vocabulary, and a new slide can
  be labelled later without refitting. The table is small (k × d), so it is a natural
  content-addressed graph node (FAIR).
- **Balance the sample per slide** (or weight it), so that a large slide does not dominate.
- **Batch effects show up as clusters.** If slides differ technically, pooled k-means will split
  by slide. Integration is out of scope here, but a cheap diagnostic is in scope: a per-cluster ×
  per-slide count table, and its entropy. A cluster that lives on one slide is a warning sign,
  unless that slide really is different.

### Environmental grids (zarr)

- **Unsupervised environmental classes.** Pixels are clustered on stacked variables: bands,
  climate and soil layers, or a per-pixel time series with d = time steps. The machinery is the
  same tiled-zarr fit-then-assign as above.
- **Area weights.** On a geographic (lat/lon) grid, pixel area shrinks with cos(latitude). Without
  weights, high latitudes are over-counted. Weighted e-means is a small change: accumulate `w·x`
  and `w` in place of `x` and 1. The same weights serve per-slide balancing.
- **Variables come in different units**, so standardise them (`standardise` /
  `whiteningMatrix` in `gram.ts`). Nodata (zarr `fill_value`) must be excluded from both fitting
  and assignment, not treated as a value.
- **k-means ignores space.** For contiguous regions, add windowed local statistics to the
  descriptor. This is the neighbourhood construction again.

### What these add to the API

The slice-1 `emeans` takes one resident array and returns everything at once. The applications
need:

- a **fit / assign split**: `fit` returns the centroid table, and `assign` takes a frozen table
  and runs per tile;
- **per-sample weights** (area, slide balancing);
- **masks** (nodata, out-of-domain), reusing `domain.ts` where the data is points.

A streaming batch *source* is not needed. Because the fit runs on a sample that fits in memory,
the resident-array `fit` is enough. Only `assign` needs to be tiled.

## Slices

1. **CPU golden** — built. It is in f64, deterministic per seed, and pinned by tests: the schedule
   against the published laws, the Lloyd corner reproducing `lloyd` exactly, escape from the line
   trap, revival, and the d ≤ 3 default.
2. **Fit/assign split, weights and masks** (CPU). This adds the API needs above. It comes before
   the op slices, because each of them needs it.
3. **Neighbourhood clustering op.** A k-hop (or radius) type-composition gather over the existing
   uniform-grid index, then `emeans`, giving a per-cell niche label and a centroid table (each
   niche's mean composition). This works on the CPU first, so it can be checked against
   squidpy/MuSpAn neighbourhood output.
4. **GPU port.** A step has three parts, and each maps onto a kernel we already have:
   - *assignment*: the batch × k nearest-centroid argmin. This is `knnGpu`'s tiled distance with
     k = 1, or a GEMM-style `‖x‖² − 2x·c + ‖c‖²` for large k·d.
   - *batch statistics*: the per-cluster sum and count, `atomicAdd` into an exact fixed-point
     accumulator. This is the `splatDensityAtomic.ts` technique. It is deterministic, which keeps
     GPU-vs-golden tests tight. Float atomics would make them order-dependent.
   - *blend*: one O(k·d) dispatch. Revival needs the k counts every 5 steps. For small k, a
     single-workgroup Gumbel-top-k kernel on the device keeps it free of readback (standing
     priority 1).

   Shuffling becomes a counter-based permutation per epoch (no host shuffle and no upload). The
   GPU should default to the `"default"` batch law, to fill the device. The test is
   loss-equivalence to the golden within a bound, not label equality, since argmin can flip on
   near-ties.
5. **Tiled assignment over zarr.** Fit on a points-zarr sampled level (or on random level-0
   tiles), then label tiles lazily as the viewport loads them. The output is a label column for
   points, or a label raster for OME-Zarr imagery and environmental grids.
