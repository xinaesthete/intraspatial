# k-means by e-means

Status: **design note** (2026-11-14). **Slice 1 is built**: the CPU golden `src/spatial/emeans.ts`
(`emeans`, `lloyd`, `emeansSchedule`, `assignNearest`), exported from `intraspatial/spatial`. Slices
2–4 are not.

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
3. **Mini-batches suit streamed data.** Pixel clustering over multiplexed imagery (IMC/CODEX,
   10⁷–10⁸ pixels × 30–50 channels, k ≈ 100) does not need to be resident. Batches can be drawn from
   tiles as the datasource delivers them, and the full-assignment pass at the end can be tiled the
   same way. This is our case where the paper's large-n speedups apply directly.
4. **The GPU kernels are ones we already have**, as listed under slice 3 below.

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

## Slices

1. **CPU golden** — built. It is in f64, deterministic per seed, and pinned by tests: the schedule
   against the published laws, the Lloyd corner reproducing `lloyd` exactly, escape from the line
   trap, revival, and the d ≤ 3 default.
2. **Neighbourhood clustering op.** A k-hop (or radius) type-composition gather over the existing
   uniform-grid index, then `emeans`, giving a per-cell niche label and a centroid table (each
   niche's mean composition). This works on the CPU first, so it can be checked against
   squidpy/MuSpAn neighbourhood output.
3. **GPU port.** A step has three parts, and each maps onto a kernel we already have:
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
4. **Streamed pixel clustering.** Batches are pulled from tiles through the datasource, with a
   tiled final assignment that writes a label raster.
