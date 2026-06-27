# Plan — a GPU primitives toolbox (image signals + discrete-cell spatial analysis on TypeGPU)

Status: **plan** (2026-06-27)

The inverse and forward DWT (5/3 + 9/7) are now both on the GPU and validated, so
we have a real wavelet *transform pair*, not just a decode step. This doc plans
turning that into a small, composable **GPU primitives toolbox** — reusable for
analysis beyond the codec (denoising, multiresolution, feature work, fusion) and
for a future GPU encode path — and how to exercise it.

A second front opened on 2026-06-27: the [MuSpAn paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627195v1.full.pdf)
(Bull et al., *MuSpAn: A Toolbox for Multiscale Spatial Analysis*, bioRxiv
2024.12.06.627195) describes spatial analysis at the level of **discrete cells**
(points and shapes — cell centroids, transcript locations, segmented boundaries)
rather than image signals. Many of its methods are exceptionally GPU-friendly
(pairwise neighbour search, Monte-Carlo null models, density estimation), and a
clean bridge — rasterise points into a grid — lets the existing image primitives
serve point data too. The discrete-cell extension is planned in
[its own section below](#extension-discrete-cell-spatial-analysis-muspan).

## Where we are (the foundation)

Already built and reusable:

- **Runtime**: headless Dawn device (`src/gpu/device.ts`); TypeGPU typed resources
  via tgpu-gen + `tgpu.resolveWithContext` (`*.gen.ts` + WGSL templates).
- **Transform pair**: `idwt53`/`fdwt53` (5/3, i32, bit-exact) and
  `idwt97`/`fdwt97` (9/7, f32, ≤1e-4). All share one kernel pattern:
  **one workgroup per line, lift in workgroup shared memory, vertical/horizontal
  passes, per-level submits**.
- **Plumbing patterns** worth promoting to shared infrastructure:
  - buffer **pooling** (reuse/grow, no per-call churn);
  - **keep-on-GPU** (`readback: false`) and on-GPU pixel/level-shift conversions
    (`idwt53` shift fold, `idwt97` `pixels` option);
  - a flat **descriptor + packed-coeffs** interface for multi-level data.

## The key idea: compose on the GPU

Today each `*Gpu` function does upload → compute → readback and owns its pool.
To use the toolbox for analysis you want to **chain** primitives without
round-tripping to the CPU each step — e.g.

```
upload → fdwt → threshold(detail) → idwt → download   (GPU wavelet denoise)
```

So the central refactor is a small handle type and a shared runner:

- **`GpuField`** — a pooled GPU buffer + shape (`width, height`, dtype i32/f32,
  optional DWT `descriptor`). Primitives take and return `GpuField`s; data stays
  on the GPU between them.
- **`upload(data) → GpuField`** and **`download(field) → TypedArray`** at the
  boundaries only (download is the Dawn-on-Node-fragile step; chain first).
- A shared **kernel runner** that owns the pipeline cache, bind-group creation,
  and dispatch, so a new primitive is "a WGSL body + a layout + dispatch sizes",
  not 150 lines of boilerplate. The shared-memory line kernel becomes a template.

This turns the current 4 DWT modules into instances of a pattern and makes new
primitives cheap.

## Primitive catalogue

Grouped; `✓` = done, `→` = planned. Keep each primitive: pooled, keep-on-GPU,
CPU-golden-validated.

**Transforms**
- ✓ inverse/forward 5/3 (i32), inverse/forward 9/7 (f32)
- → generic ATK lifting (other JPEG2000 wavelets) from the same kernel
- → multi-component color transforms (RCT lossless / ICT lossy) — codec + viz

**Coefficient-domain (exercise the pair)**
- → threshold (hard/soft) on detail bands → denoising
- → per-band scale / gain, band zeroing/selection
- → band statistics (energy/variance per subband) via reduction
- → coefficient masking / fusion (combine two images' subbands)

**Pointwise / map**
- ✓ level shift, ✓ float→pixel convert (round/clamp/shift)
- → arithmetic (add/sub/mul/blend of two fields), LUT/palette apply, clamp/abs

**Reductions**
- → min / max / sum / mean, histogram → normalization, auto-contrast, stats

**Resampling / pyramids**
- → extract LL at level k (free from the forward DWT) → multiresolution overviews
- → nearest / bilinear resize

**Spatial (shares the line-kernel pattern)**
- → separable convolution (blur, sharpen, gradient/Sobel) — same shared-mem line
  approach as the DWT lift

## First exercises (prove the pair as a tool)

1. **GPU wavelet denoise**: `fdwt97 → soft-threshold detail → idwt97`, all on
   GPU; validate vs a CPU reference and show it reduces noise on a noisy fixture.
2. **Multiresolution overview**: forward DWT, read back the LL at level k as a
   downscaled image (the "free" pyramid) — useful for SpatialData viz.
3. **Band-energy map**: forward DWT + per-band reduction → a compact feature
   descriptor; a stepping stone to edge/texture analysis.

Each doubles as a composition test of the `GpuField` chaining model.

## Extension: discrete-cell spatial analysis (MuSpAn)

Everything above treats data as a **regular grid of signal samples** — pixels,
DWT coefficients, density fields. The [MuSpAn paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627195v1.full.pdf)
points at a complementary world: **irregular sets of discrete objects** — cell
centroids, transcript points, segmented cell boundaries — analysed for how they
sit relative to one another (contact, clustering, exclusion, neighbourhood
composition, tissue architecture). Spatial biology is the motivating domain, but
the math is generic: *points and shapes with labels, in continuous 2D*.

This matters for the toolbox because MuSpAn's workhorse computations are some of
the most GPU-friendly workloads there are — all-pairs distances, neighbour
searches, and especially **Monte-Carlo null models** (the statistically
expensive, embarrassingly-parallel core of spatial statistics). And there is a
clean seam back to the existing image primitives: **rasterise points into a grid**
(density splat, quadrat/hex binning) and the convolution / reduction / pointwise
primitives already planned above apply unchanged.

### The data-model shift (two new handles + an index)

`GpuField` (grid) is not enough; point/shape data needs companions:

- **`GpuPoints`** — N points: parallel `x`, `y` (f32) buffers + optional per-point
  `label` (i32) and scalar attributes. The unit of cell-centroid / transcript work.
- **`GpuShapes`** — polygons as a flat vertex buffer (`x`, `y`) + per-shape
  `(offset, count)` ranges (the same "descriptor + packed data" shape we already
  use for multi-level DWT coeffs). The unit of cell-boundary / morphology work.
- **`GpuGridIndex`** — a **uniform-grid spatial index** over `GpuPoints`: assign
  each point a bin from `(x, y)` at cell size ≈ query radius, counting-sort points
  into bins, prefix-sum to per-bin offsets. Neighbour queries then scan the 3×3
  bin block instead of all N points. This is the foundational new kernel, and it
  is the analogue of the line kernel for this front.

Building the index hands us two classic GPU building blocks the reductions
section already wanted — **prefix sum (scan)** and **counting sort / scatter** —
so the index is high-leverage beyond neighbour search (histograms, compaction,
stream partition all fall out).

### The bridge: points → grid → existing primitives

The cheapest way to make a point method GPU-fast is often to **stop treating it as
a point method**. Splat points onto a grid and the image toolbox takes over:

```
GpuPoints → splat(kernel) → GpuField(density) → blur/getis-ord/threshold → download
            (scatter)                            (existing convolution + pointwise)
```

- **KDE** = scatter a Gaussian per point (or splat deltas then separable Gaussian
  blur — i.e. the planned convolution primitive). Output is an ordinary `GpuField`.
- **Local Getis-Ord\* / LISA / topographical correlation map** = a local
  windowed z-score over that density field → convolution + pointwise, done.
- **KL-divergence / histogram comparison** between two density grids = pointwise +
  reduction.
- **Quadrat / hexgrid binning** = scatter point labels into lattice cells; the
  **quadrat correlation matrix (QCM)** is then a small per-cell count-vector
  correlation (reduction / tiny matmul). Hexgrid is just a different bin transform.
  But hard binning is the artefact-prone special case — see
  [Windowing, not quadrats](#windowing-not-quadrats-kill-the-binning-artefacts).

So a large slice of MuSpAn's *region-based* and *distribution* modules costs us
almost nothing once splat + the planned grid primitives exist.

### Windowing, not quadrats (kill the binning artefacts)

Hard quadrat assignment — each point contributes to exactly one lattice cell, no
overlap — is a **boxcar (rectangular) window**. In spatial statistics its failure
mode is the **Modifiable Areal Unit Problem**: results swing with the arbitrary
*size*, *origin*, and *phase* of the grid. Two cells 1 µm apart but straddling a
quadrat edge are scored as unrelated; two cells at opposite corners of one quadrat
(up to √2·edge apart) are scored as co-located. The very signal we want — *who is
near whom* — is what the hard grid throws away. In signal terms the boxcar also
has the worst sidelobes of any window (sinc, ≈−13 dB) → maximal leakage and edge
artefacts.

The fix is the **Welch / STFT move**: replace the boxcar with a smooth, *overlapping*
window (Gaussian / Hann / Epanechnikov). Overlap recovers the cross-boundary pairs;
the taper removes the discontinuity; averaging over overlapping windows makes the
result nearly insensitive to grid phase.

The payoff for *this* toolbox: **the window is the primitive both fronts already
share.** On the image front a window *is* a convolution kernel; on the cell front
it is the KDE / local-stat kernel; a quadrat is just `window(shape=box, overlap=0)`.
So make the window a **parameter**, not a separate region-based family, and the
GPU-natural path is already the windowed one:

```
GpuPoints → splat onto a FINE grid → separable smooth window → [resample coarse]
            (weighted scatter)        (the planned convolution)   (optional)
```

This is the same KDE→grid bridge above — the point is that the *artefact-reducing*
method is also the *cheaper, more GPU-natural* one: hard coarse quadrats mean
atomic-contended scatter into few bins, whereas splat-fine-then-separable-blur is
the convolution primitive we already want. The artefact and the inefficiency go
away together. It also reuses infrastructure we already need: a window centred near
a tile edge must see points across the tile boundary — a **halo/apron**, the same
problem as DWT line-kernel tiling (`MAXLINE`) and large-image convolution.
Overlap-add windowed processing of a long signal *is* tiling-with-halo.

Two honest qualifications:

- **The null must follow the window.** QCM's significance and the permutation tests
  are defined on discrete quadrat counts; once counts become smooth weighted sums,
  the null has to be re-derived as a **weighted permutation null** (shuffle labels,
  recompute the windowed field). Cheap on GPU — it is the Monte-Carlo-in-one-sweep
  flagship — but it is not inherited for free.
- **For pure pairwise co-occurrence, skip the grid.** The cross-PCF and Ripley's K
  are *already* windowed — in the radial/distance domain, edge-corrected — and are
  point-native. Rule of thumb: **windowed grid when you want a spatial field/map**
  (e.g. a Getis-Ord hotspot surface); **PCF/K when you want a function of distance**;
  hard quadrats only when a downstream step genuinely needs disjoint regions.

### Render vs compute: splat by blending, not atomics

The constraint note below says core WGSL has **no f32 atomics**, so the obvious
"scatter each point's weight into a density buffer" looks awkward. But splatting is
a problem the GPU already solved in *fixed-function hardware*: draw each point as a
small kernel-textured sprite into a **float render target** (`rgba16float` /
`r32float`) with **additive blending** (`src=ONE, dst=ONE, op=ADD`). The blend unit
accumulates for us — no atomics, no contention, and it is what GPUs are fastest at.
So **density fields, KDE, quadrat/hex counts, and label-count maps are naturally a
*render* job, not a compute job.** The toolbox gains a second kernel modality: a
tiny **splat/blend pipeline** alongside the compute line-kernel.

That reframes the whole point→grid bridge as a **layer/FBO compositing** pipeline,
which is exactly how viewer stacks (deck.gl-style, and the SpatialData.js / MDV
contexts we want to feed) already think:

- render each label's density into its own float texture **layer**;
- combine layers with **blend ops or a short pointwise compute pass** — e.g. the
  **topographical correlation map (TCM)** is a windowed (productOfMeans-minus-…)
  combination of two density layers, i.e. a few texture reads per texel;
- the output **is a renderable layer**, so analysis and visualisation are the same
  artefact — no readback round-trip (which also sidesteps the Dawn-on-Node readback
  ceiling, since for viz the texture stays on the GPU and is simply displayed).

Decision axis to lock in: **compute** for irregular gather / scan / sort / reduce
and exact integer counts (the index, PCF histograms, reductions, Monte-Carlo
nulls); **render** for additive splat, resampling, and anything whose natural
output is a screen-space field (KDE, density, TCM, hotspot surfaces). A few methods
(TCM, Getis-Ord) are **render-then-compute**. WebGPU caveat: float render targets
need `rgba16float` (blendable by default) or the `float32-blendable` feature —
a capability to check at device init, alongside the existing Dawn notes.

### Distance decay = principled truncation (free performance)

Influence that falls off with distance means distant cells contribute negligibly,
so **truncating the window at a few σ is a principled approximation, not a corner
cut**. For a 2D Gaussian the mass within radius R is `1 − exp(−R²/2σ²)`: a 2σ window
keeps **86.5%**, 3σ keeps **98.9%**, and the spatial index makes "only look within
R" the natural access pattern — dropping all-pairs O(N²) to **O(N·k)** with bounded,
quantifiable error. This is also why **compact-support kernels** (Epanechnikov,
tricube) are often the better default here than a Gaussian: same smoothing, support
is *exactly* finite (zero error from truncation), and the hard cutoff is index-
friendly. The knob — window radius vs σ — is a tunable rigour/cost dial with a
closed-form error bound, not a guess.

### Point-native primitives (the new GPU wins)

These run directly on `GpuPoints` + `GpuGridIndex`, no grid:

- → **Pairwise / nearest-neighbour distances** — per point, scan neighbour bins for
  the min distance (NN) or accumulate radius-binned counts. Brute-force O(N²) tiled
  kernel (matmul-like) for small N; index-accelerated O(N·k) otherwise. Foundation
  for the next three.
- → **Cross-PCF (cross-pair correlation) & Ripley's cross-K** — for a label pair
  (A,B) and radius bins, count B within r of each A, normalise by density/area,
  edge-correct. A radius **histogram** over neighbour pairs — pure GPU reduction.
- → **Average Nearest Neighbour Index (ANNI)** — mean NN distance vs the random
  expectation → z-score. The observed part is the NN kernel; the null is below.
- → **Empty-space (spherical contact) function** — sample random locations, measure
  distance to nearest point. Same NN kernel, different query set.
- → **Proximity / contact network** — connect points within distance d (or whose
  boundaries are within d). The index gives this directly and covers the paper's
  cell-cell interaction network (Fig 3F) and adjacency analyses **without** needing
  Delaunay. Degree distribution and per-type contact frequencies are then reductions
  over the edge list.
- → **k-hop neighbourhood composition + k-means** — gather per-point neighbour
  label-count vectors (the "cellular neighbourhood" descriptor), then GPU k-means to
  assign microenvironments. Both gather and Lloyd iteration are GPU-standard.

### The flagship: Monte-Carlo null models in one sweep

Most MuSpAn statistics are only meaningful against a **random-label null** — ANNI
z-scores, QCM standardised effect sizes, the adjacency permutation test, PCF
confidence envelopes. On CPU this bootstrap (100s–1000s of relabellings, each
recomputing the whole statistic) dominates the runtime. On GPU it is close to
free: keep the points resident, and recompute the already-GPU statistic across
all permutation replicates **in parallel** — one extra dispatch dimension. This is
the single strongest reason to bring this front onto the GPU, and it makes the
*statistical rigour* (proper nulls, tight envelopes) the cheap part rather than
the expensive part.

Determinism comes from a **counter-based RNG** (PCG / Philox) seeded per replicate
from the CPU — reproducible label shuffles without `Math.random` (unavailable in
WGSL anyway).

### Beyond pairs: higher-order co-occurrence (small N)

The paper flags this as a frontier — extensions that quantify colocalisation of
**three or more** cell types within a distance of one another (neighbourhood
characterisation, ref [53]). It matters because pairwise statistics are blind to
genuinely multi-way structure: A–B and B–C can each look unremarkable while
{A,B,C} *junctions* are strongly enriched. Pairwise can be flat where the triple
lights up.

This generalises cleanly on the GPU as long as the number of types **N** (and the
order **k**) stay small:

- A pairwise PCF is a histogram over `(labelA, labelB, radius-bin)`. Higher order is
  a joint **co-occurrence tensor** over `(labelA, …, labelK, radius)` — fine while
  `Nᵏ × radii` stays modest (e.g. N ≤ ~20, k = 3 → ~8k bins per radius).
- The **per-cell neighbourhood vector** already in the plan (k-hop / windowed
  label-count per cell) is the natural substrate: higher-order colocalisation is a
  statistic over *products* of those soft counts (joint frequency of {A,B,C} in a
  neighbourhood vs the product-of-marginals null) — all GPU reductions.
- The **windowed** form drops out for free: weight neighbours by the smooth kernel
  to get soft composition vectors, then take expectations of their products. And the
  **Monte-Carlo null extends unchanged** — shuffle labels, recompute the tensor.

Keep `k` and `N` small (combinatorial blow-up otherwise); when types are many,
cluster them first (the neighbourhood-clustering primitive) and work over the
coarser label set.

### What stays on the CPU (for now)

- **Persistent homology / Vietoris-Rips** (Fig 4E–F) — the GPU can build the
  filtration (it is just thresholded pairwise distances / the neighbour graph from
  the index), but the **boundary-matrix reduction is inherently sequential**; leave
  it to a CPU library (Ripser/GUDHI) and feed it a GPU-built distance graph.
- **Exact Delaunay / Voronoi / α-shape** — irregular and hard to parallelise well.
  Substitute the uniform-grid **proximity network** above, which covers the
  contact/adjacency use-cases; revisit GPU Delaunay only if a method truly needs it.
- **Wasserstein distance** — exact optimal transport is CPU; the **entropic
  (Sinkhorn)** approximation *is* GPU-friendly (iterated matrix–vector) if an
  approximate metric is acceptable.

### First exercises (prove the discrete-cell front)

The paper ships a synthetic fixture — `Synthetic-Points-Architecture` (a ring of
two cell types with random background) — with a *known* answer (clustering at the
ring/crypt scale, short-range exclusion). That makes it an ideal golden.

1. **Windowed vs quadrat colocalisation** (the headline compromise made runnable):
   render-splat the labels to density layers, compute a windowed colocalisation
   field, and show it reproducing a QCM-style co-occurrence map **without** the
   grid-phase sensitivity — side-by-side against hard quadrats swept over grid
   origin, on the synthetic ring. Validates the windowing claim *and* the
   render/splat path in one go. (The interactive sketch of this compromise lives in
   the chat thread that motivated this section; graduate it into the `viz/` harness.)
2. **GPU cross-PCF with a Monte-Carlo envelope** on the synthetic ring: build index
   → radius histogram for a label pair → N permutation nulls in parallel →
   z-score / envelope. Lights up index + histogram + RNG + reduction, validated
   against a CPU/MuSpAn reference.
3. **GPU KDE → Getis-Ord hotspots** (render-then-compute): splat → density layer →
   local z-score (reusing the convolution + pointwise primitives). Proves the
   point→grid bridge end-to-end and lands a result the image front can keep
   processing (e.g. wavelet-denoise the density map — a composition across *both*
   fronts).
4. **Batch cell morphology**: shoelace area/perimeter, centroid, second-moment
   **principal angle** (closed-form 2×2 eigenvector) and circularity over thousands
   of `GpuShapes` polygons in one dispatch — one workgroup per polygon, reduce over
   vertices (the line-kernel pattern again). Validate vs shapely/CPU.

### Method → primitive map (MuSpAn modules)

`✓` foundation exists · `→` planned here · `~` partial/CPU-assisted · `cpu` stays CPU.

| MuSpAn module | Method (paper) | GPU primitive | Status |
| --- | --- | --- | --- |
| geometry | area, perimeter, circularity, principal angle | per-polygon reduction (one wg/shape) | → |
| spatial statistics | cross-PCF, Ripley's cross-K | radius histogram over neighbour pairs | → |
| spatial statistics | ANNI, nearest-neighbour distribution | NN search + MC null | → |
| spatial statistics | empty-space function | NN search from random samples | → |
| spatial statistics | QCM, Local Getis-Ord\*, TCM (LISA) | splat layers → blend/window (render-then-compute) | → |
| spatial statistics | higher-order co-occurrence (k≥3, small N) | joint co-occurrence tensor + MC null | → |
| spatial statistics | adjacency permutation test | proximity network + MC null | → |
| distribution | kernel density estimation | additive splat to float layer (render) | → |
| distribution | KL-divergence | pointwise + reduction on grids | → |
| distribution | Wasserstein distance | Sinkhorn OT (approx) / exact CPU | ~ |
| networks | proximity / contact network, degree | uniform-grid index + edge reduction | → |
| networks | k-hop neighbourhood + clustering | neighbour gather + GPU k-means | → |
| networks | Delaunay / Voronoi | (use proximity network instead) | cpu |
| region based | hexgrid / quadrat lattice | bin transform + scatter (prefer windowed) | → |
| region based | windowed local stats (overlapping taper) | splat → separable window → resample | → |
| topology | Vietoris-Rips, persistent homology | GPU builds filtration; reduction | cpu |
| helpers | α-shape | depends on Delaunay | cpu |

## Conventions to lock in

- **Validation**: every primitive gets a CPU golden (Rust or JS) + a GPU test
  asserting bit-exact (integer) or ≤tolerance (float), plus a round-trip where it
  applies. GPU tests are `*.gpu.test.ts`, opt-in for benches (`BENCH=1`).
- **Determinism**: integer primitives use i32 + arithmetic `>>` (bit-exact);
  float primitives bound max abs error (~1e-4) and mean error.
- **Keep-on-GPU first**: primitives never read back internally; only `download`
  does. This is also the only Dawn-on-Node-fragile op (see constraints).

## Constraints to design around (learned the hard way)

- **Dawn-on-Node readback ceiling**: `mapAsync` readback crashes the worker
  beyond ~512²; *compute* is fine to 2048². Chain on-GPU and download once;
  benchmark/validate large sizes in a **browser**.
- **Per-level / cross-pass hazards**: a buffer bound readonly in one pass and
  mutable in another within a single command encoder did **not** get the
  write→read barrier we expected (forward DWT ll0 was wrong until we used
  per-level submits). Pattern: submit per dependent stage, or double-buffer.
- **No 64-bit ints in core WGSL**: lossless >~30-bit (e.g. uint32) can't use an
  i32 GPU path; that work stays CPU (or 2×u32 emulation). Float (9/7-style)
  sidesteps it.
- **Shared-memory line cap** (`MAXLINE = 2048`): tile longer lines for very large
  images.

Additional constraints for the discrete-cell front:

- **Irregular ≠ grid**: point methods need a spatial index, and that needs **scan +
  counting-sort** — a second core kernel beyond the line kernel. Build it once.
- **No f32 atomics in core WGSL**: scatter/splat/histogram accumulation can only use
  `atomic<i32>`/`atomic<u32>`. Use **fixed-point** accumulation (scale to integer,
  atomicAdd, rescale), per-bin reduction, **or — preferred for density — a render
  pass with additive blending into a float target** (see *Render vs compute*); the
  fixed-function blend unit accumulates without atomics.
- **Float render targets**: additive-blend splat needs `rgba16float` (blendable by
  default) or the `float32-blendable` feature — check at device init. This adds a
  **render-pipeline modality** to a toolbox that is otherwise compute-only.
- **RNG**: no `Math.random` in WGSL. Monte-Carlo nulls use a **counter-based RNG**
  (PCG/Philox) seeded per replicate from the CPU — also what makes them reproducible.
- **Cell-index range**: bin indices for ≤2048² lattices fit in i32, so the index
  sidesteps the no-64-bit-int limit; very large domains tile or coarsen the lattice.

## Roadmap

1. **Extract the shared kernel runner + `GpuField`** from the 4 DWT modules
   (refactor, no behaviour change; tests stay green). This is the unlock.
2. **First exercise — GPU wavelet denoise** (`fdwt → threshold → idwt`) to prove
   composition end-to-end.
3. **Reductions + pointwise** (min/max/sum/histogram, arithmetic) — small, high
   leverage, enable normalization/stats.
4. **Color transforms (RCT/ICT)** — needed for multi-component anyway; reuse in
   viz.
5. **Spatial convolution** + **resize/overview** primitives.
6. Revisit large-size GPU validation in a **browser** harness once the toolbox is
   chaining on-GPU (where the readback ceiling stops mattering).

The discrete-cell front runs as a **parallel track** (it shares the kernel runner
but adds its own handles and index kernel):

A. **Spatial index** (`GpuPoints` + uniform-grid index = scan + counting-sort).
   This is the unlock for the whole point front, the way `GpuField` + runner is the
   unlock for the image front — and it drops scan/sort into the reductions toolbox
   for free.
B. **Splat/blend render primitive** (`GpuPoints` → float layer via additive
   blending). The second kernel modality; powers KDE, density, quadrat/hex counts,
   and the layer→FBO→blend compositing that TCM and viewer integration need.
C. **Windowed vs quadrat colocalisation** (exercise 1) — uses A+B to make the
   headline compromise runnable and validate the render path against hard quadrats
   on the synthetic fixture.
D. **Cross-PCF + Monte-Carlo envelope** (exercise 2) — proves index + histogram +
   RNG nulls together; the statistical-rigour flagship.
E. **KDE → Getis-Ord** (exercise 3, render-then-compute) — merges the two tracks
   and feeds a renderable hotspot layer straight to a viewer.
F. **Batch morphology** over `GpuShapes` (exercise 4) — reuses the line-kernel
   pattern, no new infrastructure.
G. **Proximity network + neighbourhood clustering** + **higher-order co-occurrence**
   tensor; then a CPU-assisted **VR filtration** feed (GPU builds the graph, CPU
   reduces) if topology is wanted.

See also [`performance-report.md`](performance-report.md),
[`dwt-gpu-and-high-bit-depth.md`](dwt-gpu-and-high-bit-depth.md), and the
[MuSpAn paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627195v1.full.pdf) that motivates the discrete-cell front.
