# Plan — a GPU toolbox for discrete-cell spatial analysis (TypeGPU)

Status: **building** (2026-06-28)

This is the **discrete-cell front** of the GPU primitives toolbox — a sibling to
the wavelet/image-signal front in [`gpu-primitives-toolbox.md`](gpu-primitives-toolbox.md).
Where that doc treats data as a **regular grid of signal samples** (pixels, DWT
coefficients, density fields), this one treats it as **irregular sets of discrete
objects** — cell centroids, transcript points, segmented cell boundaries — analysed
for how they sit relative to one another (contact, clustering, exclusion,
neighbourhood composition, tissue architecture).

The motivation is the [MuSpAn paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627195v1.full.pdf)
(Bull et al., *MuSpAn: A Toolbox for Multiscale Spatial Analysis*, bioRxiv
2024.12.06.627195). Spatial biology is its domain, but the math is generic:
*points and shapes with labels, in continuous 2D*. MuSpAn's workhorse computations
are some of the most GPU-friendly workloads there are — all-pairs distances,
neighbour searches, and especially **Monte-Carlo null models** (the statistically
expensive, embarrassingly-parallel core of spatial statistics). And there is a
clean seam back to the image front: **rasterise points into a grid** (density
splat, quadrat/hex binning) and the convolution / reduction / pointwise primitives
on the other front apply unchanged.

The two fronts share one runtime (the TypeGPU/Dawn device, the kernel runner) and,
increasingly, one **windowing primitive** — see [Windowing, not quadrats](#windowing-not-quadrats-kill-the-binning-artefacts).

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
| spatial statistics | nearest-neighbour distance (per point) | brute-force `"use gpu"` kernel | ✓ |
| spatial statistics | ANNI (clustered/random/dispersed) | NN distance + Clark-Evans z | ✓ |
| spatial statistics | nearest-neighbour distribution | NN distance + MC null | → |
| spatial statistics | empty-space (F) function | min-distance from random samples | ✓ |
| spatial statistics | Local Getis-Ord\* (LISA hotspots) | splat → box window → standardise | ✓ |
| spatial statistics | QCM, TCM (LISA) | splat layers → blend/window (render-then-compute) | → |
| (enabling) | separable convolution (box / Gaussian window) | `"use gpu"` two-pass gather | ✓ |
| spatial statistics | higher-order co-occurrence (k≥3, small N) | joint co-occurrence tensor + MC null | → |
| spatial statistics | adjacency permutation test | proximity network + MC null | → |
| distribution | kernel density estimation | additive splat to float layer (render) | ✓ |
| distribution | KL-divergence | pointwise + reduction on grids | → |
| distribution | Wasserstein distance | Sinkhorn OT (approx) / exact CPU | ~ |
| networks | proximity / contact network, degree | uniform-grid index + edge reduction | → |
| networks | k-hop neighbourhood + clustering | neighbour gather + GPU k-means | → |
| networks | Delaunay / Voronoi | (use proximity network instead) | cpu |
| region based | hexgrid / quadrat lattice | bin transform + scatter (prefer windowed) | → |
| region based | windowed local stats (overlapping taper) | splat → separable window → resample | → |
| networks | fuzzy / kernel-weighted adjacency | `μ_ij=exp(-d²/2σ²)` dense matrix | ✓ |
| topology | Vietoris-Rips, persistent homology | GPU builds filtration; reduction | cpu |
| topology | fuzzy / weighted VR (fuzzy simplicial set) | fuzzy adjacency → CPU reduce | ~ |
| helpers | α-shape | depends on Delaunay | cpu |

## Constraints to design around


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

The discrete-cell front runs as a **parallel track** (it shares the kernel runner
but adds its own handles and index kernel):

A. **Spatial index** (`GpuPoints` + uniform-grid index = scan + counting-sort).
   This is the unlock for the whole point front, the way `GpuField` + runner is the
   unlock for the image front — and it drops scan/sort into the reductions toolbox
   for free.
B. **Splat/blend render primitive** ✓ (`splatDensity.ts` — KDE via additive
   blending into an `r32float` layer). The second kernel modality; the same path
   extends to quadrat/hex counts and the layer→FBO→blend compositing that TCM and
   viewer integration need.
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

## Kernel authoring: `"use gpu"` (TGSL) vs WGSL templates

Per the user's steer, compute kernels are authored in TypeScript with the
`"use gpu"` directive (TGSL) **where it makes sense**, transpiled to WGSL by
`unplugin-typegpu` at build time (wired into `vitest.gpu.config.ts`). See
[ADR-0003](decisions/0003-use-gpu-tgsl-kernels.md) for the toolchain decision and
its constraints (notably: requires Node ≥ 20.11; kernels are *resolved to WGSL and
run via a raw, layout-bound pipeline* rather than the guarded-pipeline runtime,
which churns Dawn-on-Node).

Division of labour:

- **`"use gpu"` (TGSL)** — pointwise maps, per-point loops, reductions without
  shared memory: anything expressible as straight-line WGSL. Type-safe authoring,
  no string templates.
- **WGSL templates (`resolveWithContext`)** — kernels needing **workgroup shared
  memory, barriers, or atomics** (the DWT line kernel; later the spatial-index
  scan/sort and histogram splats). TGSL in this version doesn't cover those cleanly.

## Implemented primitives

- ✓ **Nearest-neighbour distance** (`src/gpu/spatial/nnDistance.ts`) — for each of
  N points, the Euclidean distance to its closest other point. Brute-force O(N²),
  authored in `"use gpu"`, validated bit-close (≤1e-3) vs a CPU golden
  (`*.gpu.test.ts`). Foundation for ANNI, the nearest-neighbour distribution, and
  the empty-space function. Pooled/grown buffers + a once-built layout-bound
  pipeline (no per-call `.destroy()`), matching the DWT modules' Dawn-stable
  discipline. Large-N validation is deferred to the browser harness (the Node+Dawn
  teardown segfaults past a few hundred points — the same instability the image
  front hit; see ADR-0003).
- ✓ **Kernel-density splat** (`src/gpu/spatial/splatDensity.ts`) — rasterise a
  weighted point cloud into a Gaussian KDE density grid: the **points → grid
  bridge**. Uses the **no-atomics additive-render path** — each point is an
  instanced quad with a Gaussian footprint, additively blended into an `r32float`
  render target (`float32-blendable`, requested in `device.ts`). Validated vs a CPU
  KDE golden sampled at the same texel centres. Output is an ordinary density grid,
  so the image-front primitives (blur, Getis-Ord, threshold, wavelet) take over
  from here. **Dependency-light on purpose:** raw WebGPU (no TypeGPU resolve, no
  deck.gl / luma.gl / MDV) — the WGSL and the "render each layer to a float target,
  then composite" shape are exactly what a deck.gl custom layer or an MDV /
  SpatialData.js overlay needs, so it *translates* later rather than locking in.
  - *Finding (added to ADR-0003):* read texture results back via TypeGPU's
    `.read()`, **not** a raw `mapAsync` on a pooled buffer — the latter crashed the
    vitest worker on teardown. Render stays raw WebGPU; only the readback borrows
    the project's Dawn-stable path.
- ✓ **ANNI** (`src/gpu/spatial/anni.ts`) — Average Nearest Neighbour Index. A pure
  *composition* of `nnDistance` + the Clark-Evans CSR test → an interpretable
  clustered / random / dispersed verdict with a z-score.
- ✓ **Empty-space (F) function** (`src/gpu/spatial/emptySpace.ts`) — distance from
  random sample locations to the nearest point; characterises void size. Same
  min-distance kernel as `nnDistance` over a separate (seeded) query set.
- ✓ **Separable convolution** (`src/gpu/spatial/convolveSeparable.ts`) — the grid
  windowing primitive (`"use gpu"`, two passes). Box window = local sum/mean,
  Gaussian = smoothing. Matches CPU; normalised Gaussian conserves mass.
- ✓ **Getis-Ord Gi\* hotspots** (`src/gpu/spatial/getisOrd.ts`) — a LISA z-score
  grid (hot/cold spots). Composes the windowed sum (convolution) with closed-form
  standardisation; `pointHotspotsGpu` chains it onto the splat for a one-call
  points → hotspot-map pipeline (render-then-compute).
- ✓ **Fuzzy adjacency** (`src/gpu/spatial/fuzzyAdjacency.ts`) — kernel-weighted
  graph `μ_ij = exp(-d²/2σ²)`, the smooth analogue of a hard within-radius edge and
  the substrate for fuzzy TDA. See [`fuzzy-tda-and-windowing.md`](fuzzy-tda-and-windowing.md).

The front now spans point-native stats (`nnDistance`, `anni`), the points→grid
render bridge (`splatDensity`), grid windowing (`convolveSeparable`), composed
interpretable maps (`getisOrd` / `pointHotspots`), and fuzzy connectivity
(`fuzzyAdjacency`). Remaining catalogue rows are still `→` planned.

A **documentation site** (Astro Starlight) presents these as composable,
interpretable primitives: `docs-site/` (`pnpm --dir docs-site build`).

See also [`gpu-primitives-toolbox.md`](gpu-primitives-toolbox.md) (the
wavelet/image-signal front), [`fuzzy-tda-and-windowing.md`](fuzzy-tda-and-windowing.md),
and the [MuSpAn paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627195v1.full.pdf).
