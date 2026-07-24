# Cell-type spatial statistics

MuSpAn-style statistics over cell centroids — the topographical correlation map (TCM), the cross
pair-correlation function (cross-PCF), and the N-way association matrix — formulated so that the
expensive part is a **render**, not a neighbour search.

This document is the derivation and the map of the code. The demo is `playground/cellstats.html`;
the implementation plan and its open questions are in [`muspan-cell-stats-plan.md`](./muspan-cell-stats-plan.md).

Source for the statistics: *Bull et al.*, Nature Communications 14, 6516 (2023) —
`natureCovid-TCMetc-s41467-023-42421-0.pdf`, equations 8–14.

---

## 1. TCM is two KDE splats

The paper defines the TCM in three steps. For cell types A and B, a neighbourhood radius `r`, a
bandwidth `σ`, and an extremity threshold `α` (the paper uses 5):

```
m_ab(a) = |{ b ∈ B : |x_a − x_b| < r }| / (ρ_B · πr²)      (eq 9)   the mark
M_ab(a) = 𝔐(m_ab(a), α)                                    (eqs 10–13) the transformed mark
Γ_ab(x) = Σ_a M_ab(a) · G_σ(x − x_a)                       (eq 14)  the map
```

Read literally that is a neighbour search. It is not one. Write the point sets as measures,
`α = Σ_a δ(x−x_a)` and `β = Σ_b δ(x−x_b)`, and let `T_r` be the disk indicator. Then

```
|{ b : |x−x_b| < r }| = (T_r ⊛ β)(x)
```

— the neighbour count **is** a KDE of B with a top-hat kernel, evaluated at `x_a`. And eq 14 is
literally a weighted splat of A:

```
Γ_ab = G_σ ⊛ (M · α)
```

So the whole statistic is **two additive splats with a pointwise nonlinearity between them**:

| pass | what | where |
|---|---|---|
| 1 | `ρ̂_B = K_r ⊛ β` — B splatted through a radial kernel | fragment shader, additive blend |
| 2 | `Γ = G_σ ⊛ (M·α)` — A splatted, weight `M = 𝔐(ρ̂_B(x_a)/ρ_B, α)` | vertex fetch + fragment |

No bucket grid, no atomics, no `O(N_A · N_B)`.

### The one part that is not per-pixel

`𝔐` is nonlinear and eq 14 applies it **per cell, before smoothing**. The fully per-pixel field

```
Γ̃(x) = (G_σ ⊛ α)(x) · 𝔐( (K_r ⊛ β)(x) / ρ_B )
```

is *not* Γ. `G_σ ⊛ (𝔐(m)·α) ≠ (G_σ ⊛ α)·𝔐(m)` — the difference is a Jensen gap, widest exactly
where a neighbourhood mixes strongly clustered and strongly excluded A cells, which is where the map
is interesting. So the mark must be evaluated **once per cell**.

That is why `computeTcmRender` fetches the mark in the **vertex** stage, one texture read per
instance. It is a point sample of a field, not a search. Moving that fetch to the fragment stage
would be faster still and would quietly compute a different statistic.

---

## 2. The kernel is a free choice, and the paper's is the worst one

Because eq 9 is a sampled KDE, the hard disk is not special — it is `n = 0` of a family
(`src/spatial/kernels.ts`):

```
K_n(u) = (n+1)/(πr²) · (1 − |u|²/r²)ⁿ        for |u| < r
```

| n | kernel | boundary |
|---|---|---|
| 0 | top-hat / disk (**the paper**) | discontinuous |
| 1 | Epanechnikov | continuous; AMISE-optimal |
| 2 | quartic / biweight | C¹ |
| 3 | triweight | C² |
| — | Gaussian, truncated at 3σ | non-compact comparison |

`n` *is* the smoothness: K and its first `n−1` derivatives vanish at the support boundary. All
kernels carry unit mass, so `m = (K_r ⊛ β)(x_a)/ρ_B` is kernel-agnostic — with the top-hat it is
exactly eq 9 (asserted to 1e-12), and swapping the kernel changes the estimator, never the meaning.

### Compare at matched scale, never at equal radius

`μ₂ = r²/(n+2)` **shrinks** as the kernel smooths. A triweight at radius `r` probes a visibly
tighter neighbourhood than a top-hat at the same `r`, so comparing at equal `r` confounds "smoother"
with "more local" and hands the smooth kernels an unearned win. Everything goes through
`equivalentRadius`, and the demo's radius box is a **top-hat-equivalent** radius that is rescaled
per kernel.

### What smoothing actually buys — measured, not asserted

`src/spatial/kernelAnalysis.ts` builds a scene with known ground truth (A cells labelled by whether
they were *placed* in a B-rich patch) and scores each kernel at matched μ₂. Weak-contrast scene:

```
kernel         radius  AUC     tied   d/dr   jitter  R·μ₂
top-hat        60.0    0.9548  98.2%  4.01%  7.19%   0.1592
Epanechnikov   73.5    0.9607   0.0%  1.14%  4.13%   0.1415
quartic        84.9    0.9612   0.0%  1.02%  4.02%   0.1432
triweight      94.9    0.9614   0.0%  0.99%  4.02%   0.1455
gaussian (3σ)  92.4    0.9609   0.0%  0.95%  4.18%   0.1545
```

- **Discrimination barely moves.** On an easy scene all five land within 0.0005 AUC. Only under weak
  contrast do the smooth kernels pull ahead, and only by ~0.006. This is the honest negative result
  and it is asserted as such in the tests.
- **The real cost of the hard disk is discretisation.** Its mark is a *count*, so 93% of A cells are
  tied on the strong scene and **99.8%** when B is sparse — the estimator cannot order them at all.
  Every smooth kernel: 0%.
- **Radius sensitivity.** A 2% nudge in `r` moves the top-hat 2.6–6.5% against ~1.0–1.6% for the
  smooth kernels. Two to five times more of the answer rides on a free parameter.
- **Positional robustness.** Jitter of 5% of `r` (segmentation / centroid error, a property of real
  data) moves the top-hat 5.6–11.4% against 4.0–6.5%.
- **R·μ₂** reproduces the textbook AMISE ordering, Epanechnikov minimal.

The discontinuity even bites numerically: rendering the top-hat, a pixel within a float ulp of
`|u| = r` flips by a **whole kernel quantum** between the f32 and f64 evaluations of the identical
formula. That is kept as a test rather than tuned away.

**Recommendation:** use the top-hat for parity with published MuSpAn numbers; use Epanechnikov or
quartic for anything you intend to interpret.

---

## 3. Three formulations, one parity chain

| formulation | where | cost | role |
|---|---|---|---|
| exact, per-cell counts | `src/spatial/tcm.ts` (CPU), `src/gpu/spatial/tcm.ts` (GPU compute) | `O(N_A · local density)` via a bucket grid | **parity oracle** — the number you quote |
| continuous, any kernel | `src/spatial/tcmKernel.ts` | same, f64 | the bridge; `TOPHAT` ≡ the paper |
| **render** | `src/gpu/spatial/tcmRender.ts` | two draw calls | **the interactive path** |

The chain each link is tested against:

```
computeTcmReference  ≡  tcmKernelField(TOPHAT)     exact, to 1e-12
tcmKernelField(K)    ≈  computeTcmRender(K)        raster sampling of the mark
```

The render path's only approximation is that the mark is bilinearly sampled from a
`markWidth × markHeight` raster rather than evaluated at `x_a`. `markWidth` is the knob that closes
it, and the demo's **check vs CPU oracle** button reports the residual on the data actually on
screen. Measured on Leap034 (30 550 A cells against 7 641 B cells, 512² mark raster):

> CPU exact 597 ms vs GPU 42 ms (**14.3×**) · max |Δ| / peak = **0.138%**

---

## 4. cross-PCF, and the N-way matrix

The single-radius cross-PCF is an inner product:

```
C_AB(r) = Σ_a Σ_b 1[|x_a−x_b| < r] = ⟨ α , T_r ⊛ β ⟩
```

`crossPCFMatrix` computes **all N² ordered pairs in one batched bucket-grid pass** rather than N²
separate calls; `crossPCFMatrixGpu` is the same statistic with the counting on the GPU (integer
atomics, so it is exact parity with the CPU, not an approximation).

The rasterised form generalises further and is not yet implemented: with `R` the N×P matrix of
per-type count rasters and `S = T_r ⊛ R`, the entire matrix is `C = R Sᵀ` — one matmul, multi-radius
for free, and the natural home for the eigen-projection idea (eigenvectors of the symmetrised
g-matrix are co-localisation modes). See plan §7.

### ROI and edge effects

Both statistics currently run **Mode 1**: a fixed ROI, a *global* `ρ_B`, and full-disk / full-annulus
areas. This is exact for anchors at least `r` inside the ROI and biased near the boundary. Mode 2 —
a viewport apron with window-local `ρ_B` and live permutation envelopes — is designed
(ADR-0018 §5) and not built.

---

## 5. Colour

Ramps are built in OKLCh (`src/color/ramps.ts`, on top of the existing `src/color/oklab.ts`; note
hues there are **radians**).

Γ is signed and is read by comparing its arms. A blue→white→red ramp interpolated in sRGB has arms
of markedly different perceived lightness — sRGB blue is far darker than sRGB red at the same
nominal distance from white — so equal clustering and exclusion do **not** look equally strong, and
the reader sees an asymmetry that is not in the data. Measured gap: >0.15 in L*.

The diverging ramp therefore derives lightness and chroma from `|t|` alone and lets **only the hue**
carry the sign. Equal magnitudes are then equally prominent by construction, and that is a test, not
a matter of taste. The sequential ramps hold the same lightness profile at every hue, so two cell
types' density maps stay comparable by brightness.

Out-of-gamut requests are resolved by **chroma reduction** (bisection at fixed L and h), not
per-channel clipping — clipping moves both hue and lightness, i.e. it corrupts exactly the two
dimensions the ramps use to carry meaning.

Two conventions worth knowing:

- Chroma **rises** to the top of a sequential ramp rather than peaking mid-way. A mid-peaking
  profile returns to zero at the top, rendering the densest region — the part anyone looks at — as
  flat white, losing the type's identity where it matters most.
- The demo inverts the diverging ramp's lightness (dark neutral, bright extremes) because on a dark
  page the neutral should recede. The symmetry property is untouched: L still depends on `|Γ|` alone.

---

## 6. Physical units

Every parameter these statistics take is a **length**. The paper's TCM uses a 100 µm neighbourhood
radius and a 50 µm bandwidth. Typing `100` against a store whose coordinates are camera pixels is a
silent error that changes the answer and looks perfectly fine, so the unit has to come from the data
where the data states it — and be visibly *unstated* where it does not.

A table is not a spatial element and carries no transform. What it carries is `region`: the name of
the element its rows annotate. That element **is** placed, and `obsm/spatial` centroids live in its
space, so its transform is the table's transform. `resolveTableSpace` performs that annotation walk
(the same one `MDV/python/mdvtools/spatial/conversion.py` does on the Python side), then
`resolveNgffXY` (`src/spatial/ngffTransform.ts`) reduces the element's `coordinateTransformations`
to a 2-D affine and reads the unit off the output axes.

Supported transform types: `identity`, `scale`, `translation`, `sequence`, `affine` (rotation and
shear included — a rotated element is not flattened to a scale). Axes are matched by **name**, not
position: a `c,y,x` scale of `[1, 0.5, 0.25]` must not read the channel scale as x. Anything else
(`byDimension`, the non-linear types) is recorded in `unsupported` and surfaced in the HUD rather
than silently approximated.

### "Unknown" is not "1"

`micrometresPer` returns `undefined` — never a default of 1 — for anything that does not name a
physical length. Two cases matter in practice:

- **`"unit"`** is what SpatialData writes when no unit was specified. It looks like a unit and is
  not one.
- **`"pixel"`** is a legitimate NGFF unit but not a physical length.

Conflating either with micrometres is exactly how a length gets reported in the wrong unit with full
confidence.

The demo therefore takes lengths in µm and shows a **µm/unit** box. When the store states a unit the
box is filled from it and readouts say `µm`. When it does not, the box is the user's *declaration*,
defaults to 1, and every readout says **`µm*`** — the asterisk is not decoration.

> **Leap034 states no scale.** Its axes read `unit: "unit"` throughout and every transform is
> identity, so `shapes/Leap034_imc_cell_shapes` resolves to identity with no physical unit. IMC is
> conventionally 1 µm/pixel, which would make the declaration correct — but the store does not say
> so, and the code will not say so on its behalf. This is a question for whoever produced the data.

---

## 7. Aspect ratio

Rasters are sized from the world box on an **area budget**, not as fixed squares (`viewDims`). A
square raster over a non-square ROI does two things, one cosmetic and one not: it stretches every
spatial view, and it makes world cells non-square, so the mark kernel is sampled at different
resolutions in x and y.

Leap034's ROI is **0.22 aspect** — three IMC blocks stacked vertically — so the square rasters were
stretching it by about 4.5×. Sizing the *longer* axis to the target is the obvious fix and a poor
one: it leaves 57 px across the short axis. Spending the same pixel count proportionally gives
121×542 for the same cost. Canvases then carry their own aspect, and the CSS caps display height
with `object-fit: contain` so a tall-thin ROI is letterboxed rather than squashed back.

---

## 8. Getting data in

Three sources, all producing the same `CellTable`, so nothing downstream knows the difference.

### SpatialData store

`listCellTables(url)` inspects the store rather than assuming: it enumerates `tables/`, reports
which have `obsm/spatial` centroids, and ranks each table's `obs` columns as candidate cell-type
columns. A store with one usable table is selected automatically; with several, the best guess is
pre-selected and the full list stays visible.

The cell-type column is **not standardised** — `cell_type_id`, `cell_type`, `phenotype`, `cluster`,
`annot_*` are all in the wild. The heuristic is deliberately transparent (`scoreColumn`) because it
only picks a default. Two rules earn their keep:

- A categorical with **fewer than two categories** is disqualified. Leap034's `annot_region` is
  exactly this — one category, i.e. it partitions nothing — and it out-scored `cell_type_id` until
  the rule existed.
- The `_id` penalty does not apply to names that already matched cell-type, so `cell_type_id` wins.

**AnnData categoricals carry names.** A categorical column is a group of `codes` + `categories`; the
category strings are read during discovery, so the UI shows *"T cell → Macrophage"* rather than
*"type 10 → type 3"*. A bare integer column cannot do this: the store simply does not say what the
numbers mean. (Leap034's `cell_type_id` is such a column — the mapping lives with whoever produced
the data.)

### CSV

`src/spatial/cellCsv.ts` — RFC-4180 parsing (quoted commas and newlines; a `split(",")` parser
corrupts such a file *silently*, shifting every later column), column inspection, and grouping. x/y
and the type column are guessed by header name, falling back to the shape of the data (coordinates
are numeric with many distinct values, a type column has few). Type values stay strings, so a CSV of
names works as well as one of integers.

A CSV carries no coordinate system, so the cloud is placed at identity in a system named after the
file — an honest "these are in their own space" rather than a fabricated registration (ADR-0018:
array-space and placed-at-identity are distinct states).

### Fixture

`syntheticCellTable()` — two Gaussian blobs through the identical path, for working offline.

---

## 9. Where things are

```
src/spatial/kernels.ts         radial kernel family; closed forms validated against quadrature
src/spatial/tcm.ts             eq 9-14 exact (reference oracle + bucket-grid path)
src/spatial/tcmKernel.ts       continuous generalisation over any kernel (f64)
src/spatial/kernelAnalysis.ts  ground-truth scene, AUC / tie / stability scoring
src/spatial/pcf.ts             cross-PCF and the N-way matrix
src/spatial/bucketGrid.ts      CSR neighbourhood index (counting sort)
src/spatial/cellCsv.ts         CSV parsing / inspection / grouping
src/spatial/ngffTransform.ts   NGFF coordinateTransformations → 2-D affine + physical unit

src/gpu/spatial/tcmRender.ts   the two render passes  ← the interactive path
src/gpu/spatial/tcm.ts         exact compute path (TGSL), the GPU parity oracle
src/gpu/spatial/crossPcf.ts    GPU cross-PCF + N-way matrix (WGSL, integer atomics)
src/color/ramps.ts             OKLCh diverging / sequential ramps

playground/cellstats.html      the demo
playground/src/cellStatsMain.ts
playground/src/datasource/cellTable.ts   store discovery + read
playground/src/datasource/cellCsv.ts     CSV → CellTable
```

---

## 10. Known limits

- **`src/gpu/spatial/tcmRender.ts` segfaults Dawn-on-Node's `atexit`** after a couple of
  render-plus-readback cycles in one test file. The assertions pass first — measured relMax 1.8e-4
  against the oracle in-process — but the fork dies before vitest flushes, so the run reports
  nothing. Caching texture views and bind groups reduced the live object count without fixing it.
  CI therefore keeps two smoke tests with teeth (mass conservation pins pass 1's normalisation and
  world mapping; sign structure pins pass 2), and the full parity sweep runs in the browser via the
  demo's oracle button. Not root-caused.
- **A declared scale is not a measured one.** Where a store states its unit, lengths are real µm;
  where it does not (Leap034), the µm/unit box is an assumption and everything is marked `µm*`.
  Nothing in the pipeline can tell the two apart for you.
- **`byDimension` and non-linear transforms are unimplemented** — reported, not approximated.
- **Mode 2 is unbuilt** — no window-local `ρ_B`, no edge correction, no permutation envelopes.
- **`crossPCFMatrixGpu` is not wired into the demo**; the matrix still runs on the CPU (one batched
  pass, fast enough at Leap034 scale).
- **The Gram-matrix / eigen-projection formulation of §4 is a plan, not code.**
