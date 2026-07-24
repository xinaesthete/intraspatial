# Cell-type spatial statistics

MuSpAn-style statistics over cell centroids — the topographical correlation map (TCM), the cross
pair-correlation function (cross-PCF), and the N-way association matrix — formulated so that the
expensive part is a **render**, not a neighbour search.

This document is the derivation and the map of the code. The demo is `playground/cellstats.html`;
the implementation plan and its open questions are in [`muspan-cell-stats-plan.md`](./muspan-cell-stats-plan.md).

## Sources

The statistics implemented here are **not ours**. They are taken, equation by equation, from:

> Weeratunga P, Denney L, Bull JA, Repapi E, *et al.* **Single cell spatial analysis reveals
> inflammatory foci of immature neutrophil and CD8 T cells in COVID-19 lungs.**
> *Nature Communications* **14**, 7216 (2023).
> [doi:10.1038/s41467-023-42421-0](https://doi.org/10.1038/s41467-023-42421-0)
> — local copy: `docs/natureCovid-TCMetc-s41467-023-42421-0.pdf`. Equations 8–14 define the
> cross-PCF and the TCM; that paper also gives the SpOOx pipeline and the MDV viewer.

and, for the wider toolbox this is measured against:

> Bull JA, Moore JW, Mulholland EJ, Leedham SJ, Byrne HM. **MuSpAn: A Toolbox for Multiscale
> Spatial Analysis.** bioRxiv (2024).
> [doi:10.1101/2024.12.06.627195](https://doi.org/10.1101/2024.12.06.627195)
> — local copy: `docs/biorxivMuSpAn2024.12.06.627195v1.full.pdf`.

Our contribution is the **formulation and the implementation** — recognising the statistic as two
splats, generalising the mark kernel, and getting it onto the GPU with a parity oracle — not the
statistics themselves.

## Scope: what has been measured, and what has not

Worth stating plainly, because two different "windowing" arguments live nearby and it would be easy
to read one as evidence for the other.

**Measured** (§2): the effect of the **mark kernel** `K_r` — the neighbourhood weighting *inside*
one statistic — on discrimination, tie rate and stability, over a **single fixed ROI**. The finding
is specific: smoothing buys little discrimination but removes a large amount of discretisation and
parameter sensitivity.

**Not measured, and not claimed here**: whether computing a statistic over a **grid of quadrat ROIs**
— MuSpAn's usual practice — differs materially from sweeping a smooth window over the domain. That
is a different axis entirely: it is about how the domain is *sampled*, not about how a neighbourhood
is *weighted*. The toolbox's windowing argument
([`fuzzy-tda-and-windowing.md`](./fuzzy-tda-and-windowing.md), and the docs-site "Windowing, not
quadrats" page) asserts a position on it; nothing in this codebase tests it.

The comparison that would settle it is sketched below (§4, "ROI and edge effects") and needs a
per-quadrat baseline, a swept-window counterpart, and a **grid-phase sweep** — slide the quadrat
origin across its own unit cell and see whether any conclusion flips, not merely whether the
variance is larger. Intended to be run against the COVID-19 lung data from the source paper, so the
baseline is the published one rather than a synthetic stand-in.

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

## 2. The mark kernel is a free choice, and the paper's is the least smooth

This section is about `K_r`, the neighbourhood weighting inside eq 9 — **not** about how the ROI is
divided (see *Scope*, above). Because eq 9 is a sampled KDE, the hard disk is not special — it is `n = 0` of a family
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

### The Gram form

The rasterised form is `src/spatial/gram.ts` (f64 reference) and `src/gpu/spatial/gramMatrix.ts`
(render + matmul). Splat each of the K channels through a unit-mass kernel `J` to get `M = J ⊛ R`,
and the whole K×K matrix is one product:

```
C = M Mᵀ,     C_ab = ∫ M_a M_b = Σ_{i∈a} Σ_{j∈b} w_i w_j · (J ⊛ J)(x_i − x_j)
```

— the same pairwise sum, with the hard `1[d < r]` replaced by the smooth `(J ⊛ J)(d)`, whose
support is **2r**, not r. Normalisation is kernel-agnostic and has no `πr²` in it at all:
`g_ab = C_ab · |ROI| / (W_a · W_b)`, where `W` is total mark mass. CSR gives exactly 1 (tested for
every kernel).

**The cost argument.** The bucket-grid path walks every neighbour inside `r`, so it is
`O(n · ρ · πr²)` — *quadratic in the radius*. The raster path is `O(K·P)` to splat and `O(K²·P)` to
multiply, and **neither term contains r**: a bigger radius only spreads each quad further. There is
therefore a crossover past which the raster form wins outright, and the paper's regime (r = 100 µm,
bins to 300 µm) is on the far side of it. The trade is exactness — the counting path is integer
arithmetic and so is *exactly* the CPU statistic, whereas this is a quadrature accumulated in f32,
measured at **< 2e-3** relative against the f64 oracle.

**Cell types are the one-hot case of a general mark.** Nothing in the derivation uses the fact that
a cell has exactly one type; `R`'s rows are arbitrary non-negative per-cell weights. Hand it a gene
column from an AnnData `X` instead and the identical code computes a spatially-smoothed gene–gene
co-expression matrix. This is the mark cross-correlation function of point-process theory, of which
the cell-type cross-PCF is a special case. See §12.

### Why the eigen-projection needs the Gram form

An eigendecomposition reads as a decomposition of variance only if the matrix is positive
semi-definite, and **symmetry is not definiteness**. This is worth stating plainly because the
plan's §7 proposed "eigenvectors of the symmetrised g-matrix", and symmetrising is not the missing
ingredient — `crossPCFMatrix`'s `g` is *already* exactly symmetric.

Two separate things cost definiteness:

1. **The mark kernel.** In operator form the matrix is `C = R K Rᵀ`, which is PSD iff the kernel is
   positive-definite — iff its Fourier transform is non-negative (Bochner). `kernelSpectrum.ts`
   measures the 2-D radial transform of the whole family and **none of them qualifies**:

   | kernel | min `K̂(z)/K̂(0)` | at `z = kr` |
   |---|---|---|
   | top-hat (the paper's) | **−13.2%** | 5.14 |
   | Epanechnikov | −5.9% | 6.38 |
   | quartic | −2.9% | 7.58 |
   | triweight | −1.6% | 8.78 |
   | gaussian (3σ) | −0.13% | 12.5 |

   Smoothness shrinks the violation monotonically and never removes it; the truncated Gaussian is
   near-PD and would be exactly PD untruncated, so what that row measures is the cost of compact
   support, not of shape. The Gram form escapes this for free: its effective kernel is `J ⊛ J`,
   whose transform is `|Ĵ|² ≥ 0`, positive-definite whatever `J` was.

2. **The normalisation — and this is the one that actually bites.** `g` divides by per-channel mass
   and drops self-pairs, removing exactly the diagonal dominance that would otherwise carry the
   matrix. On self-clustering populations it usually stays PSD *by accident*. Interdigitated ones
   destroy it: two types alternating on a lattice give `g = [[0, 2.09], [2.09, 0]]` at the pitch
   radius — eigenvalues ±2.09, maximally indefinite. That is `crossPCFMatrix`, the published
   statistic, with no Gram form involved.

So the modes are taken from **`corr`**, the centred and standardised spatial correlation of the
smoothed channel densities — a Gram matrix of real vectors, hence PSD *structurally*: exactly, for
any kernel, and even in f32 (asserted on the GPU path). Its diagonal is exactly 1 and its trace
exactly K, so `λ_k / K` is a genuine variance share. `g` remains the right thing to report and to
draw a network from; its spectrum simply is not a variance decomposition.

Mode `k` is a signed weighting over channels, and `projectMode` renders it as a pixel field
`y_k(p) = Σ_a v_ka (M_a(p) − μ_a)/σ_a` — a recombination of rasters already in hand, so the map
costs no re-splatting and no second neighbour search. An exact identity pins the three pieces
together: **the spatial variance of the projected field equals its eigenvalue**.

### The terrain, and the metric over the mode axis

`gramTerrain.ts` draws the same resident rasters as a lit, displaced surface. Height is a fourth
channel that costs none of the three the colour carries — the eye reads a shaded surface as geometry
rather than as colour — and it is driven by a mode, or by similarity to a sampled point. Nothing is
re-splatted and nothing is read back: the grid mesh samples the buffer the matrix was reduced from
in its *vertex* shader, so a camera move is a redraw of data already on the device.

The **wand** is where the eigen-decomposition pays for itself twice. Click a pixel and its
standardised channel vector `z_ref` becomes a reference; "where else looks like this?" is then a
distance in channel space, and the honest one is Mahalanobis, because two channels that always
co-occur should not count as two independent pieces of evidence. With `corr = V Λ Vᵀ`,

    d²(x) = Δzᵀ corr⁻¹ Δz = Σ_k (Δy_k)² / λ_k,      Δy = Vᵀ Δz

— project onto the co-location modes, then divide each by its own variance. The shader takes the
whitening matrix `A = Λ^{-1/2} Vᵀ` truncated to the leading `m` modes and computes `d = |A Δz|`, so
one uniform spans both ends: `m = K` is exact Mahalanobis and noisy (the trailing modes have tiny
`λ`, and dividing by them amplifies whatever they hold), while `m = 3` is distance in precisely the
space the colour is drawn from, so "looks similar" and "is selected" agree by construction. That is
ADR-0015's metric-over-an-open-axis at its full-metric end, and the Gram form is what makes the
full-metric case computable at all.

The selected region is **outlined in the flat map**, at the level set `d = tolerance` — the same
distance where the terrain's similarity ramp reaches zero, so the boundary and the shading are two
readings of one number rather than two settings to keep in step. The distance itself lives in
`similarityWgsl.ts` and both shaders call it; this is the one snippet that *must* be shared rather
than merely kept aligned, because if the two computed `d` even slightly differently the outline
would enclose a region the colour disagrees with and the picture would give no clue which was wrong.
Outline and rule lines are deliberately different colours: the outline is the similarity hue and
means "what got selected", the lines are white and mean "where you sampled".

The sample is **dragged, not clicked**: holding the pointer down on the map moves the reference and
both views follow. Sampling is a dispatch plus a buffer map, so it cannot keep up one-to-one with
pointer events and must not queue — fifty pending readbacks would land in order and repaint for
positions the pointer left long ago. The loop holds only the latest position and runs one sample at
a time, dropping intermediate positions rather than falling behind, and re-checks after each
readback so the final position is always the one sampled. That costs about 100 ms per
sample-and-redraw at a 672×219 raster, which a drag absorbs.

Where the sample came from is drawn in **both** views, by one shared WGSL snippet
(`markerWgsl.ts`, on the same principle as `kernelWgsl.ts`) so the two marks cannot drift apart. It
is a pair of full-span rule lines rather than a ring: a ring has to be found before it can be read,
and on the terrain it is a locus in the surface's XY, so over steep relief it drapes down a
near-vertical face and stops reading as a ring at all. Lines are found immediately, and draping them
is a feature — each is the surface's profile along one axis through the sample. Their width is taken
from the per-axis screen-space derivative, because a fixed model-space band smears across a wall
where model XY barely changes per pixel.

### Permutation envelopes: what the spectrum is worth

`45% of the spatial variance in mode 1` is not a claim until you know what the same cells would give
with no spatial arrangement at all. The null is **random labelling** — every cell stays exactly
where it is, the marks are shuffled between them (`src/spatial/permute.ts`). CSR is the wrong null
here and would be a straw man: tissue is nowhere near homogeneous, so a CSR test rejects for every
pair and has detected only that the section has anatomy.

**One permutation, shared across all channels.** Permuting each channel independently destroys
within-cell co-expression as well as geography, so the test would reject on cells that co-express —
which has no spatial content and is exactly the confound `selfTerm` documents. A single shuffle
moves each cell's whole profile together, leaving co-expression exact and destroying only the
geography. For one-hot types the two are the same; for `X` it is the whole ball game, and
`permute.test.ts` pins it by comparing the multiset of per-cell profiles before and after.

**The mean of the null is analytic and free.** Splitting `C_ab` at `i = j` and taking the
expectation over a uniform permutation separates the marks from the geometry completely:

    E[C_ab] = φ(0)·S_ab + Φ·(W_a·W_b − S_ab) / (n(n−1)),     φ = J ⊛ J

where the only geometric term is `Φ = Σ_{i≠j} φ(x_i − x_j) = ∫(J⊛ρ)² − n·φ(0)` — one splat of all
cells, whatever the simulation count. This is not an optimisation, it is the check that the shuffle
is uniform: a Monte Carlo mean that misses it means the permutation is wrong, and a biased
simulation produces a perfectly plausible envelope in the wrong place. Both are tested against each
other to 3%, for one-hot and weighted channels.

**The test is a global rank envelope, not pointwise quantiles.** Pointwise 2.5/97.5% bands are a 5%
test at each mode applied at K modes at once; measured here on null data at `d = 8`, that rejects
**20%** of the time at a nominal 5%. The global construction (Myllymäki et al. 2017) ranks whole
curves, so the multiplicity is handled by construction. Two implementation traps were hit and are
recorded in `envelope.ts` because both look correct: the paper's plain global rank `min_r R(r)` is
too coarse — ties at rank 1 gave a curve that was most extreme at every point p = 0.06 instead of
0.01, and shrank the band until it rejected 29% of the time — so the ERL refinement is not optional;
and building the band from the hull of the *simulated* curves only, rather than pooling the observed
in with them, over-rejects at 21%. The version that ships has a measured rejection rate inside
[2.6%, 7.4%] at a nominal 5% over 1000 replicates, and that test is the reason to trust the band.

**The first real result inverts the naive reading.** On the Xenium selection (8 genes, 900 µm range,
centre-25% window, 39 permutations): the observed spectrum is *outside* the null band with the
smallest attainable p, but mode 1 is **below** it — 45% observed against 66% under random labelling
— and modes 2 through 7 are all above. Under random labelling every channel's smoothed density is
essentially the total cell density, so all channels become the same field, `corr` goes nearly
rank-1, and the null concentrates almost everything in mode 1. Real tissue *differentiates* the
channels, so the variance spreads. Read alone, "mode 1 carries 45%" would have been quoted as
evidence of strong structure; against its null it is the opposite of remarkable, and the finding is
in modes 2–7.

### The context image, and one camera for both

The store's own image can be blended under either view. It is **draped**, not laid on a plane
beneath the terrain: the surface samples it by its own model XY, so the anatomy and the statistic
are literally the same geometry and no alignment can drift between them — which is what makes one
orbit camera serve both without anything to synchronise. The flat map takes the identical blend
through the identical shared snippet (`imageOverlayWgsl.ts`), so moving between the two views does
not change what a given mix looks like.

Two decisions in that are load-bearing. The blend happens in **OKLab**, before the sRGB conversion:
mixing encoded sRGB darkens through the midpoint and drags hue, so a slider at 0.5 would not look
halfway, and the mode colours' meaning — distance in the field ∝ perceived colour distance — would
not survive partial blending. And the **coordinates** come from the element's stored transform, not
from the loader's `ms.placements[0]`, which is a demo-normalised axis-aligned placement that centres
each image on the origin for the scene editor's staggered layout. An element with no stored
transform is refused rather than stretched to the window: an overlay that is silently in the wrong
place is worse than no overlay.

One WGSL trap is worth recording because its symptom is so unhelpful. `textureSample` takes implicit
derivatives, so it may not appear in non-uniform control flow — wrapping it in an "is this pixel on
the image?" test compiles to an invalid shader module, and the only evidence is a cascade of
"invalid due to a previous error" with the actual message nowhere in it. Sample unconditionally and
carry the test in the blend weight. `compileShader` in `src/gpu/device.ts` now asks for
`getCompilationInfo()` and throws with the WGSL diagnostic, so the next one of these names itself.

Two things this surfaced that are worth keeping. **A whitened distance is in units of mode σ**, so
the useful saturation span is ~1, not the 3 the first version defaulted to — above 2 nearly all
tissue reads as similar. And the mode views make **under-resolution** visible: if the splat radius
is fewer than about two raster pixels, the map and the terrain show aliasing rather than the
smoothed field, while the statistic itself stays perfectly well defined. `cellmodes.html` reports
pixels-per-radius and warns below 2, because nothing else in the numbers gives it away.

### ROI and edge effects

`crossPcf.ts` and `tcm.ts` still run **Mode 1**: a fixed ROI, a *global* `ρ_B`, and full-disk /
full-annulus areas. This is exact for anchors at least `r` inside the ROI and biased near the
boundary. Permutation envelopes remain unbuilt on every path.

**The Gram path has the viewport apron** (`src/spatial/gram.ts`), and the shape of the fix is worth
recording because it is smaller than it was designed to be. Two rectangles had been conflated: the
**window** (`bbox` — what is integrated, standardised and drawn) and the set of points that reach
it. `gramMatrix` now splats every point handed to it, wherever it lies, but counts `mass` only
inside the window, so `ρ_a` is estimated window-locally as `W_a(A)/|A|`. That is the whole
correction: under CSR on a window interior to the data, `g` is 1 at every radius, against the
uncorrected ladder of 0.965 (r=5) → 0.892 (r=14) → 0.814 (r=25) on the same points measured over
their own extent.

The first implementation also dilated the *raster* by `r` and integrated over the interior. Measured
against the plain version it changed nothing to four decimal places, at every radius, and it was
removed. The reason: `M_a(x)` for `x` in the window depends on points within `r` of the window, and
those points deposit onto the window's own pixels whether or not any pixels exist beyond them — the
splat's footprint is clipped to the raster, never to the point set. **So the apron is a fact about
which points you supply, not about how many pixels you rasterise**, which in a tiled reader is a
halo-fetch requirement: to measure a window, read the points covering `bbox ⊕ radius`.

What the apron cannot do is invent data. A window at the tissue edge has nothing outside it, so
`apronCoverage` reports the ring density relative to the window's own — 1 means the correction had
real cells to work with, 0 means this `g` still carries the full deficit. On the Xenium strip in
`cellmodes.html` that reads 0 for the full extent, 0.07 for a window inset from the bounding box
(whose margins are empty, because the section is not rectangular) and 1.49 for the centre 25%.

Note also which numbers move: the apron changes `mass`, so it changes `g` alone. `corr` and the
co-location modes never divide by mass and are untouched by it.

**One global number per pair is also the limit of what is spatially resolved here.** Γ is a map;
the cross-PCF and the N-way matrix are single values over the whole ROI. So the association
statistics are neither per-quadrat (MuSpAn's practice) nor swept — the comparison between those two
has not been built, let alone run. Stated as the concrete next piece of work:

- a **per-quadrat baseline**: partition the ROI on a lattice of side `q`, compute the composition /
  pair counts per quadrat. This is the published practice and the thing to match.
- a **swept counterpart**: the Gram-matrix form above, evaluated at every pixel through a window —
  a per-quadrat statistic is just that box-filtered *and decimated onto a stride-`q` lattice*, and
  those are two separable defects (leakage, and observing only at lattice points).
- a **grid-phase sweep**: slide the quadrat origin across its unit cell and recompute. The spread is
  pure artefact — the data did not move. The number worth reporting is not the variance but whether
  the phase ever **flips a conclusion**: does a pair read as clustered under one origin and excluded
  under another? A null result there is as publishable as a positive one, and should be reported as
  such.

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

## 8. Presenting: one path, no readback

Every field on screen arrives the same way: **the pass that computes it leaves it in an r32float
texture, and `paintFieldTexture` paints that texture to the canvas through a colour LUT.** Nothing is
downloaded. That holds for the op-graph too: `splatDensity` is texture-resident (ADR-0017's 2026-07-24
amendment), so `pullResident` hands the KDE panels a texture the paint pass consumes directly — the
graph feeds the display path with no copy and no adapter in between. This is ADR-0017 invariant 4 applied to display — a field that is computed on the GPU
and then *looked at* never needs to reach the host.

The alternative, which this replaced, costs per panel per frame: the GPU→CPU round trip (a pipeline
flush, not a memcpy), a JS loop over every pixel to strip the 256-byte row padding, and another JS
loop to write an `ImageData`. Measured on this demo, that was **~55 ms for a pair of KDE panels
against ~4 ms for the Γ render** — the display path cost more than ten times the statistics.

Auto-scaling stays on-device too: an atomic max over |v|, using `atomicMax` on the *bit pattern*.
That is exact rather than approximate, because for non-negative floats IEEE-754 bit order is numeric
order. Only the finished 256-entry LUT crosses to the GPU; building the ramp (OKLCh, with a gamut
bisection) stays on the host where it belongs.

With the readback gone the hover path costs **under a millisecond of main-thread time per pair**, so
the debounce that used to gate it — a relic of when Γ was seconds of CPU work — is gone. Requests
**coalesce** rather than queue: the newest pair wins and the intermediates are dropped, since
mousemove fires far faster than a GPU round trip and a queue would spend its time rendering pairs
the mouse left long ago. WebGPU has no cancellation, so a request already in flight finishes; at
worst that is one stale frame.

> **A bug worth recording.** The KDEs first went through the op-graph's `pullResident` *with a memo*,
> releasing the lease after painting. A later cache hit then returned a `ResidentBuffer` whose
> pooled buffer had been recycled — so Γ updated correctly on hover while the KDE panels showed
> stale content. Two ownership models for the same job is what made it possible; one path removed
> the class of bug along with the fork. If a resident buffer is ever cached, the memo must own the
> lease.

Note that these timings are **submission**, not GPU completion: the point is that the UI never
blocks on the GPU, and the readout says so rather than quoting a flattering number.

---

## 9. Getting data in

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

## 10. Where things are

```
src/spatial/kernels.ts         radial kernel family; closed forms validated against quadrature
src/spatial/tcm.ts             eq 9-14 exact (reference oracle + bucket-grid path)
src/spatial/tcmKernel.ts       continuous generalisation over any kernel (f64)
src/spatial/kernelAnalysis.ts  ground-truth scene, AUC / tie / stability scoring
src/spatial/pcf.ts             cross-PCF and the N-way matrix
src/spatial/gram.ts            the Gram form C = MMᵀ, g, corr, modes and mode maps (f64 reference)
src/spatial/eigenSym.ts        Jacobi symmetric eigensolver + psdDefect
src/spatial/kernelSpectrum.ts  the kernels' 2-D Fourier transforms — the positive-definiteness table
src/spatial/bucketGrid.ts      CSR neighbourhood index (counting sort)
src/spatial/cellCsv.ts         CSV parsing / inspection / grouping
src/spatial/ngffTransform.ts   NGFF coordinateTransformations → 2-D affine + physical unit

src/gpu/spatial/tcmRender.ts   the two render passes  ← the interactive path
src/gpu/spatial/paintField.ts  texture → canvas through a LUT; the single display path
src/gpu/spatial/tcm.ts         exact compute path (TGSL), the GPU parity oracle
src/gpu/spatial/crossPcf.ts    GPU cross-PCF + N-way matrix (WGSL, integer atomics)
src/gpu/spatial/gramMatrix.ts  GPU Gram form: one splat per channel + one reduction dispatch
src/spatial/envelope.ts        global rank envelopes (ERL); the coverage test is the point
src/spatial/permute.ts         the random-labelling null + its analytic mean
src/gpu/spatial/gramEnvelope.ts  N permuted spectra -> a banded scree chart
src/gpu/spatial/gramModes.ts   the flat OKLab mode map, one fragment pass, no readback
src/gpu/spatial/gramTerrain.ts the displaced surface, the orbit camera, and the wand's metric
src/gpu/spatial/markerWgsl.ts  the sample rule lines in WGSL, shared by the map and the terrain
src/gpu/spatial/similarityWgsl.ts  the wand distance + selection boundary, shared (must be, not just aligned)
src/gpu/spatial/imageOverlayWgsl.ts  the OKLab image blend, shared by the map and the terrain
playground/src/datasource/imageContext.ts  one pyramid level -> one RGBA texture + world->UV
src/gpu/spatial/kernelWgsl.ts  the kernel family in WGSL, shared by tcmRender and gramMatrix
src/color/ramps.ts             OKLCh diverging / sequential ramps

playground/cellstats.html      the demo
playground/src/cellStatsMain.ts
playground/src/datasource/cellTable.ts   store discovery + read
playground/src/datasource/cellCsv.ts     CSV → CellTable
```

---

## 11. Known limits

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
- **Mode 2 is only half built.** The Gram path has window-local mass, the viewport apron and a
  permutation envelope on its spectrum (§4); `crossPcf.ts` and `tcm.ts` still have none of the
  three, and no envelope exists for `g` or for the cross-PCF curves.
- **The envelope's remaining cost is per-INSTANCE, and the obvious optimisations are not the ones
  that pay.** 39 permutations of a 162k-cell, 8-channel selection take 3.2 s. Getting there was two
  steps, and a third that was measured and abandoned:
  - `channelPermuter` reusing its scratch buffers: 340 → 90 ms per realisation. The allocating form
    threw away ~10 MB each time and most of the difference was collection, not GPU work.
  - Culling zero-weight marks in the splat's vertex stage: 90 → 82 ms. A zero weight adds nothing
    but otherwise rasterises its whole kernel footprint; expression data is mostly zeros.
    `gramMatrix.gpu.test.ts` pins that a zero-weight cell is indistinguishable from an absent one in
    `c`, `g`, `mass` and `selfTerm`, which is what makes the cull a no-op rather than an
    approximation.
  - **Resident positions plus a permutation buffer: measured at ~1.5%, not built.** The intuition is
    that re-uploading the packed `[x, y, w]` every realisation is waste. It is, but it is small
    waste: with `n = 1` and the same 8 channels and raster, the entire fixed cost — host packing,
    upload, both submits, all three readbacks — is **2.1 ms against 141 ms**. Adding a second input
    mode and shader variant to save part of 2.1 ms is not worth the two lifetime contracts it
    introduces.
  - **Compaction — draw only the non-zero cells — was built, measured, and reverted: it is 3×
    SLOWER here.** The reasoning that led to it was that a culled zero still costs a vertex
    invocation and a primitive, so a run splats far more instances than it needs. That per-instance
    cost is real and monotone in isolation (all-offscreen instances, no fragments: 1.3 M cost ~28 ms
    over a ~4 ms floor). But it is not the bottleneck, and removing it makes things much worse. In a
    controlled A/B at a realistic 5% occupancy — the same in-window splats either way, so identical
    fragments — the cull (draw all 1.3 M, 95% collapse offscreen) and host compaction (draw only the
    ~65 k in-window) run, sustained over 39 realisations:

    | path | 39× wall clock | per realisation |
    |---|---|---|
    | vertex cull (draw all, collapse zeros) | **1.3 s** | 33 ms |
    | host compaction (draw non-zeros only) | **3.9 s** | 101 ms |

    The mechanism is the tile-based GPU, not per-instance work: cost is dominated by how *scattered*
    the in-window coverage is, and it is **non-monotone** in occupancy — 1.3 M dense instances splat
    in 110 ms while 660 k scattered ones take 235 ms. Drawing the full instance list alongside the
    splats keeps the identical in-window fragments roughly an order of magnitude cheaper (the ~1 ms
    they add to the cull vs the ~80 ms they cost alone), consistent with a clock/batching regime that
    the bulk draw triggers and the bare scattered draw does not. It is not fully root-caused, but the
    direction is unambiguous and reproduced four ways (occupancy sweep, two offscreen-coordinate
    A/Bs, and the sustained loop above), so the cull stays and there is no case for the device-side
    prefix-sum + indirect-draw version either — it would only draw fewer scattered primitives, which
    is the slow direction. Had it shipped, the permutation would still have stayed on the host
    regardless: Fisher–Yates on 162k indices is ~1 ms, while the GPU-shaped alternatives are a full
    sort or a keyed pseudorandom permutation, and the latter is **not** uniform over all `n!` — the
    assumption the test's exactness rests on.

  The earlier reading of these numbers credited GPU per-instance splatting with the whole 111 → 69 ms
  occupancy swing and called compaction "the lever." That was wrong: isolated, host packing is ~10 ms
  and the per-call fixed cost ~3 ms, the splat is insensitive to raster size, and the real cost is the
  scattered-coverage term above — which neither the cull nor compaction moves. Chasing it means
  changing the *raster*, not the instance list: a coarser splat, or a windowed formulation that
  replaces the global integral (§ the Gram-form limit below).
- **`crossPCFMatrixGpu` is not wired into the demo**; the matrix still runs on the CPU (one batched
  pass, fast enough at Leap034 scale).
- **The Gram form is global, not swept.** `C = MMᵀ` integrates over the whole raster, so it is one
  K×K matrix per view, exactly as the pair-counting path is. The *windowed* version — replace the
  integral with a convolution of the products `M_a·M_b` and you get a K×K matrix **per pixel**, an
  open-axis tensor field — is the thing plan §7 asks for and is not built. It is what would make
  the quadrat-vs-swept comparison below runnable, and what the "distance from a reference
  co-location profile" reduction needs.
- **The image overlay is one pyramid level, not a tiled pyramid.** `imageContext.ts` fetches the
  finest level whose long side fits a 2048 budget and composites it once; zoom past that and the
  overlay goes soft. `tileRenderer.ts` does proper LOD streaming for the scene editor and is not
  wired in here — the mode views draw one fixed analysis window, so there is no camera-driven LOD
  problem to solve, only a resolution ceiling.
- **The overlay's registration has not been checked against a fiducial.** It uses the element's own
  stored transform (element ∘ dataset, straight from sd.js) into the same coordinate system the
  table resolves into, and the flat map and the terrain agree with each other by construction —
  but "the two views agree" is not "the image is where the cells are". An element carrying no
  stored transform is refused outright rather than stretched to fit.
- **Any substantial CPU work in a `*.gpu.test.ts` process crashes the Dawn fork before vitest
  flushes results.** Bisected while building `gramMatrix.gpu.test.ts`: a bare `Float64Array` churn
  loop with no GPU code involved kills it just as reliably as running the CPU oracle, while the
  same GPU calls with trivial assertions pass repeatedly. The budget is severe — a CPU oracle at a
  32² raster survives, 48² does not. This is the same fragility as the `tcmRender` entry above, but
  the trigger is *host* allocation churn rather than render/readback cycles. The workaround is to
  bake oracle values in as constants (see that file's header); the general rule is that GPU test
  files must stay GPU-only. Not root-caused.
- **Quadrats vs swept windows is UNTESTED.** The association statistics are global, so the
  comparison the toolbox's windowing argument rests on has not been run here — see *Scope* at the
  top. Do not cite §2's kernel measurements as evidence for it: they are a different axis.

---

## 12. Weighted marks: gene expression as a channel

The Gram form takes per-cell weights, so a gene's expression column substitutes directly for a
one-hot type indicator — `channelsFromExpression` builds the channels, everything downstream is
unchanged, and the GPU splat already carries the weight per instance. This is tested both ways: a
one-hot `X` reproduces the cell-type matrix exactly, and doubling a channel's weights scales `C` by
4 while leaving `g` invariant.

**The confound that makes this non-trivial.** The pair sum includes `i = j`. For disjoint cell-type
channels that self term only touches the diagonal, but when every cell carries a weight in *every*
channel it lands in every entry, contributing `(J⊛J)(0) · Σ_i w_a(i) w_b(i)` — within-cell
co-expression with no spatial content at all, wearing the costume of co-location. Two genes
perfectly co-expressed in cells spread far apart still produce a large raw `C_ab`. It is reported
as `selfTerm` and subtracted in `g` (which is what `crossPCFMatrix`'s `j != i` does); `corr` cannot
subtract it and stay PSD, so for the *modes* the honest control is a permutation null — shuffle the
marks among cells and keep the positions — which is the same machinery Mode 2 needs.

**Scale.** Cost is `O(G·P)` to splat and `O(G²·P)` to reduce, so this is a *selected genes*
feature, not an all-genes one: ~50 genes at 512² is fine, 2000 is not. That matches ADR-0005's
existing position (sparse gene columns are a selection mechanism; never densify the full matrix).

**What is missing is only the loading.** Nothing in the repo reads `X`, `var` or `layers` today —
`playground/src/datasource/cellTable.ts` reads `obsm/spatial` and one `obs` column, and that is
all. The cheapest route needs no new dependency: the `zarrextra` tree the table reader already
holds contains `tables/<name>/X`, either as a dense array or as the `X/{data,indices,indptr}` CSR
triple (with `encoding-type` and `shape` in the group attrs, reachable via the existing
`symbolAttrs` helper), and `var/_index` gives the gene names through the existing `readStrings1D`.
The alternative is `@spatialdata/core`'s `getAnnDataJS()`, which exposes a `SparseArray` with a
slicing API — already a transitive dependency, but not currently imported for tables.
