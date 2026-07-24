# MuSpAn cell-type spatial stats — cross-PCF + TCM (spec + op-graph plan)

Faithful reproduction target, à la MuSpAn. **Source:** the Nature Comms COVID-lung paper
`docs/natureCovid-TCMetc-s41467-023-42421-0.pdf` (Taylor-CCB-Group / SpOOx), Methods — cross-PCF eq (8),
TCM eqs (9)–(14). (MuSpAn software preprint `docs/biorxivMuSpAn2024.12.06.627195v1.full.pdf` also in
docs; primary PCF reference is Bull 2020.) Builds on the stream-B2 centroid ingestion (per-`cell_type_id`
point clouds with placement) currently landing.

## 1. cross-PCF  g_AB(r)  — eq (8)

For cell populations A, B with N_A, N_B cells at positions x_a, x_b:

```
g(r_k) = (1/N_A) · Σ_a Σ_b  I[r_k ≤ |x_a−x_b| < r_{k+1}] / ( ρ_B · A_{r_k}(x_a) )
```

- **Bins:** r_0 = 0, r_k = r_{k−1} + 10, k = 1..30 → 10 µm annuli over [0, 300) µm. Report `g(r=20)` (bin [20,30) µm ≈ contact distance).
- **ρ_B** = density of B in the ROI = N_B / |ROI|.
- **A_{r_k}(x_a)** = area of the annulus (inner r_k, outer r_{k+1}) centred at x_a **that falls within the ROI** — the edge correction (boundary cells have a smaller in-ROI annulus, so their pairs are up-weighted).
- g > 1 clustering, g < 1 exclusion, g = 1 CSR.

## 2. TCM  Γ_ab(x)  — eqs (9)–(14)

**Step 1 — per-A-cell mark** m_ab (eq 9), the edge-corrected observed/expected B-count within 100 µm:

```
m_ab = ( Σ_j I[|x_a − x_j| < 100] ) / ( ρ_B · A_100(x_a) )
```

where A_100(x_a) = area of the radius-100 µm disk at x_a clipped to the ROI. m_ab > 1 correlation, < 1 anti.

**Step 2 — transform to M_ab ∈ [−1, 1]** with α = 5 (eqs 10–13). Note the reciprocal symmetry
**M(m) = −M(1/m)** (dispersal and clustering on the same scale — this is why the exclusion branch is *not*
a naive linear mirror):

```
M_ab =  1                       if m_ab ≥ α
        (m_ab − 1)/(α − 1)      if 1 < m_ab < α
        (1 − 1/m_ab)/(α − 1)    if 1/α < m_ab < 1     ← reciprocal form (eq 12)
       −1                       if m_ab ≤ 1/α
```

(0 at CSR; +1 = ≥α-fold clustering; −1 = ≥α-fold exclusion. e.g. m=2 → +1/4, m=1/2 → −1/4.)

**Step 3 — weighted Gaussian splat of A** (eq 14), σ = 50 µm, compact support = 300 µm square/kernel:

```
Γ_ab(x) = Σ_a  ( M_ab / (σ√(2π)) ) · exp( −|x − x_a|² / (2σ²) )
```

Γ_ab ≫ 0 = A positively associated with B there; ≪ 0 = negatively. **Γ_ab ≠ Γ_ba** (kernels centred on A).

## 3. Op-graph translation

The user's intuition ("an operation over two KDE splats") is right, made precise:

**TCM** = splat-B → mark-per-A → transform → **weighted** splat-A:
1. `splatDensity(B, kernel = top-hat r=100, normalize = count)` → a B-count field; **sample it at the A-cell positions** (a gather) = the eq-9 numerator. (Or a direct O(N_A·N_B) cross-radius count.)
2. Divide by `ρ_B · A_100(x_a)` — needs ρ_B (= N_B/|ROI|) and the **ROI-clipped disk area** per A-cell.
3. `mark→M` elementwise transform (eqs 10–13) on the per-A-cell array.
4. `splatDensity(A, kernel = Gaussian σ=50, weight = M_ab)` → Γ_ab grid.

**cross-PCF** = an **edge-corrected cross pairwise-distance histogram**: bin |x_a−x_b| into the 10 µm annuli,
accumulate `1/(ρ_B · A_{r_k}(x_a))` per pair, average over A → g(r_k) (30-vector). Same O(N_A·N_B) GPU shape
as the existing `fuzzyAdjacency`/`cKNN` pairwise ops (or a density-autocorrelation/FFT route for large N).

## 4. New / extended ops

- **`splatDensity` extensions** (the biggest reuse): (a) **per-point weights** (M_ab); (b) **kernel choice** —
  Gaussian(σ) *and* top-hat(radius); (c) **normalization** mode (count vs KDE). The Gaussian σ=50 and top-hat
  r=100 are both splats.
- **`sampleAtPoints`** — gather a grid field's value at arbitrary point positions (grid → per-point array). New, small.
- **`markToM`** — the eq 10–13 piecewise transform (per-point scalar → [−1,1]). New, trivial (elementwise).
- **edge-correction areas** `A_r(x_a)` — area of a disk/annulus at each point clipped to the ROI. The fiddly
  geometric piece; needed by *both* stats.
- **`crossPCF`** — the binned cross pairwise-distance histogram with the annulus normalization. New pairwise op.

## 5. ROI, edge-effects, and units — resolved approach (2026-07-23)

The papers' edge correction is a missing-data artifact of treating the **ROI boundary as the data boundary**
(a fixed annotation polygon = "all the cells there are"). Our view-driven, tiled datasource (ADR-0008) lets us
do better, so we build **two modes over one grid substrate** — a quadrat grid *is* a coarse splat, a KDE splat
*is* a smooth quadrat; same grid, bandwidth = smoothing; the viewport sets the extent, the resolution sets the
quadrat size.

- **Mode 1 — MuSpAn-faithful (the parity oracle; build first).** Fixed ROI = a **regular quadrat grid**
  (MuSpAn's common default) *or* an **annotation polygon** (`annot_region`), both just a mask over the grid.
  Full clipped-area edge correction `A_r(x_a)`. **`ρ_B` global** (whole-ROI `N_B/|ROI|`). Reproduces eqs (8)–(14)
  exactly; validated against a SpOOx/MuSpAn run.
- **Mode 2 — viewport-apron (the new, GPU-native mode; build second).** The viewport is a window into a larger
  resident dataset, so load an **r_max apron** (ADR-0008 halo) beyond the visible window. Then for every
  *interior* anchor the clipped area `A_r(x_a)` collapses to the **constant** full disk/annulus area — the
  fiddly per-point polygon-clip op disappears; `m_ab = (B within 100µm)/(ρ_B·π·100²)`, PCF annulus term is a
  per-bin constant. Correction is retained **only at the true tissue/slide edge** (no apron there). `ρ_B`
  **window-local** here (heterogeneity-aware; deliberately *not* the same number as Mode 1 — that's the point).
  Must prove **interior agreement** with Mode 1 where they overlap.

**`ρ_B` decision (settled):** global first for parity (Mode 1), window-local as the second mode (Mode 2).

**The payoff worth being different for:** because it is per-viewport and on the GPU, recompute the **CSR /
permutation envelope live** as the user pans/zooms — interactive significance, not a static point estimate.
This is the honest inference (permutation is the real test, compute-bound → argues for the GPU graph) and no
fixed-ROI CPU tool does it live. The paper's own null (QCM 1000-shuffle; PCF bootstrap CIs) becomes a live
per-viewport GPU pass.

**World units (µm) — deduce from metadata, do not hard-code.** σ=50, r=100, the 10 µm bins are world lengths.
Resolve µm-per-unit from the **spatialdata/NGFF coordinate-system axis units + the `worldFromArray` scale**
(the placement facet + ADR-0018 `ParamSpec.units:"world"`, slice 4). Fall back to unitless/array space only when
the store genuinely carries no unit. Wire the B2 ingestion to read this.

**Still open (minor):** which cell-type-id pairs to demo on Leap034 (pick 2–3 once ingestion enumerates them).

## 6. Sequencing

1. (in flight) stream-B2 centroid ingestion — per-`cell_type_id` clouds + placement; **also read µm-per-unit
   from the coordinate metadata** (§5).
2. **TCM, Mode 1 (faithful, global ρ_B) first** — reuses `splatDensity` heavily; increments are per-point
   weights + kernel choice + `sampleAtPoints` + `markToM` + the disk edge-correction area (fixed ROI). Ship a
   Γ_ab(x) map for a chosen A/B pair on Leap034; validate against a SpOOx/MuSpAn run (external oracle).
3. **cross-PCF, Mode 1** — the pairwise histogram + annulus edge-correction; ship g_AB(r) curves for the same
   pairs; validate.
4. **Mode 2 (viewport-apron)** for both — constant-area interior normalisation via the ADR-0008 halo,
   window-local ρ_B, and the **live per-viewport permutation envelope**. Prove interior agreement with Mode 1.

## 7. N-way extension + visualising the high-D output (direction — specifics pending external notes)

Pairwise A→B generalises to **all N cell types at once**, and the GPU makes this nearly free: one splat per
type gives N density grids; the cross-PCF's pairwise-distance accumulation is naturally a 3-D histogram
`[type_a][type_b][r_bin]`; TCM extends to N×N maps. The paper already lives here (its QCM is an N×N cell-type
association matrix; the cross-PCF network at g(r=20) is N×N with a permutation null).

**The key structuring choice: emit the N-way output as an *open-axis tensor field*, not a bespoke N×N array.**
So the value at each location is a vector over a `pair` (or `type × scale`) **open axis** — the ADR-0004/0015
`axes` facet: `{ domain: grid, axes: [{ name:"pair", length: N·N }] }` (or `type × r`). Cheap to honour when we
build the stats; expensive to retrofit. This is what lets the two visualisation reductions below drop straight
on, because both are **metric-linear-algebra over that open axis** — the ADR-0015 amendment to ADR-0004 (whose
*first* consumer was the deleted magic-wand's channel-weighted `dist2`). The wand was not a one-off; it was the
first instance of a viz primitive we reuse here.

- **Distance-from-a-reference = the generalised magic wand.** Pick a reference location / co-location profile,
  colour or select every other location by its metric distance across the pair-axis: *"where else does the
  cell-type interaction structure look like it does here?"* — the wand's `dist2`, moved from image channels to a
  stats tensor.
- **Eigenvector / spectral projection = PCA of the stats field.** Covariance over the pair-axis (a
  reduce/matmul across locations) → top eigenvectors → project each location onto the leading 2–3 → RGB or a 2-D
  embedding ("spatial map coloured by dominant co-location mode"). New op: **open-axis eigen-decomposition** —
  cheap (small N; host or GPU-Jacobi on the N×N covariance), and the projection step is the same
  `matmul-over-axis` as the TopACT SVM ([[topact-collaboration-target]]).

> **Correction (2026-07-24), now that the global case is built.** This section said the modes are
> "eigenvectors of the symmetrised g-matrix". That is wrong twice over, and `docs/cell-stats.md` §4
> carries the worked version. Symmetrising is not the missing ingredient — `crossPCFMatrix`'s `g`
> is *already* exactly symmetric — and symmetry does not confer positive semi-definiteness, without
> which the eigenvalues are not variances and the projection has no reading. `g` is genuinely
> indefinite on interdigitated populations (measured: `[[0, 2.09], [2.09, 0]]`, eigenvalues ±2.09).
> The modes must come from the **Gram** matrix `C = M Mᵀ`, or its standardised form, which is PSD
> by construction. Separately, `C = R K Rᵀ` inherits the *kernel's* definiteness, and no kernel in
> `kernels.ts` is positive-definite in 2-D (top-hat −13.2%, triweight −1.6%) — so the Gram form is
> not an optimisation of the eigen-projection, it is its precondition.
>
> The **global** K×K case is built (`src/spatial/gram.ts`, `src/gpu/spatial/gramMatrix.ts`). The
> **open-axis tensor field** this section is really about — a K×K matrix *per pixel*, from
> convolving the products `M_a·M_b` rather than integrating them — is still unbuilt, and remains
> the prerequisite for the distance-from-a-reference reduction and for the quadrat-vs-swept
> comparison.

**Interactive GUI (firm requirement):** the demo enumerates the `cell_type_id`s (labels + counts) and lets the
user pick — pairwise A/B to start, an N×N matrix / cell-type network (edges = g(r=20), à la the paper's Fig 4)
and the two reductions above once they exist. r is a slider; a selected pair drills down to its g(r) curve / Γ map.

**Status:** the global K×K case is **built and tested**; the per-location open-axis field is not. The metric
family (dot/distance/projection/matmul over an open axis) is *specified* in ADR-0015 and still unbuilt, and no op
implements `inferAxes` yet, so a windowed Gram op would be the first.

The spectral specifics that "await the user's external notes" are now settled by construction rather than by
guessing, and the reasons are in `docs/cell-stats.md` §4 — **which covariance**: the centred, standardised
correlation of the smoothed channel densities, because it is a Gram matrix of real vectors and therefore PSD,
with an exact unit diagonal so `λ_k/K` is a true variance share; **whitening**: standardising per channel, so
rare cell types are not drowned by abundant ones (the uncentred alternative makes mode 1 an abundance map);
**sign**: canonicalised so each mode's largest-magnitude loading is positive, since an eigenvector's sign is
arbitrary and re-analysis must not flip a mode map's colours. A reference would still be welcome, but these are
now falsifiable choices with tests attached rather than open questions.
