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

## 5. Cross-cutting prerequisites the papers surface (decisions needed)

These are why the ADR-0018 placement/extent work is a real prerequisite, not polish:

- **ROI / extent.** ρ_B (= N_B/|ROI|) and every edge-correction area need a concrete ROI. Options: the tissue
  **bounding box** (simplest, biased near a ragged tissue edge), a **convex hull**, or the actual
  **`annot_region` / segmentation polygon**. Faithfulness of the edge correction scales with this choice.
  → This is ADR-0018 `extent`. **Decision needed.**
- **World units (µm).** σ=50, 100, and the 10 µm bins are **world lengths**. They only mean anything if we know
  µm-per-coordinate-unit for Leap034 — i.e. the placement's `worldFromArray` scale, and the ADR-0018
  `ParamSpec.units: "world"` path (slice 4). Are the `obsm['spatial']` IMC centroids already in µm, and at what
  scale? → **Decision / datum needed.**
- **Which cell-type pairs**, and whether ρ_B / edge correction are per-`annot_region` or whole-slide.

## 6. Sequencing

1. (in flight) stream-B2 centroid ingestion — per-`cell_type_id` clouds + placement.
2. **TCM first** — it reuses `splatDensity` heavily; the increments are per-point weights + `sampleAtPoints` +
   `markToM` + the disk edge-correction area. Ship a Γ_ab(x) map for a chosen A/B pair on Leap034.
3. **cross-PCF** — the pairwise histogram + annulus edge-correction; ship g_AB(r) curves for the same pairs.
4. Validate both against a MuSpAn/SpOOx run on the same ROI (external oracle), like the TopACT box-pool plan.
