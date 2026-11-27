# Anchor screen (R5 of region/raster statistics)

**Decision.** `anchorScreen` scores one anchor column against every feature of a regions × features
table, per sample: Spearman ρ, a two-sided p from N within-sample shuffles of the anchor (default
999), and the finite-pair count. It reproduces SIMBA's continuous-anchor scorer
(`compute_feature_block_scores`). On every TNBC block, **ρ and n_finite are identical to SIMBA's**
and p agrees within Monte Carlo error. The CPU path is fast enough for live use. `anchorScreenGpu`
moves the null's dot products to the GPU (60–90× on ST genes), and by default it gives **the same p
as the CPU path**, not just a close one.

## Surface

| What | Where | Notes |
|---|---|---|
| `anchorScreen(input, opts)` | `intraspatial/spatial` (`src/spatial/anchorScreen.ts`) | CPU, f64, Fisher–Yates (mulberry32) |
| `prepareAnchorBlock`, `anchorNullExceedances`, `anchorEmpiricalP`, `averageRanks` | same | per-sample pieces: observed scores and mask groups, the null, p |
| `anchorScreenGpu`, `anchorNullExceedancesGpu` | `intraspatial/gpu/spatial` (`anchorScreenGpu.ts`) | GPU null, `shuffle: "fisher-yates"` (default) or `"feistel"` |
| `FEISTEL_WGSL`, `feistelDomain`, `feistelPermuteCpu` | `feistelWgsl.ts` | shared with `quadratCorrelationGpu`, whose output is unchanged |
| `pnpm anchor:parity <run_root> [--gpu [--feistel]]` | `scripts/anchor-screen-parity.ts` | the SIMBA comparison below; inputs are local-only |

Input: `{ anchor, features: Map<name, column>, samples? }`. The columns are one value per region,
and a non-finite value is missing. R3's `zonalTable(...).columns` is already this shape. Output: a
long table with columns `sample, feature, rho, p, nFinite, skip`, samples in first-seen order. Each
sample's stream is seeded from `seed|sampleKey`, so its null does not depend on the other samples.

## Semantics (SIMBA's rules)

| Rule | Behaviour |
|---|---|
| Pairs | Feature j uses the regions where the anchor **and** feature j are finite. |
| Support | Fewer than `minFinite` (3, `CONTINUOUS_MIN_FINITE_REGIONS`) such regions → `insufficient`, ρ NaN. |
| Ranks | Average ranks for ties, taken over those regions for both sides. ρ is the Pearson correlation of the centred ranks. |
| Constant | Either side's centred ranks all zero → `constant`, ρ NaN. |
| Shuffle | Each shuffle permutes the finite anchor values among the finite-anchor regions, and missing anchors stay put. A feature with gaps of its own sees the shuffled anchor on its regions only, **re-ranked there**. Its null is therefore a random draw of k anchor values, not a permutation of the k it was observed with. |
| Null ρ | NaN if the shuffled anchor is constant on the feature's regions. It never counts as extreme, and N stays the configured count. |
| p | Two-sided empirical p, below the table. |

**p** = (1 + #{|ρ_null| ≥ |ρ_obs| − 1e-12}) / (N + 1). The 1e-12 is the SIMBA check's tie
margin: small n gives exactly tied |ρ| that rounding would otherwise split. p is NaN where ρ is
NaN. (The SIMBA check script writes 0.001 there, an artefact of comparing against NaN.)

## Cost

Features are ranked once and grouped by finite mask.

- **Full-mask group** (every finite-anchor region; all ST genes, most patch features). Average ranks
  commute with a permutation, so a shuffle's centred ranks are the observed ones permuted, with the
  same norm. Each (shuffle, feature) costs one length-k dot product.
- **Partial masks.** The shuffled anchor is re-ranked per shuffle by counting sort over the anchor's
  dense rank ids, O(k + distinct). IMC cell-type fractions have up to ~90 distinct masks per block,
  and these always run on the CPU.

The total is N·m·k multiply-adds per sample. Measured with 999 shuffles under Node on an M-series
Mac:

| Block | k × m | CPU null | GPU null |
|---|---|---|---|
| ST genes, Leap034 (copper) | 39 × 18,460 | 861 ms | 9 ms |
| FLIM patches, Leap035 (copper) | 9,179 × 14 | 216 ms | 98 ms (host shuffles dominate) |
| IMC patches, Leap034 (copper) | 363 × 174 (52 full-mask) | 142 ms | 127 ms (partial masks stay on CPU) |
| LA-ICP-MS patches, Leap034 150 µm | 1,467 × 90 | 136 ms | 25 ms |

The GPU path pays off at cohort scale: tens of samples × 18k genes. One specimen's patch screen
already fits a live patch-size slider on the CPU.

## GPU path

The observed ρ, n_finite and skip codes always come from the CPU (`prepareAnchorBlock`). Only the
full-mask null runs on the GPU, in two dispatches per batch:

1. `permuteAnchor`, one thread per (shuffle, region): `u[t] = a[π(t)]`.
2. `tally`, one thread per (feature, shuffle): the dot product, then `atomicAdd` when the shuffle is
   extreme.

The ranks are doubled, so every value and dot product is an integer. `tally` sums them exactly as
a two-word (64-bit) integer: each product is below 2³¹ for k ≤ 46,340 (`EXACT_MAX_REGIONS`; a block
with more finite anchors runs on the CPU), and the sum is below 2⁶³.

**The cutoff.** For the full-mask group, the CPU's `|ρ_null| ≥ |ρ_obs| − tol` is, in doubled units,
`|dot_null| ≥ |dot_obs| − 4·tol·‖a‖‖f‖`. For an integer `dot_null` that is `|dot_null| ≥ T` with
`T = ceil(|dot_obs| − 4·tol·‖a‖‖f‖)`, which the host computes in f64. The GPU therefore applies the
CPU's rule at every k, up to the CPU comparison's own f64 rounding. A fixed half-unit cutoff
(`|dot_obs| − ½`) would agree with it only while `4·tol·‖a‖‖f‖ < 1`; since ‖a‖‖f‖ ≤ k(k²−1)/12, that
holds for k up to about 14,000 at tol = 1e-12, and the two would pick different integers beyond that.
The integer sum also keeps an identity shuffle and a single central swap distinguishable at any k,
where an f32 sum (exact only below 2²⁴, k ≲ 370) would not.

| `shuffle` | π from | p vs CPU |
|---|---|---|
| `"fisher-yates"` (default) | the CPU's own `mulberry32` permutations, uploaded per batch. O(N·k) host work, 1/m of the job. | identical by construction (same permutations, same cutoff) |
| `"feistel"` | the shared Feistel permutation per thread: no host shuffle and no upload. Partial-mask groups, and blocks without a full-mask group, use the host twin with the same keys. | within Monte Carlo error; identical to the CPU null run on the Feistel permutations |

Separately from that guarantee, the TNBC check found 0 differing counts across all 15 blocks
(including k = 9,179) with the default shuffle.

**The Feistel permutation is not uniform for tiny n.** The χ²/dof of (π(0), π(1)) over 200k keys is
91 at n = 5, 33 at n = 8 and 10 at n = 16, against 1.0–1.2 from n = 20 up: with ≤ 2-bit halves,
six rounds do not mix. On Leap035's 8 ST ROIs this showed up as mean z² 1.54 against SIMBA (1.03 on
the CPU). So `"feistel"` falls back to Fisher–Yates below `FEISTEL_MIN_REGIONS` = 20.
`quadratCorrelationGpu` permutes thousands of cells and is unaffected.

## Comparison with SIMBA

The inputs were exported with SIMBA's own builders (`build_patch_block`, `build_st_block`, via
`~/data/tnbc_simba_runs/scripts/export_anchor_blocks.py`). They were compared against
`simba_anchor_check.py`'s scores for collagen and copper (Leap034 and Leap035) and for collagen at
150 µm (Leap034). The comparison covers 15 blocks and ~111k (sample, feature) rows.

- **ρ:** max |Δρ| = **0** (bit-identical) on every block.
- **n_finite and NaN pattern:** 0 mismatches. This includes Leap035's IMC features, whose finite
  counts range from 11 to 140, and the 1-ROI Leap035 collagen ST block, which is all `insufficient`.
- **p:** z = Δp / √(2p(1−p)/1000), from two independent 999-shuffle estimates. The mean z² is
  0.97–1.03 on the ST blocks (expected ≈ 1). There are 45 / 46 / 17 rows with |z| > 3 out of 18,460,
  against ~50 expected. Max |z| is ≤ 2.6 on the patch blocks, where many p sit at the 1/1000 floor
  on both sides.

Spot values (ρ is identical to SIMBA's; p is ours, with SIMBA's in brackets):

| Anchor, sample | Feature | ρ | p |
|---|---|---|---|
| collagen, Leap034 | COL1A1 (ST) | +0.551 | 0.001 (0.004) |
| collagen, Leap034 | SOX18 (ST) | −0.561 | 0.003 (0.001) |
| collagen, Leap034 | Zn / Mg / Cu p50 (patches) | −0.571 / −0.529 / −0.513 | 0.001 each (0.001) |
| copper, Leap034 | COX6C (ST) | +0.367 | 0.027 (0.017) |
| copper, Leap034 | IMC tumour-lineage fraction | +0.196 | 0.001 (0.001) |

The IMC table has no feature named for panCK. `imc_lineage_tumor_fraction` is the nearest, and it is
positive, as expected.

## Not built

- The MDV chart over these results, and cohort aggregation across responders and non-responders.
- Partial-mask features on the GPU. They would need a per-(shuffle, mask) re-rank, and at the TNBC
  sizes they are cheap on the CPU.
