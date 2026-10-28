# Statistics on arbitrarily shaped domains

Status: **design note** (2026-10-27). **Slice 1 is built** (2026-10-28): `src/spatial/domain.ts`,
threaded through `crossPCF`, `crossPCFMatrixBinned` and `crossPCFBinMembership`. So are masks taken
from density fields and the Gram wand, with their selection bias measured (see "Masks from the
data"). Slices 2–4 are not. Every error number below comes from `pnpm bench:domain-mask --all`
(`scripts/bench-domain-mask.ts`).

## Decision

- **A domain is a binary raster mask in world placement.** The rectangle stays as the analytic
  special case it already is (`edgeCorrection.ts`). Polygons are rasterised, not clipped exactly.
- **The clipped annulus area is computed exactly on the mask**, as
  `A_k(x) = Σ_{mask pixels p} |annulus_k(x) ∩ p|`. Each term is an existing `diskRectArea` call. The
  integration therefore adds no error. What is left is how well the mask's staircase represents the
  true boundary, and that is measured below.
- **The same mask decides membership.** Cells outside it are dropped from A and B, and the drop is
  reported. Area, ρ_B and membership must agree, or the estimator compares counts from one region
  against expectations from another.
- **No exact-polygon path.** At the covid binning the mask costs ≤ 0.1% on `g` at h ≤ 2 µm and
  ≤ 0.34% at 4 µm. That is below the existing parity's 0.26% median. A polygon path would be exact,
  but it would cost per-edge work for every anchor and give up the GPU-friendly raster.

## Why: what today's code does on a non-rectangular ROI

Today the closest option is the bounding box clip, with or without the ROI's true area passed as
`roiArea`. Below is CSR `g` relative to the exact-domain estimator on the same points, for bins
`[0,10) [10,20) [20,30)` µm:

| domain (ρ = 0.002 µm⁻² per type)    | bbox clip, bbox area | bbox clip, true area | no correction        |
| ----------------------------------- | -------------------- | -------------------- | -------------------- |
| 805 µm square, tilted 30° (≈ 648k µm²) | 1.85 / 1.82 / 1.79   | 0.989 / 0.976 / 0.962 | 0.989 / 0.975 / 0.961 |
| disk R = 250 µm (a TMA core)         | 1.26 / 1.23 / 1.21   | 0.987 / 0.969 / 0.950 | 0.986 / 0.962 / 0.932 |

- **With the true area, the bounding-box clip barely beats no correction.** It clips at edges that
  hold no data. That leaves a signed deficit that grows with r and with perimeter/area.
- **This matches the four excluded `COVID_SAMPLE_17_*` ROIs** (`muspan-cell-stats-plan.md` §6).
  Their `roi_area` was honoured and their residuals grew 1.5 → 7.8% as the ROI shrank.

## Error budget

There are four sources of error, each measured separately. Sources 1–3 are mask artefacts. Source 4
is about whether the mask is *right*, and it dominates in practice.

**1. |W| from the mask (sets ρ_B).** Using the pixel-centre rule over 50 random grid phases:

| domain            | h = 1 µm: mean / max | h = 4 µm: mean / max |
| ----------------- | -------------------- | -------------------- |
| 805 square @ 30°  | 0.000% / 0.000%      | 0.000% / 0.009%      |
| disk R = 250      | 0.001% / 0.016%      | −0.001% / 0.107%     |
| disk R = 60       | −0.002% / 0.148%     | −0.039% / 1.11%      |

The staircase cancels in the area to first order. ρ_B is only at risk when the domain is a few
pixels across.

**2. Per-anchor A_k(x), for anchors within 30 µm of ∂W.** The table shows RMS (max) relative error
over 1500 anchors:

This is bin `[20,30)`, the published `gr20`:

| domain                  | h    | centre count    | exact-pixel     | coverage        |
| ----------------------- | ---- | --------------- | --------------- | --------------- |
| square, axis-aligned    | 1 µm | 0.74% (2.2%)    | 0.71% (1.8%)    | 0.06% (0.26%)   |
| square, axis-aligned    | 4 µm | 3.9% (14%)      | 1.6% (5.4%)     | 0.47% (2.1%)    |
| square @ 30°            | 1 µm | 0.24% (0.78%)   | 0.05% (0.20%)   | 0.02% (0.19%)   |
| square @ 30°            | 4 µm | 3.5% (11%)      | 0.58% (3.2%)    | 0.29% (1.4%)    |
| disk R = 100            | 1 µm | 0.28% (1.1%)    | 0.15% (0.57%)   | 0.02% (0.14%)   |
| disk R = 100            | 4 µm | 3.7% (15%)      | 0.94% (3.0%)    | 0.30% (1.0%)    |

Bin `[0,10)` is 2–3× worse in every cell, because a small annulus spans fewer pixels.

- **The error is zero-mean everywhere** (|mean| ≤ 0.4% in the worst row). Per-anchor error
  averages out across anchors instead of accumulating.
- **An axis-aligned edge is the worst case.** The staircase there is a coherent shift of the whole
  edge by up to h/2, so at h ≤ 2 µm exact-pixel integration barely beats centre counting. On oblique
  and curved edges it is 2–6× better.
- **Coverage (an anti-aliased mask) is 2–13× better again.** Take it when rasterising our own
  polygons. It isn't needed to meet the budget.

**3. End-to-end g.** On the domains in the first table, the ratio of masked `g` to exact-domain `g`:

| domain        | h    | exact-pixel                | centre count               |
| ------------- | ---- | -------------------------- | -------------------------- |
| square @ 30°  | 1 µm | 0.9999 / 1.0002 / 1.0001   | 0.9998 / 1.0002 / 1.0001   |
| square @ 30°  | 4 µm | 1.0006 / 1.0006 / 1.0008   | 1.0006 / 1.0013 / 1.0007   |
| disk R = 250  | 1 µm | 0.9993 / 1.0010 / 1.0001   | 0.9992 / 1.0011 / 1.0001   |
| disk R = 250  | 4 µm | 1.0032 / 1.0019 / 1.0009   | 1.0034 / 1.0018 / 1.0016   |

In `g` the two integrations are indistinguishable, because source 2 averages out. Exact-pixel is
still the default: it costs almost nothing more (see Cost) and it keeps per-anchor quantities such as
weights, TCM and highlights honest.

**4. A mis-drawn boundary.** Cells are CSR in the true W. The estimator runs exactly on a W grown
(+) or shrunk (−) by δ:

| δ (µm)       | −5                  | −1                  | +1                  | +5                  |
| ------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| square @ 30° | 1.000 / 1.003 / 1.000 | 1.000 / 1.000 / 1.000 | 1.002 / 1.002 / 1.002 | 1.015 / 1.011 / 1.010 |
| disk R = 250 | 0.994 / 1.003 / 1.001 | 0.994 / 1.005 / 1.002 | 1.003 / 1.003 / 1.002 | 1.024 / 1.019 / 1.014 |

- **Drawing the boundary too tight is safe; drawing it too loose is not.** A stationary pattern seen
  through a smaller window is still the same pattern, so a tight mask only discards data. A loose
  mask claims observed-empty area. That costs +0.2–0.5%/µm on `g` here, and the cost scales with
  perimeter/area.
- **Rule: when in doubt, erode.** This also bounds a mask derived from the cells themselves (a hull
  sits inside the true W). The rule has not been measured on a hull; see Open.
- The ~0.6% scatter in bin `[0,10)` on the disk is sampling noise, from few pairs at short range.
  It is not a trend.

## Masks from the data

**Decision:** any raster can become a mask with `maskWhere(field, keep, { derivedFrom })`. That
covers a density field, a mode projection, or the Gram wand's `similarityField` ("places like this
one"). `derivedFrom` names the types whose cells built the field. `crossPCFMatrixBinned` then flags
every pair with *both* members in it as `confounded`, because those g values are biased.
`pnpm bench:domain-selection` reproduces everything below.

**Why: a mask chosen by the cells is not a drawn ROI.** The test patterns use three mutually
independent types A, B and C, so every cross-g is truly 1. Every mask keeps about 40% of the window,
and the domain-aware estimator corrects the mask's geometry exactly, so whatever bias remains is
selection. These are the independently clustered patterns (σ 30, 120 parents per type), bin
`[10,20)`, 20 realisations, ± 2 SE:

| mask                                 | g_AB (truth 1)  | g_AC (truth 1)  | g_AA          | flagged     |
| ------------------------------------ | --------------- | --------------- | ------------- | ----------- |
| window, no mask (baseline)           | 0.980 ± 0.032   | 0.972 ± 0.026   | 2.07 ± 0.08   | —           |
| density of all cells, top 40%        | **0.838 ± 0.026** | **0.832 ± 0.024** | 1.64 ± 0.06 | all pairs   |
| density of A                         | 0.976 ± 0.019   | 0.990 ± 0.026   | 1.25 ± 0.02   | AA only     |
| density of C (independent of A, B)   | 0.997 ± 0.052   | 1.015 ± 0.022   | 2.09 ± 0.09   | none of these |
| wand at a spot rich in both A and B  | **0.845 ± 0.051** | 0.994 ± 0.037 | 1.33 ± 0.08   | all pairs   |
| wand at a random spot                | 1.045 ± 0.050   | 1.017 ± 0.045   | 1.40 ± 0.10   | all pairs   |
| density of all but B                 | 0.965 ± 0.040   | **0.769 ± 0.022** | 1.52 ± 0.04 | AC, AA      |
| wand at the A+B spot, B's channel left out | 0.979 ± 0.022 | 0.985 ± 0.035 | 1.15 ± 0.04 | AC, AA     |
| thinned 50/50: density of all        | **0.862 ± 0.036** | **0.842 ± 0.040** | 1.67 ± 0.07 | —         |
| thinned 50/50: wand at the A+B spot  | **0.887 ± 0.055** | 0.977 ± 0.056 | 1.52 ± 0.10   | —           |

The other two bins agree to within a few percent. The "flagged" column shows what the
`confounded` flag reports for each mask, given its `derivedFrom`.

- **A mask built from both members of a pair biases that pair's g low, by 14–23% here.** A mask cut
  from "where A or B is dense" includes A-territory and B-territory but drops the background where
  both are sparse. Inside it, independent types look segregated relative to the mask's own average
  density. That is a false "avoidance" result, and it is the flag's reason for existing.
- **A mask that never saw B's cells leaves g_AB unbiased** (the "density of A", "density of C" and
  "B left out" rows). The reason: for any fixed A and mask, independent B cells land in each
  anchor's clipped annulus at exactly the rate the edge correction expects.
  - Leaving out the anchor type A instead of B is weaker. "Density of C" reads g_AC 2–4% high at
    `[20,30)` in both pattern sets.
  - For this reason `confounded` flags only pairs with both members in `derivedFrom`; it does not
    flag the weaker one-sided case.
- **An auto-g measured inside a mask built from that type answers a different question.** g_AA
  falls from 2.07 to 1.25 under "density of A": this is clustering *relative to A's own territory*.
  That is legitimate if it's the question being asked. It is flagged so it isn't mistaken for the
  window-wide value.
- **Splitting cells into halves only rescues random (Poisson) patterns.** On CSR, every thinned row
  is within noise of 1, because thinning a Poisson process gives independent halves. On clustered
  patterns the two halves share cluster centres, so the mask still knows where the other half's
  clusters are (0.86–0.89). It is not offered as a fix.
- **On CSR every effect is smaller**: cross-g within ±5%, and auto-g down to 0.75 under the
  random-spot wand. Weak spatial structure is no protection, though, because tissue is rarely CSR.
- **The flag is conservative.** The random-spot wand reads close to 1 here but is still flagged:
  "both members built this mask" means "not guaranteed", not "measured biased".

**Practical rule:** to ask about A–B association inside "regions like this", build the
region from channels that exclude B. For the wand, leave B's channel out of the similarity. Pass
`derivedFrom` so the matrix flags the pairs that can't be read.

**Explainer candidate:** why "where A or B is dense" makes independent types look segregated. Draw
A's and B's clusters on a background, then the union mask, then the same cells measured against the
mask's average density.

## Interface (built)

```ts
type Domain =
  | { kind: "rect"; bbox }                                         // rectDomain(bbox)
  | { kind: "mask"; bbox; width; height; data: Uint8Array };       // maskDomain(...), checked

domainArea(d): number
domainContains(d, x, y): boolean
domainAnnulusAreasInto(out, d, x, y, dr, nBins): void   // rect → existing annulusAreasInto
inverseAnnulusAreas(d, rMax, nBins): (x, y) => Float64Array   // 1/A_k, with the interior fast path
maskWhere(field, keep, { derivedFrom? }): MaskDomain     // any ScalarField-convention raster

// gram.ts — the wand on the CPU, the twin of similarityWgsl.ts's sampleWhitened().w
pixelZ(gram, col, row): Float64Array                      // reference from a pixel
similarityField(gram, whiteningMatrix(...), m, zRef): Float64Array   // d at every window pixel
```

- **Build fields over the observation window.** `maskWhere` takes the field's `bbox` as the domain's,
  so a field padded past the window (`gaussianKdeField`'s default) can mark unobserved ground as
  inside. That is the loose-boundary bias of source 4.
- **`whiteningMatrix` and `standardise` moved from `gpu/spatial/gramTerrain.ts` to `gram.ts`**, so
  the CPU and GPU wands share one definition. `gramTerrain` re-exports them.
- **`similarityField` is checked against a direct solve of corr·x = Δz**, not against the GPU.
  CPU/GPU parity at the level set is unchecked: a mask from the CPU field and the outline from the
  GPU field can disagree by f32 rounding for pixels right at `d = tolerance`.

- **The mask uses the `ScalarField` raster convention:** row-major, row 0 at the top (maxY), and
  pixels need not be square. Non-zero means inside.
- **The interior fast path uses a summed-area table of outside pixels**, not a distance transform.
  An anchor takes it only when the square around its `rMax` disk holds no outside pixel. That is
  conservative near concave corners and never wrong.
- **`PcfParams` and `PcfMatrixBinnedParams` gain `domain?`.**
  - With it, the domain sets the area (unless `roiArea` overrides it), the clip and membership.
    Results report `outside`, the number of cells dropped.
  - Without it, behaviour is unchanged and no cell is dropped.
  - The type axis is taken from every cell before any are dropped.
- **`crossPCFGpu` and `crossPCFBootstrap` refuse a `domain`** rather than ignore it.
- **`PcfEnvelopeParams` does not have `domain` yet** (slice 3).

## Per-statistic consequences

| statistic                      | what changes                                                       | size                     |
| ------------------------------ | ------------------------------------------------------------------ | ------------------------ |
| `crossPCF`, `crossPCFMatrixBinned` | annulus areas from the domain, plus membership                   | built (slice 1)          |
| envelope, `"label"` null       | none beyond the weights (positions never move)                     | small                    |
| envelope, `"shift"` null       | the torus is undefined on W, so a new construction is needed (below) | **open, the real work**  |
| `crossPCFBootstrap` (block)    | blocks partly outside W. Needs a decision on block weight vs dropping | small, needs a check     |
| TCM (`tcm.ts`, GPU)            | `ρ_B·π r²` becomes `ρ_B·|disk ∩ W|`, i.e. mask ⊛ disk sampled at anchors | medium (GPU convolution) |
| QCM                            | quadrat coverage fraction; drop quadrats under a threshold         | small–medium             |
| Gram / apron                   | window mass counted inside the mask, not the bbox                  | small                    |
| GPU `crossPcf`                 | has no edge correction at all today; upload per-anchor `1/A_k` and weight pairs | medium        |

### The shift null on a non-rectangular W

The candidates:

- **(a)** Wrap on the bbox torus, keep `B′ ∩ W`, and recompute ρ_B per simulation. This is cheap and
  A's weights are reused. The catch is that W + v leaves holes in W that B′ never fills.
- **(b)** Analyse each simulation on `W ∩ (W − v)`. This is closer to unbiased, but A's weights
  change per simulation.
- **(c)** spatstat's `rshift(edge = "erode")`. Its behaviour should be checked against the source
  before relying on it.

**The acceptance test is the size, measured.** On independently generated self-clustered pairs
(`pointPatterns.independentClustered`) in a disk domain, the rejection rate must fall inside the
binomial CI of α. That is the same kind of measurement that caught the label null failing 20/20 in
`pcfEnvelope.ts`. It has not been run.

## Cost

A boundary anchor visits about 4(rMax/h)² pixels, 3,600 at h = 1 µm and rMax = 30 µm. Only pixels
cut by a bin-edge circle need `diskRectArea`; the rest add their area to one bin. Interior anchors
keep today's fast path.

Measured with `crossPCFMatrixBinned` on an 805 µm square tilted 30°, with 3,240 cells, 20 types,
rMax 30 and 3 bins:

| clip                  | time    |
| --------------------- | ------- |
| bbox rectangle        | 4 ms    |
| mask, h = 2 µm        | 20 ms   |
| mask, h = 1 µm        | 57 ms   |

The 1 µm mask is 1.25 M pixels, and the 57 ms includes building its summed-area table.

## Open

- **Where the masks come from.** `src/datasource` has no SpatialData `shapes` (polygon) loader.
  Polygons need one, plus a scanline rasteriser (nonzero winding, holes, multipolygons). Raster
  `labels` and tissue masks only need resampling to the world grid.
- **The `COVID_SAMPLE_17_*` geometry is unknown.** The MDV project carries region images with W×H.
  Whether it carries the acquisition outline is unchecked. The fallback is a concave hull of the
  cells, which sits on the safe (tight) side by source 4. It hugs boundary cells, though, so measure
  it on synthetic data before using it for parity.
- **Envelope calibration on non-rectangular W** (above).

## Slices

1. **Done.** `Domain` + mask annulus areas + membership in `crossPCF` / `crossPCFMatrixBinned`.
   Tests (`src/spatial/domain.test.ts`):
   - a pixel-aligned rectangle mask, square and non-square pixels, reproduces `edgeCorrection.ts`
     to 1e-9;
   - a tilted square and a disk stay within about twice the source-2 RMS and max;
   - on a tilted ROI, g is within 0.5% of the exact estimator run in the ROI's own frame (source 3);
   - a disk ROI reads g ≈ 1, where the bbox clip reads < 0.95.

   Two deliberate breakages each fail the suite: a wrong bin-edge walk, and a fast path that lets
   one outside pixel through.
2. Polygon rasteriser. Re-run the MDV parity on the five excluded ROIs if their geometry turns up.
3. Envelope nulls on W, with the size test as the gate.
4. TCM, QCM, Gram and GPU `crossPcf`.

**Explainer candidate:** an annulus over a pixel staircase, with the cut pixels highlighted. Pair it
with the erode-vs-dilate asymmetry (source 4): "a smaller window loses data; a bigger one invents
empty space".
