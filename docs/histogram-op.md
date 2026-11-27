# Histogram op (R1 of region/raster statistics)

**Decision.** One resident GPU histogram over any raster plane: `histogram` and `histogramMasked`
graph ops, backed by `histogramResident` (`src/gpu/spatial/histogram.ts`), with the f64 golden and
the host quantile helper in `src/spatial/histogram.ts`. Counts match the golden **exactly**.
Quantiles come off the CDF and are within **one bin width** of exact when the order statistics
bracketing each rank share a bin.

## Surface

| What | Where | Notes |
|---|---|---|
| `histogram`, `histogramMasked` ops | `intraspatial/graph` (registered built-ins) | Tier-2 resident, `slices: true` |
| `rasterPlaneValue(plane)` | `intraspatial/graph` | a graph source for a `ResidentRasters` plane |
| `histogramResident`, `histogramGpu`, `residentRasterPlane`, `denseRasterPlane` | `intraspatial/gpu/spatial` | caller-owned targets / host readback |
| `histogramCpu`, `histogramQuantile`, `histogramSummary`, `histogramRow`, `exactQuantile` | `intraspatial/spatial` | golden and quantiles |

Outputs are row-major `rows × bins`, so R3 can add rows rather than a new shape.

| Port | dtype | Shape | Content |
|---|---|---|---|
| `counts` | u32 | rows × bins | per-bin counts |
| `cdf` | u32 | rows × bins | inclusive, per row |
| `edges` | f32 | 1 × 2 | `[lo, hi]`, shared by all rows |
| `totals` | u32 | rows × 4 | `[inRange, under, over, nonFinite]` |
| `moments` | f32 | rows × 4 | `[min, max, sum, mean]` of finite values, NaN if a row is empty |

`rows` is 1, or 2 with a mask (row 0 is inside, i.e. non-zero; row 1 is outside).

## Rules

- **Range:** auto takes the finite `[min, max]`, with the last edge closed. A constant field gets
  `[v − ½, v + ½]` (numpy's rule). For `|v| ≥ 2²⁴` that interval rounds to a single f32, so the
  edge nearer zero moves one ulp toward zero instead. A fixed range must still have `hi > lo`
  after f32 rounding. With a fixed range, out-of-range values are counted as
  under/over and not binned.
- **Bin of v:** the largest `b` with `f32(b·range) ≤ f32(f32(v − lo)·bins)`. Division only seeds
  `b`, because GPU division is not correctly rounded. This rule is why counts are bit-exact. When
  `max(|lo|, |hi|) ≥ 2⁹⁶`, `v`, `lo` and `hi` are first scaled by the exact factor 2⁻³². That keeps
  `range` and the scaled offset from overflowing near f32 max.
- **Moments:** min and max are exact. The GPU accumulates `sum` and `mean` in f32, so cancelling
  inputs lose precision and values near f32 max can overflow to ±Inf.
- **Input:** `height × rowWords` words, as f32 texels or as two f16 texels per word. This is the
  `ResidentRasters` layout, with rows padded to 256 B and read as `at()` reads them in
  `gramMatrix.ts`. Single-channel only: a grid with open axes is refused at graph build.

## Passes (one submit)

1. A signed min/max/sum tree reduce. This is a twin of `fieldMax.ts`, which handles only
   non-negative values.
2. A one-workgroup finalize that writes the edges. Because the edges stay on the device, auto
   range has no readback.
3. Binning with `atomicAdd` into a workgroup-private histogram (up to 4088 rows×bins), then a
   global merge. Larger histograms add to global memory directly.
4. `encodeScan` over the counts, then a per-row inclusive fix that also writes the mean.

## Quantiles: error bound and the exact path

The helper uses the numpy "linear" rank `p = q·(N−1)` and interpolates linearly within the bin.
The fraction is capped at the bin's upper edge, so estimates are monotone in `q`. The estimate is
within one bin width when the order statistics around `p` share a bin. If the data has gaps wider
than a bin, or very few values, the error can grow by up to the gap. Binned estimates are clamped
to `[min, max]`, so a constant field is exact. A rank in the fixed-range under/overflow is
censored: it returns `lo` or `hi` unclamped. The IQR is within two bin widths when both quartiles
meet the same-bin condition.

**Exact median (sketched, not built).** Pass 1 uses the CDF to find the target bin `b` and the
rank `p − C_b` within it. Pass 2 stream-compacts the values with `bin(v) == b`
(`streamCompact.ts`) and sorts that small set. It is one extra dispatch plus a readback the size of
a bin. At 256 bins over a 300 µm patch, the bin-width bound is already finer than the downstream
correlations need.

## R2: gram-mode and wand histograms

**Decision.** The mode legend shows one histogram per painted mode, and by default the map's
colour range comes from it. The wand panel shows the patch's area at every tolerance, computed from
one bottleneck flood, plus the histogram of d inside and outside the selection.

| What | Where | Notes |
|---|---|---|
| `modeHistograms`, `modeProjection` | `src/gpu/spatial/modeHistograms.ts` | One dispatch writes y₁..y₃ (`sampleWhitened().xyz`) into three dense planes. `histogramGpu` then bins each plane. `halfRanges` = max(\|p1\|, \|p99\|) per mode. |
| `ModePaintOptions.ranges` | `gramModes.ts` | A per-mode half-range that replaces `saturate·σ_k`. σ still sets the chroma weighting. |
| `floodAreaByTolerance`, `floodAreaAt`, `floodLeaks`, `bottleneckCosts`, `wandSeedPixel` | `src/spatial/floodArea.ts` | CPU. `connectedRegion` now takes its start pixel from `wandSeedPixel`. |
| Legend histograms, "Colour range" select | `ModeLegend`, `useModeHistograms`, `PageControls` | Labelled as describing the current view (or the kept window), at its raster size. |
| Area curve, d histogram, medians | `WandCharts` in `WandPanel`; `useWand` | d is now read back for every sample, not only in connected mode. |

**Area vs tolerance.** Pixel p is in `connectedRegion(t)` exactly when some 4-connected path from
the start pixel has max d ≤ t. So the least such max, p's *bottleneck cost*, decides p's membership
at every t, and area(t) = #{cost ≤ t}. One minimax Dijkstra flood (an indexed binary heap with
decrease-key) gives every cost.

- **Exact at every t.** A cost is always one of the input values (a max never rounds), so the
  curve is a step function with exact breakpoints. A bucket queue over R1's bins would be exact
  only at bin edges and would smear leaks to a bin width, so it was not used.
- **Start pixel.** The fill starts at the nearest pixel if it is in range, else at the closest
  in-range corner. That splits t into at most five intervals, each with a fixed start, so at most
  four floods are needed. A corner's flood runs only for tolerances below d(nearest), and only
  when needed.
- **Validity.** NaN is never reached. ±Inf compare as numbers, as they do in the fill.
- **Tests** (`floodArea.test.ts`). The curve equals `connectedRegion`'s pixel count at every
  distinct d and at the midpoints between them. The fields are noise, ties with NaN holes, a
  fractional seed whose nearest pixel is out of range, and an edge seed. On a two-basin field
  the only leak is at the saddle, exactly.
- **Cost.** The flood is O(n log n) on the main thread, once per field (sample × metric), not
  once per tolerance.

**Leak rule** (`floodLeaks`). At a saddle, every pixel of the far basin that is lower than the
saddle gets the saddle as its cost. The whole basin therefore joins at one exact tolerance. A leak is a single
step where a region already ≥ 0.5 % of the raster grows by ≥ 50 % of itself. The size floor
keeps out the seed's first pixels and rings of tied values around it.

**Inside/outside d.** `histogramCpu(d, { mask })` is R1's two-row form. Row 0 is the selection:
the patch in connected mode, else d ≤ t. Medians come from `histogramSummary`, within one bin
of the 512-bin auto range. Both charts draw 0–5, the slider's domain, and clicking either one sets
the tolerance.

**Hover links** (transcript-modes page). A histogram bar under the pointer becomes a `band` on
`GramModesLayer`. Places whose mode value, or d, lies in that bar's range keep their colour, and the
rest are dimmed. The band reads the shader's interpolated field, so its edge falls between pixel
centres. Going the other way, the map pixel under the pointer marks its mode values and its d on
the histograms. Its tolerance for joining the patch (`FloodAreaCurve.nearestCosts`) is marked on
the area curve. For these lookups, `modeHistograms({ readback: true })` keeps the three mode planes
on the host, at 12 bytes a pixel.

**The graph wand** (`src/gpu/graph/wand.ts`, `wandPatch`). The curve does not apply as is.
Its d is `mixtureDistance`, not `wandDistance`, and `floodFrom` uses `seedIn: true` (the seed is
always in). With `rho < 0.5` the same bottleneck treatment does apply, with two changes. Pixels
that fail the Laplacian test become NaN in d, because that test does not depend on t. The start
pixel's cost is −∞, which models `seedIn`. With an opening (`rho ≥ 0.5`) it does not apply.
Opening is monotone in t, but "the opened piece holding the seed" is not a threshold of any
per-pixel cost, so area(t) would need one evaluation per tolerance. The graph wand is unchanged.

## Later steps (not built)

| Step | Scope |
|---|---|
| R3 | A polygon/label rasteriser and labelled (zonal) reductions, with rows = labels × channels. `MAX_ROWS` (2) and the `rowOf` mask rule are the seam. |
| R4 | Patch-grid generator. |
| R5 | Anchor screen with a permutation null. |
