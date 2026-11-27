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

## Later steps (not built)

| Step | Scope |
|---|---|
| R2 | Per-mode histograms in the playground's `GramModesLayer` / `ModeLegend`. A histogram of `wandDistance` in `WandPanel`, plus an area-vs-tolerance curve from one priority flood. |
| R3 | A polygon/label rasteriser and labelled (zonal) reductions, with rows = labels × channels. `MAX_ROWS` (2) and the `rowOf` mask rule are the seam. |
| R4 | Patch-grid generator. Built: see [`patch-grid.md`](patch-grid.md). |
| R5 | Anchor screen with a permutation null. |
