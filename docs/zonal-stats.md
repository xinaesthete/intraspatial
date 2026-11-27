# Zonal statistics (R3 of region/raster statistics)

**Decision.** Regions are a **u32 label raster**: polygons are rasterised into one, and a label
image (ST ROI map, segmentation, patch grid) is used as is. One labelled histogram then has
**rows = regions × channels** in R1's layout ([`histogram-op.md`](histogram-op.md)), and it is
accumulated tile by tile from a fixed pyramid level into one set of counts. Per-region count, mean,
median, IQR, percentiles and support/valid fractions come from the rows on the host. An exact second
pass gives numpy-exact quantiles. On Leap034 it reproduces SIMBA's per-ROI Cu median and IQR
bit for bit, and the collagen median to within 4·10⁻⁸ relative (see "Validation").

## Surface

| What | Where |
|---|---|
| `zonalStatsTiled({ level, regions, channels, bins, quantiles, exact })` → histogram + regions × features table | `intraspatial/gpu/spatial` (`zonalTiled.ts`) |
| `ZonalAccumulator` (`rangeTile` → `finalizeEdges` → `binTile` → `read` / `finish`, then `beginCollect` → `collectTile` → `readCollected`) | `intraspatial/gpu/spatial` (`zonal.ts`) |
| `rasterisePolygonsResident`, `rasterisePolygonBuffers`, `zonalHistogramGpu` | `intraspatial/gpu/spatial` |
| `rasterisePolygons`, `zonalHistogram` graph ops; `polygonRingSources` | `intraspatial/graph` |
| Golden and host side: `rasterisePolygonsCpu`, `polygonRingBuffers`, `polygonLayers`, `zonalHistogramCpu`, `zonalRow`, `zonalMerge`, `zonalExactPlan`, `zonalCollectCpu`, `zonalExactQuantiles`, `zonalTable` | `intraspatial/spatial` (`zonal.ts`) |

## Rules

| Topic | Rule |
|---|---|
| Labels | Label L in 1..regions is region L − 1. 0 and labels above `regions` are no region. f32 labels truncate. |
| Polygons | Polygon k is label k + 1. A pixel is inside when its **centre** is (`domainContains`, even–odd). Where polygons overlap, the highest label wins. Coordinates are level-0 array space (x = column, y = row). |
| Overlap | `polygonLayers` splits polygons greedily by bounding box. Each layer is rasterised and binned on its own, all into the same rows. A half-stride patch grid takes 4 layers; ST ROIs and a lasso take 1. |
| Per-pixel slot | Each region pixel lands in one slot per channel, checked in this order: `unsupported` (the support raster is 0; NaN counts as supported, like numpy's `!= 0`), `nonFinite`, `excluded` (`valid: "positive"` and v ≤ 0), `under` / `over` (outside a fixed range), `inRange` (binned). `pixels` is the sum of all slots. Totals stride is 8. |
| Fractions | `count = inRange + under + over`, `supported = pixels − unsupported`, `validFraction = count / supported`, `supportFraction = supported / pixels`. |
| Support | Per channel, any number of rasters, OR'd as `!= 0`. For example, LA-ICP-MS tissue is "any of the five metals non-zero". |
| Edges | Per channel, and fixed **before** binning, because tiles add into the same counts. A fixed range takes one pass. Auto range takes an extra min/max pass over every region's values, then R1's edge rule. |
| Binning | R1's f32 rule, via the shared `HIST_BIN_WGSL`. Counts, CDF, totals and edges match the golden **exactly**. |
| Moments | min and max are exact (ordered-u32 atomics). The sum is f32 within a tile, then compensated across tiles (TwoSum `(hi, lo)`), so its error does not grow with tile count. |
| Quantiles | Binned: R1's `histogramQuantile`, within one bin width under R1's bracketing condition. Exact: see below. |

## Passes

| Pass | Kernel | Per tile | Notes |
|---|---|---|---|
| labels | `rasterise` (polygons) or an upload | once per layer | Crossing parity against the ring buffers of every polygon whose box holds the centre. Vertices are pre-transformed in f64 to the tile's pixel frame. |
| range (auto only) | `range` + `finalize` | 1 dispatch per channel | Per-thread min/max, then one atomic pair. |
| bin | `bin` + `fold` | 1 dispatch per channel | Each thread walks runs of 16 pixels and flushes totals, min, max and sum once per label change. Bins go to a workgroup-private histogram when regions × bins ≤ 4088, otherwise to global atomics. |
| exact | `collect` | 1 dispatch per channel | Appends values in the planned target bins at their offsets. |

Device memory is one chunk per channel, plus labels and support, plus `rows × bins` u32. It does not
depend on slide size. A 520 Mpx resection at 512² chunks streams about 2000 tiles through the same
buffers. Each pass fetches every tile again, so give each source its own `TileCache`, sized to the
level, or use fixed ranges without the exact pass to read the slide once. Every binding uses at
most 7 storage buffers.

## Exact quantiles

`zonalExactPlan` reads each row's CDF. For each q it names the bins holding order statistics ⌊p⌋
and ⌊p⌋+1, where p = q·(N−1). Those bins are de-duplicated, at most 2 per quantile per row.
`collect` gathers only those values. That was 150 k values for 39 ROIs of about 160 k pixels each on
Leap034. `zonalExactQuantiles` sorts each small set and interpolates numpy-"linear"; that is
`np.median` at q = ½. It checks each collected count against its bin count. A rank in a fixed-range
under/overflow is censored to `lo` / `hi`, as R1's binned estimate is. This is the second pass R1
sketched, built.

## Validation: Leap034 against SIMBA

`pnpm zonal:parity` runs this. It reads the registered 1 µm/px OME-TIFFs (3788 × 16333 px)
through their own 512² tiles, with no full-image buffer. The script holds SIMBA's rules
(TNBC-MMB `feature_extraction`, `mask_dilation_pct = 0`):

- **region** is `st_day_a_roi_label_map == roi_id`. Every segment row of a ROI repeats the same value.
- **collagen** uses support `imc_roi_label_map > 0`.
- **Cu** uses support "any of Cu/Zn/Fe/Mg/Mn ≠ 0".

Both use `valid: "finite"`, 1024 bins and the exact pass. The run touched 66 tiles and took 5.5 s on
the development machine, TIFF decoding included.

| Feature | ROIs compared | Agreement |
|---|---|---|
| `laicpmsi_cu_p50` | 39 / 39 | identical (max rel diff 0) |
| `laicpmsi_cu_iqr` | 39 / 39 | identical |
| `laicpmsi_n_tissue_pixels`, `laicpmsi_tissue_fraction`, `laicpmsi_effective_region_mask_pixels` | 39 / 39 | identical |
| `imc_collagen_median` | 35 / 39 | max rel diff 3.8·10⁻⁸ |

Two differences, both explained:

- **ROIs 44–47 have no collagen value on either side.** They lie outside the IMC acquisition
  (0 acquired pixels; SIMBA's `imc_acquired_fraction` is 0). SIMBA writes an empty cell; we
  give NaN with `collagen_supported = 0`.
- **Collagen differs by about 1 f32 ulp.** For an even count, `np.median` on float32 averages
  the two middle values in f32. We interpolate in f64, so the value is the same before rounding.
  Cu matches exactly because its middle values coincide (or the count is odd).

No label with pixels is missing from SIMBA's table, and SIMBA has no ROI we lack.

The **binned** median at 1024 bins is within 0.19 (collagen) and 0.80 (Cu) bin widths of the
exact one. That is about 0.5 % of the Cu median, enough for screening; use `exact: true` to match
SIMBA.

## Not built / next

| Item | Note |
|---|---|
| R4 patch grid | It feeds in as polygons, which `polygonLayers` handles at any stride, or as a label raster per layer. Patch support rules (SIMBA's `min_tissue_frac`) are a filter on `supportFraction` / `validFraction`. |
| R5 anchor screen | It consumes `zonalTable` columns. |
| Rasteriser culling | Each pixel tests every polygon box in the tile, which is fine for hundreds per tile. A coarse bucket grid would help at thousands. |
| Device-resident support | Several support rasters are OR'd on the host, so they must be host tiles. A single device-resident support raster is bound directly. |

Explainer candidate: the run-length flush in `bin`, and the two-bin exact plan.
