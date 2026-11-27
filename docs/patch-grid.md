# Patch grid (R4 of region/raster statistics)

**Decision.** `patchGrid` (`src/spatial/patchGrid.ts`) reproduces SIMBA's patch layout and
support rules exactly on the host. It works over row runs, so a 1 µm/px canvas costs its runs,
not its pixels. Kept patches are returned as rectangles, and go to zonal statistics (R3) as
**polygons** (`patchPolygons`), so polygon k is patch k in `zonalStatsTiled`'s table. Grid to
features reproduces SIMBA's per-patch tables (below). Support fractions can also be counted on the GPU, over a resident mask, by the `patchSupport`
op (one flat prefix scan, then one subtraction per row per rectangle). Both paths give
identical integers.

## Surface

| What | Where | Notes |
|---|---|---|
| `patchGrid(opts)` | `intraspatial/spatial` | footprint → candidates → support → kept |
| `maskRuns`, `maskRunsFromRows`, `unionRuns`, `positiveFinite`, `nonZero` | `intraspatial/spatial` | masks as row runs; rows can be streamed |
| `componentBoxes`, `axisStarts`, `patchCandidates` | `intraspatial/spatial` | the layout, step by step |
| `maskCountsInRects`, `pointPixels`, `pointCountsInRects` | `intraspatial/spatial` | support counts |
| `patchPolygons` | `intraspatial/spatial` | the hand-off to R3: `regions: { kind: "polygons", polygons }` |
| `rectsArray`, `patchLayers`, `patchLabelImage` | `intraspatial/spatial` | rectangles as `n × 4` u32; label rasters |
| `patchSupport` op | `intraspatial/graph` (registered built-in) | mask grid + `n × 4` u32 rects → `n × 1` u32 counts |
| `patchSupportResident`, `patchSupportGpu` | `intraspatial/gpu/spatial` | caller-owned buffers / host readback |
| `pnpm patchgrid:parity` | `scripts/patch-grid-simba-parity.ts` | rasters → grid → R3 → SIMBA's per-patch tables |

Options: `size` and `stride` in `units` (`"px"`, the default, or `"um"` with `pixelSize` in µm/px,
rounded to whole pixels), a `footprint` mask, and `rules`. A `fraction` rule takes a mask, a
`count` rule a `pointPixels` set; each has a `min`. A patch is kept when every rule's value is
finite and `>= min`. The result has every `candidate` (SIMBA's `keep_invalid` table) and the kept
`patches`. Each patch has `id`, `x0 y0 x1 y1` (half-open pixels), `edge`, and `support[]` in rule
order.

## SIMBA's layout

From TNBC-MMB's `src/feature_extraction/patch_grid.py` (`generate_patch_bounds`, `_axis_starts`):

| Rule | Behaviour |
|---|---|
| Origin | **Per 4-connected component** of the footprint: each component's grid starts at its bounding box. There is no global lattice. |
| Wide component | Starts every `stride` from the near edge, plus a trailing start ending exactly at the far edge when the stride misses it (`edge = true`). |
| Narrow component (≤ size) | One patch, ending at the component's far edge. |
| Border | Starts are clamped into the image, so patches are never partial. They are only clipped when the image is smaller than a patch. |
| Shared patches | Patches two components both produce are kept once, and `edge` is true if either flags it. |
| Ids | The index in the `(y0, x0)` order of every candidate, before any rule: `patch_%05d`. Kept ids have gaps. |
| Fraction | Mask pixels in the patch ÷ its area (`mask[y0:y1, x0:x1].mean()`). |
| Count | Points whose pixel lies in the patch. The pixel is the centroid rounded half to even (numpy `round`) and clamped. |

The anchors' rules:

| Anchor | Footprint = fraction mask | Rules |
|---|---|---|
| Collagen | IMC acquired area, `imc_roi_label_map > 0` | `imc_acquired_fraction >= 0.5`; `n_imc_cells >= 50`, counting labelled cells (decoded type ≠ "") |
| Copper | LA tissue: any of Cu, Zn, Fe, Mg, Mn `!= 0` | `laicpmsi_tissue_fraction >= 0.5`. Copper also drops a patch whose Cu p50 is not finite. That is a feature rule, and R3's job. |

SIMBA's tissue test is `!= 0`, so NaN counts as tissue. Pass `nonZero` to match it. The default,
`positiveFinite`, gives the same mask whenever the rasters have no NaN or negatives, which is
the case on both checked specimens.

## Hand-off to zonal statistics (R3)

Patches overlap at stride < size, so one label image cannot hold them all. Use polygons:

```ts
const grid = patchGrid({ footprint, size: 300, stride: 150, units: "um", pixelSize: 1, rules });
const res = await zonalStatsTiled({
  level: 0,
  regions: { kind: "polygons", polygons: patchPolygons(grid.patches) },
  channels: [{ source: cu, support: metals, name: "cu" }],
  quantiles: [0.25, 0.5, 0.75],
  exact: true,
});
// res.table row k is grid.patches[k].
```

- **Why polygons.** R3 counts a pixel when its centre (i + ½, j + ½) is inside. So a rectangle
  with corners at its integer bounds covers exactly `[x0, x1) × [y0, y1)`, the pixels its support
  was counted over. R3 splits the overlap itself (`polygonLayers`, the same greedy split as
  `patchLayers`) and bins every layer into the same rows. No label image is built at level 0,
  where Leap035's would be 2 GB.
- **Label rasters.** `patchLayers` plus `patchLabelImage(patches, layer, w, h, scale)` give one
  u32 image per layer in R3's label convention (label L = patch L − 1, 0 = none), for a
  `regions: { kind: "labels" }` source or another consumer. At every `scale` they equal what R3
  rasterises from `patchPolygons` (tested at 1, 2 and 3). A lattice with `size = k · stride`
  needs k² layers (4 at half overlap). Real grids need more (6 to 14 below): kept trailing-edge
  patches and separately anchored components add them.
- **Batch a big grid for exact quantiles.** R3's exact pass holds every value in each row's
  target bins. Overlapping patches multiply that: Leap035's 14,174 patches at 300 µm collect
  73M values, past the default 128 MiB binding, and `beginCollect` throws. Rows are independent
  and ids run top to bottom, so call `zonalStatsTiled` on slices of `patches` (3,000 at a time
  in the parity script). Each slice is a band of the slide and touches mostly its own tiles.
- **Rectangles.** `rectsArray(patches)`, `n × 4` u32, is the `patchSupport` op's input.
- **Support.** The grid's rule values (level 0, exact) decide the kept set. R3's
  `supportFraction` re-measures them at the level it reads, and agrees at level 0 (below).

## GPU support

`patchSupport` turns the mask into 0/1 flags (f32: finite and > 0; u32/u8: non-zero) and runs
an exclusive scan (`encodeScan`). The flat scan already is a row-wise summed-area table: a
rectangle sums `P[yW + x1] − P[yW + x0]` over its rows, with `P[n]` = the scan total. Offsets
wrap past 2³² pixels, but a difference within one row cannot, so the counts stay exact.

The ceiling is one storage binding of flags: 33.5 Mpx (about 5,800²) at the default 128 MiB
limit. Past it, `checkBindingSize` throws. The layout itself (components, candidates) is
data-dependent, so it is not a graph node. It runs on the host; the op re-measures support on a
resident mask, such as a threshold or morphology output.

## Validation against SIMBA's patch tables

**Result: identical on all four region tables.** That covers the same kept set, the same bounds, the
same patch ids and edge flags, and bit-identical support values. Inputs are the registered
1 µm/px rasters (`tnbc_registered_res_1.0`). They were exported by SIMBA's own loaders: the
union-`!= 0` tissue mask, `roi_label_map > 0`, and the centroids of labelled cells keyed by
(ROI, local id) as `build_global_segmentation` keys them. The comparison was against
`tnbc_simba_runs/*/regions/*/<leap>_patch_regions.csv`.

| Specimen, anchor | Size / stride (µm) | SIMBA kept | intraspatial kept | Candidates | Differences | Label layers | Host time |
|---|---|---|---|---|---|---|---|
| Leap034 collagen | 300 / 150 | 387 | 387 | 387 | none | 9 | 11 ms |
| Leap034 collagen | 150 / 75 | 1,504 | 1,504 | 1,558 | none | 9 | 11 ms |
| Leap034 copper | 300 / 150 | 831 | 831 | 2,060 | none | 6 | 16 ms |
| Leap035 copper | 300 / 150 | 14,174 | 14,174 | 22,255 | none | 14 | 121 ms |

"Differences" covers bounds present on one side only, `patch_id`, `edge_patch`,
`imc_acquired_fraction` / `n_imc_cells` / `laicpmsi_tissue_fraction` compared with `===`, and
all were zero. Host time is the layout plus support, after the runs are built. Building the runs
takes 0.1 s for Leap034 (62 Mpx) and 0.8 s for Leap035 (522 Mpx, 16 tissue components, 187k runs).

Notes:

- The brief quoted 383 kept patches for Leap034 collagen at 300 µm. SIMBA's table and its
  `anchor_args.json` both say **387**.
- None of the five LA metals has a NaN, negative or infinite pixel on either specimen. So
  `!= 0` and `positiveFinite` give the same tissue mask there.
- There are more label layers than the 4 a single lattice needs: kept trailing-edge patches
  (26 to 207 per table) and Leap035's 16 separately anchored components add them. This is why
  rectangles are the simpler R3 input.
- Copper's finite-Cu-p50 filter is a feature rule and isn't applied by the grid. It dropped
  nothing on these specimens: every kept patch has tissue, so its Cu p50 is finite.

### Grid → R3 → SIMBA's per-patch features

`pnpm patchgrid:parity [--leap L] [--anchor copper|collagen] [--size 300|150]` runs the whole
chain from the registered OME-TIFFs, with no Python export. The masks are read in row bands, and
the centroids come from the segmentation by (ROI, local id). It lays the grid, hands the kept
patches to `zonalStatsTiled` as `patchPolygons` (1024 bins, exact quantiles), and compares with
SIMBA's `<leap>_copper_patch_features.csv` and `features/<run>/<leap>_imc_patch_features.csv`.

| Specimen, anchor | Size (µm) | Patches | Kept set, ids, edges | Median vs SIMBA | Support vs SIMBA | R3 time |
|---|---|---|---|---|---|---|
| Leap034 copper | 300 | 831 | identical | `laicpmsi_cu_p50` identical | `n_tissue_pixels`, `tissue_fraction` identical | 6.0 s |
| Leap034 collagen | 300 | 387 | identical | `imc_collagen_median` ≤ 5.6·10⁻⁸ rel | `imc_acquired_fraction` identical | 1.8 s |
| Leap034 collagen | 150 | 1,504 | identical | `imc_collagen_median` ≤ 6.0·10⁻⁸ rel | `imc_acquired_fraction` identical | 1.4 s |
| Leap035 copper | 300 | 14,174 | identical | `laicpmsi_cu_p50` identical in f32 (5 differ ≤ 5.4·10⁻⁸ rel in f64) | `n_tissue_pixels`, `tissue_fraction` identical | 205 s (5 batches) |

The collagen medians, and 5 Leap035 Cu medians, differ in f64 for the reason in
[`zonal-stats.md`](zonal-stats.md): `np.median` on float32 averages the two middle values in
f32, while R3 interpolates in f64. Each of the five Cu values rounds to SIMBA's exactly in f32.
R3 times include TIFF decoding of the tiles it touches. Leap035's 205 s is 1,759 tile visits
through a 256 MB cache per raster on a 522 Mpx slide; building its masks and grid took 64 s.

## Later

- Done as a prototype: `playground/patchgrid.html` lays the grid over an uploaded or fetched PNG
  mask (or a made-up section), re-running `patchGrid` on every slider step (size, stride,
  minimum fraction) and drawing it on WebGPU. It also sweeps the counts against patch size. On the
  1/8-resolution Leap034 collagen mock at 8 µm/px it gives SIMBA's 387 / 387 at 300 / 150 µm.
  Its `MaskLayer` (bit-packed mask) and `RectLayer` (rectangles by flag) are candidates for
  `@intraspatial/deck` if MDV wants patches drawn.
- Point counts on the GPU (bin centroids with `gridIndex`). Cell counts are cheap on the host
  today.
