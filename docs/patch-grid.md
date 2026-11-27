# Patch grid (R4 of region/raster statistics)

**Decision.** `patchGrid` (`src/spatial/patchGrid.ts`) reproduces SIMBA's patch layout and
support rules exactly on the host. It works over row runs, so a 1 µm/px canvas costs its runs,
not its pixels. Kept patches are returned as rectangles. For zonal statistics (R3) the overlap
is resolved into **label layers**: sets of patches that don't overlap, one u32 label image each.
Support fractions can also be counted on the GPU, over a resident mask, by the `patchSupport`
op (one flat prefix scan, then one subtraction per row per rectangle). Both paths give
identical integers.

## Surface

| What | Where | Notes |
|---|---|---|
| `patchGrid(opts)` | `intraspatial/spatial` | footprint → candidates → support → kept |
| `maskRuns`, `maskRunsFromRows`, `unionRuns`, `positiveFinite`, `nonZero` | `intraspatial/spatial` | masks as row runs; rows can be streamed |
| `componentBoxes`, `axisStarts`, `patchCandidates` | `intraspatial/spatial` | the layout, step by step |
| `maskCountsInRects`, `pointPixels`, `pointCountsInRects` | `intraspatial/spatial` | support counts |
| `rectsArray`, `patchLayers`, `patchLabelImage` | `intraspatial/spatial` | the hand-off to R3 |
| `patchSupport` op | `intraspatial/graph` (registered built-in) | mask grid + `n × 4` u32 rects → `n × 1` u32 counts |
| `patchSupportResident`, `patchSupportGpu` | `intraspatial/gpu/spatial` | caller-owned buffers / host readback |

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

## Overlap → label layers

Patches overlap at stride < size, so one label image cannot hold them all. R3 gets either input:

- **Rectangles**: `rectsArray(patches)`, `n × 4` u32 `[x0, y0, x1, y1]`. A zonal reduction can
  gather per rectangle directly, with no label image. This is the only form that scales to a full
  1 µm/px canvas: Leap035 is 522 Mpx, so a u32 label image would be 2 GB.
- **Label layers**: `patchLayers(patches)` greedily splits the patches into sets with no overlap.
  `patchLabelImage(patches, layer, w, h, scale)` paints one set with value = patch index + 1, and
  0 elsewhere. `scale` gives a coarser canvas, where each label pixel takes the patch holding its
  centre. A lattice with `size = k · stride` gives k² layers, so 4 at half overlap. Trailing-edge
  patches and differently anchored components add a few more. R3 reduces each layer and
  concatenates the rows, so row = label − 1 = index into `patches`.

Shared types (`Box`, `PatchRect`, `Patch`) live in `src/spatial/patchGrid.ts`. Reconcile them
with R3's region type when both land.

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

**Result: identical on all five tables.** That covers the same kept set, the same bounds, the
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
- Not rerun here: copper's finite-Cu-p50 filter (a feature rule). It dropped nothing on these
  specimens, since the kept sets already match.

## Later

- Done as a prototype: `playground/patchgrid.html` lays the grid over an uploaded or fetched PNG
  mask (or a made-up section), re-running `patchGrid` on every slider step (size, stride,
  minimum fraction) and drawing it on WebGPU. It also sweeps the counts against patch size. On the
  1/8-resolution Leap034 collagen mock at 8 µm/px it gives SIMBA's 387 / 387 at 300 / 150 µm.
  Its `MaskLayer` (bit-packed mask) and `RectLayer` (rectangles by flag) are candidates for
  `@intraspatial/deck` if MDV wants patches drawn.
- Point counts on the GPU (bin centroids with `gridIndex`). Cell counts are cheap on the host
  today.
