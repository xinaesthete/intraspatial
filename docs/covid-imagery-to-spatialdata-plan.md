# Migrating the COVID project's imagery to SpatialData zarr

**Status: planned, not started (2026-08-01).** Written while the facts were in front of me, because
every one of them below cost a probe to establish and the extension lies about the format. Nothing
here is in flight, so it is deliberately not an ADR (see `docs/decisions/README.md` — ADRs are for
decisions being acted on).

The cell *table* is already converted: `~/data/covid.mdv.zarr`, 168 MB, and it is all the cell-stats
pages need. What is still MDV-shaped, and still only on the external volume, is the imagery.

## What is actually there

`/Volumes/Crucial X8/covid/images/` — 22 GB, and it is two unrelated things sharing a directory:

| | count | size | what |
|---|---|---|---|
| `spatial_stats_roi/`, `spatial_stats_disease/` | 141,484 | 15.5 GB | **precomputed stats plots** — the PNG library this whole stream exists to replace |
| `*.ome.png` | 32 | 6.0 GB | the IMC stacks, one per ROI |
| flat `*.png` | 152 | 1.0 GB | per-ROI morphology, segmentation and composites |

**Only the bottom two rows are in scope.** The 15.5 GB of stats plots are outputs, not inputs; they
are the thing being computed live instead (and per the dataset analysis, 95.4% of them depict
non-findings). Migrating them would be preserving the problem.

## The gotcha, first

**`*.ome.png` files are not PNGs. They are OME-TIFFs.** Magic bytes `II*\0`, LZW-compressed. Any
migration script that dispatches on the extension — or hands them to a PNG decoder because the name
said so — fails on all 32, and the failure will look like corruption rather than a mislabel.

Each one is, measured from `uDFaO.ome.png` (COVID_SAMPLE_16_ROI_3):

- **49 channels**, 2000 × 2000, **float32** (`SampleFormat=3`, `SignificantBits=32`), LZW, one plane
  per IFD, `DimensionOrder="XYCZT"`.
- Channel names are real metadata and worth keeping: `80ArAr`, `89Y`, `127I`, `131Xe`, `134Xe`,
  `138Ba`, `aSMA`, `CD56`, `HLADR`, `EpCAM`, `CD107a`, `CD16`, … — i.e. calibration/background
  channels mixed in with the antibody panel, which a naive "channel 0 is the first marker" mapping
  would get wrong.
- 49 × 2000² × 4 B = **784 MB raw per ROI**, ~250 MB after LZW. So ~25 GB raw across 32 ROIs against
  6.0 GB stored.

## Per-ROI structure, from `datasources.json`

Each of the 32 entries under `regions.all_regions.<ROI>` carries:

- `ome_tiff` / `viv_image.file` → the 49-channel stack above (both keys, same file)
- `images.he` → **H&E, tissue morphology.** RGB 8-bit, ~2000 × 2000
- `images.cellmask` → **cell segmentation**
- `images.DNA1_Ir191` → 8-bit *grayscale*
- `images.MagentaaSMA_LimeCD68_WhiteCollagen1_BlueDNA`, `images.MagentaaSMA_WhiteEpCAM_LimeCD31_BlueDNA`
  → pre-rendered RGB composites

Three inconsistencies to handle rather than discover:

1. **`he` is called `un` on the HEALTHY samples** (`HEALTHY_SAMPLE_1_ROI_1`, `_1_ROI_2`,
   `_2_ROI_1`, `_2_ROI_2`). Same role, different key.
2. **Not every ROI has every image.** `COVID_SAMPLE_4_ROI_3` and `COVID_SAMPLE_6_ROI_*` have no
   `DNA1_Ir191`; `COVID_SAMPLE_6_ROI_1` has no `he` either. The migration must treat each key as
   optional, not assume a fixed set of five.
3. Names are opaque 5–6 character IDs (`uDFaO.ome.png`, `q9Qtix.png`) with no ROI in them, so the
   only route from ROI to file is `datasources.json`. It is 129 KB and already copied alongside the
   zarr.

## What the target should look like

The interesting part is that these are three genuinely different SpatialData element kinds, and
flattening them into one would be the main thing to get wrong:

- **`cellmask` is a `labels` element, not an `images` one.** It is a label image whose pixel values
  are cell IDs, so it (a) must be lossless, (b) must be resampled with nearest-neighbour at every
  pyramid level, and (c) is the natural join to the existing cell table via `instance_key` — which
  is the thing that would make the segmentation *interactive* rather than a picture. Averaging two
  neighbouring label values invents a cell that does not exist.
- **`he` / composites are `images`** — 8-bit RGB, perceptual, and the natural fit for this repo's own
  lossy HTJ2K path. This is the same shape as the Xenium `he_image` slice already landed under the
  spatialdata.js Loader work.
- **The 49-channel float32 stack is `images` too, but a different problem.** High bit depth,
  many channels, quantitative values that downstream statistics read — so the compression question is
  real rather than cosmetic, and it lands squarely on `docs/dwt-gpu-and-high-bit-depth.md`. Worth
  measuring lossless HTJ2K against the current LZW before assuming lossy is needed at all; LZW on
  float32 is close to the worst case, so the headroom may be large.

## Open questions, in the order they need answering

1. **Do the composites need migrating at all?** They are renderings of channels that are already in
   the stack. If the viewer can composite live — which is the point of the channel-mixing work — they
   are 3 of the 5 per-ROI PNGs and can simply be dropped.
2. **Lossless or lossy for the IMC stack?** Measure first. It carries the quantitative signal, and
   `cellmask` is non-negotiable, but H&E is not.
3. **Where does the pyramid come from?** Nothing here is pyramidal today; 2000² is small enough that
   it may not need to be, which would keep the conversion simple.
4. **One SpatialData zarr, or one per ROI?** The 32 ROIs are separate tissue sections and every
   statistic in this stream is per-ROI (see `docs/cell-stats.md`), so the pressure is toward
   separate — but the cell table is already one store covering all 32.

## Why it is worth doing eventually

It closes the loop the repo is already most of the way around: a SpatialData store carrying cells,
labels and images, read by the spatialdata.js Loader, with the imagery in this project's own codec —
and the segmentation joined to the table rather than flattened into a picture. It also removes the
last reason to have the external volume mounted.
