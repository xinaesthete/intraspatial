# Migrating the COVID project's imagery to SpatialData zarr

**Status: DONE (2026-08-01) — see `docs/covid-spatialdata-migration.md` for what was built and
which claims below turned out to be wrong.** This document is kept as the survey it was; the
decisions in it all held, but three of its facts did not:

1. **MDV's y axis points up.** Every `position` here, and the cell table's `y`, is measured from the
   bottom of the ROI. Read as image-space they misplace 19 of the 30 H&E images, by up to 920 µm.
2. **The IMC stacks are already 2-level pyramids** (open question 2 below says otherwise) — the
   hidden level is 20% of each file, and it is an exact 2× subsample.
3. **`cellmask` already separates every cell.** Only 2 blobs in the entire dataset contain two
   centroids, so the labels are the original segmentation recovered exactly, not reconstructed.

Also: 10 stacks are not 2000², not eight; and `COVID_SAMPLE_6_ROI_1`'s mask is 2000 × 1086 against a
2000 × 2000 stack.

<details><summary>Original status</summary>

**Status: planned, not started (2026-08-01).** Written while the facts were in front of me, because
every one of them below cost a probe to establish and the extension lies about the format. Nothing
here is in flight, so it is deliberately not an ADR (see `docs/decisions/README.md` — ADRs are for
decisions being acted on).

</details>

The cell *table* is already converted: `~/data/covid.mdv.zarr`, 168 MB, and it is all the cell-stats
pages need. What is still MDV-shaped, and still only on the external volume, is the imagery.

## What is actually there

`/Volumes/Crucial X8/covid/images/` — 22 GB, and it is two unrelated things sharing a directory:

| | count | size | what |
|---|---|---|---|
| `spatial_stats_roi/`, `spatial_stats_disease/` | 141,484 | 15.5 GB | **precomputed stats plots** — the PNG library this whole stream exists to replace |
| `*.ome.png` | 32 | 6.0 GB | the IMC stacks, one per ROI |
| flat `*.png` | 152 | 1.0 GB | per-ROI morphology, segmentation and composites — a third of it derived, see below |

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
- 49 × 2000² × 4 B = **784 MB raw per ROI**, 197.5 MB after LZW (measured, this file). **Not every
  stack is 2000²** — eight of the 32 have other extents, from 930 × 2750 to 1250 × 3500, though all
  32 do have 49 channels. So the true raw total is **22.3 GB** across 32 ROIs against 6.0 GB stored.

## Per-ROI structure, from `datasources.json`

Each of the 32 entries under `regions.all_regions.<ROI>` carries:

- `ome_tiff` / `viv_image.file` → the 49-channel stack above (both keys, same file)
- `images.he` → **H&E, tissue morphology.** 8-bit **RGBA** (alpha is constant 255 — drop it), and
  **not 2000², nor a fixed size**: the 30 images have 30 different extents and each carries its own
  offset. This ROI's is 3630 × 3630 px over 1857.2 µm at offset (26.5, 79.5), i.e. ~0.51 µm/px
  against the IMC grid's 1.0. Every one needs its own affine — which SpatialData handles natively
  and MDV's `position`/`width` pair only approximates
- `images.cellmask` → **a BINARY foreground mask**, 8-bit, exactly two values (0/255). Not a label
  image — see `docs/mdv-zarr-to-spatialdata-tables.md` finding B
- `images.DNA1_Ir191` → 8-bit *grayscale* — **not migrating, decided 2026-08-01** (see below)
- `images.MagentaaSMA_LimeCD68_WhiteCollagen1_BlueDNA`, `images.MagentaaSMA_WhiteEpCAM_LimeCD31_BlueDNA`
  → pre-rendered RGB composites — **not migrating, decided 2026-08-01** (see below)

Three inconsistencies to handle rather than discover:

1. **`he` is called `un` on the HEALTHY samples** (`HEALTHY_SAMPLE_1_ROI_1`, `_1_ROI_2`,
   `_2_ROI_1`, `_2_ROI_2`). Same role, different key.
2. **Not every ROI has every image, and this still bites after the drops below.** Counted over the
   two keys actually being kept: `cellmask` is present on all 32, but **`he`/`un` is missing on 2** —
   `COVID_SAMPLE_4_ROI_3` and `COVID_SAMPLE_6_ROI_1`. So those two ROIs get a mask and no morphology
   image at all. Treat each key as optional; do not assume a fixed set.
3. Names are opaque 5–6 character IDs (`uDFaO.ome.png`, `q9Qtix.png`) with no ROI in them, so the
   only route from ROI to file is `datasources.json`. It is 129 KB and already copied alongside the
   zarr.

## What the target should look like

The interesting part is that these are three genuinely different SpatialData element kinds, and
flattening them into one would be the main thing to get wrong:

- **`cellmask` wants to be a `labels` element and cannot be one yet.** A labels element's pixel
  values are instance IDs, joinable to the cell table via `instance_key` — the thing that would make
  the segmentation *interactive* rather than a picture. But the stored mask is **binary** (0/255,
  8-bit), so it carries no instance identity at all; every cell would point at the same blob. It is
  still lossless-and-nearest-neighbour either way — averaging neighbouring label values invents a
  cell that does not exist — but the instance identity has to be reconstructed first. **Decided
  2026-08-01: seeded watershed** from the binary mask plus the 545,400 centroids, producing `uint16`
  labels keyed to `cellID`. See `docs/mdv-zarr-to-spatialdata-tables.md` finding B. Note the mask
  therefore migrates as an *input* to that reconstruction, not as the labels element itself.
- **`he` is an `images` element** — 8-bit RGB, perceptual, and the natural fit for this repo's own
  lossy HTJ2K path. This is the same shape as the Xenium `he_image` slice already landed under the
  spatialdata.js Loader work. It is the only flat PNG that survives as an image at all, the rest
  being either derived (dropped) or a label mask.
- **The 49-channel float32 stack is `images` too, but a different problem.** High bit depth,
  many channels, quantitative values that downstream statistics read — so the compression question is
  real rather than cosmetic, and it lands squarely on `docs/dwt-gpu-and-high-bit-depth.md`.
  **Measured 2026-08-01 — see `docs/imc-image-compression-measurements.md`.** The guess that once
  stood here (LZW near worst case, large headroom) is wrong: LZW suits this data well, and lossless
  HTJ2K on the quantised stack is *53% larger* than doing nothing. The answer is to split the stack
  into a lossless float32 source (zstd-9, shuffle **off**, 1.4× better than today) and a separate
  8-bit arcsinh viewing layer.

## Decided: nothing derived from the IMC stack gets migrated (2026-08-01)

**Every flat PNG that is a rendering of channels the stack already holds is dropped** — both
composites and the single-channel `DNA1_Ir191`. Checkable rather than assumed: every marker named in
those three filenames — `aSMA`, `CD68`, `Collagen1`, `DNA1`, `EpCAM`, `CD31` — appears in the
49-channel list above. Nothing is lost that live channel mixing cannot put back.

Measured, by `images` key across the 32 ROIs:

| key | files | size | keep? |
|---|---|---|---|
| `MagentaaSMA_LimeCD68_WhiteCollagen1_BlueDNA` | 32 | 173.8 MB | **no** — derived |
| `MagentaaSMA_WhiteEpCAM_LimeCD31_BlueDNA` | 32 | 158.2 MB | **no** — derived |
| `DNA1_Ir191` | 26 | 19.9 MB | **no** — derived (`DNA1` is channel 42 of the 49) |
| `he` / `un` | 26 / 4 | 568.8 / 80.9 MB | **yes** — a separate stain, not in the IMC stack |
| `cellmask` | 32 | 7.0 MB | **yes** — segmentation output, not derivable (but binary — see above) |
| total | 152 | 1008.6 MB | **62 files / 656.7 MB kept** |

So the flat-PNG payload falls from 1008.6 MB / 152 files to **656.7 MB / 62**, and what remains is
irreducible by construction: H&E is a different stain, and `cellmask` is an analysis output. The rule
is now simple enough to state as one — *if the stack can regenerate it, it does not get stored* —
which is also the rule to apply to anything similar found later.

**This makes live channel mixing a prerequisite, not a nice-to-have.** It is the intended direction
anyway, but the consequence should be stated: until the viewer can composite from the stack, the
migrated store shows strictly less than the MDV project does today. That is a sequencing constraint
on when the old project can be retired, not a reason to keep the PNGs. Note `DNA1_Ir191` sets the
floor here — a plain single-channel greyscale view has to work before even the simplest of these
three is replaced.

## Open questions, in the order they need answering

1. ~~**Lossless or lossy for the IMC stack?**~~ **Answered 2026-08-01, measured** —
   `docs/imc-image-compression-measurements.md`. Both: a lossless float32 source array and a
   separate lossy 8-bit viewing layer, because no single browser-decodable configuration is both
   small and quantitatively faithful on this data.
2. **Where does the pyramid come from?** Nothing here is pyramidal today; 2000² is small enough that
   it may not need to be, which would keep the conversion simple. `spatialdata-js-util`'s
   `add-pyramid` is the ready-made answer if it is needed — it generates levels through
   SpatialData's own parsers, which get the half-pixel translation right.
3. **One SpatialData zarr, or one per ROI?** The 32 ROIs are separate tissue sections and every
   statistic in this stream is per-ROI (see `docs/cell-stats.md`), so the pressure is toward
   separate — but the cell table is already one store covering all 32.

## Why it is worth doing eventually

It closes the loop the repo is already most of the way around: a SpatialData store carrying cells,
labels and images, read by the spatialdata.js Loader, with the imagery in this project's own codec —
and the segmentation joined to the table rather than flattened into a picture. It also removes the
last reason to have the external volume mounted.
