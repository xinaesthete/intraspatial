# The COVID SpatialData store: what was built, and what the plan got wrong

**Status: done and validated (2026-08-01).** `~/data/covid.spatialdata.zarr`, 5.21 GB, zarr v3,
SpatialData 0.8 / OME-NGFF `0.5-dev-spatialdata`. Execution record for
`docs/covid-imagery-to-spatialdata-plan.md` (the survey),
`docs/imc-image-compression-measurements.md` (the codec decision) and
`docs/mdv-zarr-to-spatialdata-tables.md` (the table side). Not an ADR: nothing here is in flight.

The decisions those three documents took all survived contact. What did not survive is a set of
*facts* they asserted about the data — most importantly that the cell table and the imagery share a
y axis. They do not, and nothing about the resulting store looks wrong when you get it backwards.

## What is in the store

| | elements | size | codec |
|---|---|---|---|
| `images/<ROI>_imc` | 32 | 4.36 GB | zstd level 19, one channel per chunk, 2 levels |
| `images/<ROI>_he` | 30 | 0.74 GB | HTJ2K reversible, 1024² tiles, 3–4 levels |
| `labels/<ROI>_labels` | 32 | 0.009 GB | zstd level 19, uint16 |
| `tables/table` | 1 | 0.099 GB | AnnData, dense `X` |

**One store, one coordinate system per ROI**, named after the ROI. The 32 ROIs are separate tissue
sections that share no space, so a common `global` would assert an overlap that does not exist —
`sd.coordinate_systems` has 32 entries and nothing is in `global`. That answers plan open question 3
without splitting the store, which would have separated the imagery from a cell table that already
covers all 32.

Against 6.43 GB of IMC + 0.68 GB of H&E in scope today, that is **1.37×** overall — and the
comparison is honest, because both sides now include a pyramid (see finding 2). The 15.5 GB of
precomputed stats plots remain out of scope by the plan's own argument.

```bash
uv run --with spatialdata --with tifffile --with imagecodecs --with h5py --with scipy \
  --with scikit-image --with ~/code/www/SpatialData.ts/python/spatialdata-js-util \
  python scripts/covid-imagery-to-spatialdata.py he
```

then `imc`, `labels`, `table`, and `retransform` (metadata-only affine rewrite) — plus one
`spatialdata-js-util images recompress --codec experimental.openjph_htj2k --preset lossless
--pyramid` between `he` and `imc`, while the store is still small enough to copy cheaply.
`scripts/covid-imagery-inventory.py` is the survey, and `scripts/covid-spatialdata-validate.py`
is the check.

## Finding 1 — MDV's y axis points up, and every plan document assumed it did not

**This is the one that would have shipped a plausible, wrong store.** The cell table's `y`, and every
`position` in `datasources.json`, are measured from the **bottom** of the ROI. NGFF rasters are
top-down. The conversion is `y_image = H_roi − y_mdv`, with `H_roi` the cellmask's extent.

It is not subtle once you look for it, and it is invisible if you do not. Sampling the cellmask at
each centroid:

| convention | centroids landing on foreground, mean over 32 ROIs |
|---|---|
| `row = y` (as stored) | **25.1%** |
| `row = H − 1 − y` | 95.7% |
| **`row = H − y`** | **99.8%** (range 99.5–100.0) |

25.1% is not "poorly registered" — it is *exactly* the mean foreground fraction, i.e. pure chance.
The trap is that `docs/mdv-zarr-to-spatialdata-tables.md` finding B measured the same mask, found it
41% foreground, and concluded that touching cells must have merged into one blob. A reader checking
centroids against that mask would have got 41.5% hits on that ROI and read it as confirmation.

The flip is over the **extent**, not the last index: `H − 1 − y` scores 95.7%, which is high enough
to look correct and is wrong by one pixel everywhere.

It registers against the IMC stack too, not just the mask — mean `DNA1` at the centroids is 2.9× the
image mean under the flip and 0.99× without it, so the images written here need no change; it is the
table that converts.

### The rasters are placed in that same bottom-up space

The H&E `position` is a bottom-up box, so the image-space top edge is `H_roi − position_y − height`,
not `position_y`. Established by `scripts/covid-he-registration-check.py`, which resamples each H&E
onto its ROI grid and FFT-cross-correlates it against `arcsinh(DNA1)` — nuclei are dark in H&E and
bright in DNA1, so the true alignment is a correlation *minimum*.

19 of 30 H&E images land within 30 µm of the flipped prediction against 8 of 30 for the unflipped
one, and on the subset that can actually discriminate (the two predictions ≥ 100 µm apart, and a
correlation peak sharp enough to be a measurement at all) it is 3–0 with median error 16 µm against
256 µm. Every correction was in y; `dx` came out 0.0 for all 30, which is the control.

Sizes of the correction: 920 µm on `COVID_SAMPLE_5_ROI_2`, 544 µm on `COVID_SAMPLE_8_ROI_1`, 414 µm
on `COVID_SAMPLE_8_ROI_3` — against ROIs 2000 µm across.

**Residual caveat, stated rather than smoothed over:** three H&E images (`COVID_SAMPLE_4_ROI_2`,
`5_ROI_2`, `5_ROI_3`) carry `position (0,0)` and scale exactly 1.0 and have a flat correlation
surface. They look unregistered in the source. They are converted under the same rule as the rest,
which is the best available answer, but they are the three to distrust.

## Finding 2 — the IMC stacks are already pyramidal, and the level is a subsample

Plan open question 2 says "nothing here is pyramidal today". For the IMC stacks that is wrong:
`tf.pages` reports 49 planes but `tf.series[0].levels` reports **two**, and the hidden half-resolution
level is 20% of the file — which is also why the plan's 6.0 GB and the measured 6.43 GB disagree, and
why per-file LZW read 197.5 MB in the compression doc (pixel bytes of level 0) against 247.4 MB on
disk.

The source's level 1 is **exactly `level0[::2, ::2]`** — max absolute difference 0 on every channel
checked, against 444–866 for a block mean. So it is a plain subsample, and reproducing that instead
of taking SpatialData's default coarsen-mean is worth **1.67×** on the pyramid level:

| level 1 of one 49-channel stack, zstd-19 | size |
|---|---|
| exact 2× subsample (what the source stores) | **33.8 MB** |
| coarsen mean (spatialdata's default) | 56.4 MB |

Averaging four float32s manufactures bit patterns that were not in the data, which destroys the
repetitive-symbolic structure zstd is exploiting — the same mechanism as compression finding 2,
arriving from a different direction. A block mean would be the better *statistic*, but level 1 is a
viewing level and every number the analysis consumes is read from level 0. Across the dataset the
choice is worth about 0.7 GB.

Both levels are verified bit-identical to the source, including the pyramid level.

## Finding 3 — the cellmask already separates every cell, so the labels are recovered, not reconstructed

`docs/mdv-zarr-to-spatialdata-tables.md` chose a seeded watershed because "plain connected components
will not do: the mask is 41% foreground, so touching cells merge into one blob."

Measured across all 32 ROIs: **2 blobs in the whole dataset contain two centroids.** Not 2%, two.
Every other connected component holds exactly one, so each watershed basin *is* a whole component and
the watershed and connected-components agree everywhere but those two blobs. The segmentation was
written with gaps between touching cells.

That upgrades the result from "an approximation of the segmentation" to **the original per-cell masks
recovered exactly**, which is a much stronger thing to have under a table.

Corroboration from a different direction: the surplus components match the gaps in the cell
numbering. `COVID_SAMPLE_16_ROI_3` has 32,569 cells in the table, `_CELL_<n>` suffixes running to
34,878, and **34,803** connected components — the extra blobs are cells the published table filtered
out but the mask still draws.

The watershed is kept anyway: it costs about a second per ROI, it degrades gracefully if a future ROI
does have merged cells, and `multi_seed_blobs` in the per-ROI report is what would say so.

| the oracle | |
|---|---|
| cells in the table | 545,400 |
| cells with a label | **544,443 (99.82%)** |
| centroids outside the foreground | 957 (0.18%), flagged in `obs['in_labels']` |
| seed collisions | 0 |
| blobs holding two cells | 2 |

The 957 are the "coverage is not always complete" case the tables doc anticipated from Xenium's
`nucleus_boundaries`. SpatialData cannot say "this row annotates nothing", so they carry a normal
`instance_id` and a boolean saying it resolves to no pixels.

Label values are `_CELL_<n>` **+ 1**: the suffixes start at 0 in some ROIs and label 0 is background,
so a raw suffix would make those cells invisible. Global max suffix is 34,878, so uint16 keeps 47%
headroom. The suffixes are **not contiguous**, which is worth knowing before treating them as a range.

## Finding 4 — spatialdata 0.8 writes exactly what was wanted, so none of the hand-writing is needed

`docs/mdv-zarr-to-spatialdata-tables.md` part 1 works through hand-writing zarr v2, trailing-chunk
zero padding, `dimension_separator`, and the AnnData encoding contract, because `zarrita.create`
emits v3 only. None of that applies to this store: **spatialdata 0.8 writes zarr v3 natively, with
plain zstd as its default raster codec**, and `write_element(..., raster_compressor={"zstd": 19})`
sets the level directly. It also builds the AnnData encoding — `encoding-type` on every node,
categoricals as groups, the duplicated `region`/`region_key`/`instance_key` — so gotchas 3, 4 and 5
of that document are the library's problem rather than ours.

Two details worth keeping:

- The v3 path does **not** byte-shuffle (only the v2 Blosc path forces `shuffle=1`), so the "never
  shuffle" result holds without doing anything.
- `_apply_compression` maps `{"zstd": 19}` onto `zarr.codecs.ZstdCodec(level=19)` — a core zarr v3
  codec, which was the reason for preferring plain zstd over Blosc.

## Finding 5 — one experimental codec gates the whole store

The H&E is HTJ2K, so `spatialdata.read_zarr` on the store fails without `spatialdata-js-util`
installed — not just for those 30 elements, for everything:

```
zarr.errors.UnknownCodecError: Unknown codec: 'experimental.openjph_htj2k'
```

The error is clean and names the codec, and the failure is total. Stock `anndata.read_zarr` on
`tables/table` and stock `zarr.open_array` on any IMC or labels array still work, so the data is not
trapped — but the *store* is not openable by a plain SpatialData install. That is the real price of
the 1.31× the H&E gains, and it was not stated when the codec was chosen. Reversing it is a re-run of
`he` + `retransform` (about a minute), not a re-encode of the 4.4 GB.

The measurement itself reproduced exactly: 519.8 MB of HTJ2K against 681.2 MB of PNG, **1.31×**, and
lossless verified by comparing decoded pixels to the source PNG. zstd-19 on the same H&E gives
705.4 MB, i.e. *worse* than PNG — so for this one element kind the codec really is the answer.

The pyramid costs 221 MB on top of the 520 MB, which is more than the textbook 33% because small
tiles compress worse. Worth it by the compression doc's own argument: nothing here was pyramidal and
a zoomed-out view otherwise pulls full-resolution chunks.

## Finding 6 — corrections to the survey

- **10 of the 32 stacks are not 2000², not eight**, and the range is wider than recorded: from
  2000 × 324 to 4000 × 1000, 11 distinct extents. Raw total 22.30 GB, confirming the plan's revision.
- **`COVID_SAMPLE_6_ROI_1`'s cellmask is 2000 × 1086 while its IMC stack is 2000 × 2000.** The cell
  table's coordinates stop at 1085.1, so the mask — not the stack, and not the `roi` box, which
  rounds to 1100 — defines the analysis grid. The stack's extra 914 rows sit below it; the `DNA1`
  test puts the ROI at the stack's top-left, so the stack keeps an identity transform. This is the
  only ROI where the three disagree, and it is also one of the two with no H&E.
- **One H&E is genuinely anisotropic.** `COVID_SAMPLE_11_ROI_2` is 4538 × 2723 px over
  3336.5 × 2310.3 µm — 0.735 against 0.848 µm/px, a 15% difference. Every other image is isotropic to
  rounding. A single `scale` transformation cannot express it, which is why each raster gets a full
  affine with per-axis scale.
- **`he` is `un` on the four HEALTHY samples**, and missing on `COVID_SAMPLE_4_ROI_3` and
  `COVID_SAMPLE_6_ROI_1` — both as surveyed. Alpha is constant 255 on all 30, checked rather than
  assumed, and dropped.
- **The panel intersection lands exactly as predicted**: 37 of the 49 channels are table columns, and
  the 12 absent are precisely the calibration/background channels (`80ArAr`, `89Y`, `127I`, `131Xe`,
  `134Xe`, `138Ba`, `194Pt`, `195Pt`, `196Pt`, `198Pt`, `208Pb`, `209Bi`). The rule "intersect with
  the panel" needs no hand-curated list.
- **`cellID` is fine in `datafile.h5`** — the zeros are only in `~/data/covid.mdv.zarr`. This
  conversion reads the h5 directly and so does not inherit finding A's converter bug; that bug is
  still there and still worth fixing for the MDV store's own sake.

## Practical notes for anyone re-running this

- **`write_element` cannot overwrite an element it read back.** `read_zarr` leaves elements *backed
  by* the store, and spatialdata then refuses both to overwrite and to delete them (discussion #520).
  The fix is a detached `SpatialData()` with `.path` set — it writes and deletes elements happily,
  and it is safe here because the source of truth is the external volume, not the store.
- **spatialdata hardcodes `unit="unit"`** on spatial axes with no per-element API, and says in its
  own source that the user should replace it before saving. The store carries consolidated metadata,
  so the fixup has to happen *before* re-consolidating or the root keeps a stale copy that readers
  prefer.
- **`spatialdata-js-util images recompress` rewrites multiscale metadata from its own reading**, which
  resets that unit. Hence the separate `units` pass, to run after any external tool touches the store.
- **Affines are metadata.** Correcting finding 1 across 94 elements was `write_transformations`, not a
  re-encode — worth remembering before rewriting 4 GB to move an image 500 µm.
- Run the writer with `python -u` if you pipe it; a full IMC pass is ~25 minutes and buffered output
  makes it look hung.

## Validation

`scripts/covid-spatialdata-validate.py` — all checks pass. Deliberately not a re-run of the writer's
bookkeeping: each check either reads the original file on the volume or exercises the join a consumer
would make, because a writer that is wrong about the y axis will verify itself happily.

The centrepiece is check 4: sample the labels raster at each cell's own `obsm/spatial`, transformed
through SpatialData's coordinate machinery rather than the script's arithmetic, and require the pixel
to be that cell's `instance_id`. **24,000/24,000 sampled cells across 6 ROIs, 100.00%.** That single
test fails if the y flip, the affine, the `+1` label offset or the region wiring is wrong.

Also checked: IMC pixels bit-identical to the source at both levels; `X` columns identical to
`datafile.h5`; every `region` names an element that exists; every spatial axis is `micrometer`; and
the consolidated copy agrees with the elements it summarises.

## What is not done

- **Shapes circles from `obsm/spatial`** — step 3 of the tables doc's order, as the cheap-render
  fallback and the answer for cells the mask does not cover. Skipped because the labels came out at
  99.82%, which was the condition that made circles interesting. The "one annotating element per row"
  gap the tables doc describes is therefore not yet forced, and neither is
  `docs/proposals/instance-views.md`.
- **The three stats tables** (`spatial_stats`, `spatial_stats_disease`, `spatial_stats_roi`) are left
  where they are, per gotcha 7 — they annotate nothing spatial and are the outputs this stream
  computes live.
- **Retiring the MDV project still needs live channel mixing.** The plan's sequencing constraint is
  unchanged: the derived composites and `DNA1_Ir191` were not migrated, so until a viewer can
  composite from the 49-channel stack the new store shows less than the old project does.
- **`~/data/covid.spatialdata.zstd-he.zarr`** (676 MB) is the pre-HTJ2K H&E intermediate, kept only in
  case the finding-5 trade-off is worth revisiting. Nothing depends on it.
