# Generic zarr, and remodelling our MDV-shaped store into SpatialData tables

**Status: analysis, nothing started (2026-08-01). Four decisions taken, and two findings that
change the plan — see "Decisions" and "What the probing turned up" below.** Companion to
`docs/covid-imagery-to-spatialdata-plan.md`, which covers the imagery; this is the table half.
Everything below was measured against two stores actually on disk — `~/data/covid.mdv.zarr` (ours,
what `pnpm mdv:zarr` produces) and `~/data/mdv_xenium_tiny_test/spatialdata.zarr` (a real
SpatialData 0.4 store) — rather than reasoned from the spec.

## Part 1 — generic arbitrary zarr: better than expected

**zarrita 0.7.3 reads v2 and v3 through the same `open` call.** Verified by pointing one code path at
both stores:

```
v2 SpatialData: OK  shape=36     dtype=float64  (blosc-lz4 + shuffle, decoded transparently)
v3 ours:        OK  shape=545400 dtype=float32  (uncompressed)
```

So format-agnostic *reading* is close to free, and the blosc codec came along without `numcodecs` as
a direct dependency. `withConsolidatedMetadata` / `withMaybeConsolidatedMetadata` are also exported,
which matters because the real store ships a v2 `zmetadata` — use it when present (one request
instead of N) but never require it.

**The asymmetry is writing.** `zarrita.create` emits **v3 only**. Most AnnData/SpatialData stores in
the wild are v2, and a v3 array inside a v2 hierarchy is a mixed store that `anndata.read_zarr` will
not load — the write appears to succeed and the result is unreadable. Two consequences:

- Anything we *write* for external consumption needs either hand-written v2, or a verified claim that
  the consumer accepts v3. Verify with a **foreign reader** (a zarr-python venv), never by reading
  back through the library that wrote it.
- v2 hand-writing has its own trap: the **trailing chunk must be zero-padded to the full chunk
  shape**, not truncated. Uncompressed v2 is otherwise valid and simple.

Note also that v2 and v3 disagree on more than filenames: `compressor` + `filters` (v2) against a
`codecs` pipeline (v3), different `fill_value` semantics, and a `dimension_separator` that defaults
to `.` in v2 — the real store sets `/` explicitly, and a reader that assumes the default will miss
every chunk.

## Part 2 — what we have against what SpatialData wants

**Ours** (`covid.mdv.zarr`): zarr v3, five sibling groups (`cells`, `samples`, `spatial_stats`,
`spatial_stats_disease`, `spatial_stats_roi`), each holding **one flat 1-D array per column**,
uncompressed (`codecs: []`), with all semantics in a custom `mdv` attributes block. Categoricals are
a `uint8` array whose `values` list lives in that block.

**Theirs** (`tables/table/`): zarr v2, an AnnData group with `X`, `obs`, `var`, `obsm`, `layers`,
`obsp`, `varm`, `varp`, `uns`, and `encoding-type` / `encoding-version` on essentially every node.

### The column mapping is derivable, and that is the good news

`cells` has **73 columns**: 56 numeric, 16 text, 1 unique (`cellID`). Nothing in the MDV metadata says
which are *measurements* — but intersecting with the OME-TIFF panel resolves it cleanly:

- **37 of the 49 panel channels are present as columns** → these are `X`, with `var` = the marker
  names.
- The 12 that are absent are exactly the calibration/background channels — `80ArAr`, `89Y`, `127I`,
  `131Xe`, `134Xe`, `138Ba`, `194Pt`, `195Pt`, `196Pt`, `198Pt`, `208Pb`, `209Bi`. So the rule is
  "intersect with the panel", not a hand-curated list that rots.
- The remaining 36 columns are `obs` — morphology (`area`, `perimeter`, `eccentricity`,
  `major_axis_length`, `minor_axis_length`), the annotation sets, `sample_id`, `ROI`, `cellID`.

**Twelve of those 36 are not `obs` columns at all — they are four 3-D embeddings.** `UMAP_1..3`,
`lymphocyte_UMAP1..3`, `myeloid_UMAP1..3`, `structural_UMAP1..3` belong in `obsm` (`X_umap`,
`X_umap_lymphocyte`, …). Flattening them into `obs` loses that they are embeddings and hides them
from every tool that looks in `obsm`. Same for `x`/`y` → **`obsm/spatial` as an `[N,2]` array**, which
is a reshape and interleave, not a rename.

## Decisions (2026-08-01)

1. **The store is zarr v3.** Not a question to reopen — there is strong precedent in this repo and
   v3 is where the ecosystem has been for some time. The consequence is a dependency, not a risk:
   **spatialdata.js / zarrextra need updating** to read v3 before they can consume what we write. That
   is a known piece of work, not a reason to emit v2.
2. **Tables annotate `cellmask` as `labels` elements**, one per ROI, with shapes possibly derived
   alongside. *See finding B below — the current `cellmask` PNGs cannot support this, and that has to
   be solved first.*
3. **`X` stays dense.** `[545400, 37]` float32 = 78 MB. IMC is not sparse the way transcript counts
   are, and the Xenium example's `csr_matrix` should not be copied just because it is the example.
4. **Where sparse is used at all, prefer CSC over CSR.** This is right for the access pattern and
   there is a caveat worth writing down — see below.

### The CSC caveat, from our own reader

`readVarColumns` fetches *an arbitrary set of var columns*, which is exactly what CSC is for: with
CSC, `indptr` bounds each var's slice directly. But `varMatrix.ts` already notes the catch —

> For CSC only the wanted slices are needed in principle, but the arrays are chunked along their own
> length rather than by var, so a partial read would still fetch whole chunks.

So **CSC only pays off if `indices`/`data` are chunked to make a per-var slice cheap.** Choosing CSC
and then chunking those arrays in a few huge chunks gives all of CSR's I/O with none of CSC's
benefit, and it will look like it is working. If we ever write sparse, the chunk size is part of the
decision, not a detail left to a default.

## What the probing turned up

Two things found while checking whether the decisions above are actually implementable. Both are
about data we already have and neither is a blocker to the plan's shape, but both are blockers to a
naive execution of it.

### A. `cellID` is empty in our converted store — a converter bug, and it is the `instance_key`

Every one of the 9 chunks of `cells/cellID` is **zero bytes throughout** (0 nonzero of 19,464,192).
The source is fine: `h5dump` shows `/cells/cellID` as 545,400 × `H5T_STRING` STRSIZE 33, and the
first value is `"COVID_SAMPLE_4_ROI_2_CELL_0"`.

The cause is in `scripts/mdv-h5-to-zarr.ts`: it reads columns via `h5dump -b LE`, which **emits
nothing at all for `H5T_STRING` datasets**. The numeric branch catches this — it asserts
`data.length !== rows` and throws — but the fixed-width-string branch has no equivalent check, so it
wrote 18 MB of zeros and reported success. `cellID` is the only affected column in the whole store,
which is why nothing downstream noticed.

It matters here because **`cellID` is the natural `instance_key`**, and its format is informative:
`<SAMPLE>_<ROI>_CELL_<n>` — per-ROI, with an index that restarts for each ROI. A labels element's
pixel values are integers, so `instance_key` would have to be the parsed `<n>` suffix rather than the
composite string, and whether it is 0- or 1-based against the mask needs checking (label 0 is
conventionally background).

### B. `cellmask` is a BINARY mask, not a label image — decision 2 cannot be executed as stated

Decoded `q9Qtix.png` (COVID_SAMPLE_16_ROI_3's cellmask): 2000 × 2000, 8-bit grayscale, and **exactly
two distinct values** — 0 (2,341,199 px) and 255 (1,658,801 px).

It is a foreground/background rendering of the segmentation, not the segmentation. Eight bits could
not carry instance IDs anyway for an ROI with up to 32,569 cells. **So there is nothing for a table to
annotate**: every cell would point at the same undifferentiated blob, and the join that makes a
labels element worth having does not exist.

Three ways forward, in increasing cost:

- **Derive shapes from `obsm/spatial`** — circles at the centroids, which is what the Xenium store's
  `cell_circles` is. Works today, needs no new data, and gives a valid annotating store immediately.
  Loses cell morphology, which we partly have anyway as `area` / `perimeter` / axis lengths in `obs`.
- **Reconstruct labels** by seeded watershed of the binary mask from the 545,400 centroids. Plain
  connected components will not do — the mask is 41% foreground, so touching cells merge. This is the
  only route to a true labels element from what exists, and it is real work with real failure modes.
- **Re-segment from the IMC stack.** Out of scope, and it would not reproduce the published cells.

The honest reading is that decision 2's *intent* — the table joined to per-cell geometry rather than
flattened into a picture — is right, and shapes-from-centroids delivers most of it now while
reconstruction stays open.

## The gotchas, in the order they will bite

1. **A table alone is not a valid SpatialData store.** A regions table's `region` must name an element
   that *exists*, and today we have neither shapes nor labels — see finding B for why `cellmask`
   cannot currently be the answer. A table with `region: None` is legal but annotates nothing.
   Note the dependency on the imagery plan is *small*: `cellmask` is 7.0 MB of the 656.7 MB being
   migrated, so it can be pulled forward well ahead of the H&E and the 6 GB stack.

2. **`uint8` categorical codes cannot express AnnData's `-1` for missing.** Ours stores `NA` as
   category 0 of 50; AnnData reserves code `-1` and would not round-trip that. Pick the convention
   deliberately — our version is arguably more honest about `NA` being a real annotation here — but
   the codes array must become **signed** if we adopt theirs, and `uint8` silently cannot.

3. **AnnData's categorical is a GROUP, not an array plus attributes.** `codes` + `categories` as
   sibling arrays with `encoding-type: categorical`. Our whole encoding (uint8 array, `values` in a
   custom attrs block) has to be turned inside out. The reader in `playground/src/datasource/cellTable.ts`
   already detects the AnnData shape structurally, so it is the spec to write against.

4. **The `obs` dataframe contract is strict and fails opaquely.** `_index` is a mandatory array,
   `column-order` must list every column, `encoding-type` is required on the group *and* each column
   (`array`, `string-array`, `categorical`). Miss one and the load error will not name it.

5. **The region metadata is duplicated and both copies must agree** — flat `region` / `region_key` /
   `instance_key` on the table's own attrs, *and* a `uns/spatialdata_attrs` group holding the same
   three. The real store carries both; writing one is a store that half-works depending on the reader.

6. **`region_key` needs a real column, and ours maps well.** `sample_id` (32 ROIs) is the natural
   value if we emit one shapes element per ROI — which also matches the per-ROI discipline every
   statistic in this stream already follows (`docs/cell-stats.md`). One element for all 545k cells
   would make `region` constant and throw that away.

7. **Three of our five tables are not regions tables and should not be forced into the shape.**
   `spatial_stats`, `spatial_stats_disease` and `spatial_stats_roi` are derived statistics keyed by
   (sample, type 1, type 2) — 76,832 rows of pair-wise results annotating nothing spatial. They are
   the *outputs* this stream computes live. If they are carried at all it should be as plain
   non-annotating tables, not as regions tables with a fabricated `region`.

8. **`X` has to be chosen, not defaulted.** The real store's `X` is a `csr_matrix` with `shape` in its
   attrs; ours would be dense `[545400, 37]` float32 = 78 MB, which is fine dense and should stay
   dense — IMC is not sparse the way transcript counts are. Do not copy the Xenium store's sparsity
   just because it is what the example does.

9. **A cosmetic mismatch worth not chasing twice:** our `mdv` attrs say `datatype: "double"` for
   columns stored as `float32`. MDV's "double" is a generic numeric tag, not a width claim. Read the
   zarr `data_type`, not the MDV attribute.

## Suggested order, if and when this starts

1. Fix the `cellID` converter bug (finding A) — without it there is no `instance_key` to write.
2. Emit `shapes` circles from `obsm/spatial`, one element per ROI — a valid annotating store now,
   with labels to follow if the watershed reconstruction (finding B) is judged worth it.
3. Convert `cells` only: `X`/`var` by panel intersection, `obsm/spatial`, the four `obsm` embeddings,
   the rest to `obs`.
4. Validate with zarr-python + `spatialdata.read_zarr` (v3-capable versions), not by reading it back
   with zarrita.
5. Leave the three stats tables alone until there is a reason to move them.
