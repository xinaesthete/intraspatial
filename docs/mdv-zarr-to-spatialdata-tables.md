# Generic zarr, and remodelling our MDV-shaped store into SpatialData tables

**Status: analysis, nothing started (2026-08-01).** Companion to
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

## The gotchas, in the order they will bite

1. **A table alone is not a valid SpatialData store.** A regions table's `region` must name an element
   that *exists*. In the Xenium example it is `"cell_circles"` — a **shapes** element derived from the
   centroids, not an image. We have centroids and no shapes or labels element at all, so either
   synthesise circles from `obsm/spatial` (cheap, standard, and what the example does) or wait for
   `cellmask` to land as `labels` from the imagery migration. A table with `region: None` is legal but
   annotates nothing. **This is the one that turns a mechanical conversion into a design decision**,
   and it is the join between this plan and the imagery one.

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

1. Decide the v2/v3 question by testing what the intended consumer actually accepts. Everything else
   is downstream of it.
2. Emit `shapes` circles from `obsm/spatial`, one element per ROI — that makes a valid annotating
   store possible without waiting on the imagery.
3. Convert `cells` only: `X`/`var` by panel intersection, `obsm/spatial`, the four `obsm` embeddings,
   the rest to `obs`.
4. Validate with zarr-python + `spatialdata.read_zarr`, not by reading it back with zarrita.
5. Leave the three stats tables alone until there is a reason to move them.
