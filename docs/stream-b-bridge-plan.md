# Stream B — SpatialData → op-graph bridge (implementation brief)

Turns a real SpatialData element into a graph `source` so the existing spatial ops run on real
data. The audit's finding: *"Running spatial statistics on real data is blocked on a bridge from
SpatialData elements, not on missing ops."* This is that bridge.

> **Depends on stream A slices 0–1** (the `placement`/`axes`/`role` facet + builder threading). B
> stamps those facets onto the fields it produces; do not start B until 0–1 are committed. B also
> **subsumes what the stream-A plan sketched as slice 3** (the per-`(table, region)` centroid-extract
> source and the `Multiscale → placements[]` loader mapping) — those live here, so A stays purely
> graph-side (`src/gpu/graph/*`) and B owns the SpatialData→field surface (`src/datasource/*` +
> `playground/src/datasource/*`). Disjoint files ⇒ A and B can run in parallel once 0–1 land.

## The demonstrable win (staged)

**B1 — raster path (unblocked the moment 0–1 land).** A real Xenium tile already loads through the
sd.js loader. Compute a spatial stat on it *in the graph* and verify: e.g. **Getis-Ord hot-spot
Z-scores on a real morphology channel** (`splatDensity` not needed — the channel is already a grid),
or `convolveSeparable`. Success = the graph pulls a correct result from a real decoded tile, with the
tile's `placement`/channel-`axes` carried, verified against `cpuGolden`.

**B2 — centroid path (the scientifically compelling target; partially blocked).** **Cell centroids
→ `splatDensity` → `convolveSeparable` → `getisOrd`** (density → smooth → hot-spots), or
`fuzzyAdjacency` / `kthNeighborDistance` directly. This is the MuSpAn/TopACT-shaped demonstration the
positioning wants. Blocked on getting centroids into TS (see Risks: sd.js points/table loading is
WIP) — so B1 ships first and de-risks the facet plumbing on real data while B2's input is sorted.

## Op contracts (grounded — dictates the two halves)

Registered graph ops split by input shape:

| op | input | output | half |
|---|---|---|---|
| `splatDensity` | `points` | `grid` | centroid |
| `fuzzyAdjacency` | `points` | `matrix` | centroid |
| `kthNeighborDistance` | `points` | `matrix` | centroid |
| `getisOrd` | `grid` | `grid` | raster **and** post-splat |
| `convolveSeparable` | `grid` | `grid` | raster **and** post-splat |

Canonical real-data chain: `points → splatDensity → convolveSeparable → getisOrd`. (`convolveSeparable`
already returns a resident `buffer` — the raster path exercises Tier-2 edges on real-sized data for
free; see Risks.)

## Where the bridge lives (ownership boundary)

Per ADR-0015: **sd.js owns axis/transform reasoning on the playground side of the `Loader`; this repo
consumes resolved values.** So:
- **`src/datasource/` — pure conversion**, no sd.js dependency: `Tile → FieldValue(grid)` and
  `centroids → FieldValue(points)`, each stamping the facets it is *handed*. No transform composition
  here.
- **`playground/src/datasource/` — sd.js-facing resolution**: reads the store, resolves placement /
  channel metadata / (for centroids) the per-region transform, and hands the pure converter resolved
  values. `spatialDataLoader.ts` already does this for images (`SpatialDataImage` exposes `ms`,
  `loader`, `channels`, `globalFromArray`).

There is **no `Tile → FieldValue` path today** — `graph.source(value)` takes a `FieldValue`,
`graph.grid()`/`points()` build one from raw arrays, but nothing converts a datasource `Tile`. That
converter is the raster half.

## Ordered slices

### B1a — `Tile → FieldValue(grid)` converter (`src/datasource`)
- New `tileToField(tile, opts)`: `Tile.data` is already `Float32Array`; `dims` (with `dims[2]===1`)
  → `{kind:"grid", width, height}`; carry `element`/`dtype`; stamp `placement`/`axes`/`role` from
  `opts` (all optional — absent ⇒ today's behaviour). Pure, unit-tested against a synthetic `Tile`
  (mirror `syntheticLoader`/`tileCache.test.ts` fixtures).
- A `graph.sourceFromTile(tile, opts)` convenience (or just `g.source(tileToField(...))`).

### B1b — sd.js image → graph source (`playground/src/datasource`)
- Map `SpatialDataImage.globalFromArray` → `ResolvedPlacement{ system:"global", worldFromArray }`
  (absent ⇒ leave placement absent = array space, do NOT fabricate identity).
- Map `SpatialDataImage.channels` (omero) → a channel `TensorAxis` with `entries` (ADR-0015 fork B).
- Change `Multiscale.worldFromArray` → `placements: ResolvedPlacement[]` (**one `global` this pass**,
  ADR-0015 §3 scope) in `src/datasource/types.ts`; update `syntheticLoader` + consumers. (This is the
  loader half the stream-A plan had listed as slice 3.)
- Pull a level tile via the existing `Loader.getChunk`, convert (B1a), feed a graph, `pull`.

### B1c — the raster demonstration
- Run `getisOrd` (and/or `convolveSeparable`) on a real loaded morphology channel end-to-end. Verify
  against `cpuGolden`; surface it in the playground (plug into `buildGraph.ts`/the composer, or a
  small standalone demo page). Screenshot / numeric check as proof.

### B2a — centroid → `FieldValue(points)` converter (`src/datasource`)
- New `centroidsToField(xs, ys, opts)` → packed `{kind:"points", n}` (the `[x0,y0,x1,y1,…]` layout
  `graph.points` uses), stamping the per-region `placement` + the `region/region_key/instance_key`
  linkage as provenance. Pure, unit-tested.

### B2b — centroid resolution (`playground/src/datasource`, sd.js-facing)
- Obtain centroids + their per-region `ResolvedPlacement`. Port the **MDV annotation-walk + policy**
  (`~/code/www/MDV/python/mdvtools/spatial/conversion.py`: `_resolve_regions_for_table`,
  `_choose_point_transform`) to TS on the sd.js side: `get_table_keys → (region, region_key,
  instance_key)`, resolve the region's coordinate system, compose the transform via sd.js's
  CS-graph. **One field per `(table, region)` — never a merged cloud.** **Carry native coords + the
  matrix; apply on the GPU — do NOT host-bake** transformed coordinates (the MDV stopgap).
- Because sd.js table/points support is WIP, start against a **small committed fixture** (a SpatialData
  store or a synthetic table with a known transform) so B2 is testable now and swaps to the real sd.js
  path when it lands.

### B2c — the centroid demonstration
- `splatDensity → convolveSeparable → getisOrd` on real (or fixture) centroids, per region. Verify;
  surface in the playground.

## Scope

**In:** the two pure converters (`Tile→grid`, `centroids→points`) + their facet stamping; the sd.js
image mapping (placement + channel axis); `Multiscale → placements[]`; the MDV centroid-resolution
port (fixture-first); two end-to-end demonstrations with `cpuGolden` verification.

**Out — declared-for, not built:** multi-system placements beyond one `global`; the Table/instance
*join* (only carry the linkage as provenance); tiled points-element apron-splat-cache (stream-A
plan's forward-compat seam); making `bbox`/placement a UI-editable param (needs vector `ParamType`);
turning the standalone `src/gpu/spatial` ops that are *not* yet graph-registered (`anni`, `cknn`,
`emptySpace`, `nnDistance`) into graph ops — B uses the already-registered set.

## Risks / watch-items

- **sd.js points/table loading is WIP** — B2's real input may not be available yet. Mitigation:
  fixture-first (B2b), and B1 (raster) is fully unblocked and ships the facet-on-real-data proof
  independently.
- **Don't host-bake centroids**, don't merge across `(table, region)` — inherit neither MDV stopgap.
- **Real tiles are large** — the small-data assumptions (per-byte `hashSource`, full readback) bite.
  The raster path naturally exercises Tier-2 (`convolveSeparable` is resident); lean into resident
  edges where they exist, but do **not** block B on Tier-2 stages 4–5.
- **Facet dependency** — B cannot stamp `placement`/`axes`/`role` until A slices 0–1 land; the
  converters should be written so the facet args are optional and additive (absent ⇒ unchanged), so
  B1a can even be unit-tested before 0–1 merge if needed, then wired once they do.
- **Ownership boundary** — keep transform composition on the sd.js/playground side; the
  `src/datasource` converters receive resolved values only.
