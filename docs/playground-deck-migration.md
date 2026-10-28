# Playground demos onto deck.gl + SpatialData.js

**Decision.** The 2-D analysis demos move, one per PR, onto the stack `transcriptmodes` already uses:
React, one `SpatialDeckView` per page (sd.js's headless canvas — `useSpatialCanvasRendererFromLayerInputs`
\+ `SpatialViewer`), sd.js's own image layer underneath, and our analysis drawn as deck layers on the
device deck and this library share (`useSharedDevice` → `adoptDevice`). Charts beside the view are
small React/SVG components, no chart library. Each demo keeps its plain-language explanations, as
`<Explain>` blocks. The 3-D and three.js pages are out of scope.

Why: one viewer stack means one pan/zoom, one image path and one device for every analysis, and the
layers we write here are the ones that later move into `@intraspatial/deck` for MDV.

**Constraints.** One Deck per page — a second brings a second device and `adoptDevice` throws. A deck
layer must not read `res.resident` (the next `gramMatrixGpu`, or a null-envelope run, overwrites it);
it reads a per-result snapshot (`modes/gramSnapshots.ts`).

MDV has no pattern for several Decks sharing a device. Each chart creates its own Deck, which gets its
own WebGL device, and none passes `device` or `deviceProps`. Its one external-resource case,
`ImageArray`, is WebGL-only: it gets the same GL context by calling `getContext` on the same canvas.
So a page with several views (serial sections, linked charts) needs one Deck with several deck
`views`, or a library that supports more than one device. That is an open question, not decided
here.

A store need not have a `global` coordinate system (mibitof has one system per field of view). The
page picks the system and offers only the images in it: sd.js draws nothing for an image outside
the view's system, so `SpatialDeckView` would never frame.

## Order

| # | Demo | Size | Why this order |
|---|---|---|---|
| 1 | `cellmodes` | M | Same Gram/modes as `transcriptmodes`; most of the layer and legend exist. Done in this PR. |
| 2 | `rasterstat` | S | Small; introduces the one new layer the rest need, a field-texture layer. |
| 3 | `cellstats` | L | Biggest page; needs the field layer, `DensityLayer`, and views with no store behind them. |
| 4 | `umap` | M | Not tissue space: a plain deck `OrthographicView`, no sd.js. |

## Per demo

### 1. `cellmodes` — done (except the terrain)

| Now (vanilla) | On deck |
|---|---|
| `paintGramModes` onto a `<canvas>` | `GramModesLayer` (moved to `modes/`), fed a snapshot buffer |
| `loadContextImage` + OKLab image blend in the shader | sd.js image layer underneath; "map over image" opacity |
| wand: pointer → letterboxed canvas → `probeChannels` | click / shift-drag on the deck view → world → raster → `probeChannels` on the snapshot |
| corr matrix, scree + null band, loadings on 2-D canvases | `MatrixView` (now generic), `ScreeChart`, `LoadingsChart` (SVG) |
| mode legend text | `ModeLegend` (shared with `transcriptmodes`) |
| store / table / obs-or-X DOM wiring | `useSpatialData`, `useCellTables`, `useCellTable`, `useVarCatalog`; `ChannelSource` component |
| `paintGramTerrain` + orbit camera | **deferred** — the old page stays at `cellmodes-legacy.html` until it lands |

Terrain is deferred because it is its own view (a second canvas with an orbit camera), not a layer
over the map; it can be a `TerrainView` component on the shared device, reading the same snapshot.
When it lands, `cellmodes-legacy.html` and `cellModesLegacyMain.ts` are deleted.

### 2. `rasterstat`

| Now | On deck |
|---|---|
| input tile → `putImageData` | sd.js image layer (the tile *is* a store image level) |
| Gi\* z-scores → `putImageData` with a LUT | **new** `FieldTextureLayer`: a resident field + LUT, placed by its bbox (the `GramModesLayer` pattern, one channel) |
| synthetic-fixture path | same layer; `SpatialDeckView` framed by `fallbackBounds`, no `sdata` |

### 3. `cellstats`

| Now | On deck |
|---|---|
| centroid scatter (2-D canvas) | `ScatterplotLayer` over sd.js, coloured by type |
| KDE A / KDE B (`paintFieldTexture`) | `DensityLayer` from `@intraspatial/deck`, one per type |
| TCM field (`paintFieldTexture`, diverging) | `FieldTextureLayer` (from step 2) |
| type × type matrix | `MatrixView` |
| PCF curve + bootstrap band | an SVG chart in the `ScreeChart` style |
| MDV / CSV / fixture sources | `SpatialDeckView` must frame without `sdata` (it already takes `fallbackBounds`; needs testing with `sdata` absent throughout) |

### 4. `umap`

| Now | On deck |
|---|---|
| animated embedding on a 2-D canvas | `ScatterplotLayer` in a bare `DeckGL` + `OrthographicView`, positions from the GPU layout buffer as a binary attribute (no readback) — the `density` page's shape, not sd.js |
| speed / category colouring, controls | React controls; colouring as layer accessors |

### Not planned

`datasource`, `spatialdata`, `spatialvolume`, `geometry`, `raymarch` (three.js renderers, 3-D),
`spatialscene`, `r3fspike` (R3F), `hspf` (its own simulation renderer — a `FieldTextureLayer`
candidate later), `density` and `transcriptmodes` (already deck).

## Shared pieces

- `deck/` — `SpatialDeckView`, `useSharedDevice`, `worldAt` (pointer → world on the view).
- `modes/` — `GramModesLayer`, `ModeLegend`, `MatrixView`, `ScreeChart`, `LoadingsChart`,
  `WandReadout`, `gramSnapshots` (the GPU queue and snapshot pool, out of `useTranscriptGram`).
- `hooks/` — store-level hooks; `useCellTables`/`useCellTable`/`useVarCatalog` added here.

`GramModesLayer` and the future `FieldTextureLayer` are the candidates for `@intraspatial/deck`
once `cellstats` has used them from a second page.
