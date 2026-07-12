# View-driven datasource & renderer — build plan

Companion to [ADR-0008](./decisions/0008-view-driven-multiscale-datasource.md); vocabulary in
[`../CONTEXT.md`](../CONTEXT.md). Ordered so the first vertical proves the thesis end-to-end with no
external deps, then swaps in real bytes and the second render backend.

## Guiding constraints

- The `Datasource`/`Select`/`Resolve`/`Tileset`/`Loader`/`TileCache` module is **self-contained** —
  **no import of `dancer`/`evo`/`sim`** (so those can graduate out later; ADR-0008 §layering).
- Impurity only at `Loader`/`Resolve`. `Select` is pure and golden-testable on the CPU with no GPU
  and no I/O.
- Heavy deps (`zarrita`, `openjph-wasm`, `spatialdata.js`) enter **only** in the `OmeZarrLoader`
  file, behind the `Loader` interface — dynamic-imported (the ADR-0004/0006 opt-in-pack lesson:
  eager module-graph growth tips the Dawn render fork).

## Milestone 0 — types & the pure core (no GPU, no network)

1. `Multiscale`, `Level`, `Chunk`, `Selection`, `Tile`, `Tileset` types. `Loader` interface
   (`getChunk(level, chunk) → Promise<Tile>`). `Camera` (view-projection + viewport px).
2. `Select`: pure `(Multiscale, Camera) → Selection`. Frustum-cull + nearest-point Nyquist/`q`
   (ADR-0008 §4). **CPU golden test first** — a fixed camera + synthetic pyramid metadata → an
   asserted `(level, chunk)` set; oblique-plane case shows the near-fine/far-coarse gradient.
3. `SyntheticLoader`: in-memory TS mandelbulb (volume) + a plane variant; deterministic; optional
   modelled byte/latency cost so the budget HUD has real numbers.

*Exit:* `Select` is golden-tested; `SyntheticLoader.getChunk` returns correct Tiles. No renderer yet.

## Milestone 1 — the flagship demo (three.js, synthetic, plane-in-3D)

4. `Tileset` as an op-graph shape; `Resolve` node consulting a `TileCache` (LRU, chunk-identity
   keyed) → `SyntheticLoader` on miss → GPU upload.
5. Pull-time graph input (camera) threaded into `Select` via the exec context.
6. **Plain, imperative three.js — no React in the core** (ADR-0008 §8). Framework-free modules:
   a renderer host (`WebGPURenderer` + `PerspectiveCamera` + `OrbitControls`, adapted from the
   dancer harness), an attribute-less tile-mesh factory (à la `psychogeo`), and reusable **Overlay**
   factories (wireframe chunk grid, frustum, per-chunk level-tint) that any host — incl. the
   production renderer's debug mode — adds to a scene. The **Decision view** is composed from those
   Overlays; the real plane render + Decision view form a linked pair (shared camera) with `q` dial
   and fetch-budget HUD. Any React/R3F is confined to a thin demo-page host shell, never the
   primitives/Overlays.
7. Per-tile GPU `fdwt` analysis branch (reuse `waveletOps`); simplest readout (e.g. detail-energy
   per tile) surfaced in the HUD or as a tile tint toggle.

*Exit:* orbiting the camera visibly changes chunk levels and the byte budget in real time; the
oblique-plane resolution gradient is legible. This is deliverable (b)+(c).

## Milestone 2 — real bytes & the volume

8. Extend `Select`/render to the true **volume** (raymarch; slab = degenerate plane already works).
9. `OmeZarrLoader`: `zarrita` multiscales + `openjph-wasm` decode, behind the same `Loader`.
   Swapping *only this file* is the "seamless backing swap" claim made concrete.
10. `TileCache` eviction under a real **Resource ceiling**; fallible `Resolve` with degrade-to-fit
    (ADR-0008 §5). Three-tier memory (compressed / transient-decoded / GPU) verified: no persistent
    uncompressed CPU copies.

*Exit:* real OME-Zarr HTJ2K dataset streaming under camera control with a bounded working set — the
tangible perf/scalability feel. Needs a dataset (an sd.js sample or `openjph-wasm`-generated, as the
decoder fixtures already do) — sourcing it is its own small task.

## Later (design-only in ADR-0008 until picked up)

- **deck.gl-layer render backend** + the **three.js↔luma.gl shared-device (I2) spike**.
- Within-chunk resolution progression / progressive-over-frames decode (likely zarrextra).
- Second datasource family (points/AnnData → X-UMAP) to test the abstraction against non-grids.
- Anisotropic LOD; empty-space skipping; global byte-budget solver; render-in-graph node.
- **Cross-level tile compositing (follow-up to the interim fix).** Coexisting coarse/fine tiles are
  currently separated by a fixed per-level geometric z-offset (`LEVEL_Z_BIAS` in `tileRenderer.ts`)
  so the finer wins the depth test without z-fighting — `polygonOffset` would be the view-independent
  tool but three's WebGPU backend ignores it. Two improvements, in order of increasing scope:
  1. **View-dependent offset** (cheap): bias coarser levels *away from the live camera* each frame
     instead of a fixed +normal direction, so the finer tile wins from any side (removes the
     "front-side only" caveat).
  2. **Per-region masking** (the real fix): a coarse tile knows which of its footprint is already
     covered by resident finer levels and doesn't draw those parts — no overlap at all, so no depth
     trickery needed. Prior art: `psychogeo/src/geo/TileShader.ts` (attribute-less, index-only
     geometry drawing height-map meshes at different resolutions in GLSL) was heading toward exactly
     this masking; we want the same shape here (WebGPU/TSL). This is also the seam for
     **height-field-modulated tile geometry** (the `positionNode` reserved in ADR-0010 / the
     render-traits doc). NB psychogeo's separate `horn` geometry wants similar treatment — noted, not
     in scope here.

## Open task before coding

Confirm the engine call (three.js-first; ADR-0008 §8) and Milestone-2 dataset source. Everything
else is settled enough to start Milestone 0.
