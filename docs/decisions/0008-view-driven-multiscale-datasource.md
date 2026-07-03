# ADR-0008 — View-driven multiscale datasource & demand-pull rendering

Status: **proposed** (2026-07-03)

## Decision

Model a large, chunked, multi-resolution data source (flagship: an OME-Zarr image/volume
pyramid of HTJ2K chunks) as a **graph that a viewer pulls, backward, from its camera**. The
shape is:

```
pull(graph, sink, { camera })            ← a View calls this each frame
        │
        ▼
   [ sink ] ◀─Tileset─ [Resolve] ◀─Selection─ [Select] ◀─Multiscale─ [Datasource]
                          │                        ▲
                    Loader.getChunk           camera (bound at pull time)
```

with these load-bearing commitments (terms are defined in [`../../CONTEXT.md`](../../CONTEXT.md)):

1. **Demand flows backward; the camera is a pull-time argument, not a source node.** The graph
   is a *pure function of the camera* — re-pulled per frame, memoised across frames. This adds
   one small new runtime concept: a **pull-time graph input** (a bound free variable), of which
   the camera is the first instance. It reuses today's `pull(graph, field, opts)` seam directly.

2. **Impurity is quarantined at one `Loader` seam.** The graph is pure *given a Selection and the
   Tiles that selection resolved to*. The **`Datasource`** node contributes only cheap metadata
   (extent, levels, chunk grid, coordinate transforms, dtype/element) — never bulk data.
   **`Select`** is a pure `(metadata, camera) → Selection`. **`Resolve`** is the *single* effectful
   node, and its effect is exactly `Loader.getChunk(level, chunk)` — deliberately shaped like
   deck.gl's `getTileData({x,y,z})`, which is where zarrextra will eventually live.

3. **Selection is first-class data on an edge.** The set of `(level, chunk)` requests (+ a
   channel/var axis) is a value, not control flow. This is what makes interactive and cluster runs
   the *same graph*: interactive binds a `camera`; a batch job binds a region/full selector; the
   downstream `Select`/`Resolve`/analysis/render nodes are identical.

4. **The selection metric is Nyquist / projected-pixel-pitch, with one knob `q`.** Per chunk,
   frustum-cull, then at the chunk's *nearest* point pick the coarsest Level whose world
   sample-spacing `s₀·2^L ≤ worldPerPixel / q`, where `worldPerPixel = 2·depth·tan(fovy/2) /
   viewportHeightPx`. The pyramid's coarser levels are the prefilter (the DWT LL band). `q` is the
   detail budget and the dial the visualisation animates. Isotropic level per chunk; anisotropy and
   empty-space skipping are deferred.

5. **`Resolve` is fallible under a Resource ceiling.** When a Selection's working set would exceed
   the memory/bandwidth ceiling the policy is *degrade to fit* (coarsen globally), returning a
   `Result` with `Err('out of memory')` only as the honest floor. This is where a global
   byte-budget solver plugs in later — as a degradation policy, not the primary metric.

6. **`Tileset` is a new op-graph shape.** A keyed collection of Tiles (each a `grid` at its Level) is
   the currency between `Resolve` and its consumers, all of which iterate it. It is a central pillar,
   not a convenience.

7. **A dedicated `TileCache`, keyed by chunk identity, sits behind the Loader seam.** *Not* the
   per-graph content-addressed memo — it must be shared across multiple Views, persist across frames,
   hold GPU textures as the working set, and LRU-evict under the Resource ceiling. Three memory
   tiers: compressed chunks (zarrita's cache), decoded pixels (transient — dropped after GPU upload;
   this is the "don't keep uncompressed" win), GPU textures (the bounded working set). Because the
   cache is pure memoisation of a deterministic decode, it does not disturb graph purity.

8. **Render is a swappable backend over `Tileset`, parallel to the Loader over the fetch.** The
   abstraction owns *what data + what analysis*; a backend owns *pixels*. **three.js/TSL** and a
   **native deck.gl layer** are two backends. Render lives *in the viewer* for the first cut; a
   render node *in the graph* (feeding nodes downstream of a View) is kept open as a future
   extension, not built. **The reusable core and production renderer are plain, imperative
   three.js — no React.** Render primitives *and* debug overlays (wireframe chunk grid, frustum,
   level-tint) are framework-free `Object3D`/`Mesh` factories any host adds to a scene, so the
   *production* renderer can enable wireframe/debug overlays. The **Decision view reuses those same
   overlay primitives** rather than reimplementing them. **R3F, if used at all, is only a thin
   illustrative host shell** for the demo page (surrounding UI) — never where primitives or overlays
   live; React in the reusable path would be lock-in for sd.js/MDV/cluster/deck.gl hosts.

9. **The DWT-analysis hook runs post-decode, GPU-side.** OpenJPH fully decodes to pixels; *our*
   `fdwt` (+ band statistics / shrinkage / features) runs on the uploaded texture. Not a
   coefficient-domain-intercepting decoder — that only pays off once our codec beats OpenJPH, which
   current analysis says it does not yet.

## Context & provenance

Arrived at by a long grilling/domain-modelling session (2026-07-03). The playground to date has run
only trivially small synthetic data, so the project has no tangible feel yet for the scalability
that MDV & spatialdata.js actually need — where **the interactive camera view deciding what data is
processed is fundamental**. The user framed the scoring function explicitly:

1. **fit deck.gl's loader infrastructure** (adoptable, not a parallel universe);
2. **translate across interactive-viz ↔ cluster-job analysis** (the scale-equivariance thesis of
   [ADR-0004](./0004-field-type-model-and-volumetric-splat.md));
3. the **cleanest abstraction to the user** with **minimal heavy/brittle engineering**.

The purity boundary (2–3 above) is what satisfies all three at once: the impure seam is
deck.gl-shaped (1), the selection-as-data makes the graph scale-invariant (2), and quarantining
impurity at `Loader`/`Resolve` keeps the model clean without a general region-addressed executor
rewrite (3). Side-effects were treated as a first-class concern to *quarantine*, not purge — the
same boundary is where future interactive/effectful nodes would be admitted. Progressive decode is
modelled in principle as a future generator/stream of pure refinements (only *arrival timing* is
impure) and is **not** built now.

**Datasource is a family, one coat.** The flagship is image/volume, but the same abstraction is
meant to cover `points` transcripts (tiles by visible region → splat), AnnData `X` (columns by
interactive `var` selection, rows by visible area → future X-UMAP), and `obsp` connectivities — and,
later, mesh and geoarrows/parquet. Only the image/volume case is *built*; the others are designed
against on paper (the risk being grid-over-fit, mitigated by naming per-case what must differ). The
user's own caution against over-abstraction is deliberately respected: the design is not contorted
to fit the unbuilt cases.

**Prior art — this has largely been built once, in WebGL.** `psychogeo`
(`/Users/petertodd/code/www/psychogeo`) is a three.js terrain renderer that already has:
attribute-less index-only geometry; a viewport-span LOD calc that *is* the formula in (4)
(`span = 2·dist·tan(fov/2)`) but thresholded by a `2.5× tier` heuristic rather than Nyquist;
`groundViewport.ts` (frustum → plane AABB); a `TileLayerManager` (bounded concurrency, backpressure,
frustum-gated, priority-by-distance); and a `RasterChannel` interface that **is our `Loader` seam**.
The relationship is bidirectional: we adapt its renderer/LOD/load-manager (WebGL/R3F → WebGPU/TSL,
distance-heuristic → Nyquist/`q`, bespoke pyramid → OME-Zarr); psychogeo can later adopt our
OME-Zarr `Loader` + `Select` + WebGPU engine (its format "could perhaps migrate to zarr").

## Cross-repo layering & render backends

The `Loader` interface is the load-bearing seam between repositories:

| Concern | Home |
|---|---|
| Domain model (SpatialData images/labels/points/shapes/tables, AnnData, coordinate systems & transforms), OME-Zarr conventions, MDV app integration, existing deck.gl shape/point layers | **spatialdata.js** (the consumer; currently deck.gl, not three.js) |
| Storage/codec/fetch: zarr codecs incl. HTJ2K via OpenJPH, chunk fetching, compressed-chunk caching, later resolution-progressive/partial decode — the real `Loader` implementation | **zarrextra** |
| WebGPU engine: op-graph runtime, field model, DWT ops, `Select`, `Tileset`, `TileCache`, Decision view / render backends, and the dep-free `SyntheticLoader` — the abstraction itself | **this repo (tgpu-htj2k)** |

- **tgpu-htj2k is the *permanent* engine home** (it will be used here for other reasons too), not a
  temporary incubator — though restructuring/renaming is expected and refactoring cost is judged low.
  It is the **`dancer`/`evo`/`sim` code that may eventually move out**, so the datasource/renderer
  engine must stay **self-contained and free of any dependency on `dancer`/`evo`/`sim`**.
- **Packaging (deferred, not belaboured):** heavy deps (`zarrita`, `zarrextra`, `spatialdata.js`,
  `openjph-wasm`) come in as **optional/segregated** behind the `Loader` seam; the engine core
  publishes clean; a multi-package layout is likely. Prototype mode now; a clean publishing surface
  is a later concern.
- **deck.gl ↔ three.js interop (designed, not built):** target is **(I2) shared-`GPUDevice`,
  one render pass, shared depth** for true cross-layer compositing — gated on deck.gl's luma.gl-v9
  WebGPU backend maturing *and* a three.js↔luma.gl device-sharing spike (the key unknown; this repo
  already shares a Dawn device between compute and three.js). Interim: **(I1) synced separate
  canvases**. Dispreferred: **(I3) our render as a pure deck.gl/luma.gl layer** (abandons the
  three.js/TSL ecosystem). "Absorb deck.gl's own layers into our WebGPU stack" is out of scope but
  not precluded.

## Scope of the first cut

- **Build:** image/volume flagship — **both** 2D-plane-in-3D and true volume (a plane is a
  degenerate slab; they share the selection maths). `SyntheticLoader` (in-memory TS mandelbulb, no
  Python, no network) as Milestone 1; `OmeZarrLoader` (zarrita + openjph-wasm, real bytes) as
  Milestone 2 — two Loaders behind one interface *is* the proof the seam is real. **three.js-first**
  render for both the real render and the Decision view (linked pair, plane-in-3D leads). Per-tile
  GPU `fdwt` analysis hook.
- **Design-only (ADR, not code):** points/AnnData/obsp/mesh/geoarrows datasources; within-chunk
  resolution progression & progressive-over-frames decode; anisotropic LOD; empty-space skipping;
  the deck.gl-layer backend and the I2 hybrid; render-in-graph; global byte-budget solver;
  X-UMAP.

## Consequences

- **New in this repo:** the `Tileset` shape; the `TileCache` (chunk-identity keyed, ceiling-bounded
  LRU); a pull-time graph input (camera); `Datasource`/`Select`/`Resolve` nodes; the `Loader`
  interface; a three.js/WebGPU-TSL viewer with a Decision view; the `SyntheticLoader`.
- **Reuse, don't reinvent:** the lazy `pull` executor (ADR-0004's "the lazy executor is the
  small↔large seam") is the demand-pull substrate; the DWT ops + `basis` facet (ADR-0006) are the
  analysis hook and the prefilter; okLab/okLCH (recent) tints the Decision view; the dancer's
  attribute-less-TSL render harness and `psychogeo`'s LOD/load-manager/`RasterChannel` are the
  starting points.
- **Risk — abstraction over-fit to grids.** Mitigated by designing `Select`/`Selection`/`Loader`
  against the points/matrix cases on paper and recording per-case deltas, while only wiring grids.
- **Risk — WebGL→WebGPU adaptation cost.** psychogeo lowers *design* risk sharply but the port
  (vertex synthesis, material system) is real *implementation* work; three.js-first keeps it off the
  critical path of proving the datasource/selection thesis.
- **Revisit when:** a second datasource family (points/AnnData) is actually built (does the
  abstraction hold?); the deck.gl luma.gl-v9 WebGPU backend matures (the I2 spike becomes real);
  progressive/partial decode is taken on (the Loader interface grows a resolution-aware variant); or
  our own codec overtakes OpenJPH (the coefficient-domain decode hook reopens).
```
