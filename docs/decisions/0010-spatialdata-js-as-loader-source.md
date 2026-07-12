# ADR-0010 — spatialdata.js as the Milestone-2 `Loader` source

Status: **proposed** (2026-07-06). Amends the cross-repo layering table of
[ADR-0008](./0008-view-driven-multiscale-datasource.md).

## Decision

For Milestone 2 (real bytes, image-only) the `Loader` behind `Resolve` is backed by
**published spatialdata.js packages** — `@spatialdata/core` + `zarrextra` — instead of the
raw `zarrita` + `openjph-wasm` `OmeZarrLoader` the build plan originally named. A
**`SpatialDataLoader implements Loader`** adapter maps a SpatialData multiscale image element
onto our `Multiscale`/`Tile` model and implements `getChunk` via zarrextra's per-level
`getTile`. This flips spatialdata.js's role in the ADR-0008 layering table from **pure
downstream consumer** to **also the Milestone-2 Loader source**, one-way
(`tgpu-htj2k → @spatialdata/core + zarrextra`).

## The seam, confirmed against the source

- **Entry is two published npm packages, no monorepo restructure, no `link:`.**
  `@spatialdata/core@^0.2.5` and `zarrextra@^0.2.3` are on npm; `openjph-wasm` is already a
  tgpu-htj2k `optionalDependency` (`file:../openjph-wasm`). They install into the **playground**
  package — the engine core (`src/datasource`) stays dependency-free per ADR-0008 §layering.
- **`@spatialdata/core` for discovery + transforms.** `readZarr(store) → SpatialData`;
  `sdata.images['name']` → an `ImageElement` with `getStore()` and
  `getTransformationForLevel(level, 'global') → { element, dataset }` (the pixel→world affine).
- **`zarrextra` for the chunk seam.** `loadOmeZarrMultiscalesFromStore(img.getStore()) →
  VivCompatiblePixelSource[]` — one source per Level, each exposing `.shape`, `.dtype`,
  `.tileSize`, and `getTile({ x, y, selection, signal }) → { data, width, height }`. **This
  `getTile` *is* `Loader.getChunk`** — genuine `(level, chunkX, chunkY)` decoded-pixel
  granularity (open question resolved: no need to fall back to whole-image `getRaster`).
- **Metadata-only open.** Opening the multiscale reads only `.zarray`/`zarr.json`/attrs; chunk
  bytes are fetched lazily at `getTile` time — satisfies the Datasource "cheap metadata only" rule.
- **HTJ2K decode stays CPU, off-main-thread, inside zarrextra.** `registerExperimentalHtj2kCodec()`
  wires the codec into zarrita; `enableWorkerChunkDecode()` (from `zarrextra/workers`) runs
  OpenJPH decode in a worker pool. zarrextra dynamic-imports the `openjph-wasm` package the
  consumer provides. We do **not** use this repo's own `rust/htj2k-core` codec here (ADR-0008 §9:
  the GPU/experimental codec doesn't beat OpenJPH yet). Decode is not intercepted at the
  coefficient domain — full pixels come back, and our `fdwt` analysis hook (if used) runs
  post-decode, exactly as ADR-0008 §9 requires.
- **No deck.gl / React enters the render path.** The one-way dependency is on `@spatialdata/core`
  + `zarrextra` only; `@spatialdata/layers` / `@spatialdata/vis` / `@spatialdata/avivatorish`
  (deck.gl + React) are excluded, preserving ADR-0008 §8 (three.js-first, reusable core free of
  deck.gl/React). MDV consumes sd.js through those layer packages; we deliberately go one level
  lower, calling the same `getTile` those layers wrap internally.

## Why sd.js and not raw zarrita

zarrextra alone is essentially "zarrita + openjph-wasm + a viv-tile wrapper" — i.e. the original
`OmeZarrLoader`. Entering *also* through `@spatialdata/core` is what earns the reason sd.js was
chosen over raw zarrita: coordinate systems, per-level and per-element transforms, multiscale
addressing, and — later — the whole element family (images/labels/points/shapes/tables) through
one domain model. It is the fastest path to eventually exercising ADR-0008's "Datasource is a
family, one coat" without grid-over-fit. For the identity-transform blobs fixture core is
functionally redundant, but the real target (below) has a genuine non-identity affine, and 1b
(co-registration) needs core's transforms outright — so it is wired from the start.

## Scope of this slice

Real target dataset: a Xenium store served locally (CORS-enabled) with two HTJ2K image elements —
`he_image` (RGB `uint8`, 5 levels, L0 `[3, 45087, 11580]` c,y,x ≈ 1.5 Gpx, chunks `[1,1024,1024]`,
**one channel per chunk**) and `morphology_focus` (4× `uint16` fluorescence). Sourced from a real
store, not the checked-in `blobs_multiscale_image` (which is `zstd`, not HTJ2K, and too small to
make the bounded-working-set behaviour tangible).

**Slice 1a — seam proof, `he_image` only:**
- Maps onto the **existing power-of-2 `Multiscale` / `select.ts` unchanged**
  (`voxelDims0=[11580,45087,1]`, `chunkShape=[1024,1024,1]`, `levelCount=5`). The real per-level
  scale (`2.00004×…`) vs exact `2×`, and the resulting off-by-one voxel at each level's far
  border, are both negligible (chunk-grid counts are identical; the loader reports each border
  tile's actual returned dims).
- **RGB via `element: { kind: 'vec', n: 3 }`** — the datasource's first real use of the ADR-0004
  element-algebra facet. `SpatialDataLoader.getChunk` fetches all three channels for a spatial
  `(level, x, y)` (3× `getTile` with `selection: { c }`, 3 OpenJPH decodes) and interleaves them
  into one RGB `Tile`. `ChunkId` stays spatial-only — per-channel fetch is a loader-internal
  detail. The three.js tile texture branches on element: scalar → greyscale (synthetic, unchanged),
  vec3 → RGB.
- **Placed axis-aligned** — the real sd.js affine is deferred to 1b (see below).
- Reuses Select / Resolve / `TileCache` / the three.js `tileRenderer`, and **wires the existing
  `loadScheduler`** (bounded concurrency, nearest-first priority, reconciled against the live
  Selection) into the SpatialData tile path — needed because real HTJ2K decode is expensive where
  the synthetic in-memory loader was free.
- Working set bounded by **natural Nyquist + frustum + LOD gradient + `TileCache` LRU**; the
  explicit `selectWithinBudget` degrade-to-fit ceiling (ADR-0008 §5) is **deferred** until an
  actual OOM (e.g. a grazing oblique strip) can be provoked. To be reviewed against
  viv / psychogeo / others later.
- **Exit:** orbit/zoom over the real `he_image` visibly changes chunk levels and the byte budget,
  tiles stream in nearest-first under bounded concurrency, and the working set stays bounded.

**Slice 1b — same milestone, image alignment workbench:**
- Add `morphology_focus`; honor the real sd.js `global` affines (via `getTransformationForLevel`),
  so the two modalities land **co-registered** in a shared world.
- **Manual 3D alignment is a real user workflow, not a demo device** — users align images of
  tissue slices (co-registered modalities *and* separately-acquired serial sections) and later
  author shapes (e.g. tubules) through aligned image features, from which implicit surfaces are
  derived (horizon, not built here). The editable placement is therefore a first-class,
  provenance-carrying, eventually-persistable artifact — the **Alignment** (see CONTEXT.md):
  `worldFromArray = Alignment ∘ storedTransform`, `Alignment` = identity initially, edited via a 3D
  gizmo, and **fed into `Select`** (which is already oriented-capable — `select.ts` culls oriented
  boxes and computes nearest-point via `invertAffine`), so **dragging an image re-selects chunks**,
  not merely repositions pixels.
- The arbitrary **channel/var axis** in `Selection` (reserved in CONTEXT.md, unused in 1a) becomes
  real here: pick channel(s) of the fluorescence image, colormap each, and blend/opacity-composite
  across the two images. Distinct from 1a's fixed display-RGB.

## Considered and rejected

- **`zarrextra`-only (skip `@spatialdata/core`).** Rejected: it collapses back to the original
  raw-zarrita `OmeZarrLoader` and forfeits the transforms + element-family reason for choosing
  sd.js; the ADR-0008 layering flip would be cosmetic.
- **`link:` / vendored monorepo / tarball wiring.** Moot — the packages are published; plain
  `pnpm add` is cleanest. A tgpu-htj2k monorepo restructure (a dedicated spatialdata package) may
  still come, but is not needed for this slice.
- **Channel axis in `ChunkId`/`Selection` from the start.** Deferred: 1a's fixed RGB is served by a
  `vec3` element with no new addressing surface; the general channel axis is exercised in 1b where
  morphology actually needs it, avoiding model surface before it is used.
- **Real affine + Alignment in 1a.** Deferred to 1b: for a single image the placement is
  effectively invisible (aspect ratio aside) and a placement bug is untestable with nothing to
  register against; affine + Alignment + co-registration + the second image form one coherent 1b
  unit where correctness is actually visible.

## Consequences & open items

- The engine core stays dep-free; heavy deps (`@spatialdata/core`, `zarrextra`, `openjph-wasm`)
  live in the playground behind the `SpatialDataLoader`. The adapter's permanent home is a
  segregated, dynamic-imported loader module (in the engine package or a dedicated loader
  subpackage) once the seam is proven — prototype-mode placement now, per ADR-0008 §packaging.
- **`element: vec3`** support must reach the tile-upload path (texture format branch); the pure
  model already accounts for lanes (`bytesPerSample`).
- **Decode runs off-main-thread in zarrextra's codec worker (2026-07-07), matching MDV.** Two
  calls, once, before reads (mirrors MDV's `ensureChunkWorker`): `registerExperimentalHtj2kCodec()`
  (registers the `experimental.openjph_htj2k` id so the pipeline *recognises* the codec) +
  `enableWorkerChunkDecode()` from `zarrextra/workers` (routes the decode *work* to a worker pool
  that self-contains the OpenJPH/HTJ2K codec + wasm). Registration says *what*, the worker says
  *where*. This supersedes the earlier main-thread custom `openjph-wasm` decoder: the worker deps
  (`@cornerstonejs/codec-openjph@2.4.7`, `@fideus-labs/fizarrita@1.4.1`, `@fideus-labs/worker-pool@1.0.0`)
  are zarrextra optional deps already resolved in our lockfile at MDV's exact versions, so
  `registerExperimentalHtj2kCodec()` needs no custom decoder, and `openjph-wasm` is no longer
  imported on this path. The UI no longer stalls while tiles decode.
- **Vite wire-up findings (resolved while building 1a):**
  - **`optimizeDeps: { exclude: ["zarrextra", "zarrextra/workers"] }`** (one line, replacing the
    earlier `openjph-wasm` exclude). Two pre-bundling hazards, both fixed by excluding the *whole*
    package: (1) `zarrextra/workers` resolves its worker via `new URL('./codec-worker.js',
    import.meta.url)`, which pre-bundling rewrites to a `.vite/deps/codec-worker.js` that is never
    emitted → the worker 404s and decode hangs; (2) more subtly, pre-bundling `zarrextra` but not
    `zarrextra/workers` splits `chunkDecode` into **two module instances**, so
    `enableWorkerChunkDecode()` flips the worker backend on one while `getTile` reads the other
    (still inline) → decode silently falls back to the main thread and can't resolve
    `@cornerstonejs/codec-openjph` as a bare specifier. One instance from node_modules fixes both.
    (We are on Vite 5; MDV on Vite 7 uses `worker: { format: 'iife' }` and does *not* exclude — the
    trade-off differs by Vite version.)
  - `@spatialdata/core` also pulls `apache-arrow`/`parquet-wasm` (its table support) which log
    harmless warnings on image-only use.
  - The local dev server is **CORS-enabled**, so no vite proxy is required.
- **Implementation status (2026-07-06): 1a landed and verified in the browser.** `he_image`
  (1.5 Gpx RGB HTJ2K) streams from the store, decodes per-chunk, and renders on the plane; zooming
  visibly moves the selection between **L4 (3 chunks, 23 MB)** and **L0 (6 chunks, 72 MB)** with a
  bounded working set. Files: `playground/src/datasource/spatialDataLoader.ts`,
  `playground/src/spatialDataMain.ts`, `playground/spatialdata.html`, an RGB branch in
  `tileRenderer.ts`. **Not yet done in 1a:** wiring the existing `loadScheduler` into this tile
  path (still unbounded concurrency — fine for the current chunk counts, but the stated 1a item),
  and the y-flip/orientation nicety.
- **three.js is untouched** for this slice (the prompt reuses the existing renderer). The
  rendering-as-ops direction ([ADR-0009](./0009-rendering-as-ops.md)) is orthogonal and not
  triggered here. Whether we stay on three.js long-term (vs deck.gl interop / rendering-as-ops) is
  explicitly still open.
- **Geo/environmental horizon.** Image/volume rendering of non-biological spatial data
  (geo/environmental, possibly astronomical) is a real future direction — a caution against baking
  in microscopy-specific assumptions, not a reason to abstract ahead of need. Day-job is microscopy
  for now.

### Near-term horizons (noted 2026-07-07)

The channel colormap/contrast work landed after 1a: a GPU channel-composite material
(`playground/src/datasource/tileChannelMaterial.ts`) with per-channel colour/contrast/visibility
and a blend parameter, seeded from `omero.channels` with auto-contrast for >8-bit. From that, three
directions are recorded — noted, not built:

- **Arbitrary channel counts + a more extensible material.** The composite currently caps at
  `MAX_CHANNELS = 4` (packed into one RGBA tile texture) in a hand-authored TSL material. Two things
  to revisit together: (1) **> 4 channels** need a different carrier — the first-class **channel/var
  axis** in `Selection` (reserved in CONTEXT.md) plus a texture-array or multi-pass composite;
  (2) the material should become **extensible / expressible in the op-graph** rather than a fixed
  shader — the raster-as-rendering-op + blend-as-layers-panel direction in
  [ADR-0009](./0009-rendering-as-ops.md)'s horizon note. The material is built so a rendering op
  *wraps* it later (blend is already a parameter), so this door is open at no cost.
- **deck.gl integration of *our* render (investigate).** Distinct from ADR-0008's deck.gl-as-a-
  Loader/layer-backend interop: the question of compositing *this* WebGPU tile/volume render **into
  a deck.gl context** (shared device / one pass — the ADR-0008 §interop **I2** spike), so an
  sd.js/MDV deck.gl app can host our renderer directly.
- **Dataset hosting — local-only for now.** The demo assumes a SpatialData/OME-Zarr store the user
  hosts **locally** (default: a CORS-enabled server at `localhost:8080`); the demo panel now carries
  a disclaimer to that effect, and any reachable store URL works (image elements list themselves).
  A publicly-accessible sample dataset is planned so the demo runs without local setup.

## References

- [ADR-0008](./0008-view-driven-multiscale-datasource.md) — view-driven datasource; the layering
  table this ADR amends, and §4 (oriented Select), §5 (ceiling), §8 (three.js-first), §9 (CPU
  decode) it leans on.
- [ADR-0004](./0004-field-type-model-and-volumetric-splat.md) — the `element` facet that carries
  RGB as `vec3`.
- [`../datasource-renderer-plan.md`](../datasource-renderer-plan.md) — Milestone 2; the
  `OmeZarrLoader` step this ADR reshapes into `SpatialDataLoader`.
- [`../../CONTEXT.md`](../../CONTEXT.md) — glossary (`SpatialDataLoader`, `Alignment`, `stored
  transform` added alongside `Loader`).
