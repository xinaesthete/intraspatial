# Context — View-driven data loading & rendering

The ubiquitous language for the datasource / demand-pull / multiscale-rendering design
(the flagship being an image/volume pyramid renderer for spatialdata.js & MDV). This
document currently covers **only** that design; other subsystems in this repo (the dance
sim, evo/Mutator, the HTJ2K codec, the op-graph internals) are not yet glossed here.

## Language

### The data being addressed

**Resource**:
An external multi-resolution store the design reads from — an OME-Zarr multiscales group,
an AnnData `.zarr`, a zarrextra store. Lives outside this repo; reached only through a Loader.
_Avoid_: datasource (ambiguous — see below), dataset, file.

**Datasource node**:
The graph node that stands in for a Resource. It contributes cheap **metadata** only (extent,
levels, chunk grid, coordinate transforms, dtype/element) — never bulk data. Its output is a
Multiscale handle.
_Avoid_: source (means the existing eager, param-seeded playground node — a different thing),
loader.

**Multiscale**:
The lazy handle a Datasource node outputs: a tree of Levels plus coordinate transforms and
element/dtype metadata, carrying no samples. Distinct from a `GpuField` (which implies
materialisable bulk data); its *leaves*, once resolved, are ordinary `grid` fields.
_Avoid_: pyramid (fine informally, but Multiscale is the type), image, volume.

**Level**:
One resolution of a Multiscale — a grid shape + chunk shape + downsample factor relative to
full resolution. Level 0 is full resolution; higher levels are coarser overviews.
_Avoid_: LoD (use as an adjective — "LoD selection" — not as the noun for a level),
pyramid level, mip.

**Chunk**:
The unit a Resource is fetched and decoded in — one addressable block of one Level, named by
`(level, chunkIndex)`. The atom of I/O and caching.
_Avoid_: block (means the HTJ2K code-block, a codec-internal thing), brick.

**Tile**:
A resolved Chunk — the decoded samples of one `(level, chunkIndex)`, an ordinary `grid` field
at that Level's resolution. A Chunk is the request; a Tile is the answer.
_Avoid_: patch, texture (a Tile may become a GPU texture but is not one by definition).

### The pull

**View**:
An interactive viewer that *drives* pulling: it owns a camera and a render loop and, each frame,
pulls a sink with its current camera. The View is *a* demand end of the graph, not a forward
source. There may be **many** Views over one Datasource (different cameras, different channel
selections); because the cache is content-addressed by Chunk identity — not by View — they share
fetched/decoded Tiles seamlessly. A View is not necessarily terminal: a render step *may* live in
the graph downstream of it (kept open as a future extension; render lives in the viewer for now).
In a cluster job there is no View — a region/full selector is passed to the pull instead.
_Avoid_: viewer (fine informally), camera (the camera is a parameter of the View, not the View).

**Camera** (pull parameter):
The view-projection + viewport supplied *at pull time* as a bound graph input, not baked into a
source node. The graph is a pure function of the Camera argument; re-pulled per frame as it moves.
_Avoid_: viewport (that's one field of the Camera).

**Selection**:
First-class data on an edge: the set of `(level, chunkIndex)` requests — plus, where the Resource
has them, a **channel/var axis** (which image channels or AnnData `var`s are wanted) — that a
Camera (or a cluster region selector) implies. Later carries per-chunk priority / desired detail.
Naming what is wanted; it does not fetch. Making Selection *data* is what lets the identical graph
run interactive or batch.
_Avoid_: request list, visible set, query.

**Select**:
The pure op `(Multiscale metadata, Camera) → Selection`. The Nyquist / projected-pixel-pitch
heuristic lives *entirely* here. No I/O.
_Avoid_: cull, LOD-pick.

**Detail budget** (`q`):
The single user-facing knob on Select. Per chunk, Select picks the coarsest Level whose world
sample-spacing `s₀·2^L ≤ worldPerPixel / q` at the chunk's nearest point. `q=1` ≈ one
prefiltered sample per screen pixel; `q>1` supersamples; `q<1` trades sharpness for bandwidth.
The dial the trademark visualisation animates.
_Avoid_: quality, LOD bias.

**Resource ceiling**:
The memory/bandwidth bound the working set must fit. When a Selection would exceed it, the policy
is *degrade to fit* (coarsen globally), so Resolve is **fallible**: it returns a `Result`, with
`Err('out of memory')` as the honest floor when even the coarsest fit is impossible. Not a crash,
not silent truncation.
_Avoid_: quota, limit.

**Resolve**:
The single effectful op `(Multiscale, Selection) → Tileset`. Its only effect is calling
`Loader.getChunk(level, chunk)` per Chunk (memoised/cached). All impurity in the graph is
quarantined here.
_Avoid_: fetch, load (those name what the Loader does, one level down).

**Loader**:
The impure seam Resolve calls: `getChunk(level, chunk) → Promise<Tile>` — fetch the chunk's HTJ2K
codestream and *fully* decode it (OpenJPH) to pixels. Deliberately shaped like deck.gl's
`getTileData({x,y,z})` so the design plugs into deck.gl loader infrastructure and so this seam is
exactly where zarrextra will eventually live. Partial/resolution-progressive decode is a future
variant of this interface, not the current one.
_Avoid_: fetcher, store, reader.

**DWT analysis hook**:
An analysis branch of the graph *downstream* of Resolve: `fdwt` (+ band statistics / shrinkage /
features) applied per Tile, on the *already-decoded* pixels — not intercepting the codec's internal
coefficients. Runs GPU-side on the uploaded texture (so no CPU tile is retained). Reuses the
existing wavelet ops. The place the design does real work on the data beyond just displaying it.
_Avoid_: filter, transform.

**TileCache**:
The datasource-level cache behind the Loader/Resolve seam, keyed by **chunk identity**
`(Resource, level, chunkIndex)` (+ decode/analysis params) — *not* the per-graph content-addressed
memo. Holds GPU textures (the working set), shared across all Views, LRU-evicted under the Resource
ceiling; a re-appearing evicted chunk re-decodes from zarrita's compressed tier rather than
refetching. Pure memoisation of a deterministic decode, so it doesn't disturb graph purity.
_Avoid_: buffer pool, memo (that's the graph's separate content-addressed cache).

**Tileset**:
The keyed collection of Tiles that Resolve outputs — a new op-graph *shape* (the graph today has
only single grids/points/etc.). The currency between Resolve and its consumers (analysis, splat,
render), each of which iterates over it. A central pillar of the architecture, not a convenience.
_Avoid_: tile array, batch, atlas.

### The viewer

**Render backend**:
A consumer of a Tileset that turns it into pixels. Swappable — exactly parallel to the Loader being
swappable over the fetch: `Select`→`Tileset` is render-agnostic, and three.js/TSL vs a native deck.gl
layer are two backends over the same tiles. The abstraction owns *what data + what analysis*; a
backend owns *pixels*. The reusable core and production renderer are **plain imperative three.js — no
React**; R3F (if used) is only a thin illustrative host shell for the demo, never where primitives
or Overlays live.
_Avoid_: renderer (fine informally, but the point is that there are several).

**Overlay**:
A framework-free three.js `Object3D`/`Mesh` factory for debug/illustrative geometry — the wireframe
chunk grid, the frustum, the per-chunk level-tint. Reusable across hosts, so the *production*
renderer can switch overlays on for debugging. The Decision view is composed from Overlays, not a
separate reimplementation.
_Avoid_: gizmo, helper, annotation layer.

**Decision view**:
The abstract companion to the real render — a schematic of the chunk grid in the live camera, each
chunk tinted by its chosen Level (okLCH ramp), with the frustum, the per-frame fetch-budget HUD, and
the `q` dial. Shown as a **linked pair** beside the real render (shared camera); annotations can also
be toggled *onto* the real view. Where the selection is made legible. Plane-in-3D is the lead case.
_Avoid_: debug view, overlay.
