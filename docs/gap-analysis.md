# Gap analysis — what's decided/planned but not yet built

**Last refreshed: 2026-11-17** (issue [#54](https://github.com/xinaesthete/intraspatial/issues/54),
checked against `main` at 4a2e26c; previous full audit 2026-08-22). Verified against the code, not
against the documents' own status claims — several of which are stale. Items that landed since
the August audit are corrected inline below; the full per-item triage list (with doc line
references) lives in #54. Companion to the per-ADR table
in [`decisions/README.md`](decisions/README.md); this doc adds the plan-note gaps and the
cross-cutting picture.

The unbuilt work is not scattered. It clusters into a few **load-bearing primitives** that
gate large downstream cascades, a set of **science/application slices**, and the
**packaging** work. Ordered below by how much sits downstream of each item.

---

## Read this first: the two DWT paths (a correction)

"Wavelet ops are CPU" is easy to mis-state, so pin it down:

- **The codec's DWT is on the GPU — but it no longer lives in this repo.** `idwt53`,
  `fdwt53`, `fdwt97`, `idwt97` are TypeGPU/WebGPU compute kernels (one workgroup per DWT
  line, lifting in workgroup shared memory, bit-exact vs the Rust CPU golden). Since the
  2026-08-22 split they, their benches, and `dwt-gpu-and-high-bit-depth.md`'s subject are in
  the **`tgpu-htj2k`** codec repo (BSD-2-Clause). **Not a gap here.**
- **The op-graph wavelet *nodes* are CPU** — [`ops/waveletOps.ts`](../src/gpu/graph/ops/waveletOps.ts)
  (`fdwt`/`idwt`/`thresholdDetail`) call the CPU `fdwt2d`/`idwt2d` in
  [`graph/dwt.ts`](../src/gpu/graph/dwt.ts), a Float32 reference kept numerically identical
  to the docs "Draw in the DWT domain" demo (`execute` == `cpuGolden`).
- **The actual gap is a GPU wavelet op for the graph**, not a missing kernel somewhere else.
  The graph node uses the packed-Mallat Float32 layout; the codec kernels use a per-line
  subband high-bit-depth layout. Now that the two are in separate repos, the realistic path
  is a graph-native separable lifting kernel (ADR-0006 open item), not a bridge to the codec's.

---

## Terrain-derivation primitives (from the LIDAR foliage/architecture handoff, 2026-08-22)

Four generic engine ops the psychogeo/TerraCognita terrain work forces, all house-style
(`ops/` + `cpuGolden`): **(1) separable min/max morphology — landed** (`src/gpu/spatial/morphology.ts`,
graph op `morphology`, bit-exact vs a square-window oracle; erode/dilate/open/close);
**(2) decimating N×N→1 reduction — landed** (`src/gpu/spatial/decimate.ts`, graph op `decimate`,
mean/min/max, `ceil(w/f)×ceil(h/f)`, placement derived as `worldFromArray·scale(f)` per ADR-0018); (3) a nodata/validity facet — unify with the ADR-0005 `support` facet below, one mask facet
not two; (4) halo/apron on tile `getChunk` — cross-repo decision, grill before building.

## Tier 1 — Load-bearing primitives (gate cascades)

Two of these share one substrate: a **GPU prefix-sum / scan / stream-compaction kernel**.
**That kernel now exists** (2026-08-01, migrated 2026-08-22) at [`src/gpu/scan/`](../src/gpu/scan/):

```ts
// src/gpu/scan/prefixSum.ts — recursive two-level WGSL scan, 1024 elements/block, 2-D dispatch fold
export function exclusiveScanGpu(values: Uint32Array, opts?: ScanOptions): Promise<ScanResult<Uint32Array>>;
export function exclusiveScanGpu(values: Float32Array, opts?: ScanOptions): Promise<ScanResult<Float32Array>>;
export function encodeScan(ctx: ScanCtx, elem: ScanElement, src: GPUBuffer, n: number, enc: GPUCommandEncoder, maxWorkgroupsPerDim?: number, keyPrefix?: string): EncodedScan;
export function getScanCtx(): Promise<ScanCtx>;
// src/gpu/scan/streamCompact.ts — mask (u8 | u32 | f32, eq | gt) → packed index list + count
export async function streamCompactGpu(mask: MaskArray, opts?: CompactOptions): Promise<CompactResult>;
// encoder-level twin: one submit, worst-case n-sized output, count left on-device (opts.mask = u8 | u32 | f32, opts.keyPrefix)
export function encodeCompact(ctx: CompactCtx, maskBuf: GPUBuffer, n: number, enc: GPUCommandEncoder, opts?: EncodeCompactOptions): EncodedCompact;
```

It is exercised by `src/gpu/scan/*.gpu.test.ts` and `pnpm bench:scan` (33M-row scan: 46.6 ms in the file header's measurement, ~34 ms on a later run of the same bench —
compaction ~9–11 ms, readback-dominated). Companion helpers `dispatchGrid` / `checkBindingSize` /
`sized` landed in [`device.ts`](../src/gpu/device.ts). The spatial index build
(`gridIndex.ts`) now consumes it; the `support` facet does not yet.

- **GPU uniform-grid spatial index (2D + 3D)** — the single biggest hole. Decided as the
  "build-first" primitive in [ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md),
  specced in full in [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md), named the
  "foundational unlock" in [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md).
  **2D build landed** ([ADR-0022](decisions/0022-gpu-uniform-grid-index.md),
  [`src/gpu/spatial/gridIndex.ts`](../src/gpu/spatial/gridIndex.ts), 2026-08-22): on-device
  histogram → scan → scatter, consumed by `crossPcf.ts` and `tcm.ts`; 10M points in 5.5 ms.
  Since landed: indexed `nnDistance`/`knn` ([`gridIndexQuery.ts`](../src/gpu/spatial/gridIndexQuery.ts)),
  the graph op ([`ops/gridIndex.ts`](../src/gpu/graph/ops/gridIndex.ts)) and the 3D lattice build.
  Still missing (ADR-0022 follow-ups): query helpers are 2D only, CkNN/fuzzy adjacency still
  brute-force, no occupied-cell compaction, no resident `separate`/`spring` force op. Also unbuilt from ADR-0004: **3D domain**
  (`Shape` in [`handle.ts`](../src/gpu/graph/handle.ts) is 2D-only, no `dim: 2|3`) and
  **volumetric splat** (points → voxel grid).
- **The resident render op** — [ADR-0009](decisions/0009-rendering-as-ops.md) (rendering-as-ops),
  [ADR-0014](decisions/0014-procedural-geometry-render-contract.md) (surface / depth /
  distance / G-buffer contract) and [ADR-0017](decisions/0017-tier2-resident-buffer-edges.md)
  stages 4–5 are **all the same missing capability**, plus ADR-0010-geometry's no-download
  path. Zero built; rendering still lives entirely in three.js/TSL in `playground/`. Until
  this lands, geometry can't sit on resident edges and none of that chain moves.
- **The `support` facet** — [ADR-0005](decisions/0005-columnar-filters-and-sparse-support.md) /
  [`mdv-dimension-vs-support-facet.md`](mdv-dimension-vs-support-facet.md): mask⇄index
  duality, weighted reductions, the corrected t-norm operators. **Essentially zero** (only
  standalone [`sparseColumns.ts`](../src/datasource/sparseColumns.ts) host code). The
  prefix-sum/compaction kernel it was blocked on now exists (see above); a first partial step —
  mask ops as plain fields ([`filterOps.ts`](../src/gpu/graph/ops/filterOps.ts): `maskRange`,
  `maskEquals`, `maskCount`, `maskedDensity`) — landed in the composer, but no `support` facet
  in the value lattice yet.

## Tier 2 — Engine capabilities decided but unbuilt

- **Fourier / FFT basis** — [ADR-0006](decisions/0006-spectral-and-wavelet-domain-representation.md).
  Wavelet basis is built and has consumers; **Fourier is entirely absent** (no FFT kernel,
  no `fourier` variant in the `Basis` union), so the spectral-convolution/diffusion payoff
  is unreachable. (See the DWT note above for the wavelet nuance.)
- **Expression IR text-DSL + graph⇄IR duality** —
  [ADR-0007](decisions/0007-expression-ir-dsl-graph-duality.md) /
  [`render-traits-and-expression-dsl.md`](render-traits-and-expression-dsl.md). A typed IR
  *and* dual CPU/WGSL lowering exist in [`src/geometry/expr.ts`](../src/geometry/expr.ts),
  but there is **no DSL text surface, no graph⇄IR round-trip, no serialisation** — the
  "duality" itself is unbuilt. There are still **two** expression systems (geometry's `Expr`
  and the op-graph DAG), not one. Of render-traits, only okLab landed.
- **Datasource as op-graph nodes** — [ADR-0008](decisions/0008-view-driven-multiscale-datasource.md).
  The demand-pull pipeline is real and well-tested as plain functions, but `Select`/`Resolve`/
  `Tileset` are **not registered ops**, and the **pull-time camera input** (the one new
  runtime concept) was never added to the executor.
- **Facet loose ends** — `ParamSpec.units` **landed** ([`units.ts`](../src/gpu/graph/units.ts));
  still open from [ADR-0018](decisions/0018-field-domains-placement-and-resolution.md) dec 5 /
  [`stream-a-placement-plan.md`](stream-a-placement-plan.md) slice 4: ops that convert units
  and the scale-equivariance test.
  The `boundsOf(points)→domain` escape op (ADR-0018 dec 4, so `splatDensity` still
  host-falls-back for the default bbox); **vector `ParamType`**
  ([ADR-0015](decisions/0015-channel-axis-labels-coordinate-systems.md) — blocks any
  UI-editable placement/bbox/channel-weights). The `inferAxes` hook is implemented by
  [`channelOps.ts`](../src/gpu/graph/ops/channelOps.ts) and
  [`regionOps.ts`](../src/gpu/graph/ops/regionOps.ts).

## Tier 3 — Science / analysis slices

- **Windowed open-axis Gram tensor field + live permutation envelopes** —
  [`muspan-cell-stats-plan.md`](muspan-cell-stats-plan.md) §7 and
  [`cell-stats.md`](cell-stats.md). The *global* Gram/PCF/TCM path is built and
  parity-checked; the **per-pixel windowed** Gram and **Mode 2** (viewport-apron, live
  envelopes) are not. `crossPCFMatrixGpu` is **not wired into the demo** (still CPU).
  Permutation envelopes exist ([`pcfEnvelope.ts`](../src/spatial/pcfEnvelope.ts),
  [`gramEnvelope.ts`](../src/gpu/spatial/gramEnvelope.ts)); windowing is not yet validated
  against quadrats (type-I error, power). Follow-ups tracked in #45, #47, #48, #53.
- **TopACT reproduction** — [ADR-0016](decisions/0016-topact-box-vs-kde-reproduce-then-improve.md).
  **Not reproduced.** A CPU single-scale square box pool exists (`pooledBins` in
  [`channelBins.ts`](../src/spatial/channelBins.ts), built for the niche map), and
  `assignCentroids` is a nearest-centroid classifier against a frozen table. Still missing:
  per-gene lattices, multiscale pooling with a summed-area table, their calibrated SVM, a
  graph op, GPU. Mapping in ADR-0016 "Related since".
- **GPU coverage of the transcript-modes methods** (reviewed 2026-11-18). The Gram path is
  GPU-first; every clustering method is a CPU golden, with its GPU port
  ([`emeans-clustering.md`](emeans-clustering.md) slice 4) not started.

  | Method | Runs on | GPU code | Note |
  |---|---|---|---|
  | Channels from tiles (`transcriptChannels`) | CPU, main thread | — | Light bookkeeping |
  | Gram matrix (splat + pairwise reduce) | **GPU** (`gramMatrixGpu`) | ✓ | f64 oracle `gram.ts`; cost independent of radius |
  | Co-location modes (K×K eigen) | CPU | — | K ≤ a few dozen |
  | Mode map paint | **GPU** (`gramModes`) | ✓ | One fragment pass, no readback |
  | Wand | **GPU** distance + host flood fill | ✓ | Host fill is deliberate (tolerance moves without a dispatch) |
  | Permutation envelope | GPU (`gramEnvelope`) | ✓ | Wired on the cell pages, **not** the transcript page |
  | Gene modules (bin, randomized sparse PCA, e-means on genes) | CPU worker, ~6 s | ✗ | `pcaGpu` is a dense `CᵀC`; ~18k genes needs a GPU sparse × dense product we don't have |
  | Niche fit (bin, 3×3 pool, weighted e-means) | CPU worker, 0.1–0.2 s | ✗ | 25–90k bins; no need yet |
  | Niche assign per tile (bin, pool, nearest centroid, `ImageData`) | **CPU, main thread** | ✗ | Only drawing is GPU (`BitmapLayer`); per-tile cost **unmeasured** — the one likely bottleneck now |
  | `emeans` / `lloyd` / `assignCentroids` | CPU (f64) | ✗ | Slice 4 maps them onto existing kernels |

  GPU becomes necessary for per-cell neighbourhood clustering on whole slides (slice 3),
  fine-bin whole-element labelling, and TopACT-style multiscale per-gene pooling. Existing pieces
  for those ports: `knn` at k = 1 (assignment), `cellCounts` / `splatDensityAtomic` (binning,
  deterministic), `convolveSeparable` `boxKernel` or `src/gpu/scan` for summed-area tables
  (pooling), the fixed-point atomic accumulator (batch statistics). The Gram's rasters can't
  double as niche descriptors as they stand: their resolution follows the view window, while
  niche labels need one fixed bin level to stay put under pan and zoom.
- **Toolbox gaps** — [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md):
  `GpuShapes` + polygon morphology, GPU contact network (only CPU
  [`contactNetwork.ts`](../src/spatial/contactNetwork.ts) exists), higher-order co-occurrence
  tensor, k-hop/GPU k-means, KL/Sinkhorn, hexgrid, generic Monte-Carlo-in-one-sweep with a
  counter-based RNG. (crossPCF, QCM and TCM are built.) (Ten GPU spatial ops already exist in
  [`src/gpu/spatial`](../src/gpu/spatial) — the gap is the index and the shape/network family,
  not the point stats.)

## Tier 4 — Application stacks (mostly greenfield)

Three interlocking design notes, **none built**, plus the data migrations:

- **Serial-section aligner + multi-viewport** —
  [`serial-section-alignment-and-multi-viewport.md`](serial-section-alignment-and-multi-viewport.md):
  `Viewport` object, react-mosaic, scene persistence, onion-skinning. Only the R3F spike
  survives ([`playground/src/R3fSpike.tsx`](../playground/src/R3fSpike.tsx)).
- **Stain-space unmix + comparator registry** —
  [`stain-space-and-stack-transparency.md`](stain-space-and-stack-transparency.md).
- **Wand-contours → lofted geometry** —
  [`wand-contours-and-lofted-geometry.md`](wand-contours-and-lofted-geometry.md): a whole
  `src/imaging/` that doesn't exist yet; `swept` still sweeps a straight axis (no sampled
  path/profile).
- **Data migration to SpatialData** — **landed.** The imagery
  ([`covid-imagery-to-spatialdata-plan.md`](covid-imagery-to-spatialdata-plan.md)) and tables
  ([`mdv-zarr-to-spatialdata-tables.md`](mdv-zarr-to-spatialdata-tables.md)) plans are
  executed by `scripts/covid-imagery-to-spatialdata.py` (with `covid-imagery-inventory.py`,
  `covid-he-registration-check.py`, `covid-spatialdata-validate.py`); the outcome, the
  three facts that did not survive contact with the data, and the validation numbers are in
  [`covid-spatialdata-migration.md`](covid-spatialdata-migration.md). The store itself lives
  on an external volume, not in this repo. Still open from the same plans: the alignment gizmo
  now feeds `Select` (`playground/src/datasource/multiImageScene.ts`) but there is no persisted
  `Alignment` type, and `loadScheduler` still isn't wired into the SpatialData tile path
  (only the volume renderer uses it). The tile→field bridge exists
  (`datasource/tileToField.ts`, `centroidsToField.ts`).

## Tier 5 — Geometry / art deferred

- **Pick-to-feature `resolve` loop** —
  [ADR-0012](decisions/0012-geometry-provenance-and-pick-to-feature-editing.md). The decided
  instance→address→`specs()` loop is **unbuilt**; only BSP face-tag groundwork exists. This
  is the seam that makes the breeding surface directly manipulable by picking — on the
  critical path for a Mutator-controls package (see below).
- **Progen-scale parallelism / hybrid dispatch** —
  [ADR-0013](decisions/0013-hybrid-strategy-dispatch-and-progen-parallelism.md). Planar/
  non-planar *recognition* and a single depth-seam composite landed; per-building fan-out,
  raymarch proxies, and op-kind octree dispatch did not.
- **HsPf Phase 2** — [ADR-0011](decisions/0011-hspf-spatial-gpu-example.md). Phase 1 (sim +
  graphics + play) is done; geo-referencing, counts overlay, country comparison, and
  data-fit fitness are not.
- Smaller: extra sweep verbs (`curl`/`sweep`/`tilt`/`flap`), a `swept→implicit` bridge.

## Tier 6 — Packaging & consumers

[`packaging-and-consumers.md`](packaging-and-consumers.md) is **largely landed**: `package.json`
has an `exports` map, Dawn `webgpu` is an optional peer, `packages/deck`
(`@intraspatial/deck`) exists, and the npm release is at 0.3.0 via Changesets (#31); build
tooling is settled by ADR-0019. Remaining: promote `viewer-three` out of `playground/`
(and finish the playground→deck.gl move), and the open items in #46. (The renderer-agnostic
boundary holds — `src/` is verified three.js-free.)

---

## Cross-cutting through-lines

- **Two GPU primitives gate most of the cascade.** The uniform-grid spatial index (Tier 1)
  and the resident render op (Tier 1) each sit upstream of a large fraction of everything
  else. The render op is really *one* capability wearing three ADR numbers (0009, 0014,
  0017 stages 4–5).
- **One kernel unlocks two facets.** Prefix-sum / scan / stream-compaction underlies both
  the spatial index and the `support` facet — the highest-leverage single kernel, written
  at `src/gpu/scan/` and consumed by the index build since 2026-08-22; the facet is next.
- **The value lattice is half-built.** Element algebra, `axes`, `role`, and `placement` are
  real and threaded through the builder/executor; `support`, `extent`, 3D domain, and the
  `fourier` basis are not.
- **Science and art gaps are roughly balanced** — the deferred work is not lopsided toward
  either domain.

## Not covered by any ADR or plan yet

Two active directions have **no ADR or design note at all** — absent from this analysis by
omission, not by being done. If they are becoming real, they are the obvious next records to
write:

- **Mutator-controls package** — a `ParamSpec`-bound control/selection surface (sliders,
  breeding grid, pedigree view, 3D gizmo, MIDI-CC) decoupled from any one domain.
- **Audio / sonification** — the output-joint counterpart: sonify the graph (a render sink)
  and analyse incoming sound (a source). Revives the bit-rotted scsynth work from `organic`.

## Incidental bug (not a plan gap)

Resolved: the `cellID` string branch in [`scripts/mdv-h5-to-zarr.ts`](../scripts/mdv-h5-to-zarr.ts)
now has the `rows`-length assertion (line 238).

## Corrections to the 2026-07-23 ADR README audit

The [decisions README](decisions/README.md) predates the placement work; two of its "Notable
gaps" are now stale and are corrected here:

- **`ResolvedPlacement` (ADR-0015 §3 / ADR-0018) landed.** It is on both `GpuField` and
  `FieldValue` in [`handle.ts`](../src/gpu/graph/handle.ts), with `placementOf`/`systemsAgree`,
  an `inferPlacement` hook ([`op.ts`](../src/gpu/graph/op.ts)), `outPlacements` threaded and
  stamped by the executor, tests ([`placement.test.ts`](../src/gpu/graph/placement.test.ts),
  `placementAgreement.test.ts`), and a `placements[]` array on
  [`datasource/types.ts`](../src/datasource/types.ts). ADR-0018 should read **partial**, not
  open (decisions 4–5 — `boundsOf`, `ParamSpec.units` — remain).
- **The `basis` facet has consumers.** `inferBasis` is used at graph-build time and
  `registerWaveletOps()` ships `fdwt`/`idwt`/`thresholdDetail` — so "a type with no consumer"
  is stale (the remaining ADR-0006 gaps are Fourier and the wavelet GPU-layout bridge).

## Update 2026-11-17 (issue #54)

Other corrections: ADR-0022 has a graph op and a 3D build (only queries are 2D); the
`axes` facet landed via ADR-0015 and the role-threading gap is closed; the `ResolvedPlacement`
note above also applies to ADR-0018 (partial, not draft). Op-graph Tier 2 (resident buffer
edges, ADR-0017 stages 1–3) has landed with 12 resident ops; stages 4–5 (render op) have not.
Not yet in this doc, tracked in #54: memo eviction releasing resident buffers, `AbortSignal`
cancellation in `pull`/`advance`, `PortSpec` buffer usage, bundles through feedback, narrow
(u8/u16) buffers, `complexMulGpu` wiring, union-find H₀/H₁, channel axis in `Selection`, and
the UMAP "not done" list.
