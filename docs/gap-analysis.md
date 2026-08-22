# Gap analysis — what's decided/planned but not yet built

**Last refreshed: 2026-08-22.** Verified against the code (three parallel audits of ADRs
0001–0009, 0010–0018, and the forward-looking design notes in `docs/`), not against the
documents' own status claims — several of which are stale. Companion to the per-ADR table
in [`decisions/README.md`](decisions/README.md); this doc adds the plan-note gaps and the
cross-cutting picture.

The unbuilt work is not scattered. It clusters into a few **load-bearing primitives** that
gate large downstream cascades, a set of **science/application slices**, and the
**packaging** work. Ordered below by how much sits downstream of each item.

---

## Read this first: the two DWT paths (a correction)

"Wavelet ops are CPU" is easy to mis-state, so pin it down:

- **The codec's DWT is on the GPU** — [`src/gpu/idwt53.ts`](../src/gpu/idwt53.ts),
  `fdwt53`, `fdwt97`, `idwt97` are TypeGPU/WebGPU compute kernels (one workgroup per DWT
  line, lifting in workgroup shared memory, bit-exact vs the Rust CPU golden). The decode
  path runs inverse DWT + dequant + DC-shift on the GPU. These are what the benches
  (`pnpm bench:gpu`, [`test/bench_idwt.gpu.test.ts`](../test/bench_idwt.gpu.test.ts)) and
  [`dwt-gpu-and-high-bit-depth.md`](dwt-gpu-and-high-bit-depth.md) measure. **Not a gap.**
- **The op-graph wavelet *nodes* are CPU** — [`ops/waveletOps.ts`](../src/gpu/graph/ops/waveletOps.ts)
  (`fdwt`/`idwt`/`thresholdDetail`) call the CPU `fdwt2d`/`idwt2d` in
  [`graph/dwt.ts`](../src/gpu/graph/dwt.ts), a Float32 reference kept numerically identical
  to the docs "Draw in the DWT domain" demo (`execute` == `cpuGolden`).
- **The actual gap is the layout bridge**, not a missing kernel. The graph node uses the
  packed-Mallat Float32 layout; the codec kernels use the per-line subband high-bit-depth
  layout. Nothing yet reconciles them so a graph `fdwt`/`idwt` node can dispatch the
  existing GPU kernel. (ADR-0006 open item.)

---

## Tier 1 — Load-bearing primitives (gate cascades)

Two of these share one substrate: a **GPU prefix-sum / scan / stream-compaction kernel**.
Build that once and both the spatial index and the `support` facet unlock.

- **GPU uniform-grid spatial index (2D + 3D)** — the single biggest hole. Decided as the
  "build-first" primitive in [ADR-0004](decisions/0004-field-type-model-and-volumetric-splat.md),
  specced in full in [`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md), named the
  "foundational unlock" in [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md).
  **Nothing built** — only a 2D *CPU* bucket grid ([`src/spatial/bucketGrid.ts`](../src/spatial/bucketGrid.ts))
  exists; point stats are still brute-force O(N²). Also unbuilt from ADR-0004: **3D domain**
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
  standalone [`sparseColumns.ts`](../src/datasource/sparseColumns.ts) host code). Blocked on
  the same prefix-sum/compaction kernel as the spatial index.

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
- **Facet loose ends** — `ParamSpec.units` ([ADR-0018](decisions/0018-field-domains-placement-and-resolution.md)
  dec 5 / [`stream-a-placement-plan.md`](stream-a-placement-plan.md) slice 4);
  the `boundsOf(points)→domain` escape op (ADR-0018 dec 4, so `splatDensity` still
  host-falls-back for the default bbox); **vector `ParamType`**
  ([ADR-0015](decisions/0015-channel-axis-labels-coordinate-systems.md) — blocks any
  UI-editable placement/bbox/channel-weights). Also: the `inferAxes` hook exists but **no op
  implements it**.

## Tier 3 — Science / analysis slices

- **Windowed open-axis Gram tensor field + live permutation envelopes** —
  [`muspan-cell-stats-plan.md`](muspan-cell-stats-plan.md) §7 and
  [`cell-stats.md`](cell-stats.md). The *global* Gram/PCF/TCM path is built and
  parity-checked; the **per-pixel windowed** Gram and **Mode 2** (viewport-apron, live
  envelopes) are not. `crossPCFMatrixGpu` is **not wired into the demo** (still CPU), and
  "permutation envelopes remain unbuilt on every path."
- **TopACT reproduction** — [ADR-0016](decisions/0016-topact-box-vs-kde-reproduce-then-improve.md).
  **Nothing**: no box/square pool op, no summed-area-table fast path, no classifier.
- **Toolbox gaps** — [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md):
  `GpuShapes` + polygon morphology, GPU contact network (only CPU
  [`contactNetwork.ts`](../src/spatial/contactNetwork.ts) exists), higher-order co-occurrence
  tensor, generic Monte-Carlo-in-one-sweep. (Ten GPU spatial ops already exist in
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
- **Data migration to SpatialData** — imagery
  ([`covid-imagery-to-spatialdata-plan.md`](covid-imagery-to-spatialdata-plan.md)) and tables
  ([`mdv-zarr-to-spatialdata-tables.md`](mdv-zarr-to-spatialdata-tables.md)); both "not
  started," gated on watershed labels. The ADR-0010-loader **Alignment gizmo fed back into
  `Select`** isn't there, and `loadScheduler` still isn't wired into the SpatialData tile
  path.

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

[`packaging-and-consumers.md`](packaging-and-consumers.md) is **entirely unbuilt**:
`package.json` is `version 0.0.0` with **no `exports` field**, Dawn `webgpu` is still a hard
`dependencies` entry (not optional/peer), there is no `packages/` dir and no `@intraspatial/*`,
and the viewer layer still lives inside `playground/`. This is the concrete backlog behind
the "publishable modules" direction. (The renderer-agnostic boundary it depends on *does*
hold — `src/` is verified three.js-free.)

---

## Cross-cutting through-lines

- **Two GPU primitives gate most of the cascade.** The uniform-grid spatial index (Tier 1)
  and the resident render op (Tier 1) each sit upstream of a large fraction of everything
  else. The render op is really *one* capability wearing three ADR numbers (0009, 0014,
  0017 stages 4–5).
- **One kernel unlocks two facets.** Prefix-sum / scan / stream-compaction underlies both
  the spatial index and the `support` facet — the highest-leverage single kernel to write.
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

The `cellID` **string** branch in [`scripts/mdv-h5-to-zarr.ts`](../scripts/mdv-h5-to-zarr.ts)
is missing the `rows`-length assertion the numeric branch has — a latent data-corruption
risk in the MDV→zarr path.

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
