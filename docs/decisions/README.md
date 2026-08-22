# Decision records — index and implementation status

Last audited: **2026-07-23** (this table). Gap analysis refreshed **2026-08-22** across all
ADRs *and* the forward-looking design notes — see [`../gap-analysis.md`](../gap-analysis.md).
The rows below are corrected inline where the newer audit found them stale.

## When to write an ADR (and when not to)

The 2026-07-23 audit found a clean pattern, and it is the reason this file exists:

> **Every ADR that landed was written about work already in flight or just finished. Every ADR that
> did not land was written about direction.**

Landed: 0001, 0002, 0003, 0008, 0011, 0017, 0010-loader. Not landed: 0005, 0006, 0009, 0012, 0014,
0016, 0018 — none of which are *wrong*, and several of which are good. They were simply speculation
wearing the costume of a decision, which turns the log into a source of guilt rather than a map.

So:

- **ADR** — you are about to build it, or are building it. It has a `Status` and an
  `Implementation:` line kept current. Consequences are things you will actually live with.
- **Design note** (`docs/*.md`) — direction, exploration, a hunch with references. No commitment
  implied, nothing owed. `fuzzy-tda-and-windowing.md` and `gpu-spatial-analysis-toolbox.md` are the
  models. A note is *promoted* to an ADR the day someone starts implementing it.

**Three documents were demoted under this rule on the day it was written** — the
[scene note](../serial-section-alignment-and-multi-viewport.md),
[stain space](../stain-space-and-stack-transparency.md), and
[wand contours](../wand-contours-and-lofted-geometry.md), formerly ADRs 0019–0021.

The third demotion is the one that matters. 0019 was written as a decision record for work we
intended to start immediately, and then the priorities changed underneath it: the serial-section
editor now sits behind the package surface, the viewer-layer promotion, the SpatialData→ops bridge,
and the deck.gl spike. Keeping it as an ADR because it was *nearly* started is exactly how the pile
accumulated in the first place. **A record earns the ADR form by being built, not by having been
believed in.**

Numbers 0019–0021 are **retired, not reused**, so commit history keeps pointing at something real.
**The next ADR is 0022.**

## Status

`landed` = the decision is in the code. `partial` = some of it is, and the row says which.
`open` = decided on paper only.

| # | Title | Claimed | Actual | What's in the code |
|---|---|---|---|---|
| [0001](0001-language-split.md) | Rust/wasm CPU core + TS/TypeGPU GPU | accepted | **landed** | `rust/htj2k-core`, `src/gpu` |
| [0002](0002-runtime-node-not-bun.md) | Node toolchain, not Bun | accepted | **landed** | vitest configs, CLAUDE.md override |
| [0003](0003-use-gpu-tgsl-kernels.md) | `"use gpu"` TGSL kernels | accepted | **landed** | used throughout `src/gpu` |
| [0004](0004-field-type-model-and-volumetric-splat.md) | Field type model + volumetric splat | proposed | **partial** | element algebra + `axes` (via 0015) landed; `splatDensity.ts` exists |
| [0005](0005-columnar-filters-and-sparse-support.md) | `support` facet, sparse columns | proposed | **open** | no `support` in the value lattice; operator set **corrected** 2026-08-01 (see [`mdv-dimension-vs-support-facet.md`](../mdv-dimension-vs-support-facet.md)) |
| [0006](0006-spectral-and-wavelet-domain-representation.md) | `basis` facet | exploratory | **partial** | `Basis` + `inferBasis` consumed at build; `registerWaveletOps` ships `fdwt`/`idwt` (CPU nodes); **Fourier/FFT absent**; wavelet GPU-layout bridge unbuilt |
| [0007](0007-expression-ir-dsl-graph-duality.md) | Expression IR ⇄ DSL ⇄ graph | proposed | **partial** | `src/geometry/expr.ts` has its *own* `Expr`, not unified with the graph |
| [0008](0008-view-driven-multiscale-datasource.md) | View-driven multiscale datasource | proposed | **landed** | `src/datasource` (its `index.ts` says "so it can graduate") |
| [0009](0009-rendering-as-ops.md) | Rendering as ops; three.js demoted | exploratory | **open** | three.js still owns rendering |
| [0010](0010-procedural-geometry-composable-ops.md) | Procedural geometry as ops | exploratory | **landed** | `swept`, `structured`, `implicit`, `bsp`, `hybrid` |
| [0010](0010-spatialdata-js-as-loader-source.md) ⚠ | spatialdata.js as `Loader` source | proposed | **landed** | `playground/src/datasource/spatialDataLoader.ts` |
| [0011](0011-hspf-spatial-gpu-example.md) | HsPf spatial GPU example | exploratory | **landed** | `src/gpu/sim/hspf`, `hspf.html` |
| [0012](0012-geometry-provenance-and-pick-to-feature-editing.md) | Provenance / pick-to-feature | exploratory | **open** | no `resolve`, no picking channel |
| [0013](0013-hybrid-strategy-dispatch-and-progen-parallelism.md) | Hybrid strategy dispatch | exploratory | **partial** | `src/geometry/hybrid.ts` |
| [0014](0014-procedural-geometry-render-contract.md) | Render contract (depth · distance · G-buffer) | exploratory | **open** | not a contract; raymarch is playground-local |
| [0015](0015-channel-axis-labels-coordinate-systems.md) | Channel axis, labels, coordinate systems | draft | **partial** | `FieldRole`/`LabelMeta` **and** `ResolvedPlacement` now landed (via 0018); vector `ParamType` still missing |
| [0016](0016-topact-box-vs-kde-reproduce-then-improve.md) | TopACT: reproduce then improve | draft | **open** | no code |
| [0017](0017-tier2-resident-buffer-edges.md) | Tier-2 resident buffer edges | accepted | **partial** | stages 1–3 + invariant 5 landed; 4–5 remain |
| [0018](0018-field-domains-placement-and-resolution.md) | Field domains: extent, placement, resolution | draft | **partial** | `placement` on `GpuField`+`FieldValue` + `inferPlacement`/`outPlacements` landed (2026-08); `boundsOf` + `ParamSpec.units` remain |

⚠ **Two ADRs share the number 0010** (`procedural-geometry-composable-ops` and
`spatialdata-js-as-loader-source`). Renumbering breaks inbound links in `docs/gpu-resource-sync.md`
and across the ADRs themselves, so it has been left alone and is recorded here instead. Refer to them
as *ADR-0010-geometry* and *ADR-0010-loader*.

Roughly **8 landed, 7 partial, 4 open** across 19 records (0018 moved open→partial when the
placement facet landed in 2026-08). No ADR has been added since the audit — deliberately. The
three documents written on audit day all became design notes.

## Notable gaps worth knowing before planning

- **`ResolvedPlacement` (0015 §3 / 0018) landed** (2026-08). It is on `GpuField`+`FieldValue` in
  `src/gpu/graph/handle.ts` with `placementOf`/`systemsAgree`, an `inferPlacement` hook, executor
  stamping, tests, and a `placements[]` array on `src/datasource/types.ts`. The remaining pieces are
  `boundsOf` and `ParamSpec.units` (0018 dec 4–5), not the facet itself.
- **There are two expression systems** (0007). `src/geometry/expr.ts` builds `(s, θ)` expressions for
  procedural geometry; `src/gpu/graph` is a separate DAG. The claimed duality is aspirational.
- **The value lattice is half-built** (0004/0005/0006/0018): element algebra, `axes`, `role`, and
  now `placement` are real; `support`, `extent`, and the 3D domain are not, and the `basis` facet
  has consumers (`inferBasis` + wavelet ops) but no `fourier` variant.
- **Wavelet on GPU is easy to mis-state** (0006). The **codec** DWT is GPU and benchmarked
  (`src/gpu/idwt53.ts` etc.); only the op-graph wavelet *nodes* (`ops/waveletOps.ts`) are CPU. The
  gap is the layout bridge between them, not a missing kernel — see [`../gap-analysis.md`](../gap-analysis.md).
- **`src/` is entirely three.js-free** — verified, zero `from "three"` outside `playground/`. The
  renderer-agnostic boundary 0008 asserted actually holds, which is what makes a published core
  package cheap. See [`packaging-and-consumers.md`](../packaging-and-consumers.md).
- **`src/gpu/spatial` already has ten GPU-tested ops** — ANNI, cKNN, separable convolution, empty
  space, fuzzy adjacency (+ adaptive), Getis-Ord, kth-neighbour distance, NN distance, splat density.
  Running spatial statistics on real data is blocked on a *bridge from SpatialData elements*, not on
  missing ops.
- **Sharing one `GPUDevice` between three.js and deck.gl/luma.gl is already possible** — three's
  `WebGPUBackend` accepts `parameters.device`/`parameters.context`, and `luma.attachDevice` takes a
  `GPUDevice` explicitly for interleaving. What is missing is on deck's side (no WebGPU interleaving
  path, no WebGPU picking). Details in [`packaging-and-consumers.md`](../packaging-and-consumers.md).
