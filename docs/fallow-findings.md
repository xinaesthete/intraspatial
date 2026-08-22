# Fallow findings — what the dead-code / health audit says about `src/`, and what we did about it

Status: **design note** (2026-08-22). Companion to `.fallowrc.json` (which points here) and to
[ADR-0019](decisions/0019-package-surface-and-prebuilt-kernels.md), whose package-surface
decision is what makes most of fallow's "unused export" category inapplicable to `src/`.

## 1. How fallow is used in this repo

Three scripts in `package.json`, one config file, three baselines.

| Command | When | What it does |
| --- | --- | --- |
| `pnpm lint:fallow` (`fallow audit --base main`) | every PR | The gate. `audit.gate: "new-only"` means only findings a changeset *introduces* fail it; inherited findings are reported with attribution but do not block. Exit 0 on a branch that adds no new dead code. |
| `pnpm fallow:baseline` | after a cleanup PR lands | Re-saves `.fallow-baseline/{dead-code,health,dupes}.json`. Whole-repo runs (`pnpm fallow:report`) then report deltas against those, so the noise floor stays at zero and a regression is visible. |
| `pnpm fallow:report` | ad hoc | `fallow dead-code` + `fallow health --hotspots --targets`, the full picture this note is drawn from. |

### Entry points — why "exported but unimported" is not "dead" in `src/`

`.fallowrc.json` seeds **every non-test `src/**/*.ts` as an entry point**, alongside
`scripts/*` and `vite.lib.config.ts`, and declares the root package `intraspatial` as a
`publicPackage`. That is a deliberate statement, not a loophole:

- ADR-0019 publishes the whole of `src/` through a `./*` subpath wildcard plus per-catalogue
  barrels. A symbol that nothing inside this repo imports may be exactly what psychogeo, MDV
  or the art repos import. Fallow cannot see those consumers, so "unused export" under `src/`
  would be a false positive by construction. Seeding `src/` as entries tells fallow so.
- The cost is that fallow's `unused_exports` category is silent for `src/`. What remains
  useful there is everything *else* it measures: unused files, import cycles, duplicate
  exports, byte-identical clones (`fallow dupes`), and the `health --targets` list (high-impact
  files, cognitive complexity, untested complex functions).
- `playground/` and `docs-site/` are workspaces, not entries. Their pages are discovered by
  the vite/astro/vitest plugins, so "unused export" *is* meaningful there — those findings are
  real and are listed in §3 as deferred.
- `docs/**/*.html` and `viz/**` are ignored (standalone explainers with ad-hoc embedded JS).

## 2. Verdicts — every `src/` finding

Sources: `pnpm exec fallow dead-code --format json` and `pnpm exec fallow health --targets`
(fallow 3.17.0), run on `main` (before) and on this branch (after), plus `fallow dupes`.
Before: 59 issues, 2 circular dependencies. After: 58 issues, 1 circular dependency (the
remaining one is in `playground/`, §3). Dead-code categories report nothing under `src/`
(see §1); the `src/` rows below come from the cycle detector, the clone detector, the public
surface diff, and the health targets.

### 2a. Handled by this branch

Evidence was gathered per symbol with `fallow dead-code --trace <file>:<export>` on `main`
and a grep across `src`, `playground`, `docs-site/src`, `docs`, `CONTEXT.md` and `scripts`.

| File : export | Finding | Verdict | Evidence |
| --- | --- | --- | --- |
| `src/gpu/sim/jet.ts` : `Jet3`, `Body`, `neutralBody`, `SwarmBuffers`, `makeSwarmBuffers`, `readBody`, `writeBody`, `swarmFinite` | module with no in-repo consumers | **keep-as-API** (deleted in an earlier commit, reverted 2026-08-22) | `--trace`: one reference each (the barrel). Not a duplicate of `sim/body.ts` but an unreconciled overlap: `body.ts` is the struct-of-arrays *buffer* layer the ops run on; `jet.ts` is the typed DANCERL-port layer (2-jet state for C²-continuous goal approaches, quaternion `Body`) never wired since `02853d1`. Kept deliberately as the starting point for the planned richer physics; when that starts, one of the two modules should absorb the other. |
| `src/gpu/graph/ops/danceForces.ts` : `constrainOp`, `swimOp`, `vortexOp`, `solenoidOp`, `orbitOp`, `cohereOp`, `separateOp`, `springOp`, `partnerOrbitOp`, `callerOp`, `clockOp`, `bodyTapOp`, `integrateOp` | 13 per-op exports alongside `FORCE_OPS` | **keep-as-API** (privatised in an earlier commit, reverted 2026-08-22) | No in-repo importer of the individual names — the playground takes `FORCE_OPS` (`playground/src/extraOps.ts`), tests use `registerForceOps()`. But ADR-0019 makes every `src/` export public API and `.fallowrc.json` seeds all of `src/` as entry points precisely so this is *not* reported as dead; the finding came from a pre-config run. The force maths itself is the library layer (`src/gpu/sim/forces.ts`); these are its graph wrappers, and importing one by name is a legitimate downstream use. Interfaces may still change. |
| `src/geometry/swept.ts` ↔ `src/geometry/sweptGpu.ts` | circular dependency | **refactor** (done, `cf05efb`) | `gridIndices`/`gridSampleAngles` moved to `sweptGrid.ts`; `swept.ts` re-exports them so `geometry/swept` and the barrel keep the same surface; `sweptGpu.ts` now takes only `import type` from `swept.ts`. fallow reports zero cycles under `src/` after. |
| `src/spatial/umapLayout.ts` : `mulberry32` (clone of `kernelAnalysis.ts`) | byte-identical duplicate (`fallow dupes`; ADR-0019 §4 follow-up) | **refactor** (done, `d682db2`) | Declared once in `kernelAnalysis.ts`, re-exported from `umapLayout.ts`; every prior import path and the barrel still resolve; the barrel's explicit collision workaround is gone. |

Public-surface effect (diff of `export` declarations in `dist/**/*.d.ts`, `main` vs this
branch): the 21 symbols in the first two rows are removed; nothing else changes except the
declaration *site* of `gridIndices`, `gridSampleAngles` and `mulberry32`, which stay importable
from their old paths.

### 2b. Deliberately kept

| File : export | Finding (health) | Verdict | Why |
| --- | --- | --- | --- |
| `src/geometry/placement.ts` (`IDENTITY`, `Mat4`, `compose`, `mul`, `rotX/rotZ`, `scaleUniform`, `translate`, `applyPoint/Normal`) | `split_high_impact` — 80 LOC, 4 dependents, pri 24.1 | **keep-as-API** | It is 80 lines of Mat4 helpers; the "impact" is that `structured.ts`, `swept.ts` and their tests all use it, which is the point of a shared placement module. Splitting would add files without reducing coupling. |
| `src/gpu/graph/memory.ts` (`MemoryReporting`, `dtypeBytes`, `fieldBytes`, `fieldValueBytes`, `formatBytes`, `memoryBytes`) | `split_high_impact` — 46 LOC, 6 dependents, pri 22.4 | **keep-as-API** | A 46-line leaf module that `datasource/tileCache`, `gpu/graph/onePole` and both barrels consume by design (ADR memory-reporting contract). High fan-in on a tiny stable leaf is healthy, not a hotspot. |
| `src/gpu/sim/vec3.ts` | `split_high_impact` — 92 LOC, 14 dependents | **keep-as-API** | Same shape: small vector-math leaf with high fan-in. |
| `src/spatial/persistence.ts`, `src/spatial/ngffTransform.ts` | `split_high_impact` — 156 / 236 LOC, 4 / 3 dependents | **keep-as-API** | Public catalogue modules; dependents are the intended consumers. Confidence reported as medium. |
| `SCAN_PER_THREAD` and similar kernel tuning constants | — | n/a | Not reported by fallow 3.17 on this tree (no such symbol exists under `src/` today); listed here only because earlier drafts of the adoption plan anticipated it. |

### 2c. Deferred `src/` findings (complexity / coverage — not dead code)

Health targets of category `extract_complex_functions` and `add_test_coverage`. None is dead
code; each is a candidate for its own PR with tests, and none should be folded into a
cleanup commit.

| File | Finding | Verdict |
| --- | --- | --- |
| `src/gpu/graph/executor.ts` `runTick` (cognitive 102) | extract | defer — the scheduler core; touch only with the executor tests in hand |
| `src/spatial/pcfBootstrap.ts` `crossPCFBootstrap` (106), `src/spatial/pcf.ts` `crossPCFMatrixBinned` (76) / `crossPCF` (56) | extract | defer — MDV-parity scripts (`scripts/mdv-pcf-parity.ts`) pin behaviour; refactor behind them |
| `src/geometry/implicit.ts` `tessellateSdf` (91), `src/geometry/bsp.ts` `mergeCoplanar` (30) | extract | defer |
| `src/spatial/contactNetwork.ts` (82), `gram.ts` (49), `tcm.ts` (53), `umapGraph.ts` `smoothKnnDist` (50), `umapLayout.ts` `optimizeLayoutStep`/`fitAB`, `quadratCorrelation.ts` `partialCorrelation` (43), `eigenSym.ts` (47), `sublevelsetPersistence.ts` (36), `envelope.ts` (31), `knnDescent.ts` `initialiseHeap` (38) | extract | defer |
| `src/gpu/spatial/gramMatrix.ts` (56), `gramModes.ts` (49), `gramTerrain.ts` (44), `quadratCorrelationGpu.ts` (30) | extract | defer — GPU kernels, covered by `*.gpu.test.ts` |
| `src/datasource/sparseColumns.ts` (41), `annDataIo.ts` `readExpressionMatrix` (33) | extract | defer |
| `src/gpu/graph/dwt.ts`, `src/gpu/sim/hspf/hspf.gpu.test.ts` | add test coverage | defer |
| `fallow dupes` clone groups under `src/gpu/spatial/` (the kNN buffer-setup block shared by `knn.ts`, `knnDescentGpu.ts`, `kthNeighborDistance.ts`, `umapLayoutGpu.ts`; the bind-group prologue shared by six kernels) and the private `mulberry32` copies in `gramEnvelope.ts`, `emptySpace.ts`, `evo/rng.ts` | duplicates | defer — the kernel prologues are candidates for a shared helper once ADR-0019's pre-transformed kernel layout settles; the PRNG copies are intentionally local (different return types / seeding) and `fallow dupes` with `minOccurrences: 3` does not flag them |

## 3. Deferred — `playground/` and `docs-site/` findings

These are real (the workspaces are not entry points), but they are UI code outside the
published package and are not part of this cleanup.

**Unused exports (`playground/`)**: `buildGraph.ts` `buildGraph`, `createSimState`;
`datasource/brickSource.ts` `loaderBrickSource`; `datasource/imageToGraph.ts` `imageFacets`;
`datasource/umapSource.ts` `readObsLabels`; `datasource/varMatrix.ts` `denseToColumns`;
`examples.ts` `reactionDiffusionExample`, `ceilidhExample`, `reusableSubgraphExample`,
`waveletDenoiseExample` (80% of the file — likely meant to be in `EXAMPLES`; check before
deleting); `grouping.ts` `isInputNode`, `isOutputNode`, `isInterfaceNode`; `portKinds.ts`
`KIND_COLOR`; `subgraphs.ts` `DEF_SCOPE_PREFIX`.

**Unused types (`playground/`)**: `PortHover.tsx` `PortHoverApi`; `datasource/cellCsv.ts`
`CsvColumnInfo`, `CsvSchema`; `subgraphs.ts` `InstanceData`.

**Unused class members (`playground/`)**: `datasource/multiImageScene.ts` (11 members),
`naiveVolumeRenderer.ts` (4), `tileRenderer.ts` (1), `volumeRenderer.ts` (2).

**Circular dependency (`playground/`)**: `grouping.ts` ↔ `subgraphs.ts` (6 files depend on
`grouping.ts`; health's top refactoring target by priority).

**Duplicate export (`playground/`)**: `openSpatialData` declared in both
`datasource/spatialDataLoader.ts` and `datasource/spatialDataStore.ts`.

**Unused exports (`docs-site/`)**: `components/dancer/matrix.ts` `MATRIX_MAX`, `matrixSize`;
`components/dancer/sim.ts` `BODY_BLOCK_COUNT`; `components/dancer/traits.ts`
`DANCER_TRAIT_SPECS`; `lib/creatureTsl.ts` `CREATURE_RADIAL`; `lib/dwt.ts` `mirror`,
`KERNELS`, `coeffs`; `lib/dwtImages.ts` `makePlane`; `lib/gpuField.ts` `getGpuFieldContext`;
`lib/oklabTsl.ts` `oklabToLinear`, `linearToOklab`, `oklchToOklab`; `lib/tslTransform.ts`
`rotateByAxisAngle`. Unused class members: `components/dancer/sim.ts` (2).

**Dependencies (`docs-site/package.json`)**: `katex` is declared here but imported only from
`playground` (move it); `sharp` is unused (astro image service — confirm before removing).

**Other**: `scratch/webgpu-probe.ts` is an unused file (a probe, keep or move under
`scripts/`); `scripts/mdv-*.ts` parity scripts appear as complexity targets and are left as
is — they exist to mirror MDV's reference code shape.
