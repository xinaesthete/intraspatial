# ADR-0011 — HsPf: a stand-alone spatial population-genetics GPU example

Status: **proposed / exploratory** (2026-07-08)

## Context

We want to re-derive the **HsPf** simulation (prior art in
`~/code/www/hspf-spatial-analysis/theory/html/hspf-gpu`, by **Gavin Band & Andre Python**,
Oxford) as an example in this repo — **not** a code lift, but a clean re-derivation in this
repo's idiom.

**What HsPf is.** A spatial population-genetics simulation over a raster map of Africa. Two
biallelic loci — **HbS** (the sickle-cell haemoglobin allele) and a second Pf-linked locus —
give **four genotype frequencies** (`--`, `-+`, `+-`, `++`) plus a **linkage-disequilibrium**
value `r`, stored as 5 layers per cell. Each GPU step, per cell:

- gathers from a **mosquito-bite neighbourhood** — random offsets whose distances follow a
  `Beta(1, concentration)` law (biting is mostly local; larger concentration ⇒ less smoothing);
- applies a **2×4 fitness matrix** (background `A` vs sickle `S`), blended by the local HbS
  frequency into `a·A + s·S`;
- mixes **single-bite** transmission (linear in the neighbour's genotype vector) with
  **two-bite recombination** (bilinear, via a 16×4 offspring/Punnett table and a `twoBiteRate`);
- normalises, then recomputes LD `r`.

Extras in the original: **barrier line-segments** that down-weight flights crossing them,
sentinel values (`-1`/`-2`) for ocean/missing with a coastline heuristic, GeoTIFF-loaded
`HbS`/`weights` fields, and D3/Observable-Plot displays comparing simulated vs. observed allele
counts by country. Structurally it is a **selection–migration reaction–diffusion PDE on a
genotype-frequency field**, on the GPU:

```
value[cell] = ( Σ_neighbours  w · F(pf[neighbour]) ) / ( Σ_neighbours  w · Σ F(pf[neighbour]) )
r[cell]     = LD(value[cell])
```

where `F` (the reaction) is a pointwise nonlinear map on one cell's genotype vector and the
gather (the migration) is a large **sampled** neighbourhood, not a fixed stencil.

**Why re-derive it here.** Three overlapping goals: (a) prove this repo's GPU/library primitives
can carry a *real, recognisable scientific model* — the map of Africa is what makes it an
appealing demo; (b) it is a vehicle for **parameter-space exploration** (fun first, scientific
later); (c) it is a deliberate first step toward **GIS** (real raster ingestion, leaning on
libraries). It is authored partly as an agentic-coding exercise; this ADR records the design
decisions and their live tensions honestly (see [[agentic-provenance-honest-in-docs]]).

This repo already has the substrate: the iterative field-sim reference `src/gpu/sim/
reactionDiffusion.ts` (Gray–Scott — same shape: grid, N channels, ping-pong, per-cell update),
the `"use gpu"` TGSL kernel discipline (ADR-0003), the Mutator/`ParamSpec` breeding surface
(`src/evo`), and a heatmap preview (`playground/src/Preview.tsx`).

## Decision

Build HsPf as a **stand-alone example that reuses the repo's *library primitives* but not its
*op-graph framework*** — the counterpart to reaction-diffusion (which is the framework-native
sim). It operates deliberately in **"we want to render graphics" mode**, distinct from the
op-graph's "verified compute primitive" mode; that departure is a feature, not an oversight.

1. **Stand-alone artefact, not a composer op.** HsPf's core is an *irreducible fused gather*
   (thousands-of-taps sampled neighbourhood + per-bite recombination + barrier geometry) that
   does not reduce to stock ops; wrapping it as one giant graph node adds ceremony and buys
   nothing. Instead: its own page entry (`playground/hspf.html` → `hspfMain.ts`), mirroring the
   `datasource.html` precedent. It reuses library primitives — `getDevice`, the field/element
   types (ADR-0004), seeded RNG (`src/evo/rng.ts`), tgpu-managed buffers + `.read()` readback, and
   the `ParamSpec`/Mutator surface (`src/evo`). "No framework" ≠ "no library." The op-graph remains
   a *possible later on-ramp* if a decomposed variant ever wants it.

2. **One fused `hspfStep` kernel, written normally — in raw WGSL.** Reaction (`F`: fitness blend +
   single- and two-bite offspring table), the seeded neighbourhood gather, normalisation, LD, and
   the barrier segment-crossing test all live in one ordinary compute kernel. It is authored in
   **raw WGSL rather than `"use gpu"` TGSL — a deliberate, recorded deviation from ADR-0003**: the
   two-bite recombination needs dynamic vector/array indexing (`pf[g1]`, `offspring[r]`) that is
   clumsy through the transpiler, and the original is already WGSL. Readback still uses the
   tgpu-managed buffer `.read()` path (a raw staging `mapAsync` trips Dawn-on-Node's exit-teardown
   segfault). A demo showing a normal chunk of kernel code is a legitimate thing. **Decomposition is deferred wholesale.** The one piece
   flagged for eventual extraction is the **barrier mask** — and *precisely because* it wants to
   be general (3D, other dtypes) it deserves a considered design later, not a rushed extraction
   now.

3. **Bring-your-own real data; decode GeoTIFF locally.** The recognisable Africa coastline is
   intrinsic to the `HbS` raster's land/ocean mask (ocean = the `-2` sentinel), not a separate
   basemap. Decode `hbsfilter.tif` / `pf2000.tif` in-page with **`geotiff.js`** — a deliberate
   step toward GIS and toward leaning on libraries rather than bespoke raster code. **No reusable
   `geotiffLoader` / Loader abstraction is built** (the data is too thin to model a general one,
   and that work belongs to the datasource context, ADR-0008 — cross-referenced, not built here).
   **Capture** the geo-transform (lat/long ↔ pixel) into metadata now; **defer using it** until
   phase 2. Sentinel (`NaN → -2`) and `outerPadding` prep happen at the ingestion boundary, not
   in the kernel.

4. **Seed the neighbourhood generation.** The kernel is deterministic; the only randomness is in
   generating the neighbourhood (CPU-side), so there is **no RNG in the kernel** (sidestepping the
   repo's no-`Math.random` / no-f32-atomics gotchas). `Beta(1, c)` has a closed-form inverse-CDF
   `x = 1 − (1−U)^(1/c)`, so seeding it with `mulberry32` makes runs **reproducible** (required
   for a future data-fit fitness) and **drops the `@stdlib/random` dependency**.

5. **Graphics mode ⇒ proportionate verification, not a full CPU golden.** Duplicating the whole
   thousands-of-taps pipeline as a CPU oracle would drag a graphics artefact down. Instead:
   **unit-test the fiddly pure math** — `F` (single- + two-bite offspring table) and the LD
   formula — against a handful of hand/original-derived values, plus a few **invariants**
   (genotype frequencies sum to 1 after normalisation; a uniform field is a fixed point; symmetric
   input → symmetric output). This catches the silent-wrong-math failure mode (a mis-indexed
   offspring table, a wrong fitness blend, an LD-denominator slip — the class of bug behind
   [[gpu-vec3-stride-16]] and [[gpu-typecheck-before-shader-debug]]) without a field-level golden.
   The tension with this repo's CPU-golden habit is acknowledged, not papered over.

6. **GPU-resident rendering; refactor `Preview` onto a shared field-viz module.** Per-frame
   GPU→CPU readback (as today's `Preview` does) is a sync stall with no place in a graphics loop —
   the sim output stays GPU-resident and the render pass samples it directly. `Preview` /
   `drawHeatmap` are **refactored (not bypassed)** onto a shared **GPU render pass** that, in one
   shader, does: nodata-sentinel → transparent (this draws the coastline for free), a **palette
   LUT** (sequential for frequencies, diverging-about-0 for LD `r`), **GPU iso-line contours**
   (`fwidth`-antialiased — *not* CPU marching-squares, which produce ugly facets; matches the
   psychogeo/MDV approach), and channel selection. The colour-map seam is typed
   **`channels[] → RGB`** so an **okLab/okLch bivariate** view (two fields as one colour) is a
   later drop-in against the same seam. Render from a `float32-filterable` texture (request the
   feature as the original did; fall back to storage-buffer manual bilinear if absent). Nice
   contours and richer palettes are wanted for visualisation generally, so this lands as shared
   infrastructure — accepting some scope diffusion, in the mode of fleshing out basic
   functionality where we meet it. This nuances ADR-0009 (rendering-as-ops): here rendering is a
   direct GPU pass, not (yet) a graph op.

7. **`Params`/`ParamSpec` is the one seam to get right.** The sim is driven by a plain `Params`
   object built from a declared `ParamSpec[]` — cheap now, expensive to retrofit, and the bridge
   to the whole Mutator spectrum (`specimenToParams` feeds params directly; `withLocked` is
   freeze; `steer`/`advance` is casual exploration; `mutate`/`breed` is evolution; `Pedigree` is
   the dancer selection surface). Extend `ParamSpec` (additively) with optional **dotted-path
   hierarchical names** (`fitness.S.++`, `spread.concentration`) used as flat string keys — a real
   namespace with no change to `param()`/`Params` — and optional **`tags`**. Interaction model:
   **filter → apply** — filter params (by path-prefix, tag, name, locked-state) into an *ordered*
   set, then apply an operation to it by **locking the complement** (`withLocked` of everything
   else) and running `freeze`/`mutate`/`steer`/`randomize`/`reset`. The ordered filtered set is
   the designated future **MIDI-CC** binding target.

8. **The general Mutator UI and MIDI interact with the artefact by *availability*, not by being
   built into it.** HsPf's obligation is to expose a clean `ParamSpec` seam and be `Params`-driven.
   The shared param-exploration UI (manual controls, filter-then-apply, `steer`/`mutate`) is
   pointed *at* HsPf, not bespoke to it. The dancer-style population/pedigree UI and MIDI-CC are
   repo-level capabilities that operate on the seam from outside; they are **out of scope for this
   artefact** even in phase 2.

9. **Phase split, cut along the geo-referencing seam.**
   - **Phase 1 — core sim + graphics + play.** Load real `HbS` + `weights` from GeoTIFF; the fused
     `hspfStep` (barrier code path present but fed zero barriers — dormant, faithful, free);
     GPU-resident render (palette + GPU contours + nodata coastline + **channel selector** across
     the 5 layers, not multi-panel); the `Params` seam + shared exploration UI (manual / freeze /
     filter-then-apply / cheap `steer`/`mutate`); seeded neighbourhood + targeted math tests +
     invariants.
   - **Phase 2 — geo-referencing + science.** Activate the geo-transform → **barriers from TSV**
     (light up the dormant path), the **counts overlay + comparison charts**, and the **data-fit
     fitness** (simulated vs. observed country counts — the natural compute-graph-derived fitness
     the exploration story wants). Multi-panel layout as a nicety.
   - **Deferred / ambient:** MIDI-CC, okLab/okLch bivariate view, dancer population/pedigree UI.
   - **Skipped unless asked:** the `.hspf` snapshot serialisation/download.

10. **Module home.** `src/gpu/sim/hspf/` holds the library core (kernel, pure math `F`/LD, seeded
    neighbourhood, tests, and this context's `CONTEXT.md` glossary). `playground/` holds the page
    (`hspf.html` + `hspfMain.ts`), the GeoTIFF load, the shared render module, and the exploration
    UI. This mirrors the `datasource.html` split (pure core in `src/`, presentation in
    `playground/`).

## Consequences

- **HsPf becomes the canonical *stand-alone, library-reuse* sim**, complementing reaction-diffusion
  (the canonical *framework-native* sim). Together they demonstrate the lower layers carry a real
  model both inside and outside the composer — evidence the abstractions are honestly reusable.
- **The `Preview` refactor is real scope** beyond a single example — a GPU-resident field-viz module
  with palettes and GPU contours. Justified because these are broadly wanted for visualisation, but
  it diffuses this artefact's boundary. We accept that, in the mode of fleshing out basic
  functionality where we meet it.
- **A genuine tension is recorded, not resolved:** this artefact steps outside the repo's
  CPU-golden / verified-compute-primitive discipline into a graphics-first mode. That discipline can
  drag a graphics artefact down; the proportionate answer here (fiddly-math unit tests + invariants)
  is a deliberate scoping of verification, and a data point on where the golden habit does and
  doesn't earn its keep.
- **GeoTIFF is pulled in without a Loader abstraction.** We get the GIS/library step and the real
  map now; the reusable geospatial Loader is explicitly left to the datasource context (ADR-0008)
  rather than modelled prematurely on thin data.
- **`ParamSpec` grows optional `path`/`tags` metadata** — additive, so every existing op and the
  evo path are untouched — and this artefact is the first consumer of the *filter-then-apply* param
  interaction model, which the general Mutator UI and MIDI will later build on.
- **Barriers are present-but-dormant in phase 1.** Keeping the code path faithful (fed zero
  barriers) avoids ripping-and-re-adding, and the geo-referencing seam is where they and the counts
  overlay naturally light up together.
- **Provenance.** HsPf is prior art by **Gavin Band & Andre Python**; this is a re-derivation, not a
  lift. The original's GeoTIFF assets and observed-count data are reused as inputs.

## References

- ADR-0003 (`"use gpu"` TGSL kernels), ADR-0004 (field/element type model), ADR-0008 (view-driven
  multiscale datasource — where a reusable GeoTIFF Loader belongs), ADR-0009 (rendering as ops —
  nuanced here: rendering as a direct GPU pass), ADR-0010 (composable ops; `ParamSpec` breeding
  surface pattern).
- `src/gpu/sim/reactionDiffusion.ts` (the iterative field-sim reference), `src/gpu/graph/op.ts`
  (`ParamSpec`/`Params`), `src/evo` (Mutator: `traitSpaceFromParams`, `specimenToParams`,
  `withLocked`, `steer`/`mutate`/`breed`, `Pedigree`), `playground/src/Preview.tsx` (the heatmap
  being refactored), `playground/datasource.html` (the stand-alone-page precedent).
- Prior art: `~/code/www/hspf-spatial-analysis/theory/html/hspf-gpu` (`HsPfSim.wgsl`,
  `HsPfSim.ts`, `Simulation.ts`) — Gavin Band & Andre Python.
- `src/gpu/sim/hspf/CONTEXT.md` (the HsPf ubiquitous language) and root `CONTEXT-MAP.md`.
