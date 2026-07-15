# Context — HsPf spatial population-genetics example

The ubiquitous language for the **HsPf** artefact (ADR-0011): a stand-alone spatial
selection–migration simulation over a raster map of Africa, re-derived from prior art by
**Gavin Band & Andre Python**. This document glosses only that artefact; other subsystems
(datasource, geometry-ops, evo/Mutator, the codec) have their own contexts.

## Language

### The model

**HsPf**:
The simulation as a whole — **H**aemoglobin-**S** × **P**. **f**alciparum. A two-locus
selection–migration process whose state is a genotype-frequency field evolved on the GPU.
_Avoid_: "the malaria sim" (imprecise), HsPfSim (the original's class name).

**Cell**:
One raster pixel of the map. Carries a fixed **scaffold** value (`HbS`, `weights`) and an evolving
**genotype vector** + LD. Land cells simulate; ocean/missing cells hold a **sentinel**.

**Genotype vector** (`pfsa`):
The four genotype frequencies at a cell — `--`, `-+`, `+-`, `++` — over the two biallelic loci
(HbS × the Pf-linked locus). Sums to 1 on a land cell after normalisation. Stored as 4 of the 5
per-cell layers.
_Avoid_: "the four channels" (the 5th layer is LD, not a genotype).

**LD** (`r`):
Linkage disequilibrium — the 5th per-cell layer — measuring statistical association between the
two loci, `r ∈ [−1, 1]`, `0` = independence. A *derived* quantity recomputed from the genotype
vector each step; rendered with a **diverging** palette centred at 0.
_Avoid_: correlation (informal), covariance.

**Scaffold**:
The fixed background rasters a run is built on — **HbS** (sickle-cell allele frequency; its
land/ocean mask *is* the recognisable Africa coastline) and **weights** (population / Pf-endemicity,
modulating bite weight). Loaded from GeoTIFF; never evolved.
_Avoid_: "the map" (ambiguous with the rendered output), background.

**Sentinel**:
A reserved out-of-domain value in the scaffold — `-1` (missing) / `-2` (ocean) — that flags a cell
as non-simulating. Rendered as transparent (drawing the coastline). Cells early-exit on a sentinel.
_Avoid_: NaN (the original's raw form; normalised to a sentinel at ingestion), nodata (fine
informally; sentinel is the term).

### The step

**Neighbourhood** (the "mosquito bites"):
The set of sampled offsets a cell gathers from each step — random directions with distances drawn
from `Beta(1, concentration)` (biting is mostly local). Generated **CPU-side** and uploaded as a
buffer the kernel reads; the kernel itself is deterministic. Generation is **seeded** (closed-form
inverse-CDF + `mulberry32`) for reproducibility.
_Avoid_: stencil (implies a fixed kernel — this is sampled and variable), kernel (overloaded).

**Concentration**:
The `Beta` shape parameter governing how *local* biting is — higher ⇒ tighter neighbourhood ⇒ less
geographic smoothing. A headline **exploration parameter**.

**Reaction** (`F`):
The pointwise nonlinear map applied to a neighbour's genotype vector before gathering: the fitness
blend `a·A + s·S` times the **single-bite** (linear) plus **two-bite** (bilinear, recombining) terms.
Pure function of one cell's state + local HbS.
_Avoid_: "the update" (the whole step is the update; `F` is only its pointwise part).

**Fitness matrix**:
The 2×4 table (`A` = background, `S` = sickle) selecting on each genotype, blended per-cell by local
HbS into `a·A + s·S`. A headline exploration parameter group.

**Two-bite rate**:
The fraction of transmission that is **two-bite recombination** rather than single-bite. Weights the
bilinear offspring-table term against the linear term.

**Offspring table**:
The 16×4 Punnett table giving, for each ordered parent-genotype pair, the transmitted-genotype
probabilities — the mechanism of two-bite recombination.

**Barrier**:
A line segment that **down-weights** any bite-flight crossing it (segment-intersection test in the
kernel). Present-but-**dormant** (fed zero barriers) in phase 1; lit up from geo-referenced data in
phase 2. Flagged as a *future* general, dimension-polymorphic mask op — designed later, not now.

### Control & exploration

**ParamSpec seam**:
The declared `ParamSpec[]` (dotted-path **hierarchical names** + **tags**) the sim is driven from,
as a plain `Params` object. The one seam this artefact must get right — the bridge to the whole
Mutator spectrum. HsPf *exposes* it; the general Mutator UI and MIDI *consume* it by availability.

**Filter-then-apply**:
The parameter interaction model — **filter** params (by path-prefix, tag, name, or locked-state)
into an *ordered* set, then **apply** an operation to it by **locking the complement** and running
`freeze` / `mutate` / `steer` / `randomize` / `reset`. The ordered filtered set is the future
MIDI-CC binding target.

**Freeze**:
Excluding a param from mutation/steering — implemented as `withLocked` (from `src/evo`) over the
filtered set. The manual ↔ evolutionary control continuum runs: manual `Params` ⟷ `steer` ⟷
`mutate`, with freeze as the per-param gate.

**Data-fit fitness** (phase 2):
The natural objective — how well the simulated genotype frequencies match the **observed country
counts** — the concrete, scientific instance of a compute-graph-derived fitness. Where the counts
overlay earns its place.
