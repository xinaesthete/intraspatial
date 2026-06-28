# Fuzzy TDA and the windowing principle

Status: **design note** (2026-06-28)

This note follows a hunch from the user: that topological data analysis (TDA) has
"fuzzier" variants worth building on, and that they connect to the windowing theme
running through this toolbox. There's a specific paper the user means to dig out;
until then this records what the connection seems to be and what we can build now.
If the paper points somewhere else, this is cheap to redirect.

## The one idea: a hard threshold is a boxcar

We've made the same move twice already:

- **Quadrats → windows.** Hard binning assigns each point to exactly one cell — a
  boxcar in space, with the worst sidelobes of any window. Replacing it with a
  smooth, overlapping window (Gaussian/Hann) removes the grid-phase artefacts and
  recovers cross-boundary relationships. (See the toolbox doc.)
- **Hard adjacency → fuzzy adjacency.** A graph edge that exists iff two points
  are within radius `R` is a boxcar in *distance*. Replacing the 0/1 indicator with
  a smooth membership `μ_ij = exp(-d²/2σ²)` gives a weighted graph where an edge
  *fades in* with proximity instead of snapping on at one threshold.

Classical TDA — Vietoris–Rips (VR) persistent homology, as used in the MuSpAn
paper — is built on exactly that boxcar. The VR complex at scale `r` connects all
pairs within `r`; you sweep `r` and watch topological features (components `H₀`,
loops `H₁`, voids `H₂`) appear and die. It works, but it inherits every weakness of
a hard threshold: brittle to the radius, sensitive to noise and outliers (one stray
point can bridge two structures at a single `r`), and blind to *how strongly* two
points are connected.

**Fuzzier TDA replaces the boxcar threshold with a membership function.** The
topology is then read off a *weighted* complex, where each simplex carries a
strength in `[0,1]`, not a binary in/out.

## Where the "fuzzy" comes from (the lineage)

- **Fuzzy simplicial sets.** This is the construction underneath **UMAP**
  (McInnes & Healy, building on Spivak's fuzzy simplicial sets). Around each point a
  local kernel assigns membership strengths to its neighbours — adaptively scaled so
  every point is "locally connected" — and the per-point fuzzy graphs are merged
  with a probabilistic t-conorm (`a + b − a·b`). The result is a single fuzzy graph
  whose edge weights are calibrated connection probabilities. This is *exactly* a
  fuzzy adjacency with a per-point adaptive bandwidth.
- **Weighted / fuzzy Vietoris–Rips.** Run persistence on the weighted complex
  (edges ordered by membership, or filtered by a membership threshold sweeping
  1→0) rather than the crisp distance complex. Features that persist across a wide
  band of membership are the robust ones.
- **Distance-to-measure (DTM) filtrations** (Chazal, Cohen-Steiner, Mérigot).
  Replace raw distance with a density-aware, smoothed distance, so the filtration is
  provably stable to outliers — another way of saying "don't trust a single hard
  radius." A density/KDE field (which we already produce) is the natural input.

All three are the same instinct: **smooth the connectivity, then do topology.**

## What this toolbox can build

The homology reduction itself (the boundary-matrix reduction that turns a
filtration into a persistence diagram) is inherently sequential and stays on the
CPU — feed it to Ripser/GUDHI. But everything *upstream* of it — building the fuzzy
filtration — is embarrassingly parallel and is where the GPU and the windowing
primitives earn their keep:

- ✓ **Fuzzy adjacency** (`src/gpu/spatial/fuzzyAdjacency.ts`) — the kernel-weighted
  graph `μ_ij = exp(-d²/2σ²)`, truncated past a support radius. The first concrete
  fuzzy-TDA primitive: it *is* the weighted 1-skeleton.
- → **Adaptive per-point bandwidth** (UMAP-style): set `ρ_i` = nearest-neighbour
  distance (we have `nnDistance`) and solve `σ_i` so `Σ_j μ_ij = log₂(k)`. A small
  per-point root-find; GPU-friendly. Then symmetrise with the t-conorm.
- → **Membership-sweep filtration**: threshold the fuzzy graph at a decreasing
  sequence of memberships to emit a nested sequence of (weighted) complexes — the
  GPU builds each level's edge set; the CPU reduces.
- → **DTM / density-weighted distance**: reuse the KDE splat to get a density field,
  derive a smoothed distance, hand that to the filtration.

So the division of labour mirrors what we already decided for VR: **GPU builds the
(now fuzzy) filtration; CPU reduces it.** The fuzzy adjacency primitive is the first
brick, and it falls straight out of the windowing kernel we use everywhere else.

## Why it's worth it (interpretation)

A persistence diagram from a *crisp* VR complex answers "what holds together at
radius r." A diagram from a *fuzzy* complex answers "what holds together robustly,
weighting near connections more than far ones" — the same upgrade in
interpretability that windowed statistics gave us over quadrats: fewer artefacts of
an arbitrary threshold, and a confidence-graded rather than binary read of
structure. For tissue (the MuSpAn setting) that means crypt/loop features that
survive across a band of connection strengths, not ones that hinge on a single
lucky radius.

## Open question for the user's paper

The specific "fuzzier methods" paper may mean any of: fuzzy simplicial sets,
weighted/fuzzy VR persistence, DTM-style robust filtrations, persistent homology of
*fuzzy sets* proper, or something else (e.g. soft/“smooth” Euler characteristic
curves, or persistence images as fuzzy summaries). The fuzzy adjacency primitive is
useful under all of these — it's the shared substrate — so it's a safe first build
regardless of which the paper turns out to mean.

See also [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md).
