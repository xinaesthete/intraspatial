# Fuzzy TDA and the windowing principle

Status: **design note** (2026-06-28, updated with the CkNN paper)

This note follows a hunch from the user: that topological data analysis (TDA) has
"fuzzier" variants worth building on, and that they connect to the windowing theme
running through this toolbox. The user then supplied the paper they had in mind —
**Berry & Sauer, *Consistent Manifold Representation for Topological Data Analysis*
(2016)**, `docs/manifoldTDA.pdf` — which both confirms the connection and sharpens
it. The original framing is below; the [CkNN refinement](#the-cknn-refinement-berry--sauer)
adds the piece it was missing.

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

## The CkNN refinement (Berry & Sauer)

The original framing above has one axis: hard → smooth (boxcar → taper). The CkNN
paper adds the axis it was missing — **local density adaptation** — and, for
topology specifically, partly *inverts* the "fuzzier is better" intuition.

**The construction.** Let `ρ_i = d(i, k-th nearest neighbour)` — a local density
estimate (small where dense, large where sparse; on an *m*-manifold it scales like
`q(x)^(−1/m)`, with the dimension baked in for free). CkNN connects `i,j` when

```
d(i,j) < δ · √(ρ_i ρ_j)          (δ a single continuous scale knob)
```

i.e. ordinary connectivity on a **density-rescaled distance** `d̃_ij = d_ij/√(ρ_iρ_j)`.

**Two surprises worth absorbing:**

1. **For topology, keep it *unweighted*.** A fixed-bandwidth Gaussian membership
   (our `fuzzyAdjacency`) is the right tool for *geometry* — its graph Laplacian
   converges to the Laplace–de Rham operator (the self-tuning kernel is
   `K_ij = exp(−d̃_ij²)`). But to read off *homology* you want an unweighted graph,
   because then a Vietoris–Rips complex and its Betti numbers come for free; getting
   homology out of a weighted Laplacian means estimating eigenspace dimensions, which
   is fiddly. So the move for topology is not "add a taper" — it is "keep the boxcar
   but make its radius adapt locally via `ρ`." Same density-adaptive core, two
   readouts: **`d̃ < δ` for topology, `exp(−d̃²)` for geometry.**
2. **Consistent vs persistent homology.** Because `ρ` rescales away the local density,
   a *single* CkNN graph captures features at every scale at once — the right Betti
   numbers simultaneously — and it does so even on **non-compact** manifolds where no
   fixed radius can (a sparse arm and a dense core need different `ε`, but the same
   `δ`). Persistence over `δ` becomes a tool for *selecting* `δ`, not the end product.

**The practical payoff for us is small:** CkNN persistence is just standard VR
persistence on `d̃`. So the GPU builds `d̃` (one matrix), and the same CPU reducer we
already planned for fuzzy VR consumes it — no new machinery.

## What this toolbox can build

The homology reduction itself (the boundary-matrix reduction that turns a
filtration into a persistence diagram) is inherently sequential and stays on the
CPU — feed it to Ripser/GUDHI. But everything *upstream* of it — building the fuzzy
filtration — is embarrassingly parallel and is where the GPU and the windowing
primitives earn their keep:

- ✓ **Fuzzy adjacency** (`src/gpu/spatial/fuzzyAdjacency.ts`) — the fixed-bandwidth
  kernel-weighted graph `μ_ij = exp(-d²/2σ²)`. The geometry-side, weighted 1-skeleton.
- ✓ **k-th neighbour distance** (`src/gpu/spatial/kthNeighborDistance.ts`) — the local
  bandwidth `ρ_i = d(i, x_k)`. (First WGSL-template spatial kernel: the per-point
  k-smallest selection needs a local array, which TGSL can't express — see ADR-0003.)
- ✓ **CkNN rescaled distance** (`src/gpu/spatial/cknn.ts`) — `d̃_ij = d_ij/√(ρ_iρ_j)`,
  composing the two above. One matrix that yields both the topology graph
  (`cknnGraph`, `d̃ < δ`) and the geometry kernel (`selfTuningWeights`, `exp(−d̃²)`).
  This is the density-adaptive upgrade the original list called for.
- → **CPU persistence reducer**: feed `d̃` to Ripser/GUDHI for a (consistent or
  δ-swept) persistence diagram. The GPU's job ends at `d̃`.
- → **DTM / density-weighted distance**: reuse the KDE splat to get a density field,
  derive a smoothed distance, hand that to the filtration.
- → **Adaptive Gaussian membership** (UMAP-style): use `kthNeighborDistance` for the
  local `ρ` and symmetrise with the t-conorm — the fuzzy-simplicial-set variant.

So the division of labour holds: **GPU builds the (fuzzy or rescaled) filtration; CPU
reduces it.** The CkNN primitives now cover both the topology (`d̃ < δ`) and geometry
(`exp(−d̃²)`) readouts from a single density-adaptive core.

## Why it's worth it (interpretation)

A persistence diagram from a *crisp* VR complex answers "what holds together at
radius r." A diagram from a *fuzzy* complex answers "what holds together robustly,
weighting near connections more than far ones" — the same upgrade in
interpretability that windowed statistics gave us over quadrats: fewer artefacts of
an arbitrary threshold, and a confidence-graded rather than binary read of
structure. For tissue (the MuSpAn setting) that means crypt/loop features that
survive across a band of connection strengths, not ones that hinge on a single
lucky radius.

There is a subtlety the CkNN paper makes us hold both at once: **for geometry, fuzz
it (weighted, → Laplace–de Rham); for topology, keep it crisp but density-adaptive
(unweighted CkNN, → consistent homology).** Both fall out of the same `d̃`, so the
toolbox doesn't have to choose — it produces `d̃` and lets the readout decide.

## Where to take it next

With `d̃` in hand the open work is mostly downstream and CPU-side:

- Wire a **persistence reducer** (Ripser/GUDHI) over `d̃` and surface the diagram;
  validate against the paper's "figure-eight" / annulus-with-gap examples, where a
  fixed radius provably cannot recover the two holes but CkNN can.
- Use the consistent graph for **spectral clustering** (the paper's other
  application): the unweighted CkNN Laplacian's low eigenvectors segment the data.
- Revisit whether the user's "fuzzier methods" intent is exactly CkNN/self-tuning, or
  also reaches DTM-style robust filtrations or soft Euler-characteristic summaries —
  the `kthNeighborDistance` + `d̃` substrate serves all of them.

See also [`gpu-spatial-analysis-toolbox.md`](gpu-spatial-analysis-toolbox.md) and the
CkNN paper `manifoldTDA.pdf` (Berry & Sauer, 2016).
