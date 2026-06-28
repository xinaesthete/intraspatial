---
title: Fuzzy TDA
description: The windowing principle applied to topology — soft connectivity for robust persistent homology.
---

Topological data analysis asks what *shape* a point set has — connected components,
loops, voids — and how robustly those features persist as you connect points at
growing scales. The standard tool, the **Vietoris–Rips** complex, connects two
points iff they are within radius `r`. That is a [boxcar](/concepts/windowing/) in
distance, and it inherits every weakness of a hard threshold: brittle to the radius,
sensitive to outliers, and blind to *how strongly* two points are connected.

## Soften the connectivity

"Fuzzier" TDA replaces the binary edge with a smooth **membership** in `[0,1]`:

```
μ_ij = exp(-d_ij² / 2σ²)     instead of   [ d_ij ≤ r ]
```

An edge now fades in with proximity rather than snapping on at one threshold. Run
persistence on the resulting *weighted* complex and the features that survive across
a wide band of membership are the robust ones.

This is not exotic — it is the construction underneath **UMAP** (fuzzy simplicial
sets), and it has relatives in **weighted Vietoris–Rips** persistence and
**distance-to-measure** (density-aware, outlier-stable) filtrations.

## Division of labour

The homology reduction itself is inherently sequential — leave it to a CPU library
(Ripser / GUDHI). Everything *upstream* — building the fuzzy filtration — is
embarrassingly parallel and is where this toolbox helps:

- ✅ [Fuzzy adjacency](/primitives/fuzzy-adjacency/) — the kernel-weighted graph; the
  weighted 1-skeleton.
- 🔜 Adaptive per-point bandwidth (UMAP-style), built from
  [nearest-neighbour distance](/primitives/nearest-neighbour-distance/).
- 🔜 Membership-sweep filtration: the GPU builds each level's edge set, the CPU
  reduces it.

So the rule mirrors crisp VR: **GPU builds the (now fuzzy) filtration; CPU reduces
it.** Fuzzy adjacency is the first brick, and it falls straight out of the same
windowing kernel used everywhere else in the toolbox.
