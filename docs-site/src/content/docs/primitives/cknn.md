---
title: CkNN rescaled distance
description: Continuous k-Nearest Neighbours — a density-rescaled distance for consistent topology.
---

The Continuous k-Nearest Neighbours construction of Berry & Sauer (2016): a
**density-rescaled distance**

```
d̃_ij = d_ij / √(ρ_i ρ_j),   where ρ_i = distance to the k-th nearest neighbour.
```

## What it means

A fixed connection radius can't cope with varying density — too small and sparse
regions fall apart, too large and dense regions bridge together. Dividing by the
local bandwidths `√(ρ_iρ_j)` **normalises density away**, so one scale `δ` works
everywhere. Two readouts come from the same matrix:

- **Topology** — threshold `d̃ < δ` for the (unweighted) CkNN graph; feed `d̃` to a
  Vietoris–Rips persistence engine. Because density is normalised, a *single* graph
  captures features at all scales at once ("consistent homology"), and it recovers
  the correct topology even on non-compact manifolds where no fixed radius can.
- **Geometry** — the self-tuning kernel is just `exp(−d̃²)`; its graph Laplacian is a
  consistent estimator of the Laplace–de Rham operator.

This is the [fuzzy-TDA](/concepts/fuzzy-tda/) thread made concrete — with the twist
that, for *topology*, the winning move is to keep the graph unweighted but make its
radius density-adaptive, rather than to add a taper.

## GPU approach

A *composition*: [k-th neighbour distance](/primitives/kth-neighbour-distance/)
(WGSL template) gives `ρ`; a `"use gpu"` kernel then builds the N×N rescaled-distance
matrix. The persistence reduction itself stays on the CPU (Ripser/GUDHI) — the GPU's
job ends at `d̃`.

## Usage

```ts
import {
  cknnRescaledDistanceGpu, cknnGraph, selfTuningWeights,
} from 'tgpu-htj2k/gpu/spatial/cknn';

const { rescaled, rho, n } = await cknnRescaledDistanceGpu(xs, ys, { k: 5 });
const graph   = cknnGraph(rescaled, n, /* δ */ 1.2);   // unweighted adjacency (topology)
const weights = selfTuningWeights(rescaled, n);        // exp(−d̃²) (geometry)
```

**Status:** ✅ implemented and validated (matches CPU; density-normalisation property
holds) · `src/gpu/spatial/cknn.ts`
