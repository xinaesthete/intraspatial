---
title: Fuzzy adjacency
description: A kernel-weighted graph — connection strength in [0,1], not a binary edge.
---

A weighted graph over a point cloud where each pair carries a **membership** in
`[0,1]` rather than a 0/1 edge:

```
μ_ij = exp(-d_ij² / 2σ²),   μ_ii = 0,   μ = 0 beyond the support radius
```

## What it means

How strongly each pair is connected, with connection fading smoothly as distance
grows — the [tapered](/concepts/windowing/) version of a hard "edge if within radius
R" graph. It is the substrate for [fuzzy TDA](/concepts/fuzzy-tda/): the weighted
1-skeleton of a fuzzy simplicial complex, far less brittle to the choice of radius
than a crisp graph.

## GPU approach

A `"use gpu"` kernel: one thread per point evaluates the membership to every other
point, writing a dense N×N matrix. Dense is the natural form for small N (the only
regime Node + Dawn validates anyway); a sparse, spatial-index-backed version is
future work for large N.

## Usage

```ts
import { fuzzyAdjacencyGpu } from 'intraspatial/gpu/spatial/fuzzyAdjacency';

const { membership, n } = await fuzzyAdjacencyGpu(xs, ys, { sigma: 1.5 });
// membership[i * n + j] = μ_ij  (symmetric, zero diagonal)
```

## Composes with

- [Fuzzy TDA](/concepts/fuzzy-tda/) — threshold the membership to build a filtration
  the CPU reduces to a persistence diagram.
- Weighted graph statistics (planned) — weighted degree, clustering, communities.

**Status:** ✅ implemented and CPU-golden validated ·
`src/gpu/spatial/fuzzyAdjacency.ts`
