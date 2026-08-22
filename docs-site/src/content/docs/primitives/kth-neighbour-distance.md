---
title: k-th neighbour distance
description: A local density estimate — distance to the k-th nearest neighbour.
---

For each point, the distance to its **k-th** nearest other point, `ρ_i = d(i, x_k)`.

## What it means

A local density estimate: small where points are densely packed, large where they
are sparse. For points sampled from an *m*-dimensional manifold it scales like
`q(x)^(−1/m)` — it implicitly encodes the intrinsic dimension without your having to
estimate it. This is the local bandwidth that makes density-adaptive methods like
[CkNN](/primitives/cknn/) and the self-tuning kernel work.

## GPU approach

Brute force O(N·k): each thread keeps the k smallest distances it has seen in a
**local fixed-size array**, then returns the largest of them. Local mutable arrays
are not expressible with the `"use gpu"` directive in this TypeGPU version, so this
is the toolbox's first **WGSL-template** kernel (the array is sized to a compile-time
maximum; the actual `k` is a uniform).

## Usage

```ts
import { kthNeighborDistanceGpu } from 'intraspatial/gpu/spatial/kthNeighborDistance';

const rho = await kthNeighborDistanceGpu(xs, ys, 5); // ρ_i for k = 5
```

`k = 1` recovers the [nearest-neighbour distance](/primitives/nearest-neighbour-distance/).

**Status:** ✅ implemented and CPU-golden validated ·
`src/gpu/spatial/kthNeighborDistance.ts`
