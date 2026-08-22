---
title: Nearest-neighbour distance
description: For each point, the distance to its closest neighbour.
---

For each of N points, the Euclidean distance to its nearest *other* point.

## What it means

How isolated each point is. The distribution of these distances is the raw material
for several spatial statistics: a tight distribution of small distances means
clustering; large, even distances mean dispersion.

## GPU approach

Brute-force O(N²): one thread per point loops over all points, tracking the minimum.
Authored in TypeScript with the `"use gpu"` directive (transpiled to WGSL). No
atomics or shared memory, so it fits the directive directly. The
index-accelerated O(N·k) version (a uniform-grid spatial index) is planned for large
N; correctness is identical.

## Usage

```ts
import { nearestNeighborDistancesGpu } from 'intraspatial/gpu/spatial/nnDistance';

const d = await nearestNeighborDistancesGpu(xs, ys); // Float32Array, length N
```

## Composes with

- [ANNI](/primitives/anni/) — summarises these distances into a clustered / random /
  dispersed verdict.
- The empty-space function (planned) — the same kernel, queried from random sample
  locations instead of the points themselves.

**Status:** ✅ implemented and CPU-golden validated · `src/gpu/spatial/nnDistance.ts`
