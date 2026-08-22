---
title: Average Nearest Neighbour Index
description: Is a point pattern clustered, random, or dispersed?
---

A classic, fully interpretable test: compare the observed mean nearest-neighbour
distance to what you'd expect if the points were scattered at random.

## What it means

`R = observed mean NN distance / expected mean NN distance` under complete spatial
randomness:

- `R < 1` — **clustered** (points closer together than random)
- `R ≈ 1` — **random**
- `R > 1` — **dispersed** (more evenly spaced than random)

A z-score gives significance (Clark & Evans). The primitive returns the index, the
z-score, and a plain-language `interpretation` field.

## GPU approach

A *composition*: the GPU does the O(N²)
[nearest-neighbour search](/primitives/nearest-neighbour-distance/); ANNI adds the
cheap CPU summary (mean, the CSR expectation, the z-score) on top. Nothing new runs
on the GPU — it reuses an existing primitive, which is the point.

## Usage

```ts
import { anniGpu } from 'intraspatial/gpu/spatial/anni';

const r = await anniGpu(xs, ys, { area: width * height });
// { index, zScore, interpretation: 'clustered' | 'random' | 'dispersed', ... }
```

**Status:** ✅ implemented and validated (random / clustered / dispersed fixtures) ·
`src/gpu/spatial/anni.ts`
