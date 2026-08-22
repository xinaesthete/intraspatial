---
title: Empty-space function
description: How big are the voids? Distance from random locations to the nearest point.
---

For each of many random sample locations, the distance to the nearest data point —
the spherical contact / F-function.

## What it means

Where [nearest-neighbour distance](/primitives/nearest-neighbour-distance/) measures
gaps *between points*, this measures the size of *empty space*. Large values mean big
voids. It is the natural complement to [ANNI](/primitives/anni/): a clustered pattern
leaves large empty regions (big empty-space distances); an evenly spread pattern
fills the region (small ones).

## GPU approach

The same brute-force min-distance kernel as nearest-neighbour distance, but queried
from a separate set of random sample locations (and without excluding self).
Authored in `"use gpu"`. Sampling is seeded, so results are reproducible.

## Usage

```ts
import { emptySpaceGpu } from 'intraspatial/gpu/spatial/emptySpace';

const r = await emptySpaceGpu(xs, ys, { numSamples: 1024, bbox: [0, 0, 100, 100] });
// r.distances: per-sample nearest-point distance; r.mean: typical void radius
```

The full distribution of `r.distances` is the empty-space CDF; its mean is a compact
"typical void radius".

**Status:** ✅ implemented and CPU-golden validated · `src/gpu/spatial/emptySpace.ts`
