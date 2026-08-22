---
title: Getis-Ord hotspots
description: Where are the statistically significant hot and cold spots?
---

The Getis-Ord Gi* statistic — a Local Indicator of Spatial Association (LISA). For
each cell it asks whether the *neighbourhood* sum is significantly higher or lower
than a random shuffle would give.

## What it means

A z-score per cell:

- large **positive** → a **hot spot** (a cluster of high values)
- large **negative** → a **cold spot**
- near zero → unremarkable

`|z| > 1.96` is significant at p < 0.05. Directly readable as a map.

## GPU approach

A *composition* and a [render-then-compute](/concepts/render-vs-compute/) example:

1. the windowed neighbourhood sum is a
   [separable box convolution](/primitives/separable-convolution/) (GPU);
2. the global mean/variance and the closed-form standardisation are cheap CPU.

`pointHotspotsGpu` chains it onto the [KDE splat](/primitives/kde-splat/), so a raw
point cloud becomes a hotspot map in one call.

## Usage

```ts
import { getisOrdGpu, pointHotspotsGpu } from 'intraspatial/gpu/spatial/getisOrd';

// on an existing grid
const { z } = await getisOrdGpu(grid, w, h, { radius: 2 });

// or straight from points:  points → splat → window → z
const hot = await pointHotspotsGpu(xs, ys, { width: 64, height: 64, sigma: 3, radius: 2 });
```

A box window gives the classic binary-weights Gi*; a Gaussian window gives a softer,
"fuzzier" LISA — the [windowing](/concepts/windowing/) idea again.

**Status:** ✅ implemented and validated (CPU golden + hot/cold + end-to-end) ·
`src/gpu/spatial/getisOrd.ts`
