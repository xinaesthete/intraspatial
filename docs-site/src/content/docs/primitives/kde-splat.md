---
title: KDE density splat
description: Rasterise a point cloud into a smooth density field — the points-to-grid bridge.
---

Turn a weighted point cloud into a smooth Gaussian kernel-density surface on a grid.

## What it means

A continuous picture of "how much point-stuff is here" — the windowed, artefact-free
alternative to [counting points in quadrats](/concepts/windowing/). Once the cloud is
a grid, every image-front primitive (blur, hotspots, thresholding, wavelets) applies
unchanged. This is the **bridge** between the two fronts.

## GPU approach

The no-atomics [render path](/concepts/render-vs-compute/): each point is drawn as a
small instanced quad with a Gaussian footprint, **additively blended** into an
`r32float` render target. The blend hardware accumulates overlaps — no `atomic<f32>`
needed. Written in raw WebGPU so it ports to deck.gl / MDV / SpatialData.js as a
custom layer.

## Usage

```ts
import { splatDensityGpu } from 'intraspatial/gpu/spatial/splatDensity';

const field = await splatDensityGpu(xs, ys, {
  width: 256, height: 256, sigma: 8, // bandwidth in world units
  weights,                            // optional per-point weight
});
// field.data: Float32Array (row-major), field.bbox: world bounds
```

## Composes with

- [Separable convolution](/primitives/separable-convolution/) — further smooth or
  window the density.
- [Getis-Ord hotspots](/primitives/getis-ord-hotspots/) — find where the density is
  significantly elevated.

**Status:** ✅ implemented and validated vs a CPU KDE golden ·
`src/gpu/spatial/splatDensity.ts`
