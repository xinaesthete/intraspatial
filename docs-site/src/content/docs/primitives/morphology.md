---
title: Morphology (erode / dilate / open / close)
description: Separable local min/max over a square — the thin-vs-blob discriminator, and the bare-earth estimate.
---

Erode (local **min**) or dilate (local **max**) a grid over a (2r+1)² square structuring
element; **opening** is erode→dilate, **closing** is dilate→erode. Grey-scale by definition;
a 0/1 mask is the binary special case.

## What it means

- **Opening removes anything thinner than the element and keeps blobs.** The motivating
  case is LIDAR: a roof is opaque, so a building's interior has first-minus-last return
  ≈ 0 — but its one-pixel *perimeter* (first return on the roof edge, last return on the
  ground beside it) reads as several metres of canopy. It is thin and canopy is not, so an
  r = 1 opening of the canopy mask separates them. Hedgerows survive: at 1 m they are 2–4
  pixels of crown, not a line.
- **A large-radius opening of a surface is a ground estimate.** Opening the last-return
  surface with r ≈ 25 takes the local minimum and then re-expands it, flattening trees and
  buildings while tracking terrain — the classic morphological bare-earth filter.
- Closing is the dual: it fills gaps narrower than the element.

## GPU approach

A square element is **separable exactly** — the min over a square is the min of per-row
minima — so this is the same two-pass horizontal-then-vertical `"use gpu"` gather as
[separable convolution](/primitives/separable-convolution/), with `min`/`max` in place of
the weighted sum and clamp-to-edge boundaries. Because there is no arithmetic the GPU result
is **bit-exact** against a direct 2-D window (that is what the test asserts). Opening and
closing run two rounds through GPU-resident scratch; as a graph node the op is Tier-2
resident (no host round-trip between it and its neighbours).

## Usage

```ts
import { morphologyGpu, openingGpu } from 'intraspatial/gpu/spatial/morphology';

const canopy = await openingGpu(mask, w, h, 1);                 // binary opening, r = 1
const ground = await morphologyGpu(lastReturn, w, h, 25, 'open'); // bare-earth estimate
```

As a graph node: `g.op1("morphology", { grid }, { op: "open", radius: 1 })`.

## Composes with

- [Separable convolution](/primitives/separable-convolution/) — the linear sibling; a
  box blur after an opening gives the smoothed canopy-cover fraction.
- `threshold` (hard or soft) before it — the canopy mask. An untested idea from the terrain
  work: a *soft* threshold feeding a grey-scale opening may behave better at woodland edges
  than a frozen hard mask.

**Status:** ✅ implemented, bit-exact against the square-window oracle and
`psychogeo:codec_eval/foliage.py::opening` semantics · `src/gpu/spatial/morphology.ts`,
graph op `src/gpu/graph/ops/morphology.ts`. Not yet nodata-aware (terrain gap 3).
