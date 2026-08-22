---
title: Decimate (block reduce)
description: Collapse every f×f block of a grid to one cell — mean, min or max — and carry the world placement with it.
---

Reduce a `w×h` grid by an integer `factor`: every `f×f` block of input cells becomes **one**
output cell holding the block's **mean**, **min** or **max**. The output is
`ceil(w/f) × ceil(h/f)`. This is the first engine op that changes grid size, so it is also the
first that must *derive* a placement rather than pass one through.

## What it means

- **Block mean is coverage fraction.** Block-averaging a 0/1 canopy mask by `f` gives the
  canopy-cover fraction per coarse cell — the psychogeo `block_mean` step that turns a 1 m
  foliage mask into a coarser percentage layer.
- **Block min/max are coarse surfaces.** Min of a last-return surface is a cheap ground floor;
  max of a first-return surface is a coarse canopy top. Both are the non-overlapping cousins
  of the [morphology](/primitives/morphology/) erode/dilate.
- **Edge blocks reduce over the cells that exist.** When `w` or `h` is not a multiple of `f`
  the right/bottom blocks are partial; their mean is the mean of the present cells, not of a
  zero-padded full block. Factor 1 is accepted by the kernel as the identity; the graph node
  restricts `factor` to 2..64.

## Placement

Array space here is **corner-indexed**: cell `(i, j)` occupies `[i, i+1) × [j, j+1)`, so a
cell's world centre is `origin + (i+½)·axes[0] + (j+½)·axes[1]`. Output cell `(I, J)` covers
array-space `[I·f, (I+1)·f)`, hence

```
worldFromArray_out = worldFromArray_in · scale(f, f, 1)
```

— origin unchanged, in-plane axes multiplied by `f`. The output cell's centre is exactly the
centroid of its block's input-cell centres (asserted in the test). A partial edge block keeps
the nominal full-block footprint.

## GPU approach

One `"use gpu"` thread per **output** cell loops over its block — no atomics, no workgroup
memory, fixed summation order — so min/max are bit-exact against the CPU golden and the mean
agrees to f32 rounding. Tier-2 resident as a graph node: the output buffer is leased at the
smaller size and the next op reads it in place.

## Usage

```ts
import { decimateGpu } from 'intraspatial/gpu/spatial/decimate';

const cover = await decimateGpu(canopyMask, w, h, 8, 'mean'); // ceil(w/8) × ceil(h/8)
```

As a graph node: `g.op1("decimate", { grid }, { factor: 8, mode: "mean" })` — the output
handle's `placement` is already the scaled one.

## Composes with

- [Morphology](/primitives/morphology/) before it — open the mask, then block-mean it:
  canopy-cover fraction per coarse cell.
- `threshold` after it — cover ≥ 0.5 → "dense woodland" at the coarse resolution.

**Status:** ✅ implemented · `src/gpu/spatial/decimate.ts`, graph op
`src/gpu/graph/ops/decimate.ts`. Not yet nodata-aware (terrain gap 3).
