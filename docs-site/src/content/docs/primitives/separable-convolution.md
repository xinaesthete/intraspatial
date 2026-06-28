---
title: Separable convolution
description: The windowing primitive — box (local sum/mean) or Gaussian (smoothing) on a grid.
---

Convolve a grid with a 1D kernel applied on each axis. An N×N window costs O(N) per
cell, not O(N²).

## What it means

The general **window** operator on a grid:

- a **box** kernel gives a local sum (or mean) — the neighbourhood total;
- a **Gaussian** kernel gives smoothing.

It is the grid consumer the [KDE splat](/primitives/kde-splat/) feeds into, and the
concrete realisation of the [windowing principle](/concepts/windowing/) on the image
side.

## GPU approach

Two passes (horizontal then vertical), each a `"use gpu"` gather kernel: every output
cell reads its taps directly with clamp-to-edge boundaries. Box and Gaussian both
match the CPU golden; a normalised Gaussian conserves total mass.

## Usage

```ts
import {
  convolveSeparableGpu, boxKernel, gaussianKernel,
} from 'tgpu-htj2k/gpu/spatial/convolveSeparable';

const smoothed = await convolveSeparableGpu(grid, w, h, gaussianKernel(2));
const localSum = await convolveSeparableGpu(grid, w, h, boxKernel(3)); // window radius 3
```

**Status:** ✅ implemented and validated (box + Gaussian, mass-conserving) ·
`src/gpu/spatial/convolveSeparable.ts`
