---
title: Render vs compute
description: Why density splatting is a render job, and how that translates to viewer stacks.
---

The toolbox uses two GPU modalities, chosen by what a primitive naturally produces.

## The decision axis

- **Compute** — for irregular gather / scan / sort / reduce and exact integer
  counts: neighbour search, pair-count histograms, reductions, Monte-Carlo nulls.
- **Render** — for additive splat, resampling, and anything whose natural output is
  a screen-space field: KDE density, hotspot surfaces, topographical correlation.

A few methods are **render-then-compute** (splat a field, then run a windowed
statistic over it — exactly how [hotspots](/primitives/getis-ord-hotspots/) work).

## Splat by blending, not atomics

The obvious way to accumulate a density — "scatter each point's weight into a grid
with `atomicAdd`" — runs into a wall: core WGSL has no `atomic<f32>`. But GPUs
already solved additive accumulation in fixed-function hardware. Draw each point as
a small quad with a Gaussian footprint, and **additively blend** (`src = ONE,
dst = ONE, op = ADD`) into a float render target. The blend unit accumulates for
you — no atomics, no contention. See the [KDE splat](/primitives/kde-splat/).

## Why this translates

Viewer stacks — deck.gl, MDV, SpatialData.js — are render/layer based. Producing
analysis as **renderable float layers** means:

- each label's density is its own layer;
- combining layers (e.g. a topographical correlation map) is a blend or a short
  pointwise pass;
- the analysis output *is* a visual layer — no read-back round-trip.

The splat primitive is deliberately written in **raw WebGPU** (no deck.gl / luma.gl
/ MDV dependency) so the WGSL and the layer-composite shape port directly into those
contexts later, rather than locking the toolbox to one of them.
