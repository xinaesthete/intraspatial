---
title: Composable & interpretable
description: The two design commitments behind the toolbox.
---

Two properties shape every primitive here.

## Composable

A primitive takes plain typed arrays in and returns plain typed arrays out. There
is no framework object to construct, no graph to register with. That means you can
**chain** them, and the chain reads like the maths:

```ts
// a raw point cloud becomes an interpretable hotspot map
const density = await splatDensityGpu(xs, ys, { width, height, sigma });
const hot     = await getisOrdGpu(density.data, width, height, { radius: 2 });
// or, the same thing in one call:
const hot2    = await pointHotspotsGpu(xs, ys, { width, height, sigma, radius: 2 });
```

Composition currently happens at the API level (each call uploads, computes, reads
back). A future **keep-on-GPU** mode will let a whole chain run without round-trips —
the handle type is planned, the primitives are already shaped for it.

The deepest form of composition is that the two fronts share operators. A
**window** is a convolution kernel on the image side and a density/membership kernel
on the cell side; a quadrat is just `window(box, no-overlap)`. Build the window once,
reuse it everywhere.

## Interpretable

Each output has a one-sentence meaning:

- **Nearest-neighbour distance** — how isolated each point is.
- **ANNI** — is the pattern clustered, random, or dispersed (with a significance)?
- **KDE density** — a smooth surface of "how much point-stuff is here".
- **Getis-Ord** — where are the statistically significant hot and cold spots?
- **Fuzzy adjacency** — how strongly is each pair connected (0–1), not just yes/no?

This is deliberate. The point of these primitives is *mechanistic* insight into
spatial structure — the opposite of a black-box embedding. Each result links back to
a concrete spatial claim you can check.

## Validated

Every primitive ships with a CPU golden and a `*.gpu.test.ts` asserting agreement —
bit-exact for integer work, bounded error (typically ≤1e-3) for float. Correctness is
size-independent, so small tests suffice; large-scale runs are for a browser harness.
