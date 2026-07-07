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

This eager style is the simplest form of composition, and it has a cost worth naming:
**each `await` materialises to the CPU** (upload, compute, read back), so a chain
round-trips the data every step. That's fine for a one-shot analysis; it is exactly
what the op-graph exists to avoid for longer chains.

### Build lazily, evaluate once

The direction — already the model for the in-GPU op-graph and for the procedural
geometry-ops — separates *describing* a computation from *running* it. You build
a chain **synchronously and purely** (no `await`, no promises — building a description
does no I/O), and cross into async **exactly once**, at the point you demand a result:

```ts
const g = horn().radius(ramp(50)).bend(ramp(deg(30))).twist(ramp(deg(360)));  // pure, sync — nothing has run
const mesh = await g.toMesh(device);                                          // the single evaluation boundary
```

A built chain is a first-class value: inspectable, serialisable, and breedable —
none of which an in-flight promise allows. When you pull it, you choose the **form**:

- **Lazy** — the unevaluated handle itself. No work done yet; hand it to the next op.
- **Concrete GPU buffer** — pull to a resident WebGPU / TypeGPU buffer that *stays on
  device*, feeding rendering or further ops with no CPU round-trip.
- **Plain typed array** — materialise to the CPU (e.g. positions/normals/indices) with
  an explicit shape/element/basis schema — the portable, interoperable form.

**The render path is designed to stay entirely on the GPU.** Drawing uses the on-device
forms — a resident buffer, or the lazy handle lowered straight into a render op — so a whole
`generate → transform → render` chain runs GPU-resident with no round-trip through a CPU
mesh. The plain typed array is a portability / interop escape hatch (FAIR export, deck.gl
/ SpatialData.js handoff), **not** a step on the render path. (Today the runtime is Tier-1 —
ops round-trip on interior edges; the resident-buffer path and render op are the next build,
see [Operation graphs](/concepts/operation-graphs/).)

The eager `…Gpu()` calls above are the special case of "pull to a plain typed array at
every step". Same primitives; the lazy build just lets you defer *which* form you want,
and skip the round-trips in between.

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
