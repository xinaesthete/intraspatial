# @intraspatial/deck

deck.gl layers over [IntraSpatial](https://github.com/xinaesthete/intraspatial#readme)'s GPU compute. The field a layer draws is
computed on the device deck renders with and never read back. **WebGPU only.**

> **Experimental, 0.x.** Layer props may change in any `0.x` minor release; pin with `~` and see
> [`CHANGELOG.md`](https://github.com/xinaesthete/intraspatial/blob/main/packages/deck/CHANGELOG.md).

## DensityLayer

A Gaussian kernel density estimate of a point cloud, drawn as a region (density ≥ `fillLevel`) with
a fill and an outline, and iso-lines at regular intervals. Fill, outline and iso-lines each have their
own opacity (`fillOpacity`, `edgeOpacity`, `contourOpacity`), so a region can be outline-only.
Iso-lines fade out by themselves where they would crowd closer than a few line widths apart.

```ts
new DensityLayer({
  id: "density",
  x, y,                 // ArrayLike<number>, the layer's own units
  weights,              // optional; 0 leaves a point out (filters, categories)
  bandwidth: 25,        // σ, in the same units as x and y
  scale: "relative",    // levels as fractions of the peak; "absolute": weight per unit area
  fillLevel: 0.1,
  contourInterval: 0.2,
  colorRange: [[255, 255, 204], [65, 182, 196], [37, 52, 148]],
});
```

**Units are the data's.** Density is `Σ wᵢ · exp(−|p−pᵢ|²/2σ²) / 2πσ²`: weight per unit area of the
positions' own units, before any `modelMatrix`. The same σ draws the same shapes at every zoom; zoom
changes only how finely the field is sampled (`pixelsPerCell` zoomed out, `cellsPerSigma` zoomed in).
Cartesian views only.

### Deck setup

The layer needs deck on WebGPU. Ask for `float32-blendable` where the browser has it: with it the
splat accumulates by additive blending into r32float, in hardware. Without it (Firefox's WebGPU, at
the time of writing) the layer falls back to an exact fixed-point compute splat — the same field,
slower where points are dense:

```ts
import { webgpuAdapter } from "@luma.gl/webgpu";
const deviceProps = { type: "webgpu", adapters: [webgpuAdapter], optionalFeatures: ["float32-blendable"] };
```

The layer passes deck's own `GPUDevice` to the kernels it calls, which keep their state per device.
It does not adopt the device into IntraSpatial, so any number of Decks per page work, and so does a
canvas remounted with a new device (deck 9.4 does not destroy the old one).

### From a screen-pixel bandwidth

deck's `HeatmapLayer` (and MDV's contour layers built on it) take a radius in screen pixels. Its
kernel is `exp(−(r/R)²/0.0556)` with R = `radiusPixels`, which is a Gaussian with σ = R/6, peaking
at 6/√π. In an orthographic view at `zoom`, with a `modelMatrix` that scales layer units by `s`:

    σ = radiusPixels / (6 · 2^zoom · s)        (layer units)
    heatmap weight = density · 12√π · σ²       (density as this layer reports it with scale: "absolute")

so a heatmap threshold or contour spacing converts exactly to `scale: "absolute"` levels. Do the
conversion once, at the zoom the user set the bandwidth, not on every zoom.

### Cost

Points are uploaded once per change of `x`, `y` or `weights`. Each re-splat draws every point, so a
weight-0 point costs the same as any other. Filtering by weights is simple but not free at tens of
millions of rows. The grid is reused while the view pans within a margin and re-made about every
half-octave of zoom.

## TcmLayer

A topographical correlation map (TCM) between two cell types: Γ_AB(x) is positive where A cells sit
among more B cells than chance (clustered with B), negative where they sit among fewer (excluded
from B). It is drawn as a diverging fill, transparent near 0 by default, with iso-lines on both sides
of 0.

Both types are **weightings of one point set**: a table's x/y, with a weight per point for each type
(0/1 for membership). A layer per pair shares the coordinates and costs two weight arrays.

```ts
new TcmLayer({
  id: "tcm-cd8-tumour",
  x, y,                 // ArrayLike<number>, the layer's own units
  weightsA: isCd8,      // 1 for type A, 0 otherwise; Γ is about where these sit
  weightsB: isTumour,   // 1 for type B
  radius: 100,          // r: how near a B cell must be to count, in the data's units
  sigma: 50,            // Γ's Gaussian σ, in the data's units
  alpha: 5,             // the extreme threshold (the paper's default)
  region: [0, 0, w, h], // where B's expected density is averaged; default: the points' bounds
});
```

**Units are the data's**, as for `DensityLayer`. The statistic is the two-pass render in
`intraspatial/gpu/spatial/tcmRender` (see `docs/cell-stats.md`), so the same caveat applies: an A
cell's mark is sampled bilinearly from a raster (`markCellsPerRadius`, default 8 cells per r), and
that sampling converges fastest for smooth kernels (`kernel`: `TOPHAT`, the paper's disk, is the
default). Γ is rendered only for the view (plus a margin), with the marks only where A cells reach it.
Domains (ρ_B·|disk ∩ W| in place of ρ_B·πr²) are not supported yet.

Deck must be on WebGPU **with `float32-blendable`**: both passes blend into r32float, and unlike
`DensityLayer` there is no compute fallback, so on a device without it (Firefox's WebGPU, at the time
of writing) the layer reports an error and draws nothing.

Each layer renders on deck's own device with a `TcmRenderer` of its own, so any number of TCM layers
coexist. A render is synchronous — it writes, encodes and submits before the update returns — so
nothing is left waiting to be submitted when deck finalizes the layer, and it frees its buffers at
once.

## Development

In this workspace, apps resolve `@intraspatial/deck` and `intraspatial` to their `src/` through the
`@intraspatial/source` export condition (see `pnpm-workspace.yaml`), so nothing needs building. The
demo is `playground/density.html`. `pnpm build` emits `dist/` against intraspatial's published
types: run the root's `pnpm build:lib` first.
