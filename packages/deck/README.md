# @intraspatial/deck

deck.gl layers over [IntraSpatial](../../README.md)'s GPU compute. The field a layer draws is
computed on the device deck renders with and never read back. **WebGPU only.**

## DensityLayer

A Gaussian kernel density estimate of a point cloud, drawn as a colour ramp, a filled region above a
level with an edge along it, and iso-lines at regular intervals.

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

The layer needs deck on WebGPU, with the `float32-blendable` feature (the splat blends additively
into r32float):

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

## Development

In this workspace, apps resolve `@intraspatial/deck` and `intraspatial` to their `src/` through the
`@intraspatial/source` export condition (see `pnpm-workspace.yaml`), so nothing needs building. The
demo is `playground/density.html`. `pnpm build` emits `dist/` against intraspatial's published
types: run the root's `pnpm build:lib` first.
