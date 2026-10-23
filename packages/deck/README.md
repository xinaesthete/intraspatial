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

It adopts deck's `GPUDevice` into IntraSpatial on first use. IntraSpatial keeps **one device per
page**: a second Deck brings a second device, and the layer will throw. Several deck views on a page
must share one luma device; that is not yet verified with luma 9.4.

### From a screen-pixel bandwidth

deck's `HeatmapLayer` (and MDV's contour layers built on it) take a radius in screen pixels. For an
orthographic view at `zoom`, one data unit is `2^zoom` pixels, so `σ = radiusPixels / 2^zoom`. Do
that conversion once in the host, at the zoom the user set the bandwidth, not on every zoom.

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
