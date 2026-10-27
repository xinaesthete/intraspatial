# @intraspatial/deck

## 0.1.1

### Patch Changes

- 5069626: `DensityLayer`: replacing or removing a layer while its splat is in flight no longer destroys buffers that splat still uses ("[Buffer] used in submit while destroyed"). Buffers a splat holds are retired at finalize and freed when it settles, and a finalized layer submits no further work. Its GPU buffers are labelled `<layer id>:field`, `:max`, `:points` and `:ramp`, so a device error names the layer.
- Updated dependencies [d78900d]
- Updated dependencies [5069626]
  - intraspatial@0.1.1
