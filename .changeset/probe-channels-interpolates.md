---
"intraspatial": patch
---

`probeChannels` (`intraspatial/gpu/spatial`) reads between pixels and from half-precision maps:

- Fractional `col, row` are interpolated bilinearly, pixel centres at whole numbers, instead of rounded to the nearest pixel.
- f16 rasters are read; they were previously refused.
- A fourth argument, `rasters`, reads a copy of the result's rasters in place of `res.resident.buffer`, which the next compute overwrites.
- `standardise` accepts any `ArrayLike<number>` for its mean and sd.
