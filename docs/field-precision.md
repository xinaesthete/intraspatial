# Field precision: what to store things in

Design note. The first slice (f16 density fields) is built; the rest is direction.

`Dtype` is `"f32" | "i32" | "u32"` and every buffer is 4 bytes a sample. That is more than most
fields need and, in one place, less than they need.

## 1. Two axes, not one

Conflating these is how precision models go wrong.

| | What it decides | Set by |
| --- | --- | --- |
| **Storage** | bits a sample occupies in a buffer or texture | the producer, per value |
| **Compute** | what the shader does arithmetic in | the kernel |

They are independent, and the useful combination is usually **compute wide, store narrow**. The
density field below computes in f32 and stores f16; it would be wrong the other way round.

WGSL has no `u8`/`u16` scalars, so narrow *buffer* storage means packed `u32` words plus
`unpack4xU8` / `unpack2x16float` at the kernel boundary — the element a consumer sees is not the
word the buffer holds. Narrow *texture* storage needs no packing: `r16float` is a core renderable
format, which is why the first slice is a texture.

Native `f16` arithmetic needs the `shader-f16` device feature, which `getDevice()` does not
request (it asks only for `float32-blendable`). Nothing here needs it yet: f16 is storage only.

## 2. Landed: f16 density fields

`splatDensity` takes `precision: "f32" | "f16"` (default `f32`). At `f16` the field is leased as
`r16float` — **half the memory** — and the executor's bridge widens it back to f32, so no consumer,
test or readback can tell the difference.

**It accumulates in f32 regardless.** That is not caution, it is the measurement. The splat blends
additively into a render target, and f16 carries 10 mantissa bits, so once a texel's running total
exceeds ~1024× one splat's contribution, `total + c == total` and the contribution vanishes.
Measured on a 128² field (`pnpm bench:density-precision`), blending straight into `r16float`:

| points | peak | max rel err | mean rel err | **mass kept** |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 4.75 | 0.67% | 0.17% | 99.8% |
| 400 | 6.00 | 0.80% | 0.32% | 99.7% |
| 1 600 | 46.4 | 4.9% | 2.3% | 96.9% |
| 6 400 | 166 | 12.5% | 7.0% | 90.8% |
| 6 400 | 2 150 | 52% | 27% | 60.8% |
| 25 000 | 8 380 | 88% | 54% | **25.4%** |

The loss is systematic, not noise — the field quietly stops integrating to the right number, and
worst exactly where a density field is worth computing. Accumulating in f32 and narrowing once on
store costs a single rounding instead:

| points | peak | max rel err | mean rel err | mass kept |
| ---: | ---: | ---: | ---: | ---: |
| 200 … 25 000 | 4.75 … 8 380 | 0.096% | 0.035% | 100.0% |

Flat across the whole range, because there is one rounding rather than a compounding one.
~2 ulp of f16, whose relative resolution is 2⁻¹¹ ≈ 0.049%.

Cost: one extra pass, and one full-precision accumulator texture grown to the largest field and
reused. That is one field's worth of f32 however many f16 fields are alive — the saving is in the
fields a graph *retains* (a layer stack, a memo, a cache), not in the transient it renders through.

**Remaining limit:** f16 saturates at 65504. A field whose peak approaches that clips, and nothing
warns. Worth a check if a caller pushes density much past the ranges above.

### Tried on a real page: transcript co-location modes

`gramMatrixGpu` keeps one raster per gene channel in a single stacked buffer, and that buffer is
the binding this path hits the ceiling on first — the comment in the file already noted it crosses
`maxStorageBufferBindingSize` at 49 channels of 827². f16 halves it, so it doubles the channels or
the resolution that fit.

It takes `precision: "f32" | "f16"` (default f32), with the narrowing applied per channel on the
way into its slice — the same accumulate-wide/store-narrow shape, since the splat already renders
to an `r32float` target. Readers unpack behind a uniform flag (`unpack2x16float`, two texels per
word): the reduce shader, and the transcript page's `GramModesLayer`. `gramModes.ts` and
`gramTerrain.ts` have not been taught to unpack and throw if handed f16 rasters.

Measured on the Xenium store, 9 channels over a 10871×3627 µm window, 653×218 map:

| | buffer | modes | Gram |
| --- | ---: | --- | ---: |
| f32 | 5.7 MiB | 47% / 15% / 11% | ~46 ms |
| f16 | **2.9 MiB** | 47% / 15% / 11%, same memberships | ~49 ms |

The statistic is indistinguishable; the memory halves. Time is **slightly worse**, not better —
the narrowing pass runs per channel, and at this size the reduce is too small for the halved read
traffic to pay it back. Whether it turns a profit at the raster sizes that actually hurt is
untested.

One trap worth recording: the splat target needed `TEXTURE_BINDING` added before the narrowing
pass could sample it. Without it the bind group is invalid, which invalidates the command buffer,
and every raster stays zero — the page drew a blank map and reported every mode as 0% of the
variation. Exactly the silent-failure shape `checkBindingSize` exists for, in a usage flag rather
than a size.

## 3. 3D volumes: the easy case, not the hard one

Memory pressure is worse in 3D — a 256³ field is 64 MiB at f32 and 32 at f16; 512³ is 512 against
256 — and the 2D mechanism does not transfer, because there is no additive blending into a volume.
That sounds like the harder problem. It is the easier one.

**The accumulator stops being the destination.** `splatVolume` (designed in
[`gpu-spatial-index-3d.md`](gpu-spatial-index-3d.md) §4) **gathers**: one thread per output voxel,
walking the index's 27-cell stencil and summing contributions in a register. The register is f32,
the voxel is written once, and the swamping that forces the 2D split cannot happen — there is no
running total in the texture to swamp. So 3D needs **no f32 scratch volume and no extra pass**: it
writes f16 directly, for the same single rounding (~0.1%) the 2D path pays after its extra work.

The precedent is already in the repo. Volume bricks from the datasource are **already `r16float`**
(`tileTextureFormat` in `src/gpu/tiles/assemble.ts`, `rgba16float` multi-lane), chosen for exactly
this memory reason, and they sample with linear filtering.

Two mechanics to know before building it:

- **Writing f16 from a compute kernel is not core.** `r16float` with `STORAGE_BINDING` needs the
  `texture-formats-tier2` feature. The core-only route is to write packed halves into a storage
  buffer (`pack2x16float`, two voxels per `u32`) and `copyBufferToTexture` into the `r16float` 3D
  texture. One copy, and no f32 volume ever materialises.
- **f16 filtering is core; f32 filtering is not.** Trilinear sampling of `r16float` works out of
  the box, while `r32float` needs `float32-filterable`, which `getDevice()` does not request. So
  an f32 computed volume would not even be linearly samplable here without adding a feature — f16
  is the path of least resistance as well as least memory.

Caveats are the f16 range, as in 2D: it saturates at 65504, and its smallest normal is ~6·10⁻⁵, so
a very long tail flushes to zero — harmless for display, worth knowing before taking a log of it.
A gathered volume spreads the same mass over far more voxels than a 2D field, so peaks are lower
for the same data, but it is still data-dependent and unchecked.

## 4. Not built

**Narrow buffers (u8 / u16).** The win is index-shaped: `pointIds` in a bucket grid is u32, and
u16 halves it whenever n < 65536 — which is most clouds. The blocker is not the `Dtype` union
(four places branch on it) but that *4 bytes a sample* is baked into bind-group sizing — 11 sites
in the graph layer, more in every kernel. `lease(byteLength)` is already byte-based, so the pool
needs nothing. A bundle (ADR-0023) is the natural carrier for the result, since `cellOffsets` u32
beside `pointIds` u16 is a mixed-precision value that sibling ports could not express.

**Wider than f32.** Probably never needed. WebGPU has no f64, so it would mean double-single
(a pair of f32s), and the cheaper lever is already here: shift coordinates to a nearby origin via
ADR-0018's `placement` and stay in f32. deck.gl reached the same conclusion — its `fp64` module is
double-single and now "a niche technology", superseded by a 32-bit projection relative to a local
origin.

## References

- [deck.gl — 64-bit precision](https://deck.gl/docs/developer-guide/fp64)
- [WebGPU texture format tiers — `texture-formats-tier2` adds `r16float` storage](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedFeatures)
- [Chrome — filterable 32-bit float textures (`float32-filterable`)](https://developer.chrome.com/blog/new-in-webgpu-119)
