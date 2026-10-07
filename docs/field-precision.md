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

## 3. Not built

**Narrow buffers (u8 / u16).** The win is index-shaped: `pointIds` in a bucket grid is u32, and
u16 halves it whenever n < 65536 — which is most clouds. The blocker is not the `Dtype` union
(four places branch on it) but that *4 bytes a sample* is baked into bind-group sizing — 11 sites
in the graph layer, more in every kernel. `lease(byteLength)` is already byte-based, so the pool
needs nothing. A bundle (ADR-0023) is the natural carrier for the result, since `cellOffsets` u32
beside `pointIds` u16 is a mixed-precision value that sibling ports could not express.

**Wider than f32.** WebGPU has no f64 at all, so "wider" means a pair of f32s — double-single —
and belongs with ADR-0004's element algebra rather than with storage.

Before building it, note what deck.gl found. Its `fp64` module is exactly double-single, and it
is now "a niche technology": deck 6.1's improved **32-bit** projection reaches sub-centimetre
precision without it, because it transforms coordinates relative to an origin near the data
instead of carrying absolute ones. fp64 costs them ~10× shader slowdown, double the attribute
memory, slower compiles and driver-compatibility trouble; they keep it only for extreme dynamic
range, such as a whole city at sub-centimetre accuracy at once.

The same lever is already here: ADR-0018's `placement` / `worldFromArray` is origin-shifting
machinery. **Try a placement-relative origin before emulated double precision.** Slide-scale µm
coordinates are a large-offset-small-extent problem, which is the case origin-shifting solves
outright, and f32 relative to a local origin beats double-single at a tenth of the cost.

## References

- [deck.gl — 64-bit precision](https://deck.gl/docs/developer-guide/fp64)
- [luma.gl — fp64 shader module](https://luma.gl/docs/api-reference/shadertools/shader-modules/fp64-arithmetic)
- [deck.gl — project64](https://deck.gl/docs/api-reference/core/project64)
