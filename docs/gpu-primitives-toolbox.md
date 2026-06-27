# Plan — a GPU primitives toolbox (wavelet & image analysis on TypeGPU)

Status: **plan** (2026-06-27)

The inverse and forward DWT (5/3 + 9/7) are now both on the GPU and validated, so
we have a real wavelet *transform pair*, not just a decode step. This doc plans
turning that into a small, composable **GPU primitives toolbox** — reusable for
analysis beyond the codec (denoising, multiresolution, feature work, fusion) and
for a future GPU encode path — and how to exercise it.

## Where we are (the foundation)

Already built and reusable:

- **Runtime**: headless Dawn device (`src/gpu/device.ts`); TypeGPU typed resources
  via tgpu-gen + `tgpu.resolveWithContext` (`*.gen.ts` + WGSL templates).
- **Transform pair**: `idwt53`/`fdwt53` (5/3, i32, bit-exact) and
  `idwt97`/`fdwt97` (9/7, f32, ≤1e-4). All share one kernel pattern:
  **one workgroup per line, lift in workgroup shared memory, vertical/horizontal
  passes, per-level submits**.
- **Plumbing patterns** worth promoting to shared infrastructure:
  - buffer **pooling** (reuse/grow, no per-call churn);
  - **keep-on-GPU** (`readback: false`) and on-GPU pixel/level-shift conversions
    (`idwt53` shift fold, `idwt97` `pixels` option);
  - a flat **descriptor + packed-coeffs** interface for multi-level data.

## The key idea: compose on the GPU

Today each `*Gpu` function does upload → compute → readback and owns its pool.
To use the toolbox for analysis you want to **chain** primitives without
round-tripping to the CPU each step — e.g.

```
upload → fdwt → threshold(detail) → idwt → download   (GPU wavelet denoise)
```

So the central refactor is a small handle type and a shared runner:

- **`GpuField`** — a pooled GPU buffer + shape (`width, height`, dtype i32/f32,
  optional DWT `descriptor`). Primitives take and return `GpuField`s; data stays
  on the GPU between them.
- **`upload(data) → GpuField`** and **`download(field) → TypedArray`** at the
  boundaries only (download is the Dawn-on-Node-fragile step; chain first).
- A shared **kernel runner** that owns the pipeline cache, bind-group creation,
  and dispatch, so a new primitive is "a WGSL body + a layout + dispatch sizes",
  not 150 lines of boilerplate. The shared-memory line kernel becomes a template.

This turns the current 4 DWT modules into instances of a pattern and makes new
primitives cheap.

## Primitive catalogue

Grouped; `✓` = done, `→` = planned. Keep each primitive: pooled, keep-on-GPU,
CPU-golden-validated.

**Transforms**
- ✓ inverse/forward 5/3 (i32), inverse/forward 9/7 (f32)
- → generic ATK lifting (other JPEG2000 wavelets) from the same kernel
- → multi-component color transforms (RCT lossless / ICT lossy) — codec + viz

**Coefficient-domain (exercise the pair)**
- → threshold (hard/soft) on detail bands → denoising
- → per-band scale / gain, band zeroing/selection
- → band statistics (energy/variance per subband) via reduction
- → coefficient masking / fusion (combine two images' subbands)

**Pointwise / map**
- ✓ level shift, ✓ float→pixel convert (round/clamp/shift)
- → arithmetic (add/sub/mul/blend of two fields), LUT/palette apply, clamp/abs

**Reductions**
- → min / max / sum / mean, histogram → normalization, auto-contrast, stats

**Resampling / pyramids**
- → extract LL at level k (free from the forward DWT) → multiresolution overviews
- → nearest / bilinear resize

**Spatial (shares the line-kernel pattern)**
- → separable convolution (blur, sharpen, gradient/Sobel) — same shared-mem line
  approach as the DWT lift

## First exercises (prove the pair as a tool)

1. **GPU wavelet denoise**: `fdwt97 → soft-threshold detail → idwt97`, all on
   GPU; validate vs a CPU reference and show it reduces noise on a noisy fixture.
2. **Multiresolution overview**: forward DWT, read back the LL at level k as a
   downscaled image (the "free" pyramid) — useful for SpatialData viz.
3. **Band-energy map**: forward DWT + per-band reduction → a compact feature
   descriptor; a stepping stone to edge/texture analysis.

Each doubles as a composition test of the `GpuField` chaining model.

## Conventions to lock in

- **Validation**: every primitive gets a CPU golden (Rust or JS) + a GPU test
  asserting bit-exact (integer) or ≤tolerance (float), plus a round-trip where it
  applies. GPU tests are `*.gpu.test.ts`, opt-in for benches (`BENCH=1`).
- **Determinism**: integer primitives use i32 + arithmetic `>>` (bit-exact);
  float primitives bound max abs error (~1e-4) and mean error.
- **Keep-on-GPU first**: primitives never read back internally; only `download`
  does. This is also the only Dawn-on-Node-fragile op (see constraints).

## Constraints to design around (learned the hard way)

- **Dawn-on-Node readback ceiling**: `mapAsync` readback crashes the worker
  beyond ~512²; *compute* is fine to 2048². Chain on-GPU and download once;
  benchmark/validate large sizes in a **browser**.
- **Per-level / cross-pass hazards**: a buffer bound readonly in one pass and
  mutable in another within a single command encoder did **not** get the
  write→read barrier we expected (forward DWT ll0 was wrong until we used
  per-level submits). Pattern: submit per dependent stage, or double-buffer.
- **No 64-bit ints in core WGSL**: lossless >~30-bit (e.g. uint32) can't use an
  i32 GPU path; that work stays CPU (or 2×u32 emulation). Float (9/7-style)
  sidesteps it.
- **Shared-memory line cap** (`MAXLINE = 2048`): tile longer lines for very large
  images.

## Roadmap

1. **Extract the shared kernel runner + `GpuField`** from the 4 DWT modules
   (refactor, no behaviour change; tests stay green). This is the unlock.
2. **First exercise — GPU wavelet denoise** (`fdwt → threshold → idwt`) to prove
   composition end-to-end.
3. **Reductions + pointwise** (min/max/sum/histogram, arithmetic) — small, high
   leverage, enable normalization/stats.
4. **Color transforms (RCT/ICT)** — needed for multi-component anyway; reuse in
   viz.
5. **Spatial convolution** + **resize/overview** primitives.
6. Revisit large-size GPU validation in a **browser** harness once the toolbox is
   chaining on-GPU (where the readback ceiling stops mattering).

See also [`performance-report.md`](performance-report.md) and
[`dwt-gpu-and-high-bit-depth.md`](dwt-gpu-and-high-bit-depth.md).
