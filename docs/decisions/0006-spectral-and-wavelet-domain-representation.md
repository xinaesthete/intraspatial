# ADR-0006 — Spectral & wavelet domain representation (a `basis` facet)

Status: **proposed / exploratory** (2026-06-30)

This ADR is **exploratory** — it captures a design direction and a first concrete step,
not a committed type-system change. It was raised while working autonomously, explicitly
*not* as a precise architectural decision to lock in. The `basis` facet below is a
**proposal to weigh later**; the wavelet ops described under "Built now" are additive and
reversible (no type-system change) and are the safe part already landed.

## Decision (direction)

A field's values can be expressed in different **bases**, and that basis is a first-class
property orthogonal to *where* the samples live:

- **spatial** — pixels / the sampled signal itself (today's implicit default)
- **spectral (Fourier)** — frequency coefficients (FFT of the spatial field)
- **wavelet** — multiresolution subband coefficients (the DWT/Mallat pyramid)

Proposed: a **`basis` facet** alongside the four facets ADR-0004/0005 establish
(`domain` / `element` / `axes` / `support`), absent ⇒ `spatial` so existing ops are
untouched — the same optional-defaulted-facet shape as `element`. **Transform ops** move a
field between bases (`fdwt`/`idwt`, and eventually `fft`/`ifft`); analysis/edit ops declare
which basis they require, checked at graph-build time like `inferElements`.

Unlike the other facets, `basis` is **not fully independent** — it carries couplings that
are the interesting part of the design:

- **Spectral ⟺ complex element.** The FFT of a real field is a *complex* field
  (Hermitian-symmetric). So the spectral basis *uses* the `complex` `ElementType` from
  [ADR-0004](./0004-field-type-model-and-volumetric-splat.md) — that work is the
  prerequisite, and it is already built. The convolution theorem makes this concrete:
  **spatial convolution = pointwise complex multiply in the Fourier basis**, i.e.
  [`mulFields`](../../src/gpu/graph/ops/fieldArithmetic.ts) on `complex` fields *is* the
  spectral convolution kernel; a diffusion step (the RD Laplacian) is an exact pointwise
  multiply by the spectral symbol. The complex element type and the spectral basis are two
  halves of one capability.
- **Wavelet ⟺ a structured subband layout.** A wavelet-basis field is not a flat grid of
  numbers but a *packed Mallat pyramid* (LL corner + HL/LH/HH detail per level). The "DWT
  `descriptor`" the [primitives toolbox](../gpu-primitives-toolbox.md) already envisioned
  (levels, kernel, per-subband offsets) is exactly this basis metadata. Coefficient-domain
  ops (wavelet shrinkage / band gain / band statistics) operate *in* the wavelet basis. The
  ["Draw in the DWT domain"](../../docs-site/src/content/docs/concepts/dwt-draw.mdx) demo
  is the lived version: *"the coefficient pyramid is not a read-only picture — it is an
  editable representation of the image."*
- **Basis is grid-only (for now).** Spectral and wavelet apply to `grid` domains; `points`
  need a non-uniform FFT / irregular lifting (hard, deferred). 3D extends naturally (3D DWT,
  3D FFT) but is more work.
- **Basis couples to dtype.** 5/3 is integer-reversible (lossless, wants an `i32` path);
  9/7 is float-irreversible. Thresholding semantics differ by normalization (a 5/3 lift is
  *not* orthonormal, so a flat coefficient threshold is not a flat spatial-noise threshold).

### Why it matters (payoffs)

- **Denoising / compression** — wavelet shrinkage (`fdwt → thresholdDetail → idwt`) is the
  project's headline "prove the pair as a tool" exercise; HTJ2K compression is the same move.
- **Spectral convolution & diffusion** — large/global convolutions and exact diffusion become
  pointwise complex multiplies once an FFT exists; the `complex` ops are already in hand.
- **Multiresolution** — the LL band is a free image pyramid (overviews for SpatialData/MDV).
- **Windowing connection** — STFT/Welch is *spectral windowing*; wavelets are *windowed
  multiresolution*. This is the spatial front's "windowing, not quadrats" principle
  ([discrete-cell notes], [`gpu-spatial-analysis-toolbox.md`](../gpu-spatial-analysis-toolbox.md))
  applied along the basis axis. And spectral/wavelet are intrinsically multiscale, so they sit
  naturally under the **scale-equivariance** design goal from ADR-0004.

### Built now (the safe, reversible first step — no type-system change)

The wavelet basis is implemented **without** committing the `basis` facet: `fdwt`/`idwt`
produce/consume an ordinary `grid` whose values are packed Mallat coefficients, with the
band layout derived from `(width, height, levels)`. The basis is *implicit in the convention*,
exactly as the docs demo treats it. This proves the direction and the composition payoff
without an architectural commitment.

- [`dwt.ts`](../../src/gpu/graph/dwt.ts) — a Float32 port of the project's reference DWT
  ([`docs-site/src/lib/dwt.ts`](../../docs-site/src/lib/dwt.ts)): separable 2D Mallat, 5/3
  (reversible) + 9/7 (irreversible), whole-sample-symmetric boundary, `dwtBands` layout
  descriptor. 5/3 round-trips exactly on integer data; 9/7 within f32 tolerance.
- [`waveletOps.ts`](../../src/gpu/graph/ops/waveletOps.ts) — `fdwt` / `idwt` (basis transforms)
  + `thresholdDetail` (wavelet shrinkage; soft = `sign(x)·max(|x|−t, 0)` on detail bands,
  LL untouched). CPU Tier-1.
- Registered **opt-in** via `registerWaveletOps()` (dynamic import), *not* eagerly — the same
  Dawn-on-Node fragility lesson as the element ops (ADR-0004 implementation notes): eager
  module-graph growth on the base registry tips unrelated render-fork GPU tests over.
- Tests: [`dwt.test.ts`](../../src/gpu/graph/dwt.test.ts) (round-trip + band geometry) and
  [`waveletOps.test.ts`](../../src/gpu/graph/waveletOps.test.ts) (graph round-trip + a
  `fdwt → thresholdDetail → idwt` denoise that measurably beats the noisy input). CPU, 8 tests.

## Context & provenance

Raised by the user while the session ran autonomously: *"consider also spectral/wavelet domain
representation"* — with the explicit caveat that they were *not in a space to make precise
architectural decisions*. Hence: exploratory ADR + a reversible first step, no facet committed.

Grounding in existing assets (mapped this session):

- **Wavelet transform already exists, both CPU and GPU.** GPU kernels: 5/3
  [`fdwt53.ts`](../../src/gpu/fdwt53.ts)/[`idwt53.ts`](../../src/gpu/idwt53.ts) (i32,
  **bit-exact** vs CPU) and 9/7 [`fdwt97.ts`](../../src/gpu/fdwt97.ts)/[`idwt97.ts`](../../src/gpu/idwt97.ts)
  (f32, ≤1e-4). CPU reference: [`docs-site/src/lib/dwt.ts`](../../docs-site/src/lib/dwt.ts).
  The [toolbox doc](../gpu-primitives-toolbox.md) already plans the `GpuField` + DWT
  `descriptor`, the coefficient-domain ops, and the `fdwt → threshold → idwt` denoise exercise.
- **No FFT exists anywhere** in `src/`, `rust/`, or `docs/` — so the spectral basis is, for
  now, design-only: it needs a GPU FFT kernel (Stockham / Cooley–Tukey, real-FFT
  Hermitian half-spectrum, bit-reversal) that has not been written.
- **The complex element type ([ADR-0004](./0004-field-type-model-and-volumetric-splat.md)) is
  the spectral value type** — built this session, which is what makes the spectral basis
  reachable later without re-deriving complex arithmetic.
- **Layout mismatch to reconcile:** the CPU port (and the docs demo) use an *in-place Mallat*
  packed layout; the GPU kernels use a *descriptor-packed* layout (per-level subband offsets
  `hl_off`/`lh_off`/`hh_off`). A GPU-backed wavelet op must reconcile these (or adopt the
  descriptor as the basis metadata). The current CPU ops use the in-place layout for
  demo-parity.

## Update (2026-06-30): the `basis` facet is now built (for wavelet)

The implicit-convention approach below was superseded almost immediately: a field now
carries an explicit **`basis` facet** (`spatial` | `wavelet{wavelet, levels}`) on
`GpuField`/`FieldValue`, threaded through the builder alongside `element` (new
`OpType.inferBasis`, builder stores `outBases`, executor stamps it onto runtime values).
This was prompted by the concrete contract problem the user raised: `thresholdDetail`/`idwt`
should not each re-declare `levels` (and `idwt` the kernel) as params that must *match* the
producing `fdwt` — `fdwt` now tags its output `wavelet{kernel, levels}` and the consumers
read it (`idwt` has no params; `thresholdDetail` keeps only `thresh`/`soft`). Build-time
`inferBasis` rejects e.g. `idwt` on a `spatial` field. The default is **pass-through** (first
input's basis), so editing coefficients with a generic op (`scaleField`, …) and then `idwt`
still works. The `basisOf`/`basisLabel` helpers + the playground preview ("grid 96×96 ·
wavelet 5/3·L3") make it visible. Fourier/`fft` remain unbuilt (no FFT kernel); the facet is
ready for them.

## Consequences

- **Built:** `dwt.ts` + `fdwt`/`idwt`/`thresholdDetail` ops + tests, registered opt-in via
  `registerWaveletOps()`. The playground composer must call it to surface these ops (same as
  `registerElementOps`). **Plus the `basis` facet itself** (see the update above).
- **Proposed / deferred (the real decisions, for when you are ready):**
  - The **Fourier basis** + a **GPU FFT kernel** — the facet now exists; `fft`/`ifft` would add
    a `fourier` basis variant and bridge to the `complex` element type.
  - A **GPU FFT kernel** (the gate for the entire spectral basis), then `fft`/`ifft` ops that
    bridge to the `complex` element type and turn `mulFields`-complex into spectral convolution.
  - A **GPU-backed wavelet op** wrapping the existing 5/3/9/7 kernels (reconciling the layout).
  - **Normalization conventions** to lock in (orthonormal vs lifting scale; what a coefficient
    threshold means), and the `i32` (5/3) vs `f32` (9/7) dtype interaction.
  - **`points` and 3D** bases (NUFFT / irregular lifting; 3D DWT/FFT).
- Revisit when an FFT kernel is on the table (spectral becomes real), when a second basis-aware
  op family appears (the implicit convention starts to strain and the explicit facet earns its
  keep), or when GPU-backing the wavelet ops forces the layout-reconciliation decision.
