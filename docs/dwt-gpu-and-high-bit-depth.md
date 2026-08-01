# Evaluation — GPU DWT payoff & high-bit-depth (uint32) support

Status: **findings** (2026-06-27)

Captures the inverse-DWT benchmark results and two forward-looking questions the
numbers raise: (a) is moving the DWT to the GPU worth it — here, or inside
OpenJPH; (b) is this codec useful for SpatialData **label** volumes, which want
`uint32`, and how hard is that to support.

All measurements are from `test/bench_idwt.test.ts` (CPU) and
`test/bench_idwt.gpu.test.ts` (GPU), single component, reversible 5/3, 5 levels,
high-detail pseudo-random content, on the dev machine (Node + Dawn). Numbers are
medians in ms; treat them as ratios, not absolutes.

> ⚠️ Build matters: measure against a **release** wasm build (`pnpm build:wasm`).
> An earlier draft of this doc quoted dev-build numbers (≈10× slower) and wrongly
> concluded our DWT was ~20× off OpenJPH. The release numbers below correct that.

> ⚠️ **Corrected 2026-08-01 — the readback ceiling was our own bug, not Dawn's.**
> This doc originally reported that "the `mapAsync` full readback crashes the
> worker beyond ~512²" and benchmarked with readback disabled for that reason.
> It was written 2026-06-27, the same day `src/gpu/device.ts` was scaffolded and
> a month before the **Dawn Instance-lifetime fix** (2026-07-29, commit
> `4e326b0`): the Instance was being GC'd out from under a live device, and
> *allocation* is what triggered collection — which a large staging buffer is.
> Retested, readback is clean to at least **4096² / 67 MB**. §1 and §2 below carry
> the corrected numbers. The conclusions did not change; they got stronger.

## 1. Benchmark findings

CPU decode breakdown vs OpenJPH (its fully optimised C/SIMD decoder), release
build (`test/bench_idwt.test.ts`):

| size  | OpenJPH full | our entropy¹ | our CPU DWT |
|-------|-------------:|-------------:|------------:|
| 64²   |        0.22  |        0.10  |       0.09  |
| 128²  |        0.63  |        0.26  |       0.20  |
| 256²  |        0.71  |        0.73  |       0.77  |
| 512²  |        2.80  |        2.85  |       3.23  |

¹ "entropy" = HT block decode + reassembly (`decode_dwt_input_53`).

CPU DWT vs GPU DWT **compute** (no readback, buffers pooled — the keep-on-GPU
case; `pnpm bench:gpu`). Two kernels: the original *naive* one-thread-per-line
with global scratch, and the *optimised* one-workgroup-per-line with workgroup
shared memory and intra-line parallelism (current):

| size  | CPU DWT | GPU naive | naive ratio | GPU **optimised** | **opt ratio** |
|-------|--------:|----------:|------------:|------------------:|--------------:|
| 128²  |    ~0.2 |      ~0.8 |       0.25× |              ~0.4 |         0.4×  |
| 256²  |    ~0.8 |      ~1.5 |        0.5× |              ~0.5 |         1.4×  |
| 512²  |    ~3.3 |      ~3.0 |        1.1× |              ~1.0 |         3.0×  |
| 1024² |   ~13.5 |   (crash) |           — |              ~1.2 |        11.7×  |

Reading the numbers:

- **DWT ≈ entropy in our pipeline** (entropy/DWT ≈ 0.9–1.3×), so the DWT is
  roughly half the work and a reasonable offload target.
- **Our CPU code is competitive, not terrible.** Whole pipeline ≈ 6 ms at 512²
  (entropy 2.85 + DWT 3.23) vs OpenJPH's *entire* decode 2.80 ms — about **2×**,
  good for a from-scratch implementation (OpenJPH streams lines with SIMD; ours
  copies `Band`s and gathers columns, so there is headroom).
- **The optimised kernel proves the GPU thesis.** Shared-memory lifting +
  intra-line parallelism dropped the crossover to **~256²** and gives **~3×** at
  512² and **~12×** at 1024². GPU compute barely scales with size (0.4 → 1.2 ms
  for 128² → 1024²) — it has become overhead/dispatch-bound, i.e. the actual DWT
  is nearly free; at 1024² it is ~1.2 ms vs ~13.5 ms on the CPU.
- ~~**Readback, not compute, is the Dawn-on-Node ceiling.**~~ **Wrong — see the
  correction above.** Readback works at every size tested; the crash was the
  Instance-lifetime bug. The measurement it was blocking is below.

### Readback, measured (2026-08-01, `pnpm bench:readback`)

Every row verified against `idwt53_cpu` element-by-element, so a fast number
cannot be a silently empty buffer. 2048² is new — the old ceiling stopped at 1024².

| size  | CPU DWT | GPU compute | GPU + readback | readback alone | keep-on-GPU | decode-to-CPU |
|-------|--------:|------------:|---------------:|---------------:|------------:|--------------:|
| 128²  |    0.21 |        0.47 |           0.74 |           0.27 |       0.44× |     **0.28×** |
| 256²  |    0.84 |        0.54 |           1.32 |           0.78 |       1.55× |     **0.64×** |
| 512²  |    3.38 |        0.86 |           3.64 |           2.78 |       3.93× |     **0.93×** |
| 1024² |   15.33 |        1.49 |          31.86 |          30.37 |      10.27× |     **0.48×** |
| 2048² |   72.72 |        4.67 |         136.10 |         131.42 |      15.57× |     **0.53×** |

**The readback costs 6–28× the compute it enables**, and the decode-to-CPU column
is below 1.00× at every size — i.e. running the DWT on the GPU *and bringing the
result back* is slower than never leaving the CPU, everywhere we measured. That is
the same conclusion the original doc reached, but from measurement rather than
from a crash: the keep-on-GPU premise is not a convenience, it is the whole margin.

> **Measurement trap, worth not re-learning.** Readback is timed in a plain
> process (`pnpm bench:readback`), *not* in the vitest benchmark. Inside the
> vitest fork, `mapAsync` completion is only observed on a coarse tick: every size
> from 128² to 1024² reports a flat ~125 ms — a fixed wait, not bandwidth — and
> the ordering against size inverts. The identical code as a standalone process
> scales cleanly. `test/bench_idwt.gpu.test.ts` therefore *verifies* readback but
> does not time it.

**Conclusion for our layer:** the GPU DWT is correct, an architectural fit, and —
with the shared-memory kernel — a clear win from ~256² up (≈12× at 1024²),
result kept on-GPU. Remaining headroom: (1) coalesce the vertical pass (it reads
columns strided; a transposed or tiled pass would help), (2) tile lines longer
than the shared-memory cap (currently 2048 samples/line), and (3) keep the result
on-GPU for display (no readback). That last point is the whole premise: the GPU
path pays off in a **GPU-resident viz pipeline** (zarr → decode → render), not in
decode-to-CPU.

## 2. Would a GPU DWT help OpenJPH?

Short answer: **not as a general decode speed-up, yes only for a GPU-display
pipeline** — which is exactly the niche our separate TS/TypeGPU layer fills.

- OpenJPH's DWT is already SIMD-vectorised (SSE/AVX2/AVX512/NEON/WASM-SIMD) and
  streams line-by-line, fused with the rest of decode — its whole decode is on
  par with our DWT stage alone, so its DWT is a fast bar to clear on CPU.
- A C library that decodes **to CPU memory** would have to add CPU→GPU→CPU
  round-trips to use a GPU DWT, and the readback costs more than the compute
  saves — measured, §1: **decode-to-CPU is 0.28–0.93× the plain CPU path at every
  size tested**, i.e. always a loss. So a decode-to-CPU library gains nothing
  here, and this no longer rests on the Dawn crash that originally suggested it.
- The bit-serial **HT entropy decode stays on the CPU** regardless (it is
  inherently sequential), so the GPU could only ever offload the back half.
- The real opportunity is a pipeline that **renders on the GPU**: decode HT on
  CPU, upload coefficients once, run DWT + dequant + level-shift on the GPU, and
  keep the image in a GPU texture for display — no readback. That is our layer,
  not OpenJPH's job. So "integrate GPU DWT into OpenJPH" is the wrong framing;
  "pair OpenJPH-style HT decode with a GPU DWT for viz" is what we are building.

## 3. High bit-depth and `uint32`

### What the formats allow
- **JPEG 2000 / HTJ2K**: a component's `Ssiz` encodes sign + (depth − 1), so the
  codestream can declare well beyond 16-bit; up-to-32-bit integer is the
  commonly-supported range, and the HT block coder is bit-plane based so it is
  not intrinsically limited to 32.
- **OpenJPH core**: supports `precision > 32` via 64-bit code-block buffers —
  `ojph_codeblock.cpp` picks `BUF32` (`precision <= 32`) vs `BUF64`, and the
  transform has `si64` reversible lifting (`gen_rev_vert_step64`,
  `gen_rev_horz_syn64`). The 64-bit path is **reversible-only** (irreversible
  uses 32-bit float). So lossless >32-bit is supported in principle.
- **openjph-wasm (our reference wrapper)**: its `Dtype` is
  `uint8 | int8 | uint16 | int16 | int32` — **no `uint32`**. Full `uint32`
  labels (IDs ≥ 2³¹) overflow `int32`. This is the concrete limitation the user
  hit; it is a *wrapper* gap (OpenJPH core has the 64-bit path), fixable by
  exposing it.

### Where *our* codec stands
- **Parser** reads `bit_depth = (Ssiz & 0x7F) + 1` with no ceiling — it accepts
  a high-bit-depth declaration.
- **HT block decoder** is **BUF32-only**: `p = 30 − missing_msbs`, magnitudes
  packed as `(v + 2) << (p − 1)` with the sign in bit 31 (`block_decoder.rs`).
  Effective ceiling ≈ 30-bit precision. `uint16` (our test path) is comfortable;
  `int32`/`uint32` would need a ported 64-bit HT path (mirror OpenJPH `BUF64`).
- **Inverse DWT** is `i32` (CPU `idwt_1d_53`, WGSL `idwt53`). Reversible 5/3 on
  32-bit input grows the coefficient range by a few bits per level (to ~37-bit),
  so the DWT needs **64-bit integers** end-to-end.

### The GPU wall: WebGPU has no 64-bit integers
Core WGSL has `i32/u32/f32` (and `f16` via extension) — **no native `i64`/`u64`,
and no ratified 64-bit-integer extension**. A lossless `uint32` GPU DWT would
need 2×`u32` emulated 64-bit arithmetic in the shader (carry handling on every
add/shift in the lifting) — a meaningful rewrite and slower. So:

- CPU lossless `uint32` is a well-scoped port (64-bit HT + `i64` DWT).
- **GPU lossless `uint32` is the hard part** — emulated 64-bit, or give up the
  GPU DWT for that path and do it on CPU.
- High-dynamic-range **intensity as float** sidesteps this: the irreversible
  9/7 path is `f32`, which WGSL has natively (already working on GPU).

## 4. Is this codec useful for SpatialData *labels*?

Tentative answer: **probably not the right tool for label volumes**, for reasons
deeper than bit-depth.

- **Labels are categorical, not numeric.** A wavelet transform assumes the
  sample value is a magnitude on a continuum; label IDs are names (label 3 is not
  "between" 2 and 4). The 5/3 transform produces large high-frequency
  coefficients at *every* segment boundary and cannot exploit the run/region
  structure that label data actually has. It stays lossless, but typically
  compresses labels **worse** than purpose-built schemes (RLE,
  Neuroglancer `compressed_segmentation`, or generic `zstd`/`blosc`).
- **Multiresolution is meaningless for labels.** The headline differentiators of
  a wavelet codec — progressive / lower-resolution decode for overviews, and
  lossy rate control — both require *averaging* samples, which is nonsensical for
  label IDs (you cannot down-sample a segmentation by averaging). Lossy is out
  entirely. So the main reasons to pick a wavelet codec evaporate for labels.
- **Only the reversible path is usable, and only at the cost above**, plus the
  `uint32`/64-bit work in §3.

Where this codec *does* fit SpatialData is **intensity** data — microscopy and
volumetric continuous-tone images — where multiresolution overviews, GPU decode
for interactive viz, and a lossy↔lossless choice are all genuinely valuable.
That is the sweet spot to optimise for; labels are better served by a
categorical/RLE codec (which could live alongside this one) — see next.

### What to use for labels instead

Labels are categorical and must be **lossless**; the values are usually
**sparse** (a few thousand IDs in a uint16/uint32 container). Options, by effort:

- **First, relabel to a dense palette.** Remap distinct IDs to `0..N-1`, store the
  palette separately (paletted-image style). This collapses the effective
  bit-depth and makes *every* downstream compressor far better. Do it first.
- **Tier 1 — zarr-native baseline (low effort):** blosc + **zstd** (high level) +
  **bitshuffle** via `numcodecs`. Piecewise-constant regions compress well under
  zstd; bitshuffle zeros the high bytes of small IDs. With the palette remap this
  is often close to purpose-built codecs. *Avoid* delta filters (meaningless for
  categorical IDs) and any wavelet/DCT codec (this one).
- **Tier 2 — purpose-built, random-access, GPU-friendly (recommended):**
  Neuroglancer **`compressed_segmentation`** (PyPI `compressed-segmentation`;
  TensorStore supports it). Splits the volume into blocks, each storing a LUT of
  the block's distinct labels + bit-packed per-voxel indices. uint32/uint64,
  block-level random access, multiscale-friendly, and **designed to be decoded on
  the GPU** (Neuroglancer renders it in shaders).
- **Tier 3 — maximum ratio (3D connectomics-tuned):** **Crackle** (`crackle-codec`)
  and **Compresso** separate per-segment boundary structure from labels (crack
  codes / windowed features), reaching very high ratios on dense 3D segmentation.
  Best when the data is segmentation-like and large; overkill for 2D masks.

**Multiscale caveat:** downsample labels with **nearest / mode (majority)**, never
averaging — averaging label IDs is meaningless. OME-Zarr label pyramids do this.

**Synthesis for this project:** `compressed_segmentation` is the *label analog* of
our intensity path — block-structured, random-access, GPU-decodable. If the value
is GPU-resident decode for zarr/SpatialData viz, the natural split is **intensity →
HTJ2K + GPU inverse-DWT** (this work) and **labels → a TypeGPU
`compressed_segmentation` decoder** (LUT + bit-unpack in a shader; no wavelet, no
64-bit-integer problem). That is a much simpler, on-theme extension than forcing
labels through the wavelet pipeline.

## 5. Difficulty summary & recommendation

| Capability | Needs | Difficulty |
|---|---|---|
| `uint16` lossless/lossy (current) | — | done |
| `int32` lossless, CPU | 64-bit HT decode + `i64` CPU DWT | moderate, well-scoped (port OpenJPH `BUF64`) |
| `uint32` lossless, CPU | as above + 33-bit-range handling | moderate |
| `uint32` lossless, **GPU DWT** | emulated 64-bit in WGSL | hard / low payoff |
| HDR intensity lossy, GPU | none (9/7 is f32) | already works |

**Recommendations**

1. **GPU 5/3 kernel optimised — thesis proven.** The shared-memory kernel wins
   from ~256² (≈4× at 512², ≈10× at 1024², ≈16× at 2048²), result kept on-GPU.
   Next: apply the same shared-memory rewrite to the 9/7 kernel; coalesce the
   vertical pass; and tile lines beyond the 2048-sample shared-memory cap. The
   "validate at >1024² in a browser" item is **done on Node instead** — the
   readback ceiling that motivated it was our own bug (see the correction at the
   top), and 2048² now measures here directly.
2. **Treat labels as a separate problem.** If SpatialData needs compressed
   `uint32` labels, evaluate a categorical/RLE codec rather than forcing them
   through a wavelet pipeline; reserve this codec for intensity data.
3. If `int32`/`uint32` *intensity* is needed, port OpenJPH's 64-bit reversible HT
   path on the **CPU** and keep the GPU DWT for the ≤32-bit and float paths.
4. The "GPU DWT in OpenJPH" idea is better realised as our existing split:
   OpenJPH-style HT decode on CPU + a GPU DWT/dequant/level-shift stage for
   GPU-resident visualisation.

Source: benchmarks in `test/bench_idwt*.ts` and `scripts/bench-idwt-readback.ts`; bit-depth ceilings in
`rust/htj2k-core/src/block_decoder.rs` and OpenJPH `ojph_codeblock.cpp` /
`ojph_transform.cpp`; wrapper `Dtype` in `openjph-wasm`.
