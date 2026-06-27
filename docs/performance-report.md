# Performance report — HTJ2K decode (CPU core + GPU inverse DWT)

Status: **report** (2026-06-27)

A synthesis of the performance work so far: how our from-scratch HTJ2K decoder
compares to OpenJPH on the CPU, and what moving the inverse DWT to the GPU
actually buys. Companion to the architecture/scope notes in
[`dwt-gpu-and-high-bit-depth.md`](dwt-gpu-and-high-bit-depth.md).

## TL;DR

- **CPU pipeline is ~2× OpenJPH**, not far off for a from-scratch decoder.
  Entropy decode and the inverse DWT each cost about half.
- **The GPU inverse 5/3 DWT wins from ~256² up**: ≈3× at 512² and ≈9–12× at
  1024² vs our CPU DWT, with the result kept on the GPU.
- The optimised kernel made the difference: a **shared-memory, one-workgroup-per-
  line** design dropped the crossover from ~512² (naive) to ~256² and made GPU
  compute essentially **size-independent** (~0.4 → ~1.5 ms over 128²→1024²).
- The GPU path is now **overhead/upload-bound**, not compute-bound — the DWT
  itself is nearly free; what's left is fixed launch cost and the input upload.
- Both DWT kernels stay **numerically faithful**: 5/3 bit-exact vs OpenJPH, 9/7
  within 1 LSB.

## Methodology & caveats

- **Machine**: the dev machine, Node + the Dawn (`webgpu`) binding. Figures are
  **medians in ms**; treat them as **ratios and trends**, not absolutes — a
  different GPU/driver will move them.
- **Build matters.** All numbers are from a **release** wasm build
  (`pnpm build:wasm`). An early draft used a dev build and was ~10× slower,
  which produced a badly wrong "20× slower than OpenJPH" conclusion. Always
  benchmark release.
- **What's timed.**
  - *OpenJPH full* — `openjph-wasm` `decode()` (entire decode, optimised C/SIMD).
  - *ours entropy* — `decode_dwt_input_53` (HT block decode + reassembly).
  - *ours CPU DWT* — `idwt53_cpu` (inverse DWT only, on the packed coefficients).
  - *GPU compute* — `idwt53Gpu(..., {readback:false})`: upload + dispatch + sync,
    **result left on the GPU** (no readback). This is the realistic
    visualisation case.
- **The Dawn-on-Node readback ceiling.** The `mapAsync` full readback crashes the
  worker beyond ~512²; the *compute* runs fine to 2048². Large-size GPU timing
  therefore uses the no-readback path — which is also exactly how a GPU-resident
  viz pipeline would run. Benchmarks are opt-in (`pnpm bench:gpu`) so this
  fragility stays out of the default test suite.

## CPU pipeline vs OpenJPH

Single component, reversible 5/3, 5 levels, high-detail content
(`test/bench_idwt.test.ts`):

| size  | OpenJPH full | ours: entropy | ours: CPU DWT | ours total |
|-------|-------------:|--------------:|--------------:|-----------:|
| 64²   |        0.18  |         0.10  |         0.08  |      ~0.18 |
| 128²  |        0.60  |         0.24  |         0.19  |      ~0.43 |
| 256²  |        0.70  |         0.69  |         0.72  |      ~1.41 |
| 512²  |        2.84  |         2.90  |         3.32  |      ~6.22 |

Observations:

- **~2× OpenJPH at 512²** (≈6.2 ms vs 2.84 ms). OpenJPH streams lines with SIMD
  and fuses stages; ours copies `Band`s and gathers columns, so there is real
  but unsurprising headroom. For a from-scratch decoder this is a respectable
  starting point, not the order-of-magnitude gap an earlier (dev-build) draft
  implied.
- **Entropy ≈ DWT** (the `entropy/DWT` ratio sits at ~0.9–1.2). The DWT is about
  half the pipeline, so it's a worthwhile thing to offload — but only half the
  story; the HT entropy decode is the other half and is **bit-serial, so it
  stays on the CPU**.

## GPU inverse 5/3 DWT — naive vs optimised

CPU DWT vs GPU compute (no readback, buffers pooled). Two kernels:

**Naive** — one thread per line, lifting in a global scratch buffer:

| size  | ratio (CPU/GPU) |
|-------|----------------:|
| 128²  |          0.25×  |
| 256²  |          0.5×   |
| 512²  |          1.1×   |
| 1024² |  (readback/mem crash) |

**Optimised** — one workgroup per line, lifting in workgroup **shared memory**
with intra-line parallelism, no global scratch (`test/bench_idwt.gpu.test.ts`):

| size  | CPU DWT | GPU compute | speedup |
|-------|--------:|------------:|--------:|
| 128²  |    0.20 |        0.47 |   0.42× |
| 256²  |    0.73 |        0.57 |   1.29× |
| 512²  |    3.11 |        0.98 |   3.17× |
| 1024² |   14.18 |        1.58 |  ~9–12× |

(1024² speedup varies ~9–12× run-to-run, driven by CPU-side thermal noise; GPU
compute is steady at ~1.2–1.6 ms.)

What the optimisation changed and why it works:

- **Crossover dropped from ~512² to ~256²**, and the win grows with size.
- **GPU compute is nearly size-independent** (~0.4 ms at 128² → ~1.5 ms at
  1024², a 16× pixel increase). The lifting work scales, but it's hidden behind
  a fixed launch/upload floor — i.e. the DWT compute itself is essentially free.
- The mechanism: the old kernel did every lifting read/write through **global
  memory** and serialised each line in one thread. The new kernel loads a line
  into **shared memory** once, then runs each lifting step in parallel across the
  workgroup (steps are independent within themselves; only step→step ordering
  needs a `workgroupBarrier`). Dropping the global scratch buffer also lowered
  GPU memory enough to run the no-readback path to 1024² in this harness.
- **Output is display-ready.** The vertical pass folds in the DC level shift, so
  the GPU emits final pixel values — no CPU post-pass, and a viz pipeline can use
  the buffer/texture directly.

## What didn't work (so it isn't re-tried)

- **Batching all levels into one command submission.** Intuitively this should
  cut per-level submit overhead, but it was **no faster and ~1.5× slower at
  1024²** — the per-level submits already pipeline on the queue, and one big
  command buffer with automatic inter-pass barriers synchronised more, not less.
  Reverted.
- **The dev-build measurement trap** (noted above): a 10× error that inverted the
  CPU conclusion. Caught by sanity-checking against OpenJPH.

## Where the time actually goes (and the architectural read)

At the sizes we can benchmark, the GPU DWT is **overhead/upload-bound**:

- a **fixed ~0.4 ms floor** (visible at 128²/256², where the upload is tiny) —
  command encoding, bind groups, dispatch, submit, sync;
- plus **input upload** (~0.3 ms/MB) of the coefficients, which is unavoidable
  (they're produced by the CPU entropy stage and must reach the GPU);
- plus **near-zero actual DWT compute**.

The takeaways:

1. **The GPU thesis holds for a GPU-resident viz pipeline.** Decode HT on the
   CPU → upload coefficients once → DWT + level-shift on the GPU → keep the image
   on-GPU for display. In that flow the DWT is effectively free and scales to
   large images; the only real cost is the upload, which any GPU display path
   pays anyway.
2. **It does *not* pay off for decode-to-CPU.** Adding a readback turns the win
   into a loss (the readback is also the Dawn-on-Node stability ceiling). This is
   why "add a GPU DWT to OpenJPH" is the wrong framing — OpenJPH decodes to CPU
   memory with fast SIMD; the GPU value is specifically keep-on-GPU.
3. **Entropy is the CPU floor.** Since the bit-serial HT decode can't go to the
   GPU and is ~half the pipeline, the *end-to-end* decode-to-display time is
   gated by entropy + upload, not the DWT. Further DWT speedups have diminishing
   end-to-end impact unless entropy is also sped up (CPU SIMD/threads).

## Numerical fidelity

- **5/3 (lossless)**: GPU output **bit-exact** vs the CPU golden / OpenJPH across
  odd / non-power-of-two / multi-level sizes (`gpu_idwt53.gpu.test.ts`).
- **9/7 (lossy)**: GPU within **≤1 LSB** of the CPU (`gpu_idwt97.gpu.test.ts`) —
  the expected cross-implementation float tolerance; `maxd ≤ 1` is the
  correctness signal.
- Underneath, the HT block decoder and parser are bit-exact vs OpenJPH (separate
  tests), so the DWT is the only place numerics could drift, and it doesn't.

## Remaining headroom

- **9/7 pixel-fold**: fold `irvToPixels` into the GPU output (→ i32 pixel buffer)
  to complete end-to-end GPU for the lossy path.
- **Coalesce the vertical pass**: it reads columns strided; a transposed/tiled
  pass would help — but only bites >1024², which the readback ceiling hides here.
- **Tile lines > 2048** (the shared-memory cap) for very large images.
- **Browser benchmark**: validate/measure >1024² where Dawn-on-Node can't, and
  confirm the size-independence trend continues.
- **CPU side**: a streaming, copy-free DWT and SIMD entropy would close the ~2×
  gap to OpenJPH if CPU decode speed matters.

## Reproduce

```sh
pnpm build:wasm          # IMPORTANT: release build
pnpm test                # correctness (CPU + GPU), incl. bit-exact / ≤1 LSB
npx vitest run test/bench_idwt.test.ts   # CPU breakdown vs OpenJPH (to 512²)
pnpm bench:gpu                            # GPU compute vs CPU DWT (to 1024²)
```
