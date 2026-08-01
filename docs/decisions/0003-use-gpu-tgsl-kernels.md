# ADR-0003 — Author compute kernels in `"use gpu"` (TGSL), run via resolved WGSL

Status: **accepted** (2026-06-28) — **three Dawn limits below superseded 2026-08-01**

> ⚠️ **The three "Dawn instability" constraints in this ADR were all one bug of ours,
> and it is fixed.** They were recorded 2026-06-28; `src/gpu/device.ts` was letting
> Dawn's Instance be GC'd out from under a live device, fixed 2026-07-29 (`4e326b0`).
> Re-tested 2026-08-01 on Node 24.18.0, 5 runs of 5 clean, probes in
> [`test/dawn-limits-sweep.gpu.test.ts`](../../test/dawn-limits-sweep.gpu.test.ts):
>
> | claimed limit | re-tested | result |
> |---|---|---|
> | 2nd guarded pipeline segfaults teardown | 4 pipelines, each dispatched + read | **gone** |
> | nnDistance segfaults past ~256 points | 256 → 8192 points, checked vs CPU golden | **gone** |
> | raw `mapAsync` on a pooled MAP_READ buffer kills the vitest worker | pooled, reused 3×, then grown | **gone** |
>
> **The decision itself still stands** — resolved WGSL + a layout-bound pipeline is
> still the right pattern, because it builds once and survives buffer growth. What
> changes is that it is now a design preference rather than a crash workaround, and
> the *test-size* caps taken on its account can be lifted (`nnDistance.gpu.test.ts`
> went 160 → 2048). The Node ≥ 20.11 requirement below is unrelated and still real;
> the Volta pin moved 22.23.1 → 24.18.0.

## Decision

For the spatial-analysis front, compute kernels that don't need workgroup shared
memory / barriers / atomics are authored in **TypeScript with the `"use gpu"`
directive** (TGSL), transpiled to WGSL by **`unplugin-typegpu`** (wired into
`vitest.gpu.config.ts` as a Vite plugin). Kernels are then **resolved to WGSL
(`tgpu.resolveWithContext`) and executed via a raw, layout-bound compute pipeline**
— *not* the `createGuardedComputePipeline(...).dispatchThreads()` runtime.

Kernels that DO need shared memory / barriers / atomics (the DWT line kernel; later
the spatial-index scan/counting-sort and histogram splats) stay as **WGSL
templates** — TGSL in TypeGPU 0.11.x doesn't cover those cleanly.

This is in response to the user's instruction: *"Please `"use gpu"` for compute
kernels where it makes sense to do so."*

## Context & provenance

Validated with a spike, then the first real primitive
(`src/gpu/spatial/nnDistance.ts`, nearest-neighbour distance):

- **TypeGPU 0.11.9** supports the `"use gpu"` directive (TGSL → WGSL via `tinyest`),
  but the JS→AST step needs the **`unplugin-typegpu`** build plugin (installed
  `0.11.6`, matches the 0.11.x line). Without it, `"use gpu"` functions throw at
  resolve time.
- **Node ≥ 20.11 required.** `unplugin` evaluates `import.meta.dirname` at import;
  on Node 18 that is `undefined` and the plugin crashes on load. The repo's Volta
  pin (Node 22.23.1) is the intended runtime — run GPU tests under it. `engines.node`
  bumped to `>=20.11` to make this explicit.
- **Use resolved WGSL + raw pipeline, not the guarded pipeline.** The guarded
  `"use gpu"` pipeline closes over specific buffer instances, so it must be rebuilt
  when pooled buffers grow — and creating a second guarded pipeline in one process
  **segfaulted Dawn-on-Node's exit teardown**. Resolving the same TGSL function to
  WGSL and binding a *layout* lets the pipeline be built **once** (survives buffer
  growth → only the bind group is recreated). This matches the DWT modules.
- **Dawn teardown threshold (unchanged Node limit).** Even on the clean path, a
  process that does "enough" GPU work segfaults its exit teardown (the same Dawn
  instability ADR-0002 / `vitest.gpu.config.ts` already isolate per-file). For
  nnDistance this showed up past ~256 brute-force points. Node tests are kept below
  that; large-N validation belongs in the **browser** harness. Correctness is
  N-independent, so small-N tests still validate the kernel against the CPU golden.
- TGSL gotchas found: requires `!==` (not `!=`); `noUncheckedIndexedAccess` means
  storage-array reads need `!` (the assertion is stripped by the transpiler).
- **Read results back via TypeGPU `.read()`, not raw `mapAsync`.** Building the
  density splat (`splatDensity.ts`, a raw-WebGPU render primitive) showed that a raw
  `mapAsync` on a pooled `MAP_READ` buffer **segfaulted the vitest worker on
  teardown**, even though the same render+readback exits cleanly outside vitest
  (natural Node exit). Every Dawn-stable test in the repo reads via TypeGPU's
  `.read()`. Fix: keep the render raw, but `copyTextureToBuffer` into a GPUBuffer we
  own and wrap with `root.createBuffer(schema, rawBuffer)`, then `.read()` it.

## Consequences

- New dev dependency: `unplugin-typegpu` (build-time only). GPU tests require
  Node ≥ 20.11 (Volta already pins 22). CPU tests are unaffected.
- Kernel-authoring is a two-track choice (TGSL vs WGSL template) decided by whether
  the kernel needs shared memory / barriers / atomics. Documented in
  [`gpu-spatial-analysis-toolbox.md`](../gpu-spatial-analysis-toolbox.md).
- Pattern to reuse for new TGSL primitives: define a `tgpu.bindGroupLayout`, write
  the `tgpu.computeFn(...)((input) => { "use gpu"; ... })` referencing `layout.$`,
  `resolveWithContext([fn])` → WGSL, build the pipeline once, pool/grow buffers
  without `.destroy()`, recreate only the bind group per call.
- Revisit if a later TypeGPU version makes the guarded pipeline Dawn-stable, or
  brings shared memory / atomics into TGSL (then the WGSL templates could migrate).
