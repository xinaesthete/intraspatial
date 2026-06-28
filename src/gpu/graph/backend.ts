// The device/pool/readback seam.
//
// The op definitions (kernels, ports, params, goldens) and the executor are
// backend-agnostic. Only three things differ between running under Node (Dawn
// N-API, `.read()` readback, explicit layout-bound pipelines — ADR-0003) and in
// the browser (`navigator.gpu`, `mapAsync`, single-flight + validate-or-null
// fallback): device acquisition, buffer allocation, and readback. Those three are
// exactly what already diverged between `src/gpu/spatial/*` and the demo's
// `docs-site/src/lib/gpuField.ts`; this interface unifies them.
//
// Phase 0 keeps the surface small: enough for the native `threshold` op to run a
// real compute pass and read it back. Tier-2 resident ops will grow the pool API
// (lease/return by liveness) on top.

import type tgpu from "typegpu";

export type Root = ReturnType<typeof tgpu.initFromDevice>;

export interface GpuBackend {
  /** Human-readable tag for diagnostics ("node" | "browser"). */
  readonly kind: string;
  /** Get (and cache) the GPUDevice. */
  getDevice(): Promise<GPUDevice>;
  /** Get (and cache) a TypeGPU root bound to that device. */
  getRoot(): Promise<Root>;
  /** Read back a storage buffer of `n` f32s as a host Float32Array, using the
   *  backend's Dawn-stable path. */
  readbackF32(buffer: GPUBuffer, n: number): Promise<Float32Array>;
}
