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
// Phase 0 kept the surface small: enough for the native `threshold` op to run a
// real compute pass and read it back. ADR-0017 grows it with the pool API this
// header anticipated — lease/return by liveness — which is what lets a value stay
// GPU-resident across a graph edge instead of round-tripping through the host.

import type tgpu from "typegpu";
import type { ResidentBuffer, ResidentTexture } from "./handle";
import type { PoolStats } from "./pool";

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
  /** Read back `bytes` of a storage buffer verbatim, for a dtype the f32 path would mangle
   *  (ADR-0017 stage 1's limit; see `readBackBytes` in `device.ts` for why it is a separate
   *  path rather than a second TypeGPU wrapper). */
  readbackBytes(buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer>;

  // --- Tier-2 resident pool (ADR-0017, invariant 3) ---

  /** Lease a resident buffer of at least `byteLength` bytes from the pool. Defaults to the
   *  resident usage class (`STORAGE | COPY_SRC | COPY_DST`); pass explicit flags for a class the
   *  pool doesn't name (e.g. `| VERTEX`). Free lists are keyed on the flags, so classes never
   *  alias — a lease can only be served by a buffer physically created with the same usage. */
  lease(byteLength: number, usage?: number): Promise<ResidentBuffer>;
  /** Return a leased buffer to the pool. NEVER destroys it — mid-process destruction
   *  segfaults Dawn-on-Node (ADR-0002/0003). */
  release(b: ResidentBuffer): void;
  /** Lease a buffer and upload `data` into it — the "upload at sources" half of invariant 4. */
  upload(data: ArrayBufferView, usage?: number): Promise<ResidentBuffer>;
  /** Lease a resident TEXTURE — what a render-producing op writes into. Same liveness contract as
   *  `lease`: the holder returns it, and the pool never destroys. */
  leaseTexture(width: number, height: number, format?: GPUTextureFormat, usage?: number): Promise<ResidentTexture>;
  /** Return a leased texture to the pool. */
  releaseTexture(t: ResidentTexture): void;
  /** Pool occupancy, for tests and the debug overlay. */
  poolStats(): PoolStats;
}
