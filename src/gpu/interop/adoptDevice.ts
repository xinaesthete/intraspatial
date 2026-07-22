// Interop seam #1 — adopt an *externally owned* GPUDevice.
//
// The whole compute stack is backend-agnostic: every kernel is handed a `GPUDevice`
// and builds a TypeGPU `root` from it (`tgpu.initFromDevice`). The Node backend gets
// that device from Dawn; the browser normally gets it from `navigator.gpu`. But a host
// renderer — three.js `WebGPURenderer` (`renderer.backend.device`), or deck.gl/luma.gl —
// creates and owns its OWN device. To share GPUBuffers with such a host WITHOUT readback,
// our compute must run on the SAME device. This util wraps any device we're handed as a
// `GpuBackend`, so the rest of the stack is none the wiser about who created it.
//
// Pure WebGPU + TypeGPU — no three.js/deck.gl import here, so it stays in the tested `src`
// library and works under Node (Dawn) too. The renderer-specific buffer bridging lives
// alongside the host (e.g. the dancer's three.js bridge), built on top of this.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { GpuBackend, Root } from "../graph/backend";
import type { ResidentBuffer } from "../graph/handle";
import { BufferPool, type PoolStats, residentUsage } from "../graph/pool";

/** Wrap an externally-owned device as a `GpuBackend` bound to it. `kind` is a diagnostic
 *  tag for the host ("three" | "deck" | …). The returned backend caches its root so all
 *  compute shares one root on the shared device. */
export function adoptDevice(device: GPUDevice, kind = "adopted"): GpuBackend {
  let root: Root | undefined;
  const getRoot = (): Root => (root ??= tgpu.initFromDevice({ device }));
  // Tier-2 leases are allocated on the *host's* device, which is the point: a buffer leased
  // here can be bound straight into the host renderer's pass with no transfer at all.
  const pool = new BufferPool(device);
  return {
    kind,
    async getDevice() {
      return device;
    },
    async getRoot() {
      return getRoot();
    },
    async readbackF32(buffer: GPUBuffer, n: number): Promise<Float32Array> {
      // Wrap the caller's raw buffer and read via TypeGPU (uniform with the Node path).
      // On a shared host device this is the *occasional* / off-hot-path readback (e.g. a
      // low-frequency snapshot for a CPU-side panel); the render path reads the buffer on
      // the GPU directly and never calls this.
      const wrap = getRoot().createBuffer(d.arrayOf(d.f32, Math.max(1, n)), buffer);
      const got = (await wrap.read()) as ArrayLike<number>;
      return Float32Array.from({ length: n }, (_, i) => got[i] ?? 0);
    },
    async lease(byteLength: number, usage: number = residentUsage()): Promise<ResidentBuffer> {
      return pool.lease(byteLength, usage);
    },
    release(b: ResidentBuffer): void {
      pool.release(b);
    },
    async upload(data: ArrayBufferView, usage: number = residentUsage()): Promise<ResidentBuffer> {
      const res = pool.lease(data.byteLength, usage);
      // dataOffset/size are in view *elements*, not bytes — omit them and write the whole view.
      device.queue.writeBuffer(res.buffer, 0, data as BufferSource);
      return res;
    },
    poolStats(): PoolStats {
      return pool.stats();
    },
  };
}
