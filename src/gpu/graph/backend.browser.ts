// Browser backend for the operation-graph runtime. Shares the device that
// src/gpu/device.ts resolves to `navigator.gpu` (so Tier-1 ops and native ops use
// one device), and reads back through TypeGPU `.read()` — the same Dawn-stable
// pattern, which works unchanged in the browser. This is the only file that differs
// from the Node backend; the op definitions and executor are imported verbatim.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../device";
import type { GpuBackend, Root } from "./backend";
import type { ResidentBuffer } from "./handle";
import { BufferPool, type PoolStats, residentUsage } from "./pool";

let rootP: Promise<Root> | undefined;
function getRoot(): Promise<Root> {
  rootP ??= (async () => {
    const device = await getDevice();
    return tgpu.initFromDevice({ device });
  })();
  return rootP;
}

// Tier-2 pool (ADR-0017). Created on first use, since the device is resolved asynchronously.
let poolP: Promise<BufferPool> | undefined;
function getPool(): Promise<BufferPool> {
  poolP ??= getDevice().then((device) => new BufferPool(device));
  return poolP;
}
let poolRef: BufferPool | undefined;

/** One readback wrapper per raw buffer — see backend.node.ts for why this must not be per-call:
 *  wrapping makes the TypeGPU root a second owner of a pooled buffer, and a recycling pool would
 *  otherwise mint a fresh owner on every download. */
const wrappers = new WeakMap<GPUBuffer, ReturnType<Root["createBuffer"]>>();

function wrapperFor(root: Root, buffer: GPUBuffer) {
  let w = wrappers.get(buffer);
  if (!w) {
    w = root.createBuffer(d.arrayOf(d.f32, Math.max(1, Math.floor(buffer.size / 4))), buffer);
    wrappers.set(buffer, w);
  }
  return w;
}

export const browserBackend: GpuBackend = {
  kind: "browser",
  getDevice,
  getRoot,
  async readbackF32(buffer: GPUBuffer, n: number): Promise<Float32Array> {
    const root = await getRoot();
    const got = (await wrapperFor(root, buffer).read()) as ArrayLike<number>;
    return Float32Array.from({ length: n }, (_, i) => got[i] ?? 0);
  },

  async lease(byteLength: number, usage: number = residentUsage()): Promise<ResidentBuffer> {
    const pool = await getPool();
    poolRef = pool;
    return pool.lease(byteLength, usage);
  },

  release(b: ResidentBuffer): void {
    // A release can only follow a lease, so the pool exists by now.
    if (!poolRef) throw new Error("browserBackend.release: no pool — release without a preceding lease");
    poolRef.release(b);
  },

  async upload(data: ArrayBufferView, usage: number = residentUsage()): Promise<ResidentBuffer> {
    const [device, pool] = await Promise.all([getDevice(), getPool()]);
    poolRef = pool;
    const res = pool.lease(data.byteLength, usage);
    // dataOffset/size are in view *elements*, not bytes — omit them and write the whole view.
    device.queue.writeBuffer(res.buffer, 0, data as BufferSource);
    return res;
  },

  poolStats(): PoolStats {
    return poolRef?.stats() ?? { live: 0, free: 0, created: 0, bytes: 0 };
  },
};
