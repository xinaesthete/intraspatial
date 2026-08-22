// Node (Dawn N-API) backend. Wraps the project's headless device + a TypeGPU root,
// and reads back through TypeGPU's `.read()` (wrapping a raw GPUBuffer), the
// Dawn-on-Node-stable path — a raw `mapAsync` on a pooled buffer segfaults the
// vitest worker on teardown (ADR-0003 / splatDensity notes).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice, readBackBytes, writeView } from "../device";
import type { GpuBackend, Root } from "./backend";
import type { ResidentBuffer, ResidentTexture } from "./handle";
import { BufferPool, type PoolStats, residentTextureUsage, residentUsage, TexturePool } from "./pool";

let cached: Promise<{ device: GPUDevice; root: Root; pool: BufferPool }> | undefined;

function init(): Promise<{ device: GPUDevice; root: Root; pool: BufferPool }> {
  cached ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    return { device, root, pool: new BufferPool(device) };
  })();
  return cached;
}

/** The pool is created with the device, so synchronous `poolStats()` needs a handle to it once
 *  acquired. Before the first `getDevice()` there is nothing allocated and stats are zero. */
let poolRef: BufferPool | undefined;

// The texture half of the Tier-2 pool, resolved lazily for the same reason.
let texPoolP: Promise<TexturePool> | undefined;
let texPoolRef: TexturePool | undefined;
function getTexPool(): Promise<TexturePool> {
  texPoolP ??= getDevice().then((device) => new TexturePool(device));
  return texPoolP;
}

/** One readback wrapper per raw buffer — see `readbackF32` for why this must not be per-call.
 *  Sized to the buffer's physical capacity so the wrapper stays valid whatever logical length a
 *  lease is later used at; callers take the prefix they asked for. */
const wrappers = new WeakMap<GPUBuffer, ReturnType<Root["createBuffer"]>>();

function wrapperFor(root: Root, buffer: GPUBuffer) {
  let w = wrappers.get(buffer);
  if (!w) {
    w = root.createBuffer(d.arrayOf(d.f32, Math.max(1, Math.floor(buffer.size / 4))), buffer);
    wrappers.set(buffer, w);
  }
  return w;
}

export const nodeBackend: GpuBackend = {
  kind: "node",
  async getDevice() {
    return (await init()).device;
  },
  async getRoot() {
    return (await init()).root;
  },
  async readbackF32(buffer: GPUBuffer, n: number): Promise<Float32Array> {
    const { root } = await init();
    // Wrap the caller's raw buffer so `.read()` (not raw mapAsync) does the readback.
    //
    // The wrapper is CACHED PER BUFFER, and that is load-bearing since ADR-0017. Wrapping makes
    // the TypeGPU root a second owner of a buffer someone else allocated, and every wrapper
    // frees its buffer when the root is torn down. That was harmless while readback targets
    // were module-scoped singletons — one wrapper each, created once. The Tier-2 pool recycles
    // buffers, so an uncached wrap mints a fresh owner for the *same* Dawn handle on every
    // download, and process exit then double-frees it: the atexit segfault of ADR-0002/0003.
    // One wrapper per buffer, forever, keeps ownership single.
    const got = (await wrapperFor(root, buffer).read()) as ArrayLike<number>;
    return Float32Array.from({ length: n }, (_, i) => got[i] ?? 0);
  },

  async readbackBytes(buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const { device } = await init();
    return readBackBytes(device, "backend.node:staging", buffer, 0, bytes);
  },

  async lease(byteLength: number, usage: number = residentUsage()): Promise<ResidentBuffer> {
    const { pool } = await init();
    poolRef = pool;
    return pool.lease(byteLength, usage);
  },

  release(b: ResidentBuffer): void {
    // A release can only follow a lease, so the pool exists by now.
    if (!poolRef) throw new Error("nodeBackend.release: no pool — release without a preceding lease");
    poolRef.release(b);
  },

  async upload(data: ArrayBufferView, usage: number = residentUsage()): Promise<ResidentBuffer> {
    const { device, pool } = await init();
    poolRef = pool;
    const res = pool.lease(data.byteLength, usage);
    // Whole-view write: `writeBuffer`'s dataOffset/size are counted in the view's *elements*,
    // not bytes, so passing byte counts there overruns. Omitting them writes exactly the view.
    writeView(device.queue, res.buffer, data);
    return res;
  },

  async leaseTexture(
    width: number,
    height: number,
    format: GPUTextureFormat = "r32float",
    usage: number = residentTextureUsage(),
  ): Promise<ResidentTexture> {
    const pool = await getTexPool();
    texPoolRef = pool;
    return pool.lease(width, height, format, usage);
  },

  releaseTexture(t: ResidentTexture): void {
    if (!texPoolRef) throw new Error("nodeBackend.releaseTexture: no pool — release without a preceding lease");
    texPoolRef.release(t);
  },

  poolStats(): PoolStats {
    const b = poolRef?.stats() ?? { live: 0, free: 0, created: 0, bytes: 0 };
    const t = texPoolRef?.stats();
    if (!t) return b;
    // Textures and buffers share one figure: callers want "how much is the graph holding", not a
    // breakdown by resource kind.
    return { live: b.live + t.live, free: b.free + t.free, created: b.created + t.created, bytes: b.bytes + t.bytes };
  },
};
