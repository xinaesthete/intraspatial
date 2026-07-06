// Node (Dawn N-API) backend. Wraps the project's headless device + a TypeGPU root,
// and reads back through TypeGPU's `.read()` (wrapping a raw GPUBuffer), the
// Dawn-on-Node-stable path — a raw `mapAsync` on a pooled buffer segfaults the
// vitest worker on teardown (ADR-0003 / splatDensity notes).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../device";
import type { GpuBackend, Root } from "./backend";

let cached: Promise<{ device: GPUDevice; root: Root }> | undefined;

function init(): Promise<{ device: GPUDevice; root: Root }> {
  cached ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    return { device, root };
  })();
  return cached;
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
    const wrap = root.createBuffer(d.arrayOf(d.f32, Math.max(1, n)), buffer);
    const got = (await wrap.read()) as ArrayLike<number>;
    return Float32Array.from({ length: n }, (_, i) => got[i] ?? 0);
  },
};
