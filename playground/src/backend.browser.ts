// Browser backend for the operation-graph runtime. Shares the device that
// src/gpu/device.ts resolves to `navigator.gpu` (so Tier-1 ops and native ops use
// one device), and reads back through TypeGPU `.read()` — the same Dawn-stable
// pattern, which works unchanged in the browser. This is the only file that differs
// from the Node backend; the op definitions and executor are imported verbatim.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../../src/gpu/device";
import type { GpuBackend, Root } from "../../src/gpu/graph/backend";

let rootP: Promise<Root> | undefined;
function getRoot(): Promise<Root> {
  rootP ??= (async () => {
    const device = await getDevice();
    return tgpu.initFromDevice({ device });
  })();
  return rootP;
}

export const browserBackend: GpuBackend = {
  kind: "browser",
  getDevice,
  getRoot,
  async readbackF32(buffer: GPUBuffer, n: number): Promise<Float32Array> {
    const root = await getRoot();
    const wrap = root.createBuffer(d.arrayOf(d.f32, Math.max(1, n)), buffer);
    const got = (await wrap.read()) as ArrayLike<number>;
    return Float32Array.from({ length: n }, (_, i) => got[i]!);
  },
};
