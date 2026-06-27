// Headless WebGPU device for Node via the Dawn (`webgpu`) binding.
//
// In the browser this module is replaced by `navigator.gpu`. Under Node we use
// the Dawn N-API addon. (Bun segfaults on the compute path — see
// docs/decisions/0002-runtime-node-not-bun.md.)
import { create, globals } from "webgpu";

let devicePromise: Promise<GPUDevice> | undefined;

/** Get (and cache) a headless GPUDevice. */
export function getDevice(): Promise<GPUDevice> {
  devicePromise ??= (async () => {
    // Make GPUBufferUsage, GPUMapMode, etc. available as globals.
    Object.assign(globalThis, globals);
    const gpu: GPU = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu ?? create([]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("tgpu-htj2k: no WebGPU adapter available");
    return adapter.requestDevice();
  })();
  return devicePromise;
}
