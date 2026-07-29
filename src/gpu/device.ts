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
    // Request float32-blendable when available so the spatial front can additively
    // blend density into an r32float render target (the no-atomics splat path).
    // Harmless for the compute kernels; falls back cleanly if unsupported.
    const requiredFeatures = (["float32-blendable"] as GPUFeatureName[]).filter((f) => adapter.features.has(f));
    const device = await adapter.requestDevice({ requiredFeatures });
    // WebGPU validation errors are NOT exceptions and do not reach the console on their own: an
    // invalid pipeline or bind group simply produces nothing. That failure mode is a blank canvas
    // with no diagnostic, which is the worst kind to debug, so surface them.
    const report = (err: unknown) => console.error("[webgpu]", (err as GPUError | undefined)?.message ?? err);
    if (typeof device.addEventListener === "function") {
      device.addEventListener("uncapturederror", (e) => report((e as GPUUncapturedErrorEvent).error));
    } else {
      // Dawn's Node binding is not an EventTarget, so the listener above silently never attaches —
      // which is how a shader that failed to compile produced an all-zero field and no diagnostic
      // at all. Fall back to the property form.
      (device as { onuncapturederror?: (e: GPUUncapturedErrorEvent) => void }).onuncapturederror = (e) => report(e.error);
    }
    return device;
  })();
  return devicePromise;
}

/**
 * Compile a shader module and surface its diagnostics as an exception.
 *
 * `createShaderModule` never throws. A module that failed to compile yields an invalid pipeline,
 * which yields an invalid bind group, which yields an invalid command buffer — and the only thing
 * the console shows is a cascade of "invalid due to a previous error" with the actual WGSL message
 * nowhere in it. The root cause is in `getCompilationInfo()`, so ask for it.
 *
 * Awaiting this costs one round-trip per pipeline, at pipeline-construction time only, which is
 * cached in every consumer here.
 */
export async function compileShader(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length > 0) {
    const detail = errors.map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join("\n");
    throw new Error(`WGSL compile failed in '${label}':\n${detail}`);
  }
  return module;
}
