// Headless WebGPU device for Node via the Dawn (`webgpu`) binding.
//
// In the browser `navigator.gpu` is used directly. Under Node we load the Dawn N-API addon
// — but only then, and only via a dynamic import: `webgpu` is an OPTIONAL peer of the
// published package, so a browser bundle must never try to resolve it. The specifier is
// held in a variable so bundlers leave the import alone (`@vite-ignore` covers Vite).
// (Bun segfaults on the compute path — see docs/decisions/0002-runtime-node-not-bun.md.)

type DawnModule = { create(args: string[]): GPU; globals: Record<string, unknown> };

async function loadDawn(): Promise<GPU> {
  const specifier = "webgpu";
  let dawn: DawnModule;
  try {
    dawn = (await import(/* @vite-ignore */ specifier)) as DawnModule;
  } catch (e) {
    throw new Error(
      "intraspatial: no `navigator.gpu` and the optional `webgpu` (Dawn) package is not installed — " +
        "install it for headless Node use, or run in a WebGPU-capable browser.",
      { cause: e },
    );
  }
  // Make GPUBufferUsage, GPUMapMode, etc. available as globals (the browser has them already).
  Object.assign(globalThis, dawn.globals);
  return dawn.create([]);
}

let devicePromise: Promise<GPUDevice> | undefined;

/**
 * Strong references to the Dawn **Instance** and adapter, held for the process lifetime.
 *
 * These are not tidiness — they are the fix for the crash that made GPU work outside
 * vitest look impossible. `create([])` returns Dawn's Instance, which owns the native
 * event loop and the mutexes every later device call takes. If it is left as a local it
 * becomes unreachable the moment `getDevice()` resolves, V8 collects it whenever it next
 * feels like it, and the N-API finaliser destroys the Instance *out from under the still
 * live device*. The next dispatch then locks a destroyed mutex:
 *
 *     libc++abi: terminating due to uncaught exception of type std::__1::system_error:
 *     mutex lock failed: Invalid argument
 *
 * — or simply segfaults, depending on where the collection lands. That is why the
 * failures looked random and environment-dependent: they were GC-timing dependent. It is
 * also why a *non-allocating* busy loop never reproduced it while `knnBruteForceCpu` or
 * writing zarr chunks did — allocation is what triggers the collection.
 *
 * The adapter is retained for the same reason: it sits between the Instance and the
 * device and is likewise dropped by the spec-shaped API once a device exists.
 */
let instanceRef: GPU | undefined;
let adapterRef: GPUAdapter | undefined;

/** Get (and cache) a headless GPUDevice. */
export function getDevice(): Promise<GPUDevice> {
  devicePromise ??= (async () => {
    const gpu: GPU = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu ?? (await loadDawn());
    instanceRef = gpu;
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("intraspatial: no WebGPU adapter available");
    adapterRef = adapter;
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
 * Release the device, adapter and Dawn Instance so a batch process can exit.
 *
 * **Call this exactly once, as the last thing the process does.** It is not a general
 * "I'm done with the GPU for now" — releasing while any GPU object is still reachable
 * (a pooled buffer, a cached pipeline, a TypeGPU root) faults when those objects are
 * finalised against a destroyed Instance. Measured: releasing right after the k-NN and
 * then writing zarr crashed 3 runs of 3; moving the same call to the end of the process
 * gave exit 0, 3 of 3.
 *
 * The pairing with `instanceRef` is the whole story:
 *   • hold the Instance while any GPU work or GPU object may still exist — otherwise
 *     the finaliser can destroy it mid-flight and the next dispatch faults;
 *   • drop it once the process is finished — otherwise the retained handle keeps the
 *     event loop alive and the process computes its answer correctly and never exits.
 *
 * A browser page wants neither half: it holds one device for its lifetime and there is
 * no process to exit. Long-running servers should simply never call this.
 *
 * Best-effort: `destroy()` failures are swallowed, because by the time a caller asks for
 * this it already has its readbacks and a noisy teardown should not fail work that has
 * already succeeded.
 */
export async function releaseDevice(): Promise<void> {
  const pending = devicePromise;
  devicePromise = undefined;
  adapterRef = undefined;
  instanceRef = undefined;
  if (!pending) return;
  try {
    (await pending).destroy();
  } catch {
    // Teardown is advisory; the caller already has its readbacks.
  }
}

/** WebGPU's floor guarantee for `maxComputeWorkgroupsPerDimension`. Used rather than the
 *  adapter's reported limit so the dispatch shape is identical on every machine — a grid that
 *  folds differently per device is one more thing to rule out when two runs disagree. */
export const MAX_WORKGROUPS_PER_DIM = 65535;

/**
 * Fold a linear workgroup count into a 2-D dispatch grid.
 *
 * A 1-D dispatch past `maxComputeWorkgroupsPerDimension` is NOT clamped: Dawn invalidates the
 * command buffer, the pass never runs, and the only outward sign is that it got faster. The
 * kernel rebuilds the linear index as `wid.x + wid.y * gridX`, where `gridX` is `x` here (or
 * `num_workgroups.x`, which saves a uniform field).
 */
export function dispatchGrid(workgroups: number, maxPerDim: number = MAX_WORKGROUPS_PER_DIM): { x: number; y: number } {
  const wg = Math.max(1, workgroups);
  const x = Math.min(wg, maxPerDim);
  const y = Math.ceil(wg / x);
  if (y > maxPerDim) {
    throw new Error(`dispatchGrid: ${workgroups} workgroups exceeds ${maxPerDim}^2 and cannot be folded into a 2-D grid`);
  }
  return { x, y };
}

/**
 * Refuse a storage binding larger than the device allows, rather than discovering it as a
 * wrong answer.
 *
 * An over-large binding is a *validation* error, and a validation error is silence: the bind
 * group is invalid, the command buffer is invalid, every dispatch does nothing, and the
 * readback returns whatever the buffer held before. No exception, and the timing looks fast
 * and healthy — a killed dispatch is quicker than a real one.
 *
 * The default limit is 128 MiB (2^27), i.e. ~33.5M f32. Adapters routinely support far more —
 * the Dawn adapter here reports 4 GiB — but `getDevice()` requests default limits, so raising
 * it is a device-wide decision rather than something an individual kernel can take.
 */
export function checkBindingSize(device: GPUDevice, who: string, bytes: number): void {
  const max = device.limits.maxStorageBufferBindingSize;
  if (bytes > max) {
    throw new Error(
      `${who}: needs a ${bytes} byte storage binding, over this device's maxStorageBufferBindingSize of ${max}. ` +
        `Request a higher limit in getDevice(), or split the input.`,
    );
  }
}

/**
 * A buffer binding sized to **this call**, not to the pooled buffer behind it.
 *
 * Every pool in this repo is grow-only, and most grow by doubling, so `{ buffer }` on its own
 * binds however big some earlier, larger call made the buffer. Once that capacity passes
 * `maxStorageBufferBindingSize` the bind group is invalid and the dispatch silently returns
 * the previous call's data — see `checkBindingSize` above. Measured 2026-08-01 in
 * `streamCompact`: 132 MB of data in a 160 MB pooled buffer reported the *previous* call's
 * answer, 3,913,615 rows, in 7 ms.
 *
 * Doubling is what makes this bite early: a buffer holding 70 MB can have a 134 MB capacity,
 * so the binding crosses the limit while the data is only half way there.
 *
 * (Note the Tier-2 pool in `graph/pool.ts` is *not* exposed to this, because it buckets to
 * powers of two and the limit is itself a power of two — `bucket(x) <= 2^27` whenever
 * `x <= 2^27`. That is an arithmetic accident rather than a design, and it stops holding if
 * either the bucketing or the limit stops being a power of two.)
 */
export function sized(buffer: GPUBuffer, bytes: number): GPUBufferBinding {
  if (bytes <= 0 || bytes > buffer.size) {
    throw new Error(`sized: ${bytes} is not a valid binding length into the ${buffer.size} byte buffer '${buffer.label || "unlabelled"}'`);
  }
  return { buffer, size: bytes };
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
