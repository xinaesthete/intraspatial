// Readback (download) instrumentation for the operation graph.
//
// WHY THIS EXISTS. `docs/gpu-resource-sync.md` invariant 4 — "Boundary-only transfer:
// `upload` at sources, `download` at sinks; interior edges stay on-GPU" — is a
// **graph-level** property, while the repo's test discipline is **per-op** (every node has
// a `cpuGolden` the GPU must match). So no existing test can fail because an interior edge
// round-tripped to the host: each op passes its golden either way. The invariant was
// unenforceable by construction, and the executor has been violating it on every edge.
// This module makes it falsifiable by counting the downloads a pull actually performs.
//
// HOW. `GPUBuffer.prototype.mapAsync` is the universal chokepoint — every readback path
// (TypeGPU `.read()`, a raw `mapAsync`, the legacy `*Gpu` helpers) bottoms out there — so
// counting needs **no changes to any op**. Uploads via `queue.writeBuffer` are not maps and
// are not counted; `GPUMapMode.WRITE` maps are counted separately as staging uploads.
//
// Requires a device to exist first: `getDevice()` installs the `webgpu` globals that make
// `GPUBuffer` reachable (see src/gpu/device.ts).

export interface ReadbackStats {
  /** Number of `mapAsync(READ)` calls — i.e. GPU→host downloads. */
  downloads: number;
  /** Bytes mapped for reading. */
  downloadBytes: number;
  /** Number of `mapAsync(WRITE)` calls — staging uploads (rare; most uploads use writeBuffer). */
  uploadMaps: number;
}

type MapAsyncFn = (...args: unknown[]) => Promise<void>;

let installed = false;
let original: MapAsyncFn | undefined;
let downloads = 0;
let downloadBytes = 0;
let uploadMaps = 0;

function protoOf(): Record<string, unknown> {
  const G = globalThis as unknown as { GPUBuffer?: { prototype: Record<string, unknown> } };
  const proto = G.GPUBuffer?.prototype;
  if (!proto) {
    throw new Error("instrument: GPUBuffer is unavailable — acquire a device first (getDevice() installs the webgpu globals).");
  }
  return proto;
}

/** Patch `GPUBuffer.prototype.mapAsync` to count downloads. Idempotent. Returns an uninstall fn. */
export function installReadbackCounter(): () => void {
  if (installed) return uninstallReadbackCounter;
  const proto = protoOf();
  original = proto.mapAsync as MapAsyncFn;
  const orig = original;
  const READ = (globalThis as { GPUMapMode?: { READ: number } }).GPUMapMode?.READ ?? 1;

  proto.mapAsync = function (this: GPUBuffer, ...args: unknown[]) {
    const mode = Number(args[0] ?? 0);
    const offset = args[1] as number | undefined;
    const size = args[2] as number | undefined;
    if ((mode & READ) !== 0) {
      downloads++;
      downloadBytes += size ?? Math.max(0, this.size - (offset ?? 0));
    } else {
      uploadMaps++;
    }
    // Preserve exact arity — the Dawn binding is sensitive to trailing undefineds.
    return orig.apply(this, args);
  } as MapAsyncFn;

  installed = true;
  return uninstallReadbackCounter;
}

/** Restore the original `mapAsync`. Safe to call when not installed. */
export function uninstallReadbackCounter(): void {
  if (!installed || !original) return;
  protoOf().mapAsync = original;
  installed = false;
  original = undefined;
}

export function resetReadbacks(): void {
  downloads = 0;
  downloadBytes = 0;
  uploadMaps = 0;
}

export function readbackStats(): ReadbackStats {
  return { downloads, downloadBytes, uploadMaps };
}

/** Reset, run `fn`, and report the readbacks it performed. Counter must be installed. */
export async function measureReadbacks<T>(fn: () => Promise<T>): Promise<{ result: T; stats: ReadbackStats }> {
  resetReadbacks();
  const result = await fn();
  return { result, stats: readbackStats() };
}
