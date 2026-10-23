import { describe, expect, it } from "vitest";
import { getDevice } from "../src/gpu/device";
import { exclusiveScanGpu } from "../src/gpu/scan/prefixSum";
import { releaseScratchTextures, scratchTexture, scratchTextureBytes } from "../src/gpu/scratchTexture";

// Does destroying a GPU resource mid-process actually segfault Dawn-on-Node?
//
// Twenty-one files in `src/` assert that it does, citing ADR-0002/0003. But ADR-0003's own
// results table marks two of the three symptoms it was inferred from as **gone**, and
// `device.ts` later found the real cause of the random crashes: the Dawn **Instance** was being
// garbage-collected out from under a live device, so the failures were GC-timing dependent. The
// raw-`mapAsync` crash, blamed on pooled buffers at the time, is explicitly attributed to that
// bug in `prefixSum.ts` — which now uses raw `mapAsync` deliberately.
//
// So the rule may be a precaution that outlived its reason. This probe is the cheap way to find
// out, and a regression test either way: the claimed failure is at TEARDOWN, so it destroys
// resources mid-process, keeps doing real GPU work afterwards, and lets the worker exit. A clean
// exit is the assertion.

describe("destroying GPU resources mid-process", () => {
  it("survives destroying many buffers and textures, with real work after", async () => {
    const device = await getDevice();

    // Enough to be well past "enough GPU work" by the old threshold, and varied in size so none
    // of it is a single cached allocation.
    for (let round = 0; round < 4; round++) {
      const buffers: GPUBuffer[] = [];
      const textures: GPUTexture[] = [];
      for (let i = 0; i < 32; i++) {
        buffers.push(
          device.createBuffer({
            size: 1024 * (i + 1),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          }),
        );
        textures.push(
          device.createTexture({
            size: { width: 16 + i, height: 16 + i },
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
          }),
        );
      }
      // Touch them, so they are not destroyed straight after creation with no native work done.
      const enc = device.createCommandEncoder();
      for (const b of buffers) enc.clearBuffer(b);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      for (const b of buffers) b.destroy();
      for (const t of textures) t.destroy();
    }

    // Real work AFTER the destruction — a wrong answer here would mean destruction had corrupted
    // something, which is a different failure from the segfault and worth catching too.
    const n = 4096;
    const { scan, total } = await exclusiveScanGpu(new Uint32Array(n).fill(1));
    expect(total).toBe(n);
    expect(scan[n - 1]).toBe(n - 1);
  });

  it("survives destroying a buffer that was just read back", async () => {
    // The specific pattern ADR-0003 blamed: a MAP_READ buffer, mapped and then released. The
    // Instance-lifetime fix is what made this safe; destroying it is the extra step.
    const device = await getDevice();
    for (let i = 0; i < 16; i++) {
      const src = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const staging = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(src, 0, new Uint32Array([i, i + 1, i + 2, i + 3]));
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, 256);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const got = new Uint32Array(staging.getMappedRange().slice(0, 16))[0];
      staging.unmap();
      expect(got).toBe(i);
      staging.destroy();
      src.destroy();
    }

    const { total } = await exclusiveScanGpu(new Uint32Array(1024).fill(2));
    expect(total).toBe(2048);
  });

  it("replaces a scratch target instead of leaking it, and hands the memory back", async () => {
    const device = await getDevice();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const before = scratchTextureBytes();

    const a = scratchTexture(device, "probe:scratch", { width: 256, height: 256, format: "r32float", usage });
    expect(scratchTextureBytes() - before).toBe(256 * 256 * 4);

    // Within the 2x reuse window: same texture back, no new allocation.
    const b = scratchTexture(device, "probe:scratch", { width: 200, height: 200, format: "r32float", usage });
    expect(b).toBe(a);
    expect(scratchTextureBytes() - before).toBe(256 * 256 * 4);

    // Past it: a new one, and crucially the old is DESTROYED rather than dropped for a collector
    // that cannot see the 256 KiB behind the handle. The accounting is flat, not cumulative.
    const c = scratchTexture(device, "probe:scratch", { width: 512, height: 512, format: "r32float", usage });
    expect(c).not.toBe(a);
    expect(scratchTextureBytes() - before).toBe(512 * 512 * 4);

    // A smaller request outside the window shrinks it — growth is not one-way.
    scratchTexture(device, "probe:scratch", { width: 64, height: 64, format: "r32float", usage });
    expect(scratchTextureBytes() - before).toBe(64 * 64 * 4);

    expect(releaseScratchTextures("probe:")).toBe(64 * 64 * 4);
    expect(scratchTextureBytes()).toBe(before);

    // Still healthy afterwards.
    const { total } = await exclusiveScanGpu(new Uint32Array(512).fill(1));
    expect(total).toBe(512);
  });
});
