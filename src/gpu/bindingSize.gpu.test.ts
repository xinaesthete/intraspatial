// The two guards that stand between a grow-only buffer pool and a silently wrong answer.
//
// Background: every pool in this repo is grow-only and most double on growth, so a bind group
// written `{ buffer }` binds however big an earlier, larger call made the buffer. Past
// `maxStorageBufferBindingSize` that bind group is invalid, the command buffer is invalid,
// every dispatch does nothing, and the readback returns the PREVIOUS call's contents — with no
// exception and a healthy-looking (in fact faster) timing. Measured 2026-08-01 in
// `streamCompact`: 132 MB of data in a 160 MB pooled buffer reported the previous call's
// answer, 3,913,615 rows, in 7 ms.
//
// This is worse than the other silent-failure modes the repo has recorded, because it does not
// produce zeroes: the non-zero assertion that catches the 65535 workgroup cap, the ~2 s
// watchdog and the 9th storage binding does not catch this one.
import { describe, expect, it } from "vitest";
import { checkBindingSize, getDevice, sized } from "./device";

describe("sized", () => {
  it("binds the requested length, not the buffer's capacity", async () => {
    const device = await getDevice();
    // A pooled buffer that has been grown well past what this call needs.
    const buf = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE, label: "pooled" });
    expect(sized(buf, 256)).toEqual({ buffer: buf, size: 256 });
    buf.destroy();
  });

  it("refuses a length the buffer cannot hold", async () => {
    const device = await getDevice();
    const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE, label: "small" });
    // This is the guard against the OTHER way to get it wrong: hand-computing a binding size
    // that has drifted from the size the buffer was allocated with. WebGPU would reject it as
    // a validation error, i.e. silently.
    expect(() => sized(buf, 512)).toThrow(/not a valid binding length/);
    expect(() => sized(buf, 0)).toThrow(/not a valid binding length/);
    buf.destroy();
  });
});

describe("checkBindingSize", () => {
  it("passes at the limit and throws one byte past it", async () => {
    const device = await getDevice();
    const max = device.limits.maxStorageBufferBindingSize;
    // Boundary is inclusive — a binding of exactly the limit is legal.
    expect(() => checkBindingSize(device, "probe", max)).not.toThrow();
    expect(() => checkBindingSize(device, "probe", max + 1)).toThrow(/maxStorageBufferBindingSize/);
  });

  it("names the caller and the remedy, because the alternative is a silent wrong answer", async () => {
    const device = await getDevice();
    const max = device.limits.maxStorageBufferBindingSize;
    expect(() => checkBindingSize(device, "myKernel: 40M rows", max + 1)).toThrow(/myKernel: 40M rows/);
    expect(() => checkBindingSize(device, "myKernel", max + 1)).toThrow(/getDevice/);
  });

  it("reports the default limit this device was created with", async () => {
    const device = await getDevice();
    // Pinned because the whole audit turns on this number: 2^27, which is also why the Tier-2
    // pool's power-of-two bucketing cannot push a legal size over it. If `getDevice()` ever
    // requests a higher limit (MDV's 35M rows needs one), this is the test that should change
    // and prompt a re-read of `sized`'s note about that coincidence.
    expect(device.limits.maxStorageBufferBindingSize).toBe(134_217_728);
    expect(Math.floor(device.limits.maxStorageBufferBindingSize / 4)).toBe(33_554_432);
  });
});
