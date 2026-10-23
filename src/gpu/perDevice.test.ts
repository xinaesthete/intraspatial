import { describe, expect, it } from "vitest";
import { deviceScoped, perDevice } from "./device";

// The property the per-device kernel caches rest on, which no GPU test here can show: the GPU suite
// has exactly one device (a second Dawn instance is off limits), so "state made on device A is never
// handed to device B" is only checkable with stand-in devices. Nothing below touches the GPU — the
// memo only ever uses a device as a WeakMap key.
const fakeDevice = (): GPUDevice => ({}) as GPUDevice;

describe("perDevice", () => {
  it("builds once per device and never hands one device's state to another", () => {
    const a = fakeDevice();
    const b = fakeDevice();
    const built: GPUDevice[] = [];
    const state = perDevice((device) => {
      built.push(device);
      return { owner: device };
    });

    const sa = state(a);
    const sb = state(b);
    expect(state(a)).toBe(sa);
    expect(state(b)).toBe(sb);
    expect(sa).not.toBe(sb);
    expect(sa.owner).toBe(a);
    expect(sb.owner).toBe(b);
    expect(built).toEqual([a, b]);
  });

  it("caches a falsy value rather than rebuilding it", () => {
    let calls = 0;
    const state = perDevice(() => {
      calls++;
      return 0;
    });
    const a = fakeDevice();
    state(a);
    state(a);
    expect(calls).toBe(1);
  });
});

describe("deviceScoped", () => {
  it("shares one build between concurrent first callers on a device", async () => {
    const a = fakeDevice();
    let calls = 0;
    const ctx = deviceScoped(async (device) => {
      calls++;
      await Promise.resolve();
      return { owner: device };
    });
    const [x, y] = await Promise.all([ctx(a), ctx(a)]);
    expect(x).toBe(y);
    expect(calls).toBe(1);
  });

  it("keeps an explicit device's state apart from another's", async () => {
    const a = fakeDevice();
    const b = fakeDevice();
    const ctx = deviceScoped(async (device) => ({ owner: device }));
    const [sa, sb] = await Promise.all([ctx(a), ctx(b)]);
    expect(sa.owner).toBe(a);
    expect(sb.owner).toBe(b);
  });
});
