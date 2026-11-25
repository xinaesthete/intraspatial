// The wrapper cache must hand a ping-pong pair the same two wrappers for ever, and must never
// destroy a buffer it was given: that buffer is the simulation's.
import type { Device } from "@luma.gl/core";
import { describe, expect, it } from "vite-plus/test";
import { HandleCache, resolveBuffer } from "./handles";

type Raw = { size: number; destroyed: boolean };
type Wrapper = { handle: Raw; destroyed: boolean; destroy(): void };

function fakeDevice() {
  const wrappers: Wrapper[] = [];
  const device = {
    createBuffer({ handle }: { handle: Raw }) {
      const w: Wrapper = {
        handle,
        destroyed: false,
        destroy() {
          w.destroyed = true;
        },
      };
      wrappers.push(w);
      return w;
    },
  };
  // The cache only calls createBuffer; the rest of luma's Device is not needed here.
  return { device: device as object as Device, wrappers };
}

const raw = (): GPUBuffer => ({ size: 64, destroyed: false }) as object as GPUBuffer;

describe("HandleCache", () => {
  it("wraps each buffer of a ping-pong pair once", () => {
    const { device, wrappers } = fakeDevice();
    const cache = new HandleCache(device);
    const a = raw();
    const b = raw();
    const seen = new Set<unknown>();
    for (let frame = 0; frame < 20; frame++) seen.add(cache.wrap(frame % 2 ? a : b));
    expect(seen.size).toBe(2);
    expect(wrappers).toHaveLength(2);
  });

  it("drops the oldest wrapper when full, and on destroy frees wrappers only", () => {
    const { device, wrappers } = fakeDevice();
    const cache = new HandleCache(device, 2);
    const buffers = [raw(), raw(), raw()];
    for (const b of buffers) cache.wrap(b);
    expect(wrappers.map((w) => w.destroyed)).toEqual([true, false, false]);
    cache.destroy();
    expect(wrappers.every((w) => w.destroyed)).toBe(true);
    expect(buffers.some((b) => (b as object as Raw).destroyed)).toBe(false);
  });
});

describe("resolveBuffer", () => {
  it("asks a function each time and passes a buffer through", () => {
    const a = raw();
    const b = raw();
    let current = a;
    const source = () => current;
    expect(resolveBuffer(source)).toBe(a);
    current = b;
    expect(resolveBuffer(source)).toBe(b);
    expect(resolveBuffer(a)).toBe(a);
    expect(resolveBuffer(null)).toBeNull();
    expect(resolveBuffer(() => undefined)).toBeNull();
  });
});
