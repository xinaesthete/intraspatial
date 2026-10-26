import { describe, expect, it } from "vite-plus/test";
import { releaseScratchTextures, scratchTexture, scratchTextureBytes } from "./scratchTexture";

// Slots are per device. The GPU suite has one device, so this uses stand-ins: `scratchTexture` only
// ever calls `createTexture` on a device and `destroy` on a texture.

interface Fake {
  device: GPUDevice;
  /** Textures this device made, and which of them have been destroyed. */
  made: GPUTexture[];
  destroyed: Set<GPUTexture>;
}

function fakeDevice(): Fake {
  const made: GPUTexture[] = [];
  const destroyed = new Set<GPUTexture>();
  // An empty stand-in with the one method assigned, rather than a literal cast to `GPUDevice`, which
  // TypeScript rejects without a double cast.
  const device = {} as GPUDevice;
  device.createTexture = (): GPUTexture => {
    const tex = {
      destroy: () => {
        destroyed.add(tex);
      },
    } as GPUTexture;
    made.push(tex);
    return tex;
  };
  return { device, made, destroyed };
}

// usage is passed through untouched, so any number will do; `GPUTextureUsage` only exists once Dawn
// has installed its globals, which a CPU test never does.
const desc = (width: number, height: number) => ({ width, height, format: "r32float" as const, usage: 1 });

describe("scratchTexture", () => {
  it("keeps each device's slot for a key apart, so neither destroys the other's", () => {
    const a = fakeDevice();
    const b = fakeDevice();
    const ta = scratchTexture(a.device, "test:shared", desc(32, 32));
    const tb = scratchTexture(b.device, "test:shared", desc(32, 32));
    // In turn, as two Decks on two devices would: both are reused, nothing is replaced.
    expect(scratchTexture(a.device, "test:shared", desc(32, 32))).toBe(ta);
    expect(scratchTexture(b.device, "test:shared", desc(32, 32))).toBe(tb);
    expect(a.made).toHaveLength(1);
    expect(b.made).toHaveLength(1);
    expect(a.destroyed.size + b.destroyed.size).toBe(0);
    releaseScratchTextures("test:");
  });

  it("replaces a slot that does not fit on its own device only", () => {
    const a = fakeDevice();
    const b = fakeDevice();
    const ta = scratchTexture(a.device, "test:grow", desc(16, 16));
    const tb = scratchTexture(b.device, "test:grow", desc(16, 16));
    scratchTexture(a.device, "test:grow", desc(64, 64));
    expect(a.destroyed.has(ta)).toBe(true);
    expect(b.destroyed.has(tb)).toBe(false);
    expect(scratchTexture(b.device, "test:grow", desc(16, 16))).toBe(tb);
    releaseScratchTextures("test:");
  });

  it("counts and releases slots on every device", () => {
    const a = fakeDevice();
    const b = fakeDevice();
    const before = scratchTextureBytes();
    scratchTexture(a.device, "test:count", desc(8, 8));
    scratchTexture(b.device, "test:count", desc(16, 16));
    expect(scratchTextureBytes() - before).toBe((8 * 8 + 16 * 16) * 4);
    expect(releaseScratchTextures("test:count")).toBe((8 * 8 + 16 * 16) * 4);
    expect(a.destroyed.size).toBe(1);
    expect(b.destroyed.size).toBe(1);
    expect(scratchTextureBytes()).toBe(before);
  });
});
