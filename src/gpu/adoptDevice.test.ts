// adoptDevice: a renderer's device becomes the one getDevice() returns, so resources can be shared.
import { afterEach, describe, expect, it, vi } from "vitest";
import { adoptDevice, getDevice, releaseDevice } from "./device";

/** Just the members adoptDevice and releaseDevice touch; identity is what the tests check. */
type FakeDevice = Pick<GPUDevice, "features" | "addEventListener" | "destroy" | "label">;
const fakeDevice = (features: string[]): GPUDevice => {
  const fake: FakeDevice = {
    features: new Set(features),
    addEventListener: () => {},
    destroy() {
      fake.label = "destroyed";
    },
    label: "live",
  };
  return fake as GPUDevice;
};

describe("adoptDevice", () => {
  afterEach(() => releaseDevice());

  it("makes getDevice() return the adopted device, and is idempotent for it", async () => {
    const d = fakeDevice(["float32-blendable"]);
    adoptDevice(d);
    adoptDevice(d);
    expect(await getDevice()).toBe(d);
  });

  it("refuses a second, different device — resources cannot cross devices", () => {
    adoptDevice(fakeDevice(["float32-blendable"]));
    expect(() => adoptDevice(fakeDevice(["float32-blendable"]))).toThrow(/already exists/);
  });

  it("warns when the device cannot blend r32float", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    adoptDevice(fakeDevice([]));
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/float32-blendable/));
    warn.mockRestore();
  });

  it("releases without destroying a borrowed device", async () => {
    const d = fakeDevice(["float32-blendable"]);
    adoptDevice(d);
    await releaseDevice();
    expect(d.label).toBe("live");
  });
});
