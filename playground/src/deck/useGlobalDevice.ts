// One GPU device for the whole page: this library's own, which every Deck on the page draws on.
// So anything the library computes (a simulation run as graph ops, a density field) can be bound
// by any deck layer with no copy, and several Decks can show the same buffers.
//
// The device is made here, before anything else asks for one, with everything the machine's
// adapter offers (the default 128 MiB storage binding is too small for a whole Xenium table), and
// handed to the library with `adoptDevice`. If the library already has a device, that one is used.
// Deck is given `GLOBAL_DEVICE_PROPS`, whose adapter wraps this device instead of making another.

import { sharedDeviceAdapter, type SharedGpu } from "@intraspatial/deck";
import { useEffect, useState } from "react";
import { adoptDevice, getDevice } from "../../../src/gpu/device";

let shared: Promise<SharedGpu> | undefined;

/** The page's device. The same one on every call. */
function globalGpu(): Promise<SharedGpu> {
  shared ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("This page needs WebGPU, and the browser offers no adapter.");
    const requiredLimits: Record<string, number> = {};
    for (const key in adapter.limits) {
      const value: unknown = adapter.limits[key as keyof GPUSupportedLimits];
      if (typeof value === "number") requiredLimits[key] = value;
    }
    const device = await adapter.requestDevice({
      label: "intraspatial:page",
      requiredFeatures: [...adapter.features] as GPUFeatureName[],
      requiredLimits,
    });
    try {
      adoptDevice(device);
      return { device, adapter };
    } catch {
      // Something used the library before this ran, so the library's device is the page's.
      device.destroy();
      return { device: await getDevice(), adapter };
    }
  })();
  return shared;
}

/** For Deck's `deviceProps`. Module-level: a new object each render would make deck start over. */
export const GLOBAL_DEVICE_PROPS = { type: "webgpu" as const, adapters: [sharedDeviceAdapter(globalGpu)] };

export interface GlobalDevice {
  /** Null until the device exists. It does not wait for a Deck. */
  readonly device: GPUDevice | null;
  /** A line for a status bar. */
  readonly status: string;
}

export function useGlobalDevice(): GlobalDevice {
  const [state, setState] = useState<GlobalDevice>({ device: null, status: "waiting for the GPU…" });
  useEffect(() => {
    let live = true;
    globalGpu().then(
      ({ device }) => live && setState({ device, status: "WebGPU, one device for compute and drawing" }),
      (e: unknown) => live && setState({ device: null, status: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
  }, []);
  return state;
}
