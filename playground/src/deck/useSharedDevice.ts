// One WebGPU device for deck.gl and this library, so a deck layer can bind buffers our compute left
// on the GPU — no readback, no copy across devices.
//
// Deck creates the device (luma 9.4 cannot attach an existing one) and this library adopts it
// before any GPU work of its own; `ready` says when that has happened, so callers hold their compute
// until then. The library keeps one device per page, so this suits a page with one Deck.

import type { Device } from "@luma.gl/core";
import { webgpuAdapter } from "@luma.gl/webgpu";
import { useCallback, useState } from "react";
import { adoptDevice } from "../../../src/gpu/device";

/** Module-level: a new object each render would make deck rebuild its device. `float32-blendable`
 *  is what the Gram splat needs to blend into r32float. */
export const SHARED_DEVICE_PROPS = {
  type: "webgpu" as const,
  adapters: [webgpuAdapter],
  optionalFeatures: ["float32-blendable" as const],
};

export interface SharedDevice {
  /** True once deck's device is this library's: start GPU work after this, not before. */
  readonly ready: boolean;
  /** A line for a status bar: shared or not, and whether the needed feature was granted. */
  readonly status: string;
  /** For `deckProps.onDeviceInitialized`. Stable. */
  readonly onDeviceInitialized: (device: Device) => void;
}

export function useSharedDevice(): SharedDevice {
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("waiting for deck's device…");

  const onDeviceInitialized = useCallback((device: Device) => {
    const handle = device.handle;
    if (device.type !== "webgpu" || !(handle instanceof GPUDevice)) {
      setStatus(`deck chose ${device.type}; this page needs WebGPU`);
      return;
    }
    try {
      adoptDevice(handle);
    } catch (e) {
      // A second Deck brings a second device, and this library keeps one per page. In dev that is
      // an HMR remount, so start over.
      if (import.meta.hot) location.reload();
      setStatus(`not shared — ${(e as Error).message}`);
      return;
    }
    setStatus(
      `WebGPU, shared with deck · float32-blendable ${handle.features.has("float32-blendable") ? "✓" : "✗ (the Gram will be blank)"}`,
    );
    setReady(true);
  }, []);

  return { ready, status, onDeviceInitialized };
}
