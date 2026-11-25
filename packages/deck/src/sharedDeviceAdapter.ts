// Lets deck draw on a GPU device that already exists, so compute that ran on that device can be
// drawn without copying: a luma adapter that wraps the device it is given instead of making one.
//
// luma 9.4's own `webgpuAdapter.attach()` is not implemented, but its `create()` does only two
// things: ask the browser for a device, and wrap it in a `WebGPUDevice`. This does the second
// with a device from the caller. Pass it to Deck as `deviceProps: { type: "webgpu", adapters: [it] }`.
//
// Several Decks can share the one device: each gets its own luma wrapper and canvas. A wrapper
// made here never destroys the device, since it is not luma's to destroy.

import { Adapter, type Device, type DeviceProps } from "@luma.gl/core";
import { WebGPUDevice } from "@luma.gl/webgpu";

export interface SharedGpu {
  readonly device: GPUDevice;
  /** The adapter the device came from: luma reads its description, and asks nothing else of it. */
  readonly adapter: GPUAdapter;
}

/** A luma device over a `GPUDevice` someone else owns. */
class BorrowedWebGPUDevice extends WebGPUDevice {
  override destroy(): void {
    // Deck calls this when it goes away. The device stays: others are using it.
    this.commandEncoder?.destroy();
  }
}

class SharedDeviceAdapter extends Adapter {
  readonly type = "webgpu";
  constructor(private readonly gpu: () => Promise<SharedGpu>) {
    super();
  }
  isSupported(): boolean {
    return typeof navigator !== "undefined" && Boolean(navigator.gpu);
  }
  isDeviceHandle(handle: unknown): boolean {
    return typeof GPUDevice !== "undefined" && handle instanceof GPUDevice;
  }
  async create(props: DeviceProps): Promise<Device> {
    const { device, adapter } = await this.gpu();
    return new BorrowedWebGPUDevice({ ...props, featureLevel: "core" }, device, adapter, adapter.info);
  }
  attach(): Promise<Device> {
    return Promise.reject(new Error("sharedDeviceAdapter: use create(); the device is the one given when the adapter was made"));
  }
}

/** An adapter for Deck's `deviceProps.adapters` that draws on `gpu()`'s device. `gpu` is asked once
 *  per Deck and should return the same device every time. */
export function sharedDeviceAdapter(gpu: () => Promise<SharedGpu>): Adapter {
  return new SharedDeviceAdapter(gpu);
}
