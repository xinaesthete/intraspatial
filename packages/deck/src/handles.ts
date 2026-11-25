// Binding GPU buffers that something else owns (a simulation, a compute kernel) into a deck layer.
// luma wraps a raw `GPUBuffer` without copying it, and never destroys a handle it was given, so the
// wrapper is ours to free and the buffer is its owner's.

import { Buffer, type Device } from "@luma.gl/core";

/** A buffer, or a function giving the buffer to draw this frame. A simulation that ping-pongs two
 *  buffers passes the function, so the layer never draws the one being written. */
export type BufferSource = GPUBuffer | (() => GPUBuffer | null | undefined) | null | undefined;

export const resolveBuffer = (source: BufferSource): GPUBuffer | null => (typeof source === "function" ? source() : source) ?? null;

/** luma wrappers for the last few raw buffers seen, so a ping-pong pair is wrapped once each. */
export class HandleCache {
  private readonly wrapped = new Map<GPUBuffer, Buffer>();

  constructor(
    private readonly device: Device,
    private readonly keep = 4,
  ) {}

  wrap(handle: GPUBuffer): Buffer {
    const found = this.wrapped.get(handle);
    if (found) return found;
    if (this.wrapped.size >= this.keep) {
      // Oldest first: a Map iterates in insertion order.
      const [oldest, wrapper] = this.wrapped.entries().next().value!;
      wrapper.destroy();
      this.wrapped.delete(oldest);
    }
    const wrapper = this.device.createBuffer({
      id: `${handle.label || "buffer"}:wrap`,
      handle,
      byteLength: handle.size,
      usage: Buffer.STORAGE,
    });
    this.wrapped.set(handle, wrapper);
    return wrapper;
  }

  /** Free the wrappers. The raw buffers are untouched. */
  destroy(): void {
    for (const wrapper of this.wrapped.values()) wrapper.destroy();
    this.wrapped.clear();
  }
}
