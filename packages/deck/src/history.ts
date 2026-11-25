// A ring of past snapshots of a position buffer, kept on the GPU: where every particle has been.
// A snapshot is a buffer-to-buffer copy, so nothing comes back to the host. `TraceLayer` draws it.

export interface PositionHistoryOptions {
  /** Particles per snapshot. */
  readonly count: number;
  /** Snapshots kept. */
  readonly capacity: number;
  /** f32 per particle in the source buffer. */
  readonly stride?: number;
  readonly label?: string;
}

/** Which slot of a ring holds the snapshot `age` steps back (0 is the newest). */
export function traceSlot(head: number, age: number, capacity: number): number {
  return (((head - 1 - age) % capacity) + capacity) % capacity;
}

export class PositionHistory {
  /** `capacity` snapshots of `count * stride` f32, slot after slot. */
  readonly buffer: GPUBuffer;
  readonly count: number;
  readonly capacity: number;
  readonly stride: number;
  /** The slot the next snapshot goes into; the newest is the one before it. */
  head = 0;
  /** Snapshots taken so far, up to `capacity`. */
  filled = 0;
  private readonly label: string;

  constructor(
    private readonly device: GPUDevice,
    o: PositionHistoryOptions,
  ) {
    this.count = o.count;
    this.capacity = Math.max(2, Math.floor(o.capacity));
    this.stride = o.stride ?? 4;
    this.label = o.label ?? "history";
    this.buffer = device.createBuffer({
      label: `${this.label}:ring`,
      size: this.capacity * this.slotBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private get slotBytes(): number {
    return this.count * this.stride * 4;
  }

  /** Copy the first `count` particles of `src` (which needs `COPY_SRC`) in as the newest snapshot. */
  capture(src: GPUBuffer): void {
    const enc = this.device.createCommandEncoder({ label: `${this.label}:capture` });
    enc.copyBufferToBuffer(src, 0, this.buffer, this.head * this.slotBytes, this.slotBytes);
    this.device.queue.submit([enc.finish()]);
    this.head = (this.head + 1) % this.capacity;
    this.filled = Math.min(this.filled + 1, this.capacity);
  }

  /** Forget the snapshots (after a restart). */
  clear(): void {
    this.head = 0;
    this.filled = 0;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
