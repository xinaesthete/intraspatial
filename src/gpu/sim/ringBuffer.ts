// A ring buffer over *field frames* — the general primitive behind trails, the `feedback`
// delay, and temporal analysis. `RingBuffer` is field-agnostic: it stores the last
// `capacity` frames of a fixed-length float record in a circular buffer, knowing nothing
// about what a frame means. `FieldRing` wraps it with a field's shape/element so it
// records the history of *any* `FieldValue` — a point cloud (→ trails), a grid (→ temporal
// echo / a spacetime volume / a temporal DWT along the new time axis), a matrix (→ the
// history of an adjacency structure). `feedback` is the depth-1 case of this.
//
// Checked access, no `!`; the backing store is a single flat buffer (allocation-light).
import { elementLanes, numCells, SCALAR, type Dtype, type ElementType, type FieldValue, type Shape } from "../graph/handle";

export class RingBuffer {
  readonly frameLength: number;
  readonly capacity: number;
  private readonly buf: Float32Array;
  private cursor = 0; // next slot to write
  private filled = 0; // frames stored (≤ capacity)

  constructor(frameLength: number, capacity: number) {
    if (frameLength < 1) throw new Error("RingBuffer: frameLength must be ≥ 1");
    this.frameLength = frameLength;
    this.capacity = Math.max(1, capacity);
    this.buf = new Float32Array(this.frameLength * this.capacity);
  }

  /** Record one frame (its first `frameLength` values). */
  push(data: ArrayLike<number>): void {
    if (data.length < this.frameLength) {
      throw new Error(`RingBuffer.push: data length ${data.length} < frameLength ${this.frameLength}`);
    }
    const base = this.cursor * this.frameLength;
    if (data instanceof Float32Array) {
      this.buf.set(data.subarray(0, this.frameLength), base);
    } else {
      for (let i = 0; i < this.frameLength; i++) this.buf[base + i] = data[i] ?? 0;
    }
    this.cursor = (this.cursor + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;
  }

  get frames(): number {
    return this.filled;
  }

  /** Frame `k` back from the newest (0 = newest, `frames-1` = oldest). Returns a view into
   *  the ring — valid until the next `push`; copy if you need to retain it. */
  frame(k: number): Float32Array {
    if (k < 0 || k >= this.filled) throw new RangeError(`RingBuffer.frame: ${k} out of range for ${this.filled} frames`);
    const slot = (((this.cursor - 1 - k) % this.capacity) + this.capacity) % this.capacity;
    const o = slot * this.frameLength;
    return this.buf.subarray(o, o + this.frameLength);
  }

  reset(): void {
    this.cursor = 0;
    this.filled = 0;
    this.buf.fill(0);
  }
}

/** A ring buffer of `FieldValue`s — the history of a field. The shape/element are fixed at
 *  construction; each `push` records that field's data, and `frame(k)` reconstructs the
 *  field from `k` ticks ago. This is what makes the ring buffer apply to fields, not just
 *  points. */
export class FieldRing {
  readonly ring: RingBuffer;
  readonly shape: Shape;
  readonly element: ElementType;
  readonly dtype: Dtype;

  constructor(shape: Shape, capacity: number, element: ElementType = SCALAR, dtype: Dtype = "f32") {
    this.shape = shape;
    this.element = element;
    this.dtype = dtype;
    this.ring = new RingBuffer(numCells(shape) * elementLanes(element), capacity);
  }

  push(field: FieldValue): void {
    if (!field.data) throw new Error("FieldRing.push: field has no data");
    this.ring.push(field.data);
  }

  get frames(): number {
    return this.ring.frames;
  }

  /** The field `k` ticks back (0 = newest) as a fresh `FieldValue`. */
  frame(k: number): FieldValue {
    return { shape: this.shape, dtype: this.dtype, element: this.element, data: Float32Array.from(this.ring.frame(k)) };
  }
}
