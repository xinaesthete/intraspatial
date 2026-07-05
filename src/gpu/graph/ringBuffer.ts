// A ring buffer over *field frames* — the general primitive behind trails, the `feedback`
// delay, and temporal analysis. `RingBuffer` is field-agnostic: it stores the last
// `capacity` frames of a fixed-length float record in a circular buffer, knowing nothing
// about what a frame means. `FieldRing` wraps it with a field's shape/element so it records
// the history of *any* `FieldValue` — a point cloud (→ trails), a grid (→ temporal echo / a
// spacetime volume / a temporal DWT along the new time axis), a matrix (→ the history of an
// adjacency structure). `feedback` is the depth-1 case of this.
//
// The history can be *sampled continuously* (`sample(k)` interpolates between frames), so a
// trail, a delay, or a resampler can read at sub-frame resolution. Everything reports its
// resident `byteLength` (MemoryReporting) — a delay keeps k copies of its field, so its
// footprint matters. Checked access, no `!`.
import { type Dtype, type ElementType, elementLanes, type FieldValue, numCells, SCALAR, type Shape } from "./handle";
import type { MemoryReporting } from "./memory";

export class RingBuffer implements MemoryReporting {
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

  /** Resident bytes of the backing store. */
  get byteLength(): number {
    return this.buf.byteLength;
  }

  /** Frame `k` back from the newest (0 = newest, `frames-1` = oldest). Returns a view into
   *  the ring — valid until the next `push`; copy if you need to retain it. */
  frame(k: number): Float32Array {
    if (k < 0 || k >= this.filled) throw new RangeError(`RingBuffer.frame: ${k} out of range for ${this.filled} frames`);
    const slot = (((this.cursor - 1 - k) % this.capacity) + this.capacity) % this.capacity;
    const o = slot * this.frameLength;
    return this.buf.subarray(o, o + this.frameLength);
  }

  /** Sample the history at a *fractional* age `k` (0 = newest .. frames-1 = oldest),
   *  linearly interpolating between the two bracketing frames. Clamps to the valid range.
   *  Writes into `out` (allocated if omitted) and returns it. */
  sample(k: number, out?: Float32Array): Float32Array {
    if (this.filled < 1) throw new RangeError("RingBuffer.sample: empty buffer");
    const dst = out ?? new Float32Array(this.frameLength);
    const maxK = this.filled - 1;
    const kk = k < 0 ? 0 : k > maxK ? maxK : k;
    const lo = Math.floor(kk);
    const hi = Math.min(maxK, lo + 1);
    const t = kk - lo;
    const a = this.frame(lo);
    const b = this.frame(hi);
    for (let i = 0; i < this.frameLength; i++) {
      const av = a[i] ?? 0;
      const bv = b[i] ?? 0;
      dst[i] = av + (bv - av) * t;
    }
    return dst;
  }

  /** Sample the history at a *fractional* age `k`, using **Catmull-Rom cubic** interpolation
   *  through the four frames bracketing `k` (a C1-continuous curve that passes through every
   *  stored frame — smoother than `sample`'s piecewise-linear read for trails/resampling).
   *  Ends are handled by clamping the neighbour indices. Writes into `out` (allocated if
   *  omitted) and returns it. */
  sampleCubic(k: number, out?: Float32Array): Float32Array {
    if (this.filled < 1) throw new RangeError("RingBuffer.sampleCubic: empty buffer");
    const dst = out ?? new Float32Array(this.frameLength);
    const maxK = this.filled - 1;
    const kk = k < 0 ? 0 : k > maxK ? maxK : k;
    const lo = Math.floor(kk);
    const t = kk - lo;
    const clamp = (j: number): number => (j < 0 ? 0 : j > maxK ? maxK : j);
    // p1 = frame(lo) (newer knot), p2 = frame(lo+1) (older knot); p0/p3 the outer neighbours.
    const p0 = this.frame(clamp(lo - 1));
    const p1 = this.frame(clamp(lo));
    const p2 = this.frame(clamp(lo + 1));
    const p3 = this.frame(clamp(lo + 2));
    const t2 = t * t;
    const t3 = t2 * t;
    for (let i = 0; i < this.frameLength; i++) {
      const a = p0[i] ?? 0,
        b = p1[i] ?? 0,
        c = p2[i] ?? 0,
        d = p3[i] ?? 0;
      dst[i] = 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
    }
    return dst;
  }

  reset(): void {
    this.cursor = 0;
    this.filled = 0;
    this.buf.fill(0);
  }
}

/** A ring buffer of `FieldValue`s — the history of a field. The shape/element are fixed at
 *  construction; each `push` records that field's data, `frame(k)` reconstructs an exact
 *  past field, and `sample(k)` reads a continuously-interpolated one. This is what makes the
 *  ring buffer apply to fields, not just points. (Component-wise lerp; a quaternion field
 *  would want slerp for a true rotation blend — a future refinement.) */
export class FieldRing implements MemoryReporting {
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

  get byteLength(): number {
    return this.ring.byteLength;
  }

  private wrap(data: Float32Array): FieldValue {
    return { shape: this.shape, dtype: this.dtype, element: this.element, data };
  }

  /** The field `k` ticks back (0 = newest) as a fresh `FieldValue`. */
  frame(k: number): FieldValue {
    return this.wrap(Float32Array.from(this.ring.frame(k)));
  }

  /** The field at a fractional age `k`, linearly interpolated between frames. */
  sample(k: number): FieldValue {
    return this.wrap(this.ring.sample(k));
  }

  /** The field at a fractional age `k`, Catmull-Rom cubic interpolated (C1-smooth). */
  sampleCubic(k: number): FieldValue {
    return this.wrap(this.ring.sampleCubic(k));
  }
}
