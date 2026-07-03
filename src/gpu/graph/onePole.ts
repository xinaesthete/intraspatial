// A one-pole (exponential-lag) low-pass filter over field frames — the general primitive for easing
// a value/field toward a moving target. `OnePole` is field-agnostic: it holds one smoothed float
// record and relaxes it toward each pushed target by y += α·(x − y), with α = 1 − exp(−dt/τ) from a
// time constant τ — so the response is frame-rate independent (equal wall-clock, equal smoothing).
// `FieldOnePole` wraps it with a field's shape/element to smooth any `FieldValue`: a param vector
// (→ eased TraitSpace transitions in the dancer), a point cloud, a grid, a matrix. The sibling of
// `RingBuffer` (history) — this keeps a single relaxing state, not a window. Checked access, no `!`.
import { type Dtype, type ElementType, elementLanes, type FieldValue, numCells, SCALAR, type Shape } from "./handle";
import type { MemoryReporting } from "./memory";

export interface OnePoleOptions {
  /** Time constant τ (same units as the `dt` passed to `push`; larger = slower/smoother). */
  tau: number;
  /** Initial state; when given the filter starts primed at these values (no snap on first push). */
  initial?: ArrayLike<number>;
  /** When not `initial`-primed, whether the first push jumps straight to the target (default true)
   *  rather than ramping up from zero. */
  snapFirst?: boolean;
}

export class OnePole implements MemoryReporting {
  readonly frameLength: number;
  /** Time constant; mutable so the smoothing can be retuned live. */
  tau: number;
  private readonly y: Float32Array;
  private readonly snapFirst: boolean;
  private primed: boolean;

  constructor(frameLength: number, opts: OnePoleOptions) {
    if (frameLength < 1) throw new Error("OnePole: frameLength must be ≥ 1");
    if (opts.tau <= 0) throw new Error("OnePole: tau must be > 0");
    this.frameLength = frameLength;
    this.tau = opts.tau;
    this.snapFirst = opts.snapFirst ?? true;
    this.y = new Float32Array(frameLength);
    if (opts.initial) {
      if (opts.initial.length < frameLength) {
        throw new Error(`OnePole: initial length ${opts.initial.length} < frameLength ${frameLength}`);
      }
      for (let i = 0; i < frameLength; i++) this.y[i] = opts.initial[i] ?? 0;
      this.primed = true;
    } else {
      this.primed = false;
    }
  }

  /** Ease toward `target` over `dt` of time; returns the smoothed state (a view — copy to retain).
   *  `α = 1 − exp(−dt/τ)`, so equal wall-clock steps give equal smoothing regardless of frame rate,
   *  and one step of `dt=2` equals two of `dt=1` (the filter composes). */
  push(target: ArrayLike<number>, dt = 1): Float32Array {
    if (target.length < this.frameLength) {
      throw new Error(`OnePole.push: target length ${target.length} < frameLength ${this.frameLength}`);
    }
    if (!this.primed) {
      this.primed = true;
      if (this.snapFirst) {
        for (let i = 0; i < this.frameLength; i++) this.y[i] = target[i] ?? 0;
        return this.y;
      }
    }
    const a = 1 - Math.exp(-Math.max(dt, 0) / this.tau);
    for (let i = 0; i < this.frameLength; i++) {
      const yi = this.y[i] ?? 0;
      this.y[i] = yi + a * ((target[i] ?? 0) - yi);
    }
    return this.y;
  }

  /** The current smoothed state (a view — copy to retain). */
  value(): Float32Array {
    return this.y;
  }

  /** Jump straight to `to`, or (with no argument) re-arm so the next push snaps to its target. */
  reset(to?: ArrayLike<number>): void {
    if (to) {
      if (to.length < this.frameLength) throw new Error(`OnePole.reset: length ${to.length} < frameLength ${this.frameLength}`);
      for (let i = 0; i < this.frameLength; i++) this.y[i] = to[i] ?? 0;
      this.primed = true;
    } else {
      this.primed = false;
    }
  }

  get byteLength(): number {
    return this.y.byteLength;
  }
}

/** Smooth any `FieldValue` toward a target field with a one-pole lag (per-lane exponential ease) —
 *  the field-typed wrapper, like `FieldRing` over `RingBuffer`. */
export class FieldOnePole implements MemoryReporting {
  readonly filter: OnePole;
  readonly shape: Shape;
  readonly element: ElementType;
  readonly dtype: Dtype;

  constructor(shape: Shape, opts: OnePoleOptions, element: ElementType = SCALAR, dtype: Dtype = "f32") {
    this.shape = shape;
    this.element = element;
    this.dtype = dtype;
    this.filter = new OnePole(numCells(shape) * elementLanes(element), opts);
  }

  /** Ease toward `field` and return the smoothed field (a fresh `FieldValue`). */
  push(field: FieldValue, dt = 1): FieldValue {
    if (!field.data) throw new Error("FieldOnePole.push: field has no data");
    return this.wrap(Float32Array.from(this.filter.push(field.data, dt)));
  }

  /** The current smoothed field (a fresh `FieldValue`). */
  value(): FieldValue {
    return this.wrap(Float32Array.from(this.filter.value()));
  }

  private wrap(data: Float32Array): FieldValue {
    return { shape: this.shape, dtype: this.dtype, element: this.element, data };
  }

  get byteLength(): number {
    return this.filter.byteLength;
  }
}
