// Per-element arithmetic and linear algebra over interleaved lane data (ADR-0004).
//
// A field's `data` stores samples interleaved lane-major: sample `i` of an element
// with `L = elementLanes` lanes occupies `data[i*L .. i*L + L)`. These functions are
// the CPU reference for the element algebra; the eventual WGSL kernels mirror them
// lane-for-lane. They are pure (allocate and return a fresh array) and shape-agnostic
// — the op wrappers attach the shape/element metadata.
//
// Lane conventions: complex = [re, im]; quaternion = [w, x, y, z]; vec = [c0..c{n-1}].
import type { ElementType } from "./handle";
import { elementLabel, elementLanes } from "./handle";

const sameLen = (a: ArrayLike<number>, b: ArrayLike<number>, op: string) => {
  if (a.length !== b.length) throw new Error(`${op}: field length mismatch (${a.length} vs ${b.length})`);
};

/** Lane-wise sum a+b. Linear, so valid for every element (scalar/complex/vec/quat). */
export function addFields(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  sameLen(a, b, "addFields");
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

/** Lane-wise difference a−b. Linear, valid for every element. */
export function subFields(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  sameLen(a, b, "subFields");
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i]! - b[i]!;
  return out;
}

/** Multiply every lane by a real scalar. Linear, valid for every element. */
export function scaleField(a: ArrayLike<number>, s: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = a[i]! * s;
  return out;
}

/** The element's *algebra product* a·b, sample by sample: ordinary product for
 *  scalar, complex multiply for complex, Hamilton product for quaternion. `vec` has
 *  no canonical product (use `dotFields`/`crossFields`), so it is rejected. */
export function mulFields(el: ElementType, a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  sameLen(a, b, "mulFields");
  const out = new Float32Array(a.length);
  switch (el.kind) {
    case "scalar":
      for (let i = 0; i < out.length; i++) out[i] = a[i]! * b[i]!;
      return out;
    case "complex":
      for (let i = 0; i < out.length; i += 2) {
        const ar = a[i]!, ai = a[i + 1]!, br = b[i]!, bi = b[i + 1]!;
        out[i] = ar * br - ai * bi;
        out[i + 1] = ar * bi + ai * br;
      }
      return out;
    case "quaternion":
      for (let i = 0; i < out.length; i += 4) {
        const aw = a[i]!, ax = a[i + 1]!, ay = a[i + 2]!, az = a[i + 3]!;
        const bw = b[i]!, bx = b[i + 1]!, by = b[i + 2]!, bz = b[i + 3]!;
        out[i] = aw * bw - ax * bx - ay * by - az * bz;
        out[i + 1] = aw * bx + ax * bw + ay * bz - az * by;
        out[i + 2] = aw * by - ax * bz + ay * bw + az * bx;
        out[i + 3] = aw * bz + ax * by - ay * bx + az * bw;
      }
      return out;
    case "vec":
      throw new Error("mulFields: vec has no algebra product — use dotFields or crossFields");
  }
}

/** Conjugate: identity for scalar (real), negate imaginary for complex, negate the
 *  vector part for quaternion. `vec` has no conjugate. */
export function conjugate(el: ElementType, a: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(a);
  switch (el.kind) {
    case "scalar":
      return out;
    case "complex":
      for (let i = 1; i < out.length; i += 2) out[i] = -out[i]!;
      return out;
    case "quaternion":
      for (let i = 0; i < out.length; i += 4) {
        out[i + 1] = -out[i + 1]!;
        out[i + 2] = -out[i + 2]!;
        out[i + 3] = -out[i + 3]!;
      }
      return out;
    case "vec":
      throw new Error("conjugate: undefined for vec");
  }
}

/** Magnitude |a| per sample → a scalar field (one lane out per sample). |x| for
 *  scalar, modulus for complex, Euclidean norm for quaternion/vec. */
export function magnitude(el: ElementType, a: ArrayLike<number>): Float32Array {
  const L = elementLanes(el);
  const n = a.length / L;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < L; c++) {
      const x = a[i * L + c]!;
      s += x * x;
    }
    out[i] = Math.sqrt(s);
  }
  return out;
}

/** Pointwise vector dot product a·b → a scalar field. Defined for `vec` only. */
export function dotFields(el: ElementType, a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  if (el.kind !== "vec") throw new Error(`dotFields: requires vec, got ${elementLabel(el)}`);
  sameLen(a, b, "dotFields");
  const L = el.n;
  const n = a.length / L;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < L; c++) s += a[i * L + c]! * b[i * L + c]!;
    out[i] = s;
  }
  return out;
}

/** Pointwise cross product a×b for vec3 → a vec3 field. */
export function crossFields(el: ElementType, a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  if (el.kind !== "vec" || el.n !== 3) throw new Error(`crossFields: requires vec3, got ${elementLabel(el)}`);
  sameLen(a, b, "crossFields");
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i += 3) {
    const ax = a[i]!, ay = a[i + 1]!, az = a[i + 2]!;
    const bx = b[i]!, by = b[i + 1]!, bz = b[i + 2]!;
    out[i] = ay * bz - az * by;
    out[i + 1] = az * bx - ax * bz;
    out[i + 2] = ax * by - ay * bx;
  }
  return out;
}

/** Normalise each sample to unit magnitude (zero samples left as zero). Defined for
 *  vec and quaternion. */
export function normalize(el: ElementType, a: ArrayLike<number>): Float32Array {
  if (el.kind !== "vec" && el.kind !== "quaternion") {
    throw new Error(`normalize: requires vec or quaternion, got ${elementLabel(el)}`);
  }
  const L = elementLanes(el);
  const out = Float32Array.from(a);
  for (let i = 0; i < out.length; i += L) {
    let s = 0;
    for (let c = 0; c < L; c++) s += out[i + c]! * out[i + c]!;
    const m = Math.sqrt(s);
    if (m > 0) for (let c = 0; c < L; c++) out[i + c] = out[i + c]! / m;
  }
  return out;
}

/** Interleave two scalar fields (real, imag) into a complex field [re, im, ...]. */
export function packComplex(re: ArrayLike<number>, im: ArrayLike<number>): Float32Array {
  sameLen(re, im, "packComplex");
  const out = new Float32Array(re.length * 2);
  for (let i = 0; i < re.length; i++) {
    out[i * 2] = re[i]!;
    out[i * 2 + 1] = im[i]!;
  }
  return out;
}

/** Extract one lane (e.g. real=0, imag=1) of a multi-lane element as a scalar field. */
export function extractLane(el: ElementType, a: ArrayLike<number>, lane: number): Float32Array {
  const L = elementLanes(el);
  if (lane < 0 || lane >= L) throw new Error(`extractLane: lane ${lane} out of range for ${elementLabel(el)}`);
  const n = a.length / L;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i * L + lane]!;
  return out;
}
