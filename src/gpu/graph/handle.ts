// Resource handles for the in-GPU operation graph.
//
// A `GpuField` is a *lazy descriptor*, not data: it names the shape/dtype of a
// value and the node+port that will produce it, but holds no buffer. The executor
// binds a pooled buffer at run time, so the same handle may alias different
// physical buffers across runs (resource-sync invariant 3 — reuse by liveness, not
// by identity). See docs/gpu-resource-sync.md.
//
// At execution time the value flowing along an edge is a `FieldValue` — for the
// Tier-1 (boundary-granularity) ops this is a host typed array + shape; a future
// Tier-2 resident op carries a GPU buffer handle instead. The shape vocabulary is
// shared by both fronts (grid for the image side, points/matrix for the spatial
// side) plus scalar and an opaque payload (e.g. a persistence diagram).

export type Dtype = "f32" | "i32" | "u32";

/** The algebraic type of a single sample's value (ADR-0004). A *closed* set of small
 *  algebras, each with its own arithmetic (complex multiply, Hamilton product, dot) —
 *  deliberately distinct from an *open* tensor axis (e.g. genes), which is bulk data
 *  with no per-element algebra. Stored interleaved: a sample occupies `elementLanes`
 *  contiguous lanes of the field's flat `data`. Absent on a field ⇒ `scalar`. */
export type ElementType =
  | { kind: "scalar" }
  | { kind: "complex" }
  | { kind: "vec"; n: 2 | 3 | 4 }
  | { kind: "quaternion" };

/** The implicit element of any field that doesn't declare one. */
export const SCALAR: ElementType = { kind: "scalar" };

/** Contiguous f32 lanes one sample of this element occupies. */
export function elementLanes(e: ElementType): number {
  switch (e.kind) {
    case "scalar":
      return 1;
    case "complex":
      return 2;
    case "vec":
      return e.n;
    case "quaternion":
      return 4;
  }
}

export function elementsEqual(a: ElementType, b: ElementType): boolean {
  if (a.kind !== b.kind) return false;
  return a.kind === "vec" ? b.kind === "vec" && a.n === b.n : true;
}

export function elementLabel(e: ElementType): string {
  return e.kind === "vec" ? `vec${e.n}` : e.kind;
}

/** The *basis* a field's values are expressed in (ADR-0006), orthogonal to domain and
 *  element. `spatial` (the default) is the sampled signal itself; `wavelet` is a packed
 *  Mallat coefficient pyramid that carries its own decomposition contract (kernel +
 *  levels) so downstream ops (idwt, detail thresholding) read it from the input rather
 *  than re-declaring it as a parameter. Absent on a field ⇒ `spatial`. */
export type Basis =
  | { kind: "spatial" }
  | { kind: "wavelet"; wavelet: "5/3" | "9/7"; levels: number };

/** The implicit basis of any field that doesn't declare one. */
export const SPATIAL: Basis = { kind: "spatial" };

export function basisLabel(b: Basis): string {
  return b.kind === "wavelet" ? `wavelet ${b.wavelet}·L${b.levels}` : "spatial";
}

export type ShapeKind = "grid" | "points" | "matrix" | "scalar" | "opaque";

export type Shape =
  | { kind: "grid"; width: number; height: number }
  | { kind: "points"; n: number }
  | { kind: "matrix"; rows: number; cols: number }
  | { kind: "scalar" }
  /** A non-numeric payload (e.g. a persistence diagram); `name` tags the concrete
   *  type so the UI can pick a preview renderer. */
  | { kind: "opaque"; name: string };

export type NodeId = string;

/** A lazy reference to a node's output. The builder returns these; `pull(field)`
 *  resolves the transitive dependencies and executes them. */
export interface GpuField {
  /** Stable identity for this logical value. */
  readonly id: number;
  readonly shape: Shape;
  readonly dtype: Dtype;
  /** Element algebra of each sample (ADR-0004). Absent ⇒ `scalar`. */
  readonly element?: ElementType;
  /** Basis the values are expressed in (ADR-0006). Absent ⇒ `spatial`. */
  readonly basis?: Basis;
  /** The node that writes this value. */
  readonly producer: NodeId;
  /** Which output port of that node. */
  readonly outPort: string;
  /** Bumped when an upstream param/source changes; drives memoisation. */
  version: number;
}

/** The concrete value carried along an edge during execution. For Tier-1 ops this
 *  is host-resident (a typed array); `payload` holds non-numeric opaque results. */
export interface FieldValue {
  shape: Shape;
  dtype: Dtype;
  /** Element algebra of each sample (ADR-0004). Absent ⇒ `scalar`. */
  element?: ElementType;
  /** Basis the values are expressed in (ADR-0006). Absent ⇒ `spatial`. */
  basis?: Basis;
  /** Host data for numeric shapes (grid/points/matrix/scalar). Length is
   *  `numCells(shape) * elementLanes(element)` — samples interleaved lane-major. */
  data?: Float32Array | Int32Array | Uint32Array;
  /** Arbitrary payload for `opaque` shapes. */
  payload?: unknown;
}

/** The element of a field or value, defaulting to `scalar` when undeclared. */
export function elementOf(v: { element?: ElementType }): ElementType {
  return v.element ?? SCALAR;
}

/** The basis of a field or value, defaulting to `spatial` when undeclared. */
export function basisOf(v: { basis?: Basis }): Basis {
  return v.basis ?? SPATIAL;
}

/** Unpack a points `FieldValue` (packed [x0,y0,x1,y1,...]) into parallel arrays. */
export function unpackPoints(v: FieldValue): { xs: number[]; ys: number[]; n: number } {
  const data = v.data;
  if (!data) throw new Error("unpackPoints: field has no data");
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i + 1 < data.length; i += 2) {
    xs.push(data[i]!);
    ys.push(data[i + 1]!);
  }
  return { xs, ys, n: xs.length };
}

export function numCells(shape: Shape): number {
  switch (shape.kind) {
    case "grid":
      return shape.width * shape.height;
    case "points":
      return shape.n;
    case "matrix":
      return shape.rows * shape.cols;
    case "scalar":
      return 1;
    case "opaque":
      return 0;
  }
}

export function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "grid":
      return b.kind === "grid" && a.width === b.width && a.height === b.height;
    case "points":
      return b.kind === "points" && a.n === b.n;
    case "matrix":
      return b.kind === "matrix" && a.rows === b.rows && a.cols === b.cols;
    case "scalar":
      return b.kind === "scalar";
    case "opaque":
      return b.kind === "opaque" && a.name === b.name;
  }
}
