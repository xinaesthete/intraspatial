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

import type { Affine3 } from "../../coords";

export type Dtype = "f32" | "i32" | "u32";

/** Where a field's samples sit in world space (ADR-0015 §3, ADR-0018). A **resolved**
 *  placement handed across the `Loader` — sd.js owns the transform algebra and collapses
 *  Sequence/Affine/Rotation into `worldFromArray`; this repo only *consumes* the matrix,
 *  never composes it. `system` names the target coordinate-system (default `"global"`) and
 *  is the string binary-op agreement checks against; `worldFromArray` maps array space
 *  (level-0 voxel/cell units) to world.
 *
 *  Absent on a field ⇒ **array space**: the field is unitless and cell-indexed, exactly as
 *  every field is today. Note this is distinct from an *identity* placement, which asserts
 *  "already in system S" — do not conflate the two (see `placementOf`). */
export interface ResolvedPlacement {
  /** Target coordinate-system name (default `"global"`). */
  system: string;
  /** sd.js-composed array→world matrix (Sequence/Affine/Rotation already collapsed). */
  worldFromArray: Affine3;
}

/** The algebraic type of a single sample's value (ADR-0004). A *closed* set of small
 *  algebras, each with its own arithmetic (complex multiply, Hamilton product, dot) —
 *  deliberately distinct from an *open* tensor axis (e.g. genes), which is bulk data
 *  with no per-element algebra. Stored interleaved: a sample occupies `elementLanes`
 *  contiguous lanes of the field's flat `data`. Absent on a field ⇒ `scalar`. */
export type ElementType = { kind: "scalar" } | { kind: "complex" } | { kind: "vec"; n: 2 | 3 | 4 } | { kind: "quaternion" };

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
export type Basis = { kind: "spatial" } | { kind: "wavelet"; wavelet: "5/3" | "9/7"; levels: number };

/** The implicit basis of any field that doesn't declare one. */
export const SPATIAL: Basis = { kind: "spatial" };

export function basisLabel(b: Basis): string {
  return b.kind === "wavelet" ? `wavelet ${b.wavelet}·L${b.levels}` : "spatial";
}

/** An *open* tensor axis (ADR-0004/0015): a named, runtime-length bulk dimension with **no**
 *  per-element algebra — deliberately distinct from the closed `ElementType`. `channel`/`time`
 *  axes carry NGFF semantics; their *interpretation* is resolved by sd.js and consumed here,
 *  never parsed (ADR-0015 ownership boundary). Axis-parametric ops (reduce/select/composite over
 *  a named axis) operate on these; the algebraic element ops are rejected on them at build time. */
export type AxisType = "channel" | "time" | "gene" | "custom";

/** Per-index metadata for a `channel` axis, index-aligned to the axis `length`. Resolved from
 *  NGFF `omero.channels` on the sd.js side of the Loader (ADR-0015 fork B: it lives on the axis). */
export interface ChannelEntry {
  /** Marker/stain name — the channel's semantic identity (NGFF has no wavelength/ontology field). */
  label?: string;
  /** Display colour, hex e.g. `"0000FF"`. */
  color?: string;
  window?: { min: number; max: number; start: number; end: number };
  active?: boolean;
}

export interface TensorAxis {
  /** Axis name, e.g. `"c"`, `"t"`, `"gene"`. */
  name: string;
  type: AxisType;
  /** Open, runtime length. */
  length: number;
  /** UDUNITS-2 string (time axes); channels usually none. */
  unit?: string;
  /** Channel axis: per-index metadata, index-aligned to `length`. */
  entries?: ChannelEntry[];
}

/** Product of all open-axis lengths (1 if none). A field's flat sample count is
 *  `numCells(domain) · elementLanes(element) · axesProduct(axes)`. */
export function axesProduct(axes?: readonly TensorAxis[]): number {
  return axes ? axes.reduce((p, a) => p * a.length, 1) : 1;
}

/** The `channel` axis of a field, if it has one. */
export function channelAxis(v: { axes?: readonly TensorAxis[] }): TensorAxis | undefined {
  return v.axes?.find((a) => a.type === "channel");
}

/** Field *polarity* (ADR-0015): an ordinary intensity field vs a label/segmentation image. A label
 *  is **not** a per-sample element — it is a whole-field kind with structural invariants (integer
 *  dtype, 0 = background, no channel axis, nearest-only resampling) plus a value→properties map. */
export type FieldRole = { kind: "intensity" } | { kind: "label"; labels: LabelMeta };

export interface LabelMeta {
  /** Ref to the parent intensity field (NGFF `image-label.source`). */
  source?: string;
  /** value → rgba (NGFF `image-label.colors`). */
  colors?: Array<{ value: number; rgba: [number, number, number, number] }>;
  /** value → arbitrary property bag (NGFF `image-label.properties`). */
  properties?: Array<{ value: number; [k: string]: unknown }>;
  /** INVARIANT (ADR-0015 fork D): label ids are categorical — resample **nearest**, never linear;
   *  averaging fabricates non-existent ids. The multiscale/pyramid path must honour this. */
  resample: "nearest";
  /** Link to a `Table` by instance id (region/region_key/instance_key). */
  instanceKey?: string;
}

/** The implicit polarity of any field that doesn't declare one. */
export const INTENSITY: FieldRole = { kind: "intensity" };

/** Pool identity of a leased buffer (ADR-0017, invariant 3). `release` returns the buffer to
 *  the free list of exactly the `usage` class and `capacity` bucket it was drawn from — the
 *  pool never calls `destroy()`, because mid-process buffer destruction segfaults Dawn-on-Node
 *  (ADR-0002/0003). */
export interface LeaseToken {
  /** Monotonic id, for debugging and double-release detection. */
  readonly id: number;
  /** The `GPUBufferUsage` flags this buffer was created with. */
  readonly usage: number;
  /** Bucketed byte capacity of the physical buffer (≥ the requested `byteLength`). */
  readonly capacity: number;
}

/** A GPU-resident field payload (ADR-0017, Tier-2). Carried on a `FieldValue` *instead of*
 *  host `data`, so an interior graph edge stays on-device — invariant 4's "interior edges stay
 *  on-GPU". `byteLength` is the *logical* size of the value; the physical buffer may be larger
 *  (see `lease.capacity`), so readers must bound themselves by the logical length. */
export interface ResidentBuffer {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly lease: LeaseToken;
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
  /** Open tensor axes — channel/time/gene (ADR-0004/0015). Absent ⇒ none. */
  readonly axes?: readonly TensorAxis[];
  /** Field polarity (ADR-0015). Absent ⇒ `intensity`. */
  readonly role?: FieldRole;
  /** Where this field's samples sit in world space (ADR-0015/0018). Absent ⇒ array space:
   *  the field is unitless and cell-indexed, exactly as every field is today. */
  readonly placement?: ResolvedPlacement;
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
  /** Open tensor axes — channel/time/gene (ADR-0004/0015). Absent ⇒ none. */
  axes?: readonly TensorAxis[];
  /** Field polarity (ADR-0015). Absent ⇒ `intensity`. */
  role?: FieldRole;
  /** Where this field's samples sit in world space (ADR-0015/0018). Absent ⇒ array space:
   *  the field is unitless and cell-indexed, exactly as every field is today. */
  placement?: ResolvedPlacement;
  /** Host data for numeric shapes (grid/points/matrix/scalar). Length is
   *  `numCells(shape) * elementLanes(element) * axesProduct(axes)`. Element lanes are
   *  interleaved (lane-major); open axes are **planar** (outermost, per-index contiguous —
   *  matching the zarr/xarray `(c,y,x)` layout the sd.js loader delivers). */
  data?: Float32Array | Int32Array | Uint32Array;
  /** GPU-resident payload (ADR-0017, Tier-2), same logical contents and layout as `data`.
   *  Present on interior edges between `resident` ops, which is what keeps them off the host
   *  (invariant 4). The executor bridges the two representations on demand: it downloads into
   *  `data` before a non-resident op and uploads into `buffer` before a resident one.
   *
   *  INVARIANT: at least one of `data` / `buffer` / `payload` is present. A value may carry
   *  both transiently — immediately after a bridge in either direction. */
  buffer?: ResidentBuffer;
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

/** The polarity of a field or value, defaulting to `intensity` when undeclared. */
export function roleOf(v: { role?: FieldRole }): FieldRole {
  return v.role ?? INTENSITY;
}

/** The placement of a field or value, or `undefined` when it has none (ADR-0018).
 *
 *  Deliberately does **not** default to an identity placement: array-space (absent) and
 *  placed-at-identity-in-system-S are *distinct* states, and conflating them would claim every
 *  bare test grid lives in `global`. Absent means unitless/cell-indexed; identity means "already
 *  in system S". Callers that need to tell them apart must see the `undefined`. */
export function placementOf(v: { placement?: ResolvedPlacement }): ResolvedPlacement | undefined {
  return v.placement;
}

/** Whether two placements may combine in a binary op (ADR-0018 decision 3): the build-time
 *  "reject `add` across systems" check, on the `system` name (always statically known).
 *
 *  - both absent ⇒ `true` (two array-space fields combine, today's behaviour);
 *  - both present ⇒ `a.system === b.system` (same system ⇒ ok);
 *  - exactly one present ⇒ **throws** (a placed field and an unplaced one can't combine — one is
 *    in world space, the other is unitless, and there is no transform to reconcile them).
 *
 *  Agreement is only on the system string; the matrices are not compared (two levels of one
 *  pyramid share a system but differ in `worldFromArray`, and both are correct). */
export function systemsAgree(a?: ResolvedPlacement, b?: ResolvedPlacement): boolean {
  if (a === undefined && b === undefined) return true;
  if (a !== undefined && b !== undefined) return a.system === b.system;
  throw new Error("placement mismatch: cannot combine a placed field with an unplaced (array-space) one");
}

/** Enforce the ADR-0015 label invariants (integer dtype, no channel axis); throws on violation.
 *  Nearest-only resampling is a separate constraint enforced by the multiscale path. A no-op on
 *  intensity fields. Call at the point a label field is produced (op output / builder). */
export function assertLabelInvariants(v: { role?: FieldRole; dtype: Dtype; axes?: readonly TensorAxis[] }): void {
  if (roleOf(v).kind !== "label") return;
  if (v.dtype === "f32") throw new Error("label field must have integer dtype (u32/i32), not f32");
  if (channelAxis(v)) throw new Error("label field must not carry a channel axis");
}

/** Unpack a points `FieldValue` (packed [x0,y0,x1,y1,...]) into parallel arrays. */
export function unpackPoints(v: FieldValue): { xs: number[]; ys: number[]; n: number } {
  const data = v.data;
  if (!data) throw new Error("unpackPoints: field has no data");
  const xs: number[] = [],
    ys: number[] = [];
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
