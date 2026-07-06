// A general memory-accounting interface for the runtime. Fields — and especially the
// resident state that history/`delay` nodes hold — can grow large (a `delay(k)` keeps k
// copies of its input), so anything that holds host memory should be able to report it,
// and a field's cost should be computable statically from its shape/element/dtype.
import { type Dtype, type ElementType, elementLanes, type FieldValue, numCells, SCALAR, type Shape } from "./handle";

/** Bytes per component of a dtype (f32/i32/u32 are all 32-bit). */
export function dtypeBytes(_dtype: Dtype): number {
  return 4;
}

/** Resident bytes of a field of this shape/element/dtype — computable without the data. */
export function fieldBytes(shape: Shape, element: ElementType = SCALAR, dtype: Dtype = "f32"): number {
  return numCells(shape) * elementLanes(element) * dtypeBytes(dtype);
}

/** Resident bytes of a concrete value (its typed array, or the static estimate). */
export function fieldValueBytes(v: FieldValue): number {
  return v.data ? v.data.byteLength : fieldBytes(v.shape, v.element ?? SCALAR, v.dtype);
}

/** Anything holding resident host memory can report it in bytes (mirrors TypedArray). */
export interface MemoryReporting {
  readonly byteLength: number;
}

/** Bytes held by an object that either reports `byteLength` or is a `FieldValue`. */
export function memoryBytes(x: MemoryReporting | FieldValue): number {
  if ("byteLength" in x && typeof x.byteLength === "number") return x.byteLength;
  return fieldValueBytes(x as FieldValue);
}

const UNITS = ["B", "KiB", "MiB", "GiB", "TiB"];

/** Human-readable byte count, e.g. `12.4 MiB`. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  let u = 0;
  let x = n;
  while (x >= 1024 && u < UNITS.length - 1) {
    x /= 1024;
    u++;
  }
  return `${u === 0 ? x : x < 10 ? x.toFixed(1) : Math.round(x)} ${UNITS[u]}`;
}
