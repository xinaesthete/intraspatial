// The operation abstraction: a named node type with typed input/output ports,
// UI-discoverable params, shape inference, and an execute body. Optional
// `cpuGolden`/`sanity` implement resource-sync invariant 5 (validate + fall back).
//
// An `OpType` is backend-agnostic: `execute` receives a `GpuBackend` via the
// context and resolved input `FieldValue`s, and returns output `FieldValue`s. The
// registry keys these by name so the React Flow palette can list and instantiate
// them.
import type { Dtype, FieldValue, Shape, ShapeKind } from "./handle";
import type { GpuBackend } from "./backend";

export interface PortSpec {
  name: string;
  /** Which shape kinds may connect here. A single kind, or `"any"` for pass-through
   *  ops; used by the UI to validate edges. */
  kind: ShapeKind | "any";
  dtype?: Dtype;
}

export type ParamType = "number" | "int" | "enum" | "bool";

export interface ParamSpec {
  name: string;
  type: ParamType;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  /** For `enum`. */
  options?: string[];
  /** Short help shown in the UI. */
  describe?: string;
}

export type Params = Record<string, unknown>;

export interface ExecCtx {
  backend: GpuBackend;
}

export interface OpType {
  /** Registry key, e.g. "convolveSeparable". */
  name: string;
  /** Display label + one-line description for the palette. */
  label: string;
  describe?: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  /** Derive output shapes from input shapes + params, so the graph can validate
   *  connections and size pools before running. */
  inferShapes(inputs: Shape[], params: Params): Shape[];
  /** Run the op. Inputs are positional, matching `inputs`; outputs positional,
   *  matching `outputs`. */
  execute(ctx: ExecCtx, inputs: FieldValue[], params: Params): Promise<FieldValue[]>;
  /** CPU reference (invariant 5). Pure; same positional contract as `execute`. */
  cpuGolden?(inputs: FieldValue[], params: Params): FieldValue[];
  /** Cheap output sanity gate (finite / plausible range). Returning false triggers
   *  the CPU fallback when one is available. */
  sanity?(outputs: FieldValue[]): boolean;
}

/** Resolve a param with its declared default. */
export function param<T>(params: Params, spec: ParamSpec): T {
  const v = params[spec.name];
  return (v === undefined ? spec.default : v) as T;
}

/** Default params for an op type (all declared defaults). */
export function defaultParams(op: OpType): Params {
  const out: Params = {};
  for (const p of op.params) out[p.name] = p.default;
  return out;
}

/** A finite-numbers sanity check usable as a default `sanity` for numeric ops. */
export function allFinite(outputs: FieldValue[]): boolean {
  for (const o of outputs) {
    if (!o.data) continue;
    for (let i = 0; i < o.data.length; i++) {
      if (!Number.isFinite(o.data[i]!)) return false;
    }
  }
  return true;
}
