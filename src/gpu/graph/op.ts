// The operation abstraction: a named node type with typed input/output ports,
// UI-discoverable params, shape inference, and an execute body. Optional
// `cpuGolden`/`sanity` implement resource-sync invariant 5 (validate + fall back).
//
// An `OpType` is backend-agnostic: `execute` receives a `GpuBackend` via the
// context and resolved input `FieldValue`s, and returns output `FieldValue`s. The
// registry keys these by name so the React Flow palette can list and instantiate
// them.

import type { GpuBackend } from "./backend";
import type { Basis, Dtype, ElementType, FieldValue, Shape, ShapeKind } from "./handle";

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

/** Rich help for a node, shown in a tooltip/inspector beyond the one-line `describe`.
 *  `math` is a LaTeX/KaTeX source string rendered as display math by the UI. */
export interface OpHelp {
  /** Longer prose explanation. */
  detail?: string;
  /** Display-math source (LaTeX), e.g. `z\\,w=(ac-bd)+(ad+bc)i`. */
  math?: string;
}

export interface ExecCtx {
  backend: GpuBackend;
}

export interface OpType {
  /** Registry key, e.g. "convolveSeparable". */
  name: string;
  /** Display label + one-line description for the palette. */
  label: string;
  describe?: string;
  /** Palette grouping, e.g. "Arithmetic", "Wavelet". Absent ⇒ grouped as "Other". */
  category?: string;
  /** Rich help (prose + display math) for the node tooltip. */
  help?: OpHelp;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  /** Derive output shapes from input shapes + params, so the graph can validate
   *  connections and size pools before running. */
  inferShapes(inputs: Shape[], params: Params): Shape[];
  /** Derive output element algebras from input elements + params (ADR-0004). Absent
   *  ⇒ every output is `scalar` (the legacy default). Runs at graph-build time, so an
   *  op rejects a wrong element here (e.g. `complexMul` on a vec field) — this is the
   *  build-time element type-check. Positional inputs match `inputs`. */
  inferElements?(inputs: ElementType[], params: Params): ElementType[];
  /** Derive output bases from input bases + params (ADR-0006). Absent ⇒ the output
   *  passes through the first input's basis (or `spatial` for a source). Runs at
   *  graph-build time, so an op rejects a wrong basis here (e.g. `idwt` on a spatial
   *  field). The executor then stamps the inferred basis onto runtime values, so
   *  generic ops carry a wavelet field through to `idwt` without knowing about it. */
  inferBasis?(inputs: Basis[], params: Params): Basis[];
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
