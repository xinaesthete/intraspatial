// The operation abstraction: a named node type with typed input/output ports,
// UI-discoverable params, shape inference, and an execute body. The optional
// `cpuGolden` is the reference oracle tests validate `execute` against, and the
// implementation `mode: "cpu"` runs — not a runtime fallback (resource-sync
// invariant 5, as revised by ADR-0017).
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
  /** Free-form tags for grouping/filtering params (orthogonal to the dotted-path `name`
   *  namespace). Used by param-exploration UIs to filter a set and apply operations to it
   *  (freeze/mutate/steer/…). Optional and additive — ops that don't set it are unaffected. */
  tags?: string[];
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
  /** Tier-2 opt-in (ADR-0017). When true, `execute` accepts inputs carrying `buffer` instead
   *  of host `data` and is expected to return outputs that do the same — so an edge between two
   *  resident ops never touches the host (invariant 4). Absent ⇒ host-only (Tier-1), today's
   *  behaviour, which is what makes the migration incremental: the executor bridges between the
   *  two representations, so a resident op and a host op can sit next to each other and every
   *  unconverted op keeps working unchanged.
   *
   *  A resident op MUST lease its outputs from `ctx.backend.lease` rather than returning a
   *  module-scoped scratch buffer: the executor owns the returned buffer's lifetime and will
   *  release it once its last consumer has run.
   *
   *  ONE LEASE PER OUTPUT PORT. The executor tracks ownership per `(node, port)`, so a
   *  multi-output resident op works — but each output must carry its *own* lease. Returning the
   *  same `ResidentBuffer` on two ports makes the executor release it twice, which the pool
   *  rejects (a double release means two live values would share one buffer). Alias by copying,
   *  or emit one port and let a downstream op derive the rest. */
  resident?: boolean;
  /** Run the op. Inputs are positional, matching `inputs`; outputs positional,
   *  matching `outputs`. */
  execute(ctx: ExecCtx, inputs: FieldValue[], params: Params): Promise<FieldValue[]>;
  /** CPU reference implementation. Pure; same positional contract as `execute`.
   *
   *  A **test-time oracle**, not a production fallback (gpu-resource-sync invariant 5, revised
   *  by ADR-0017). It is what makes a kernel trustworthy — tests compare `execute` against it,
   *  where an explicit download is free and correct — and it is what `mode: "cpu"` runs when a
   *  caller deliberately asks for the reference path. The executor never substitutes it for a
   *  failed GPU op: that is a fail-state and the error propagates. */
  cpuGolden?(inputs: FieldValue[], params: Params): FieldValue[];
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

/** True when every host-resident output is all-finite. A **test** helper: the executor no
 *  longer scans outputs (that would force a download on every pull, breaking invariant 4), so
 *  this is for assertions, not the run path. Note it skips values carrying only a GPU `buffer`,
 *  since checking those would mean downloading them. */
export function allFinite(outputs: FieldValue[]): boolean {
  for (const o of outputs) {
    if (!o.data) continue;
    for (let i = 0; i < o.data.length; i++) {
      if (!Number.isFinite(o.data[i]!)) return false;
    }
  }
  return true;
}
