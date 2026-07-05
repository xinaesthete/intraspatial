// Unified node metadata for both source generators and registry ops, so the
// palette, the custom node, and the inspector treat them the same way.

import type { OpHelp, ParamSpec } from "../../src/gpu/graph";
import { getOp, listOps } from "../../src/gpu/graph";
import { OP_CATEGORY, OP_HELP } from "./opMeta";
import { getSource, isSource, SOURCES } from "./sources";

export interface PortMeta {
  name: string;
  kind: string;
}

export interface NodeSpec {
  name: string;
  label: string;
  describe?: string;
  /** Palette grouping (resolved from the op's own `category` or the OP_CATEGORY map). */
  category: string;
  /** Rich help (prose + display math) for the tooltip, if any. */
  help?: OpHelp;
  isSource: boolean;
  inputs: PortMeta[];
  outputs: PortMeta[];
  params: ParamSpec[];
}

const categoryOf = (name: string, own?: string): string => own ?? OP_CATEGORY[name] ?? "Other";

export function getSpec(name: string): NodeSpec {
  if (isSource(name)) {
    const s = getSource(name)!;
    return {
      name,
      label: s.label,
      describe: s.describe,
      category: categoryOf(name),
      help: OP_HELP[name],
      isSource: true,
      inputs: [],
      outputs: s.outputs,
      params: s.params,
    };
  }
  const op = getOp(name);
  return {
    name,
    label: op.label,
    describe: op.describe,
    category: categoryOf(name, op.category),
    help: op.help ?? OP_HELP[name],
    isSource: false,
    inputs: op.inputs.map((p) => ({ name: p.name, kind: String(p.kind) })),
    outputs: op.outputs.map((p) => ({ name: p.name, kind: String(p.kind) })),
    params: op.params,
  };
}

export function listSourceSpecs(): NodeSpec[] {
  return SOURCES.map((s) => getSpec(s.name));
}
export function listOpSpecs(): NodeSpec[] {
  return listOps().map((o) => getSpec(o.name));
}

export function defaultParamsFor(spec: NodeSpec): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of spec.params) out[p.name] = p.default;
  return out;
}
