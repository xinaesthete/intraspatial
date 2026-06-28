// An explicit Input/Output interface node inside a subgraph. You wire it to members;
// the parent group node exposes a matching port. Coloured by the shape kind it
// carries (attached by deriveDisplay).
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { kindColor } from "./portKinds";
import type { IODataShape } from "./grouping";

export function InterfaceNode({ data, selected }: NodeProps) {
  const d = data as unknown as IODataShape & { kind?: string };
  const color = kindColor(d.kind ?? "any");
  const isInput = d.io === "input";
  return (
    <div className={`io-node ${d.io} ${selected ? "selected" : ""}`} style={{ borderColor: color }}>
      <span className="io-arrow" style={{ color }}>{isInput ? "▸" : ""}</span>
      <span className="io-name">{d.label}</span>
      <span className="kind-tag" style={{ color }}>{d.kind ?? "any"}</span>
      <span className="io-arrow" style={{ color }}>{isInput ? "" : "▸"}</span>
      {isInput
        ? <Handle id="out" type="source" position={Position.Right} style={{ background: color }} />
        : <Handle id="in" type="target" position={Position.Left} style={{ background: color }} />}
    </div>
  );
}
