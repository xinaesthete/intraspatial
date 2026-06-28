// A boundary "port" stub shown inside a group's scope — it represents a connection
// that enters (side: "in") or leaves (side: "out") the group. Coloured + tagged by
// shape kind so the type is clear while editing the subgraph.
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { kindColor } from "./portKinds";

export function GroupPortNode({ data }: NodeProps) {
  const d = data as { label: string; side: "in" | "out"; kind: string };
  const color = kindColor(d.kind);
  return (
    <div className={`port-stub ${d.side}`} style={{ borderColor: color }} title={d.side === "in" ? "input from outside" : "output to outside"}>
      <span className="kind-dot" style={{ background: color }} />
      <span className="port-name">{d.label}</span>
      <span className="kind-tag" style={{ color }}>{d.kind}</span>
      {d.side === "in"
        ? <Handle id="out" type="source" position={Position.Right} style={{ background: color }} />
        : <Handle id="in" type="target" position={Position.Left} style={{ background: color }} />}
    </div>
  );
}
