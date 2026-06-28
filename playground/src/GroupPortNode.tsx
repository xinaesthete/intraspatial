// A boundary "port" stub shown inside a group's scope — it represents a connection
// that enters (side: "in") or leaves (side: "out") the group, so you can see how the
// subgraph wires to the outside while editing it.
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";

export function GroupPortNode({ data }: NodeProps) {
  const d = data as { label: string; side: "in" | "out" };
  return (
    <div className={`port-stub ${d.side}`} title={d.side === "in" ? "input from outside" : "output to outside"}>
      {d.side === "in" ? "▸ " : ""}{d.label}{d.side === "out" ? " ▸" : ""}
      {d.side === "in"
        ? <Handle id="out" type="source" position={Position.Right} />
        : <Handle id="in" type="target" position={Position.Left} />}
    </div>
  );
}
