// The collapsed-group proxy node: a single node standing in for a subgraph, with one
// handle per boundary port. Double-click expands it (handled in App).
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { GroupData } from "./grouping";

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

export function GroupNode({ data, selected }: NodeProps) {
  const d = data as unknown as GroupData;
  const rows = Math.max(d.inputs.length, d.outputs.length, 1);

  return (
    <div className={`opnode group ${selected ? "selected" : ""}`} style={{ minHeight: 30 + rows * 18 }} title="Double-click to expand">
      <div className="opnode-title">▦ {d.label}</div>
      <div className="group-sub">{d.members.length} nodes</div>
      {d.inputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="target" position={Position.Left} title={p.label} style={{ top: portTop(i, d.inputs.length) }} />
      ))}
      {d.outputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="source" position={Position.Right} title={p.label} style={{ top: portTop(i, d.outputs.length) }} />
      ))}
      <div className="opnode-ports">
        <div className="opnode-in">{d.inputs.map((p) => <span key={p.id}>{p.label}</span>)}</div>
        <div className="opnode-out">{d.outputs.map((p) => <span key={p.id}>{p.label}</span>)}</div>
      </div>
    </div>
  );
}
