// The group node (shown in its parent scope): a single node standing in for a
// subgraph, with one handle per boundary port. Double-click navigates into it.
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { GroupData } from "./grouping";

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

export function GroupNode({ data, selected }: NodeProps) {
  const d = data as unknown as GroupData;
  const inputs = d.inputs ?? [];
  const outputs = d.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length, 1);

  return (
    <div className={`opnode group ${selected ? "selected" : ""}`} style={{ minHeight: 30 + rows * 18 }} title="Double-click to open">
      <div className="opnode-title">▦ {d.label}</div>
      <div className="group-sub">{d.members ?? 0} nodes · double-click to open</div>
      {inputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="target" position={Position.Left} title={p.label} style={{ top: portTop(i, inputs.length) }} />
      ))}
      {outputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="source" position={Position.Right} title={p.label} style={{ top: portTop(i, outputs.length) }} />
      ))}
      <div className="opnode-ports">
        <div className="opnode-in">{inputs.map((p) => <span key={p.id}>{p.label}</span>)}</div>
        <div className="opnode-out">{outputs.map((p) => <span key={p.id}>{p.label}</span>)}</div>
      </div>
    </div>
  );
}
