// The group node (shown in its parent scope): a single node standing in for a
// subgraph, with one handle per boundary port — coloured + tagged by shape kind so
// the port types are clear. Double-click navigates into it.
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { BoundaryPort, GroupData } from "./grouping";
import { kindColor } from "./portKinds";

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

function PortRow({ p, side }: { p: BoundaryPort; side: "in" | "out" }) {
  const dot = <span className="kind-dot" style={{ background: kindColor(p.kind) }} />;
  return (
    <span className={`port-row ${side}`}>
      {side === "in" && dot}
      <span className="port-name">{p.label}</span>
      <span className="kind-tag" style={{ color: kindColor(p.kind) }}>{p.kind}</span>
      {side === "out" && dot}
    </span>
  );
}

export function GroupNode({ data, selected }: NodeProps) {
  const d = data as unknown as GroupData;
  const inputs = d.inputs ?? [];
  const outputs = d.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length, 1);

  return (
    <div className={`opnode group ${selected ? "selected" : ""}`} style={{ minHeight: 30 + rows * 20 }} title="Double-click to open">
      <div className="opnode-title">▦ {d.label}</div>
      <div className="group-sub">{d.members ?? 0} nodes · double-click to open</div>
      {inputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="target" position={Position.Left} title={`${p.label}: ${p.kind}`} style={{ top: portTop(i, inputs.length), background: kindColor(p.kind) }} />
      ))}
      {outputs.map((p, i) => (
        <Handle key={p.id} id={p.id} type="source" position={Position.Right} title={`${p.label}: ${p.kind}`} style={{ top: portTop(i, outputs.length), background: kindColor(p.kind) }} />
      ))}
      <div className="opnode-ports">
        <div className="opnode-in">{inputs.map((p) => <PortRow key={p.id} p={p} side="in" />)}</div>
        <div className="opnode-out">{outputs.map((p) => <PortRow key={p.id} p={p} side="out" />)}</div>
      </div>
    </div>
  );
}
