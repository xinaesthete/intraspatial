// An INSTANCE of a reusable named subgraph (shown in its parent scope). Visually like
// a group node, but tagged ⬡ "linked" — it references a shared definition, so editing
// the definition (double-click to open) updates every instance. Ports derive from the
// definition's interface (attached by deriveDisplay).
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { GroupData, Port } from "./grouping";
import { kindColor } from "./portKinds";

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

function PortRow({ p, side }: { p: Port; side: "in" | "out" }) {
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

export function InstanceNode({ data, selected }: NodeProps) {
  const d = data as unknown as GroupData & { def?: string };
  const inputs = d.inputs ?? [];
  const outputs = d.outputs ?? [];
  const rows = Math.max(inputs.length, outputs.length, 1);

  return (
    <div className={`opnode instance ${selected ? "selected" : ""}`} style={{ minHeight: 30 + rows * 20 }} title="Reusable subgraph — double-click to edit the shared definition">
      <div className="opnode-title">⬡ {d.label}</div>
      <div className="group-sub">linked · {d.members ?? 0} nodes · double-click to edit</div>
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
