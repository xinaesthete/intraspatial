// Custom React Flow node: title + one Handle per declared input/output port. Ports
// are typed by shape kind (shown in the handle tooltip); editing of params happens
// in the inspector panel, so the node itself stays compact.
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { getSpec } from "./specs";
import type { NodeData } from "./buildGraph";

const KIND_COLOR: Record<string, string> = {
  points: "#7cc4ff",
  grid: "#9be29b",
  matrix: "#e2b85b",
  scalar: "#d79bff",
  opaque: "#ff9bb5",
  any: "#bbbbbb",
};

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

export function OpNode({ data, selected }: NodeProps) {
  const d = data as unknown as NodeData;
  const spec = getSpec(d.opName);
  const rows = Math.max(spec.inputs.length, spec.outputs.length, 1);

  return (
    <div className={`opnode ${spec.isSource ? "source" : ""} ${selected ? "selected" : ""}`} style={{ minHeight: 28 + rows * 18 }}>
      <div className="opnode-title">{spec.label}</div>
      {spec.inputs.map((p, i) => (
        <Handle
          key={`in-${p.name}`}
          id={p.name}
          type="target"
          position={Position.Left}
          title={`${p.name}: ${p.kind}`}
          style={{ top: portTop(i, spec.inputs.length), background: KIND_COLOR[p.kind] ?? "#bbb" }}
        />
      ))}
      {spec.outputs.map((p, i) => (
        <Handle
          key={`out-${p.name}`}
          id={p.name}
          type="source"
          position={Position.Right}
          title={`${p.name}: ${p.kind}`}
          style={{ top: portTop(i, spec.outputs.length), background: KIND_COLOR[p.kind] ?? "#bbb" }}
        />
      ))}
      <div className="opnode-ports">
        <div className="opnode-in">{spec.inputs.map((p) => <span key={p.name}>{p.name}</span>)}</div>
        <div className="opnode-out">{spec.outputs.map((p) => <span key={p.name}>{p.name}</span>)}</div>
      </div>
    </div>
  );
}
