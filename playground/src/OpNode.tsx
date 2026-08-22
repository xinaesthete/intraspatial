// Custom React Flow node: title + one Handle per declared input/output port. Ports
// are typed by shape kind (shown in the handle tooltip); editing of params happens
// in the inspector panel, so the node itself stays compact.

import type { NodeProps } from "@xyflow/react";
import { Handle, Position } from "@xyflow/react";
import { useContext } from "react";
import type { NodeData } from "./buildGraph";
import { PortHoverContext } from "./PortHover";
import { kindColor } from "./portKinds";
import { getSpec } from "./specs";

function portTop(i: number, count: number): string {
  return `${((i + 1) / (count + 1)) * 100}%`;
}

export function OpNode({ id, data, selected }: NodeProps) {
  const d = data as unknown as NodeData;
  const spec = getSpec(d.opName);
  const rows = Math.max(spec.inputs.length, spec.outputs.length, 1);
  const hover = useContext(PortHoverContext);

  return (
    <div className={`opnode ${spec.isSource ? "source" : ""} ${selected ? "selected" : ""}`} style={{ minHeight: 28 + rows * 18 }}>
      <div className="opnode-title">{spec.label}</div>
      {spec.inputs.map((p, i) => (
        <Handle
          key={`in-${p.name}`}
          id={p.name}
          type="target"
          position={Position.Left}
          onMouseEnter={(e) => hover?.onPortEnter(id, p.name, true, p.kind, e.currentTarget.getBoundingClientRect(), p.bundle)}
          onMouseLeave={() => hover?.onPortLeave()}
          style={{ top: portTop(i, spec.inputs.length), background: kindColor(p.kind) }}
        />
      ))}
      {spec.outputs.map((p, i) => (
        <Handle
          key={`out-${p.name}`}
          id={p.name}
          type="source"
          position={Position.Right}
          onMouseEnter={(e) => hover?.onPortEnter(id, p.name, false, p.kind, e.currentTarget.getBoundingClientRect(), p.bundle)}
          onMouseLeave={() => hover?.onPortLeave()}
          style={{ top: portTop(i, spec.outputs.length), background: kindColor(p.kind) }}
        />
      ))}
      <div className="opnode-ports">
        <div className="opnode-in">
          {spec.inputs.map((p) => (
            <span key={p.name}>{p.name}</span>
          ))}
        </div>
        <div className="opnode-out">
          {spec.outputs.map((p) => (
            <span key={p.name}>{p.name}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
