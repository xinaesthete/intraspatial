// A rich hover tooltip for a palette node: label, category, description, the longer
// help detail, rendered KaTeX math, and a port summary. Fixed-positioned just to the
// right of the hovered button, clamped to the viewport. Non-interactive (pointer-events
// none) so moving the cursor onto it never causes flicker.
import type { NodeSpec } from "./specs";
import { MathTex } from "./Math";

export function HelpTooltip({ spec, rect }: { spec: NodeSpec; rect: DOMRect }) {
  const left = Math.min(rect.right + 10, window.innerWidth - 330);
  const top = Math.max(8, Math.min(rect.top, window.innerHeight - 240));
  const io = [
    spec.inputs.length ? `in: ${spec.inputs.map((p) => p.name).join(", ")}` : null,
    spec.outputs.length ? `out: ${spec.outputs.map((p) => p.name).join(", ")}` : null,
  ]
    .filter(Boolean)
    .join("   ·   ");

  return (
    <div className="help-tip" style={{ left, top }}>
      <div className="help-tip-head">
        <span className="help-tip-title">{spec.label}</span>
        <span className="help-tip-cat">{spec.category}</span>
      </div>
      {spec.describe && <div className="help-tip-desc">{spec.describe}</div>}
      {spec.help?.detail && <div className="help-tip-detail">{spec.help.detail}</div>}
      {spec.help?.math && (
        <div className="help-tip-math">
          <MathTex tex={spec.help.math} />
        </div>
      )}
      {io && <div className="help-tip-io">{io}</div>}
    </div>
  );
}
