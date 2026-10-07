// A collapsible plain-language explanation, with the formula underneath for those who want it.

import type { ReactNode } from "react";
import { MathTex } from "../Math";

export function Explain({ title, children, math }: { title: string; children: ReactNode; math?: string }) {
  return (
    <details className="explain">
      <summary>{title}</summary>
      <div className="explain-body">{children}</div>
      {math && (
        <div className="explain-math">
          <MathTex tex={math} />
        </div>
      )}
    </details>
  );
}
