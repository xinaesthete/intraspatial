// Render a LaTeX string as math via KaTeX. `display` = block (centered) vs inline.

import katex from "katex";
import { useMemo } from "react";
import "katex/dist/katex.min.css";

export function MathTex({ tex, display = true }: { tex: string; display?: boolean }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, output: "html" });
    } catch {
      return tex;
    }
  }, [tex, display]);
  // biome-ignore lint/security/noDangerouslySetInnerHtml: html from katex should be ok
  return <span className="katex-host" dangerouslySetInnerHTML={{ __html: html }} />;
}
