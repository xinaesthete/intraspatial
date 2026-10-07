// The K×K co-location matrix g: 1 = together as often as chance, >1 together, <1 apart. Colour is
// log₂ g, so doubling and halving look equally strong.

import { cssRgb, diverging } from "../../../src/color/ramps";
import type { TranscriptGram } from "./useTranscriptGram";

const colour = (g: number): string => {
  const t = g > 0 ? Math.max(-1, Math.min(1, Math.log2(g) / 3)) : -1;
  return cssRgb(diverging(t, { centreL: 0.19, endL: 0.8, endC: 0.17 }));
};

export function MatrixView({ gram }: { gram: TranscriptGram }) {
  const { labels } = gram.modes;
  const K = labels.length;
  return (
    <div className="matrix-wrap">
      <div className="matrix" style={{ gridTemplateColumns: `minmax(90px, max-content) repeat(${K}, 18px)` }}>
        {labels.map((a, i) => [
          <div key={`l${a}`} className="matrix-label">
            {i + 1}. {a}
          </div>,
          ...labels.map((b, j) => {
            const g = gram.res.g[i * K + j] ?? 0;
            const text = `${a} × ${b}: g = ${g.toFixed(2)}`;
            return <div key={`${a}|${b}`} className="matrix-cell" style={{ background: colour(g) }} title={text} />;
          }),
        ])}
      </div>
      <p className="hint">Hover a cell for its value. Columns follow the row numbering.</p>
    </div>
  );
}
