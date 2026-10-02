// The legend for the co-location mode map (mode 1 → lightness, modes 2–3 → hue): which channels
// push each mode which way, and how much of the variation each carries.

import { oklabToSrgb } from "../../../src/color/oklab";
import { cssRgb } from "../../../src/color/ramps";
import { modeSwatch } from "../../../src/gpu/spatial/gramModes";
import type { TranscriptGram } from "./useTranscriptGram";

const swatch = (k: number, sign: 1 | -1): string =>
  cssRgb(oklabToSrgb(modeSwatch([k === 0 ? sign : 0, k === 1 ? sign : 0, k === 2 ? sign : 0])));

export function ModeLegend({ gram }: { gram: TranscriptGram }) {
  const { vectors, explained, labels } = gram.modes;
  const K = labels.length;
  const roles = ["brightness", "hue (green ↔ red)", "hue (blue ↔ yellow)"];
  return (
    <div className="legend">
      {[0, 1, 2]
        .filter((k) => k < K)
        .map((k) => {
          const load = labels.map((label, a) => ({ label, v: vectors[k * K + a] ?? 0 })).sort((p, q) => q.v - p.v);
          const top = (sign: 1 | -1) =>
            (sign > 0 ? load : [...load].reverse())
              .filter((l) => l.v * sign > 0.15)
              .slice(0, 4)
              .map((l) => l.label)
              .join(", ") || "(none)";
          return (
            <div key={k} className="legend-mode">
              <div className="legend-head">
                Mode {k + 1} · {roles[k]} · {((explained[k] ?? 0) * 100).toFixed(0)}% of the variation
              </div>
              <div className="legend-row">
                <span className="swatch" style={{ background: swatch(k, 1) }} /> more: {top(1)}
              </div>
              <div className="legend-row">
                <span className="swatch" style={{ background: swatch(k, -1) }} /> more: {top(-1)}
              </div>
            </div>
          );
        })}
    </div>
  );
}
