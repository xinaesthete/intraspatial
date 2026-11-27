// The legend for the co-location mode map (mode 1 → lightness, modes 2–3 → hue): which channels
// push each mode which way, how much of the variation each carries, and — when given — how each
// mode's values are spread over the map's window, with the range the colour ramp spans.

import type { CoLocationModes } from "../../../src/spatial/gram";
import { HistogramChart } from "./HistogramChart";
import { modeColour as swatch } from "./modeColour";
import type { ModeHistogramState } from "./useModeHistograms";

export interface LegendHistograms {
  readonly hist: ModeHistogramState;
  /** The half-range the ramp saturates at, per mode, as the map is painted now. */
  readonly ramp: readonly [number, number, number];
  /** What the window is, for the caption: "the current view", "this map's window". */
  readonly window: string;
}

export function ModeLegend({ modes, histograms }: { modes: CoLocationModes; histograms?: LegendHistograms }) {
  const { vectors, explained, labels } = modes;
  const K = labels.length;
  const roles = ["brightness", "hue (green ↔ red)", "hue (blue ↔ yellow)"];
  return (
    <>
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
            const h = histograms?.hist.value.modes[k];
            const r = histograms?.ramp[k] ?? 0;
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
                {h && (
                  <HistogramChart
                    h={h}
                    label={`mode ${k + 1} values over ${histograms.window}`}
                    {...(r > 0 ? { view: { lo: Math.max(h.lo, -1.6 * r), hi: Math.min(h.hi, 1.6 * r), bars: 56 } } : {})}
                    fills={[(v) => (v === 0 ? "#64748b" : swatch(k, v > 0 ? 1 : -1))]}
                    markers={[
                      { at: 0, colour: "#475569", title: "0: the window's mean mixture" },
                      ...(r > 0
                        ? [
                            { at: -r, colour: "#e2e8f0", dashed: true, title: `colour saturates below −${r.toFixed(2)}` },
                            { at: r, colour: "#e2e8f0", dashed: true, title: `colour saturates above +${r.toFixed(2)}` },
                          ]
                        : []),
                    ]}
                  />
                )}
              </div>
            );
          })}
      </div>
      {histograms && (
        <p className="hint">
          Histograms: each mode's value at every pixel of {histograms.window} ({histograms.hist.width}×{histograms.hist.height} px), not the
          whole slide{histograms.hist.forCurrent ? "" : " — updating for the new window"}. Dashed lines: where the colour ramp saturates;
          tails past 1.6× that are not drawn.
        </p>
      )}
    </>
  );
}
