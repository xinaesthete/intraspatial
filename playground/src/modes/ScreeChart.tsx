// The eigenvalue spectrum: each mode's share of the spatial variance, with the null band from a
// permutation envelope behind it when one has been run. Modes 1–3 take their map colours.

import type { SpectrumEnvelopeResult } from "../../../src/gpu/spatial/gramEnvelope";
import { modeColour } from "./modeColour";

const W = 460;
const H = 200;
const PAD = 28;
const MAX_MODES = 24;

export function ScreeChart({ explained, envelope }: { explained: Float64Array; envelope?: SpectrumEnvelopeResult }) {
  const n = Math.min(explained.length, MAX_MODES);
  const modes = Array.from({ length: n }, (_, i) => i);
  const env = envelope?.envelope;
  const bw = (W - 2 * PAD) / n;
  // The band shares the bar scale, or "outside the envelope" would be a claim about the drawing.
  const peak = Math.max(...modes.map((i) => explained[i]!), ...(env ? modes.map((i) => env.hi[i] ?? 0) : []), 1e-9);
  const yOf = (v: number) => H - PAD - ((H - 2 * PAD) * Math.max(0, v)) / peak;
  // Bars drawn narrow when there is a band, so the grey shows on both sides of each.
  const inset = env ? 4 : 1;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="eigenvalue spectrum">
      {env &&
        modes.map((i) => {
          const hi = yOf(env.hi[i] ?? 0);
          return (
            <rect
              key={`band${i}`}
              x={PAD + i * bw + 1}
              y={hi}
              width={bw - 2}
              height={Math.max(yOf(env.lo[i] ?? 0) - hi, 1)}
              fill="rgba(148,163,184,0.22)"
            />
          );
        })}
      {modes.map((i) => {
        const v = explained[i]!;
        const y = yOf(v);
        // A mode outside the band is the finding; mark it rather than leaving it to be eyeballed.
        const out = env && (v > (env.hi[i] ?? 0) || v < (env.lo[i] ?? 0));
        return (
          <g key={i}>
            <rect
              x={PAD + i * bw + inset}
              y={y}
              width={Math.max(bw - 2 * inset, 1)}
              height={H - PAD - y}
              fill={i < 3 ? modeColour(i) : "#5a6b87"}
            >
              <title>
                mode {i + 1}: {(v * 100).toFixed(1)}%
                {env ? ` (null ${((env.lo[i] ?? 0) * 100).toFixed(1)}–${((env.hi[i] ?? 0) * 100).toFixed(1)}%)` : ""}
              </title>
            </rect>
            {out && <circle cx={PAD + i * bw + bw / 2} cy={y - 6} r={2.5} fill="#e2e8f0" />}
            {i < 12 && (
              <text x={PAD + i * bw + bw / 2} y={H - PAD + 12} textAnchor="middle">
                {i + 1}
              </text>
            )}
          </g>
        );
      })}
      <line x1={PAD} x2={W - PAD} y1={H - PAD} y2={H - PAD} stroke="#334155" />
      <text x={PAD} y={16}>
        top mode {((explained[0] ?? 0) * 100).toFixed(1)}% of variance
      </text>
      {env && (
        <text x={W - PAD} y={16} textAnchor="end" className="strong">
          null band {((1 - env.alpha) * 100).toFixed(0)}% · p = {env.p.toFixed(3)}
        </text>
      )}
    </svg>
  );
}
