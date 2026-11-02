// Mode loadings: rows are modes 1–3, bars each channel's signed weight. Hovering a channel lights
// its column across all three modes and spells out its loadings — "gene X drives mode 1 and opposes
// mode 2" is the reading the chart is for, and the flat rows cannot give it on their own.

import { useState } from "react";
import type { CoLocationModes } from "../../../src/spatial/gram";
import { modeColour } from "./modeColour";

const G = { W: 460, rowH: 54, pad: 46, right: 10, top: 20, labelH: 44 } as const;

export function LoadingsChart({ modes }: { modes: CoLocationModes }) {
  const [hover, setHover] = useState<number>();
  const { labels, vectors } = modes;
  const K = labels.length;
  const rows = Math.min(3, K);
  const H = G.top + rows * G.rowH + G.labelH;
  const bw = (G.W - G.pad - G.right) / K;
  // Labels thinned when bars are too narrow to carry every name; the hover covers what is dropped.
  const everyN = bw >= 34 ? 1 : bw >= 16 ? 2 : Math.ceil(11 / bw);
  const axisY = G.top + rows * G.rowH + 4;
  const h = hover !== undefined && hover < K ? hover : undefined;
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${G.W} ${H}`} role="img" aria-label="mode loadings" onMouseLeave={() => setHover(undefined)}>
        {h !== undefined && <rect x={G.pad + h * bw} y={G.top - 4} width={bw} height={rows * G.rowH + 4} fill="rgba(56,189,248,0.14)" />}
        {Array.from({ length: rows }, (_, k) => {
          const mid = G.top + k * G.rowH + G.rowH / 2;
          return (
            <g key={k} fill={modeColour(k)}>
              <line x1={G.pad} x2={G.W - G.right} y1={mid} y2={mid} stroke="#334155" />
              <text x={G.pad - 6} y={mid + 3} textAnchor="end">
                mode {k + 1}
              </text>
              {labels.map((label, a) => {
                const v = (vectors[k * K + a] ?? 0) * (G.rowH / 2 - 4);
                return (
                  <rect key={label} x={G.pad + a * bw + 1} y={mid - Math.max(v, 0)} width={Math.max(bw - 2, 1)} height={Math.abs(v)} />
                );
              })}
            </g>
          );
        })}
        {labels.map((label, a) =>
          a === h || a % everyN === 0 ? (
            <text
              key={label}
              transform={`translate(${G.pad + a * bw + bw / 2} ${axisY}) rotate(-69)`}
              textAnchor="end"
              dominantBaseline="middle"
              fill={a === h ? "#e2e8f0" : "#64748b"}
            >
              {label}
            </text>
          ) : null,
        )}
        {/* Hit targets last, full height, so hovering anywhere in a column selects it. */}
        {labels.map((label, a) => (
          <rect key={`hit${label}`} x={G.pad + a * bw} y={0} width={bw} height={H} fill="transparent" onMouseEnter={() => setHover(a)} />
        ))}
      </svg>
      <p className="hint">
        {h !== undefined ? (
          <>
            <b>{labels[h]}</b> —{" "}
            {Array.from({ length: rows }, (_, k) => {
              const v = vectors[k * K + h] ?? 0;
              return (
                <span key={k} style={{ color: modeColour(k) }}>
                  mode {k + 1} {v >= 0 ? "+" : "−"}
                  {Math.abs(v).toFixed(2)}
                  {k < rows - 1 ? " · " : ""}
                </span>
              );
            })}{" "}
            (unit eigenvector components; the sign is the direction of co-variation).
          </>
        ) : (
          "Hover a column to read one channel across the modes."
        )}
      </p>
    </>
  );
}
