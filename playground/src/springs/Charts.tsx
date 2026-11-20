// Small chart pieces for the springs page: a line chart and the phase-diagram grid.
import type { PhasePoint } from "../../../src/gpu/sim/springs/index";

export interface Series {
  readonly label: string;
  readonly colour: string;
  readonly values: readonly number[];
}

/** One or more series against a shared x, each on the chart's common y range. */
export function LineChart(props: {
  title: string;
  xs: readonly number[];
  series: readonly Series[];
  /** Horizontal guide lines, e.g. a predicted value. */
  guides?: readonly { y: number; label: string }[];
  yMin?: number;
  yMax?: number;
}) {
  const W = 320;
  const H = 130;
  const pad = { l: 38, r: 8, t: 18, b: 18 };
  const all = props.series.flatMap((s) => s.values).filter(Number.isFinite);
  const lo = props.yMin ?? Math.min(0, ...all);
  const hi = Math.max(props.yMax ?? -Infinity, lo + 1e-9, ...(props.yMax === undefined ? all : []));
  const x0 = props.xs[0] ?? 0;
  const x1 = Math.max(x0 + 1, props.xs.at(-1) ?? 1);
  const px = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const py = (y: number) => H - pad.b - ((y - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={props.title}>
      <text x={pad.l} y={12} className="chart-title">
        {props.title}
      </text>
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} className="chart-axis" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} className="chart-axis" />
      <text x={pad.l - 4} y={pad.t + 4} className="chart-tick" textAnchor="end">
        {fmt(hi)}
      </text>
      <text x={pad.l - 4} y={H - pad.b} className="chart-tick" textAnchor="end">
        {fmt(lo)}
      </text>
      <text x={W - pad.r} y={H - 4} className="chart-tick" textAnchor="end">
        step {x1}
      </text>
      {props.guides?.map((g) => (
        <g key={g.label}>
          <line x1={pad.l} y1={py(g.y)} x2={W - pad.r} y2={py(g.y)} className="chart-guide" />
          <text x={W - pad.r} y={py(g.y) - 3} className="chart-tick" textAnchor="end">
            {g.label}
          </text>
        </g>
      ))}
      {props.series.map((s, k) => (
        <g key={s.label}>
          <polyline
            fill="none"
            stroke={s.colour}
            strokeWidth={1.5}
            points={s.values.map((v, i) => `${px(props.xs[i] ?? i)},${py(v)}`).join(" ")}
          />
          <text x={pad.l + 6 + k * 96} y={pad.t + 10} fill={s.colour} className="chart-tick">
            {s.label} {s.values.length > 0 ? fmt(s.values.at(-1) ?? 0) : ""}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Teal for a round outline through to magenta for a fingered one. */
function circularityColour(c: number): string {
  const t = Math.max(0, Math.min(1, (c - 0.55) / 0.4));
  return `rgb(${Math.round(217 - 172 * t)},${Math.round(70 + 142 * t)},${Math.round(239 - 48 * t)})`;
}

const label = (v: number) => (Number.isFinite(v) ? String(v) : "∞");

/** The phase diagram: one square per run, coloured by how round the tissue ended up. */
export function PhaseGrid(props: {
  xLabel: string;
  yLabel: string;
  xs: readonly number[];
  ys: readonly number[];
  points: readonly PhasePoint[];
  onPick: (point: PhasePoint) => void;
}) {
  const at = new Map(props.points.map((q) => [`${q.ix},${q.iy}`, q]));
  return (
    <div className="phase">
      <div className="phase-y">{props.yLabel}</div>
      <div className="phase-grid" style={{ gridTemplateColumns: `auto repeat(${props.xs.length}, 46px)` }}>
        {[...props.ys.keys()].reverse().map((iy) => (
          <div key={iy} style={{ display: "contents" }}>
            <div className="phase-tick">{label(props.ys[iy] ?? 0)}</div>
            {props.xs.map((_, ix) => {
              const q = at.get(`${ix},${iy}`);
              return (
                <button
                  type="button"
                  key={`${ix},${iy}`}
                  className="phase-cell"
                  disabled={!q}
                  style={{ background: q ? circularityColour(q.shape.circularity) : "#111827" }}
                  title={
                    q
                      ? `circularity ${q.shape.circularity.toFixed(2)}, grew ×${q.growth.toFixed(1)}, ${q.shape.count} cells`
                      : "not run yet"
                  }
                  onClick={() => q && props.onPick(q)}
                >
                  {q ? `×${q.growth.toFixed(1)}` : ""}
                </button>
              );
            })}
          </div>
        ))}
        <div />
        {props.xs.map((x) => (
          <div key={x} className="phase-tick phase-tick-x">
            {label(x)}
          </div>
        ))}
      </div>
      <div className="phase-x">{props.xLabel}</div>
    </div>
  );
}
