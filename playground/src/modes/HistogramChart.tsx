// A small value histogram as SVG: one or two rows of a `HistogramResult` (masked: inside drawn over
// outside), vertical markers, and an optional click that reports the value under the pointer. The
// mode legend and the wand panel share it.

import { useState } from "react";
import type { HistogramResult } from "../../../src/spatial/histogram";

/** A drawn bar as a value range: [lo, hi), or [lo, hi] for the histogram's own last bin, whose
 *  upper edge is closed (R1's binning rule). */
export interface HoveredBar {
  readonly lo: number;
  readonly hi: number;
  readonly closed: boolean;
}

export interface HistogramMarker {
  readonly at: number;
  readonly colour: string;
  readonly dashed?: boolean;
  readonly title?: string;
}

export interface HistogramChartProps {
  readonly h: HistogramResult;
  /** Bar fill per row, by bin centre. Row 0 is drawn last (on top). */
  readonly fills: readonly ((centre: number) => string)[];
  readonly markers?: readonly HistogramMarker[];
  readonly width?: number;
  readonly height?: number;
  readonly label: string;
  /** The value range drawn, and how many bars across it: source bins are summed into the bar their
   *  centre falls in, and those outside are left out (a long tail would squash the rest into a few
   *  bars). Default: the histogram's own range and bins. */
  readonly view?: { readonly lo: number; readonly hi: number; readonly bars: number };
  /** Called with the value at the click. */
  readonly onPick?: (value: number) => void;
  /** Called with the drawn bar under the pointer as a value range [lo, hi), and null on leaving. */
  readonly onHoverBar?: (bar: HoveredBar | null) => void;
}

/** Each row is scaled to its own peak, so a small inside region's shape is readable beside a large outside. */
export function HistogramChart({ h, fills, markers = [], width = 220, height = 54, label, view, onPick, onHoverBar }: HistogramChartProps) {
  /** The hovered bar, with the domain it was a bar of: a new domain makes the same index another range. */
  const [hovered, setHovered] = useState<{ bar: number; domain: string }>();
  const PAD = 2;
  const AXIS = 11;
  const plotH = height - AXIS - PAD;
  const lo = view?.lo ?? h.lo;
  const hi = view?.hi ?? h.hi;
  const bars = view?.bars ?? h.bins;
  const xOf = (v: number) => PAD + ((v - lo) / (hi - lo || 1)) * (width - 2 * PAD);
  const bw = (width - 2 * PAD) / bars;
  const domain = `${lo},${hi},${bars}`;
  const hoverBar = hovered && hovered.domain === domain ? hovered.bar : undefined;
  /** Row `r`'s counts summed into the drawn bars. */
  const barCounts = (r: number) => {
    const out = new Float64Array(bars);
    for (let b = 0; b < h.bins; b++) {
      const centre = h.lo + ((b + 0.5) / h.bins) * (h.hi - h.lo);
      const k = Math.floor(((centre - lo) / (hi - lo || 1)) * bars);
      if (k >= 0 && k < bars) out[k]! += h.counts[r * h.bins + b]!;
    }
    return out;
  };
  const rows = Array.from({ length: Math.min(h.rows, fills.length) }, (_, r) => r).reverse();
  const fmt = (v: number) => (Math.abs(v) >= 100 || v === 0 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
  return (
    <svg
      className="chart hist"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
      style={onPick ? { cursor: "crosshair" } : undefined}
      onPointerMove={
        onHoverBar
          ? (e) => {
              const box = e.currentTarget.getBoundingClientRect();
              const x = ((e.clientX - box.left) / box.width) * width;
              const b = Math.floor((x - PAD) / bw);
              if (b === hoverBar) return;
              if (b < 0 || b >= bars) {
                setHovered(undefined);
                onHoverBar(null);
                return;
              }
              setHovered({ bar: b, domain });
              onHoverBar({ lo: lo + (b / bars) * (hi - lo), hi: lo + ((b + 1) / bars) * (hi - lo), closed: b === bars - 1 && hi === h.hi });
            }
          : undefined
      }
      onPointerLeave={
        onHoverBar
          ? () => {
              setHovered(undefined);
              onHoverBar(null);
            }
          : undefined
      }
      onClick={
        onPick
          ? (e) => {
              const box = e.currentTarget.getBoundingClientRect();
              const x = ((e.clientX - box.left) / box.width) * width;
              onPick(lo + ((x - PAD) / (width - 2 * PAD)) * (hi - lo));
            }
          : undefined
      }
    >
      <title>{label}</title>
      {rows.map((r) => {
        const counts = barCounts(r);
        const peak = Math.max(1, ...counts);
        const fill = fills[r]!;
        return (
          <g key={r} opacity={h.rows > 1 && r === 1 ? 0.55 : 1}>
            {Array.from(counts, (c, b) => {
              if (c === 0) return null;
              const bh = Math.max(1, (c / peak) * plotH);
              const centre = lo + ((b + 0.5) / bars) * (hi - lo);
              return <rect key={b} x={PAD + b * bw} y={PAD + plotH - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill={fill(centre)} />;
            })}
          </g>
        );
      })}
      {hoverBar !== undefined && (
        <rect x={PAD + hoverBar * bw} y={PAD} width={bw} height={plotH} fill="rgba(251,191,36,0.25)" stroke="#fbbf24" strokeWidth={0.75} />
      )}
      {markers
        .filter((m) => m.at >= lo && m.at <= hi)
        .map((m, i) => (
          <line
            key={i}
            x1={xOf(m.at)}
            x2={xOf(m.at)}
            y1={PAD}
            y2={PAD + plotH}
            stroke={m.colour}
            strokeWidth={1.25}
            strokeDasharray={m.dashed ? "3 2" : undefined}
          >
            {m.title && <title>{m.title}</title>}
          </line>
        ))}
      <line x1={PAD} x2={width - PAD} y1={PAD + plotH + 0.5} y2={PAD + plotH + 0.5} stroke="#334155" />
      <text x={PAD} y={height - 1}>
        {fmt(lo)}
      </text>
      <text x={width - PAD} y={height - 1} textAnchor="end">
        {fmt(hi)}
      </text>
    </svg>
  );
}
