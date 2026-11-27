// The wand's tolerance guidance, under its slider: how big the sample's patch is at every tolerance
// (with the points where it leaks into a neighbouring basin), and how the distance d is spread
// inside and outside the selection. All of it is over the map's window, at its raster's pixels.

import { floodAreaAt } from "../../../src/spatial/floodArea";
import { histogramRow, histogramSummary } from "../../../src/spatial/histogram";
import type { MapBand } from "./GramModesLayer";
import { HistogramChart, type HoveredBar } from "./HistogramChart";
import type { Wand } from "./useWand";

/** Hover links between the charts and the map, on a page that wires them. */
export interface WandHover {
  /** Raster pixel index under the map pointer: its d and the tolerance it joins at are marked. */
  readonly hoverPixel?: number | undefined;
  /** Called with the d histogram's bar under the pointer, to show its places on the map. */
  readonly onBand?: ((band: MapBand | null) => void) | undefined;
}

/** The slider's domain, which the curve shares so a click lands where the slider would. */
export const TOL_MIN = 0.05;
export const TOL_MAX = 5;

const W = 280;
const H = 110;
const L = 30;
const B = 14;
const T = 6;
const R = 6;

function AreaToleranceChart({ wand, joins }: { wand: Wand; joins?: number | undefined }) {
  const ac = wand.areaCurve;
  if (!ac) return null;
  const { curve, leaks } = ac;
  const xOf = (t: number) => L + (t / TOL_MAX) * (W - L - R);
  const tOf = (x: number) => ((x - L) / (W - L - R)) * TOL_MAX;
  const maxLog = Math.log10(Math.max(10, curve.pixels));
  const yOf = (a: number) => T + (1 - Math.log10(Math.max(1, a)) / maxLog) * (H - T - B);
  // One point per screen column, read off the exact step function at the column's right edge.
  const cols = W - L - R;
  let path = "";
  for (let i = 0; i <= cols; i++) {
    const x = L + i;
    const a = floodAreaAt(curve, tOf(x));
    path += `${i === 0 ? "M" : "L"}${x},${a > 0 ? yOf(a).toFixed(1) : (H - B).toFixed(1)}`;
  }
  const decades = Array.from({ length: Math.floor(maxLog) + 1 }, (_, k) => k);
  const shownLeaks = leaks.map((k) => curve.tol[k]!).filter((t) => t <= TOL_MAX);
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="patch area against tolerance"
      style={{ cursor: "crosshair" }}
      onClick={(e) => {
        const box = e.currentTarget.getBoundingClientRect();
        const t = tOf(((e.clientX - box.left) / box.width) * W);
        wand.setTolerance(Math.round(Math.min(TOL_MAX, Math.max(TOL_MIN, t)) * 100) / 100);
      }}
    >
      {decades.map((k) => (
        <g key={k}>
          <line x1={L} x2={W - R} y1={yOf(10 ** k)} y2={yOf(10 ** k)} stroke="#1e293b" />
          <text x={L - 3} y={yOf(10 ** k) + 3} textAnchor="end">
            {k === 0 ? "1" : `1e${k}`}
          </text>
        </g>
      ))}
      {shownLeaks.map((t, i) => (
        <line key={i} x1={xOf(t)} x2={xOf(t)} y1={T} y2={H - B} stroke="#f87171" strokeWidth={1}>
          <title>leak at tolerance {t.toFixed(3)}</title>
        </line>
      ))}
      <path d={path} fill="none" stroke="#38bdf8" strokeWidth={1.5} />
      {joins !== undefined && joins <= TOL_MAX && (
        <line x1={xOf(joins)} x2={xOf(joins)} y1={T} y2={H - B} stroke="#fbbf24" strokeWidth={1.25}>
          <title>the place under the pointer joins the patch at {joins.toFixed(2)}</title>
        </line>
      )}
      <line x1={xOf(wand.tolerance)} x2={xOf(wand.tolerance)} y1={T} y2={H - B} stroke="#e2e8f0" strokeDasharray="3 2" />
      <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="#334155" />
      {[0, 1, 2, 3, 4, 5].map((t) => (
        <text key={t} x={xOf(t)} y={H - 2} textAnchor="middle">
          {t}
        </text>
      ))}
      <text x={W - R} y={T + 8} textAnchor="end">
        px in the patch (log)
      </text>
    </svg>
  );
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "—");

export function WandCharts({ wand, hoverPixel, onBand }: { wand: Wand } & WandHover) {
  const h = wand.distanceHist;
  const atD = hoverPixel !== undefined ? wand.distance?.[hoverPixel] : undefined;
  const joinCost = hoverPixel !== undefined ? wand.areaCurve?.curve.nearestCosts?.[hoverPixel] : undefined;
  const joins = joinCost !== undefined && Number.isFinite(joinCost) ? joinCost : undefined;
  if (!wand.sample || (!h && !wand.areaCurve)) return null;
  const inside = h && h.totals[0]! > 0 ? histogramSummary(histogramRow(h, 0)) : undefined;
  const outside = h && h.totals[4]! > 0 ? histogramSummary(histogramRow(h, 1)) : undefined;
  const pixels = wand.areaCurve?.curve.pixels ?? 0;
  const leaks = wand.areaCurve ? wand.areaCurve.leaks.map((k) => wand.areaCurve!.curve.tol[k]!).filter((t) => t <= TOL_MAX) : [];
  return (
    <div className="wand-charts">
      <div className="picker-head">
        Patch size at every tolerance{" "}
        <span className="dim">
          (click to set it
          {wand.patchArea !== undefined && pixels > 0
            ? `; now ${wand.patchArea} px, ${((100 * wand.patchArea) / pixels).toPrecision(2)}% of the window`
            : ""}
          )
        </span>
      </div>
      <AreaToleranceChart wand={wand} joins={joins} />
      {atD !== undefined && (
        <p className="hint">
          <span className="raised">Under the pointer</span>: d {atD.toFixed(2)}
          {joins !== undefined ? `, joins the patch at tolerance ${joins.toFixed(2)}` : ", not reachable from the sample"}
        </p>
      )}
      {leaks.length > 0 && (
        <p className="hint">
          <span className="error">Red</span>: leaks, where the patch grows by half its size or more at a single tolerance (
          {leaks
            .slice(0, 4)
            .map((t) => t.toFixed(2))
            .join(", ")}
          {leaks.length > 4 ? ", …" : ""}). Just below one keeps the patch in its own basin.
        </p>
      )}
      {h && (
        <>
          <div className="picker-head">
            Distance from the sample over the window{" "}
            <span className="dim">
              ({wand.connected ? "blue: in the patch" : "blue: within tolerance"}, grey: the rest; 0–{TOL_MAX} shown)
            </span>
          </div>
          <HistogramChart
            h={h}
            width={280}
            height={60}
            label="histogram of the wand's distance d, inside and outside the selection"
            view={{ lo: 0, hi: TOL_MAX, bars: 70 }}
            fills={[() => "#38bdf8", () => "#64748b"]}
            markers={[
              { at: wand.tolerance, colour: "#e2e8f0", dashed: true, title: `tolerance ${wand.tolerance.toFixed(2)}` },
              ...(atD !== undefined ? [{ at: atD, colour: "#fbbf24", title: `the place under the pointer: d ${atD.toFixed(2)}` }] : []),
            ]}
            {...(onBand ? { onHoverBar: (bar: HoveredBar | null) => onBand(bar && { value: 3 as const, ...bar }) } : {})}
            onPick={(t) => wand.setTolerance(Math.round(Math.min(TOL_MAX, Math.max(TOL_MIN, t)) * 100) / 100)}
          />
          <p className="hint">
            median d inside {fmt(inside?.median ?? Number.NaN)} · outside {fmt(outside?.median ?? Number.NaN)}
            {inside ? ` (within ${inside.binWidth.toFixed(2)})` : ""}
          </p>
        </>
      )}
    </div>
  );
}
