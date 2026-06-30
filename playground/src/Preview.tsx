// Render a pulled FieldValue: grids/matrices as a heatmap, a persistence diagram as
// a birth/death scatter, scalars/points as a compact summary. Multi-lane elements
// (complex / vec / quaternion) are projected to a scalar before drawing — magnitude
// by default, or a chosen lane (re/im, x/y/z…) via the selector.
import { useEffect, useRef, useState } from "react";
import type { ElementType, FieldValue } from "../../src/gpu/graph";
import { basisLabel, basisOf, elementLanes } from "../../src/gpu/graph";
import { PointsScatter } from "./PointsScatter";

interface PersistencePair { dim: number; birth: number; death: number }

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

// black -> red -> yellow -> white "hot" ramp
function hot(t: number): [number, number, number] {
  const u = clamp01(t);
  return [255 * clamp01(u * 3), 255 * clamp01(u * 3 - 1), 255 * clamp01(u * 3 - 2)];
}

function drawHeatmap(canvas: HTMLCanvasElement, data: ArrayLike<number>, w: number, h: number) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) { const v = data[i]!; if (v < min) min = v; if (v > max) max = v; }
  const span = max - min || 1;
  const scale = Math.max(1, Math.floor(320 / Math.max(w, h)));
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = hot((data[i]! - min) / span);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  tmp.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

/** Project an interleaved multi-lane field to a scalar field of `cells` values:
 *  lane < 0 → per-sample magnitude, otherwise that lane. */
function project(data: ArrayLike<number>, cells: number, lanes: number, lane: number): Float32Array {
  if (lanes <= 1) return data instanceof Float32Array ? data : Float32Array.from(data);
  const out = new Float32Array(cells);
  if (lane < 0) {
    for (let i = 0; i < cells; i++) {
      let s = 0;
      for (let c = 0; c < lanes; c++) { const v = data[i * lanes + c]!; s += v * v; }
      out[i] = Math.sqrt(s);
    }
  } else {
    for (let i = 0; i < cells; i++) out[i] = data[i * lanes + lane]!;
  }
  return out;
}

function laneLabel(el: ElementType | undefined, i: number): string {
  if (el?.kind === "complex") return ["re", "im"][i] ?? `lane ${i}`;
  if (el?.kind === "quaternion") return ["w", "x", "y", "z"][i] ?? `lane ${i}`;
  if (el?.kind === "vec") return ["x", "y", "z", "w"][i] ?? `lane ${i}`;
  return `lane ${i}`;
}

function drawPersistence(canvas: HTMLCanvasElement, pairs: PersistencePair[]) {
  const S = 320, pad = 28;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#11131a"; ctx.fillRect(0, 0, S, S);
  let maxV = 0.001;
  for (const p of pairs) maxV = Math.max(maxV, p.birth, Number.isFinite(p.death) ? p.death : 0);
  const map = (v: number) => pad + (v / maxV) * (S - 2 * pad);
  ctx.strokeStyle = "#333a4a"; ctx.beginPath(); ctx.moveTo(pad, S - pad); ctx.lineTo(S - pad, pad); ctx.stroke();
  for (const p of pairs) {
    const x = map(p.birth);
    const y = S - map(Number.isFinite(p.death) ? p.death : maxV);
    ctx.fillStyle = p.dim === 1 ? "#ff6b8a" : "#6ba8ff";
    ctx.beginPath(); ctx.arc(x, y, p.dim === 1 ? 5 : 3.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = "#8a93a6"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("birth →", S - 64, S - 8);
  ctx.save(); ctx.translate(10, 60); ctx.rotate(-Math.PI / 2); ctx.fillText("death →", 0, 0); ctx.restore();
}

export function Preview({ value, error, stale }: { value: FieldValue | null; error?: string | null; stale?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  // -1 = magnitude; 0..lanes-1 = a specific lane (re/im, x/y/z…).
  const [lane, setLane] = useState(-1);
  const lanes = value?.element ? elementLanes(value.element) : 1;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !value) return;
    const s = value.shape;
    if (s.kind === "grid" && value.data) drawHeatmap(canvas, project(value.data, s.width * s.height, lanes, lane), s.width, s.height);
    else if (s.kind === "matrix" && value.data) drawHeatmap(canvas, value.data, s.cols, s.rows);
    else if (s.kind === "opaque") {
      const pairs = (value.payload as { pairs?: PersistencePair[] })?.pairs ?? [];
      drawPersistence(canvas, pairs);
    }
  }, [value, lane, lanes]);

  const s = value?.shape;
  const summary = value && s ? describe(value, s, lanes) : "";
  const showCanvas = s ? s.kind === "grid" || s.kind === "matrix" || s.kind === "opaque" : false;

  return (
    <div className="preview">
      {error ? (
        <div className="preview-error">⚠ {error}</div>
      ) : !value || !s ? (
        <div className="preview-empty">Select a node and press <b>Run</b> to preview its output.</div>
      ) : (
        <>
          <div className="preview-summary">
            {summary}
            {stale && <span className="stale"> · stale — re-run</span>}
          </div>
          {lanes > 1 && s.kind === "grid" && (
            <select className="preview-lane" value={lane} onChange={(e) => setLane(Number(e.target.value))}>
              <option value={-1}>magnitude |·|</option>
              {Array.from({ length: lanes }, (_, i) => (
                <option key={i} value={i}>{laneLabel(value.element, i)}</option>
              ))}
            </select>
          )}
          {showCanvas && <canvas ref={ref} className={`preview-canvas${stale ? " is-stale" : ""}`} />}
          {s.kind === "points" && value.data && <PointsScatter data={value.data} />}
        </>
      )}
    </div>
  );
}

function describe(value: FieldValue, s: NonNullable<FieldValue["shape"]>, lanes: number): string {
  const el = lanes > 1 && value.element ? `${value.element.kind === "vec" ? `vec${value.element.n}` : value.element.kind} ` : "";
  const b = basisOf(value);
  const basis = b.kind !== "spatial" ? ` · ${basisLabel(b)}` : "";
  if (s.kind === "grid") return `${el}grid ${s.width}×${s.height}${basis}`;
  if (s.kind === "matrix") return `matrix ${s.rows}×${s.cols}`;
  if (s.kind === "points") return `points ×${s.n}`;
  if (s.kind === "scalar") return `scalar ${value.data?.[0]?.toFixed(4) ?? "?"}`;
  if (s.kind === "opaque") {
    const pairs = (value.payload as { pairs?: PersistencePair[] })?.pairs ?? [];
    const h1 = pairs.filter((p) => p.dim === 1 && Number.isFinite(p.death)).length;
    return `${s.name} — ${pairs.length} features, ${h1} loops (H₁)`;
  }
  return "value";
}
