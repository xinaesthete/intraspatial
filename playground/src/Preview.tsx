// Render a pulled FieldValue: grids/matrices as a heatmap, a persistence diagram as
// a birth/death scatter, scalars/points as a compact summary.
import { useEffect, useRef } from "react";
import type { FieldValue } from "../../src/gpu/graph";

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

function drawScatter(canvas: HTMLCanvasElement, data: ArrayLike<number>) {
  const S = 320, pad = 16;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#11131a"; ctx.fillRect(0, 0, S, S);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i + 1 < data.length; i += 2) {
    const x = data[i]!, y = data[i + 1]!;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const mx = (minX + maxX) / 2, my = (minY + maxY) / 2; // centre, keep aspect square
  const px = (x: number) => pad + ((x - mx) / span + 0.5) * (S - 2 * pad);
  const py = (y: number) => S - (pad + ((y - my) / span + 0.5) * (S - 2 * pad)); // world +Y up
  ctx.fillStyle = "#6ba8ff";
  for (let i = 0; i + 1 < data.length; i += 2) {
    ctx.beginPath(); ctx.arc(px(data[i]!), py(data[i + 1]!), 2.5, 0, Math.PI * 2); ctx.fill();
  }
}

function drawPersistence(canvas: HTMLCanvasElement, pairs: PersistencePair[]) {
  const S = 320, pad = 28;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#11131a"; ctx.fillRect(0, 0, S, S);
  let maxV = 0.001;
  for (const p of pairs) maxV = Math.max(maxV, p.birth, Number.isFinite(p.death) ? p.death : 0);
  const map = (v: number) => pad + (v / maxV) * (S - 2 * pad);
  // diagonal
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

export function Preview({ value, error }: { value: FieldValue | null; error?: string | null }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !value) return;
    const s = value.shape;
    if (s.kind === "grid" && value.data) drawHeatmap(canvas, value.data, s.width, s.height);
    else if (s.kind === "matrix" && value.data) drawHeatmap(canvas, value.data, s.cols, s.rows);
    else if (s.kind === "points" && value.data) drawScatter(canvas, value.data);
    else if (s.kind === "opaque") {
      const pairs = (value.payload as { pairs?: PersistencePair[] })?.pairs ?? [];
      drawPersistence(canvas, pairs);
    }
  }, [value]);

  if (error) return <div className="preview-error">⚠ {error}</div>;
  if (!value) return <div className="preview-empty">Select a node and press <b>Run</b> to preview its output.</div>;

  const s = value.shape;
  const summary = (() => {
    if (s.kind === "grid") return `grid ${s.width}×${s.height}`;
    if (s.kind === "matrix") return `matrix ${s.rows}×${s.cols}`;
    if (s.kind === "points") return `points ×${s.n}`;
    if (s.kind === "scalar") return `scalar ${value.data?.[0]?.toFixed(4) ?? "?"}`;
    if (s.kind === "opaque") {
      const pairs = (value.payload as { pairs?: PersistencePair[] })?.pairs ?? [];
      const h1 = pairs.filter((p) => p.dim === 1 && Number.isFinite(p.death)).length;
      return `${s.name} — ${pairs.length} features, ${h1} loops (H₁)`;
    }
    return "value";
  })();

  const showCanvas = s.kind === "grid" || s.kind === "matrix" || s.kind === "opaque" || s.kind === "points";
  return (
    <div className="preview">
      <div className="preview-summary">{summary}</div>
      {showCanvas && <canvas ref={ref} className="preview-canvas" />}
    </div>
  );
}
