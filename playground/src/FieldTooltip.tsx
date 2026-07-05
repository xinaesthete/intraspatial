// Hover tooltip for a port or edge: the full resolved type (shape · element · basis ·
// dtype) and a small view of the data flowing through it (a mini heatmap for grids /
// matrices, the value for scalars). The value comes from the last run; before running,
// only the declared port kind is known.
import { useEffect, useRef } from "react";
import type { FieldValue } from "../../src/gpu/graph";
import { basisLabel, basisOf, elementLanes } from "../../src/gpu/graph";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
function hot(t: number): [number, number, number] {
  const u = clamp01(t);
  return [255 * clamp01(u * 3), 255 * clamp01(u * 3 - 1), 255 * clamp01(u * 3 - 2)];
}

/** Project interleaved multi-lane data to a scalar magnitude field. */
function project(data: ArrayLike<number>, cells: number, lanes: number): Float32Array {
  if (lanes <= 1) return data instanceof Float32Array ? data : Float32Array.from(data);
  const out = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    let s = 0;
    for (let c = 0; c < lanes; c++) {
      const v = data[i * lanes + c];
      if (v === undefined) throw new Error("Unexpected");
      s += v * v;
    }
    out[i] = Math.sqrt(s);
  }
  return out;
}

function typeLine(v: FieldValue): string {
  const s = v.shape;
  const shape =
    s.kind === "grid"
      ? `grid ${s.width}×${s.height}`
      : s.kind === "points"
        ? `points ×${s.n}`
        : s.kind === "matrix"
          ? `matrix ${s.rows}×${s.cols}`
          : s.kind === "scalar"
            ? "scalar"
            : s.name;
  const el = v.element && v.element.kind !== "scalar" ? ` · ${v.element.kind === "vec" ? `vec${v.element.n}` : v.element.kind}` : "";
  const b = basisOf(v);
  const basis = b.kind !== "spatial" ? ` · ${basisLabel(b)}` : "";
  return `${shape}${el}${basis} · ${v.dtype}`;
}

export function FieldTooltip({ title, kind, value, rect }: { title: string; kind: string; value?: FieldValue; rect: DOMRect }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const s = value?.shape;
  const drawable = !!value?.data && (s?.kind === "grid" || s?.kind === "matrix");

  useEffect(() => {
    const c = ref.current;
    if (!c || !value?.data || !s) return;
    let w = 0,
      h = 0;
    if (s.kind === "grid") {
      w = s.width;
      h = s.height;
    } else if (s.kind === "matrix") {
      w = s.cols;
      h = s.rows;
    } else return;
    const lanes = value.element ? elementLanes(value.element) : 1;
    const data = project(value.data, w * h, lanes);
    let min = Infinity,
      max = -Infinity;
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      if (v === undefined) throw new Error("Unexpected");
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const span = max - min || 1;
    const scale = Math.max(1, Math.floor(96 / Math.max(w, h)));
    c.width = w * scale;
    c.height = h * scale;
    const ctx = c.getContext("2d");
    if (!ctx) throw new Error("Couldn't get canvas context");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const [r, g, b] = hot(((data[i] ?? 0) - min) / span);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = w;
    tmp.height = h;
    tmp.getContext("2d")?.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, c.width, c.height);
  }, [value, s]);

  const left = Math.min(rect.right + 12, window.innerWidth - 230);
  const top = Math.max(8, Math.min(rect.top - 6, window.innerHeight - 190));

  return (
    <div className="field-tip" style={{ left, top }}>
      <div className="field-tip-head">
        <span className="field-tip-title">{title}</span>
        <span className="field-tip-kind">{kind}</span>
      </div>
      {value ? (
        <div className="field-tip-type">{typeLine(value)}</div>
      ) : (
        <div className="field-tip-type muted">run to inspect the data</div>
      )}
      {drawable && <canvas ref={ref} className="field-tip-canvas" />}
      {value && s?.kind === "scalar" && <div className="field-tip-scalar">{value.data?.[0]?.toFixed(5) ?? "—"}</div>}
      {value && s?.kind === "points" && <div className="field-tip-note">{((value.data?.length ?? 0) / 2) | 0} points</div>}
    </div>
  );
}
