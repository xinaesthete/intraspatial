// Hover tooltip for a port or edge: the full resolved type (shape · element · basis ·
// dtype) and a small view of the data flowing through it (a mini heatmap for grids /
// matrices, the value for scalars). The value comes from the last run; before running,
// only the declared port kind is known.
import { useEffect, useRef } from "react";
import type { FieldValue } from "../../src/gpu/graph";
import { basisLabel, basisOf, elementLanes } from "../../src/gpu/graph";
import type { BundleInfo } from "./PortHover";

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

/** The shape alone, as one line. Shared by the type line and the bundle member rows. */
function shapeLine(s: FieldValue["shape"]): string {
  switch (s.kind) {
    case "grid":
      return `grid ${s.width}×${s.height}`;
    case "points":
      return `points ×${s.n}`;
    case "matrix":
      return `matrix ${s.rows}×${s.cols}`;
    case "scalar":
      return "scalar";
    case "bundle":
      return `${s.name} ×${Object.keys(s.parts).length}`;
    default:
      return s.name;
  }
}

function typeLine(v: FieldValue): string {
  const s = v.shape;
  const shape = shapeLine(s);
  const el = v.element && v.element.kind !== "scalar" ? ` · ${v.element.kind === "vec" ? `vec${v.element.n}` : v.element.kind}` : "";
  const b = basisOf(v);
  const basis = b.kind !== "spatial" ? ` · ${basisLabel(b)}` : "";
  return `${shape}${el}${basis} · ${v.dtype}`;
}

/** The mini heatmap. Lifted out of the component so the component is a render function and this
 *  is the drawing, rather than one function doing both. */
function drawMini(c: HTMLCanvasElement, value: FieldValue, s: NonNullable<FieldValue["shape"]>): void {
  if (!value.data) return;
  let w = 0;
  let h = 0;
  if (s.kind === "grid") {
    w = s.width;
    h = s.height;
  } else if (s.kind === "matrix") {
    w = s.cols;
    h = s.rows;
  } else return;
  const lanes = value.element ? elementLanes(value.element) : 1;
  const data = project(value.data, w * h, lanes);
  let min = Infinity;
  let max = -Infinity;
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
}

/** A bundle is several values that are one value (ADR-0023), so "bundle" on its own says nothing
 *  useful on hover — list what is inside. The shapes come from the declared shape, which exists
 *  whether or not the graph has run; dtypes come from the run, when there is one. */
function BundleMembers({
  shape,
  value,
  bundle,
}: {
  shape?: Extract<FieldValue["shape"], { kind: "bundle" }>;
  value?: FieldValue;
  bundle?: BundleInfo;
}) {
  // After a run the shape knows each part's size; before one, the port still declares the member
  // NAMES, which is the half that never changes — and the half a reader most needs.
  const rows = shape
    ? Object.entries(shape.parts).map(([name, partShape]) => ({
        name,
        detail: `${shapeLine(partShape)}${value?.parts?.[name] ? ` · ${value.parts[name]!.dtype}` : ""}`,
      }))
    : (bundle?.parts ?? []).map((name) => ({ name, detail: "" }));
  if (!rows.length) return null;
  return (
    <div className="field-tip-parts">
      {rows.map((r) => (
        <div className="field-tip-part" key={r.name}>
          <span className="field-tip-part-name">{r.name}</span>
          <span className="field-tip-part-type">{r.detail}</span>
        </div>
      ))}
    </div>
  );
}

/** Everything below the header: the type line, then whichever view fits the value. Split out of
 *  `FieldTooltip` so the component that positions the tooltip is not also the one branching over
 *  five shape kinds. */
function ValueDetail({
  value,
  bundle,
  canvasRef,
  drawable,
}: {
  value?: FieldValue;
  bundle?: BundleInfo;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  drawable: boolean;
}) {
  const s = value?.shape;
  const showParts = s?.kind === "bundle" || (!value && !!bundle);
  return (
    <>
      {value ? (
        <div className="field-tip-type">{typeLine(value)}</div>
      ) : (
        <div className="field-tip-type muted">{bundle ? bundle.name : "run to inspect the data"}</div>
      )}
      {drawable && <canvas ref={canvasRef} className="field-tip-canvas" />}
      {s?.kind === "scalar" && <div className="field-tip-scalar">{value?.data?.[0]?.toFixed(5) ?? "—"}</div>}
      {s?.kind === "points" && <div className="field-tip-note">{((value?.data?.length ?? 0) / 2) | 0} points</div>}
      {showParts && <BundleMembers shape={s?.kind === "bundle" ? s : undefined} value={value} bundle={bundle} />}
    </>
  );
}

export function FieldTooltip({
  title,
  kind,
  value,
  rect,
  bundle,
}: {
  title: string;
  kind: string;
  value?: FieldValue;
  rect: DOMRect;
  bundle?: BundleInfo;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const s = value?.shape;
  const drawable = !!value?.data && (s?.kind === "grid" || s?.kind === "matrix");

  useEffect(() => {
    const c = ref.current;
    if (c && value && s) drawMini(c, value, s);
  }, [value, s]);

  const left = Math.min(rect.right + 12, window.innerWidth - 230);
  const top = Math.max(8, Math.min(rect.top - 6, window.innerHeight - 190));

  return (
    <div className="field-tip" style={{ left, top }}>
      <div className="field-tip-head">
        <span className="field-tip-title">{title}</span>
        <span className="field-tip-kind">{kind}</span>
      </div>
      <ValueDetail value={value} bundle={bundle} canvasRef={ref} drawable={drawable} />
    </div>
  );
}
