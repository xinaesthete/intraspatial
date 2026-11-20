// Draw structure onto a chain: FoldSynth's matrix lines. The square has a cell for every pair of
// particles; a line across it ties each pair it crosses with a spring. Drag on empty space to
// draw a line, drag an end or the body to move it, double-click or right-click to remove it.
import { useEffect, useMemo, useRef } from "react";
import { lineCells, type MatrixLine, snapEnd } from "../../../src/gpu/sim/springs/index";
import { drawMatrix } from "./draw";

type Part = "start" | "end" | "body";

interface Drag {
  readonly index: number;
  readonly part: Part;
  /** Pointer position, in cells, when the drag began. */
  readonly origin: readonly [number, number];
  /** The line as it was when the drag began. */
  readonly line: MatrixLine;
}

export function MapEditor(props: {
  /** Particles in the chain: the map is `size × size`. */
  size: number;
  lines: readonly MatrixLine[];
  onChange: (lines: MatrixLine[]) => void;
  /** Current model distances, row-major, shown behind the lines. */
  distances: Float32Array | null;
}) {
  const { size, lines, onChange, distances } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx && distances && distances.length === size * size) drawMatrix(ctx, distances, size, 12);
  }, [distances, size]);

  const cells = useMemo(
    () => lines.flatMap((line) => lineCells(line)).filter(([a, b]) => a >= 0 && b >= 0 && a < size && b < size && Math.abs(a - b) > 1),
    [lines, size],
  );

  /** Pointer position as a cell index, clamped to the map. */
  const cellAt = (e: React.PointerEvent): [number, number] => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    const clamp = (v: number) => Math.max(0, Math.min(size - 1, Math.round(v - 0.5)));
    return [clamp(((e.clientX - rect.left) / rect.width) * size), clamp(((e.clientY - rect.top) / rect.height) * size)];
  };

  const begin = (e: React.PointerEvent, index: number, part: Part, line: MatrixLine) => {
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    drag.current = { index, part, origin: cellAt(e), line };
  };

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const [a, b] = cellAt(e);
    const line = { a0: a, b0: b, a1: a, b1: b };
    onChange([...lines, line]);
    begin(e, lines.length, "end", line);
  };

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const [a, b] = cellAt(e);
    let next: MatrixLine;
    if (d.part === "body") {
      const da = a - d.origin[0];
      const db = b - d.origin[1];
      next = { a0: d.line.a0 + da, b0: d.line.b0 + db, a1: d.line.a1 + da, b1: d.line.b1 + db };
    } else if (d.part === "start") {
      const [sa, sb] = e.ctrlKey ? [a, b] : snapEnd(d.line.a1, d.line.b1, a, b);
      next = { ...d.line, a0: sa, b0: sb };
    } else {
      const [sa, sb] = e.ctrlKey ? [a, b] : snapEnd(d.line.a0, d.line.b0, a, b);
      next = { ...d.line, a1: sa, b1: sb };
    }
    onChange(lines.map((l, k) => (k === d.index ? next : l)));
  };

  const onUp = () => {
    drag.current = null;
  };

  const remove = (e: React.SyntheticEvent, index: number) => {
    e.preventDefault();
    e.stopPropagation();
    onChange(lines.filter((_, k) => k !== index));
  };

  return (
    <div className="map-editor">
      <canvas ref={canvasRef} width={size} height={size} className="map" />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${size} ${size}`}
        className="map-overlay"
        role="img"
        aria-label="Pair matrix editor"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onContextMenu={(e) => e.preventDefault()}
      >
        <line x1={0} y1={0} x2={size} y2={size} className="map-diagonal" />
        {cells.map(([a, b]) => (
          <g key={`${a},${b}`}>
            <rect x={a} y={b} width={1} height={1} className="map-cell" />
            <rect x={b} y={a} width={1} height={1} className="map-cell map-cell-mirror" />
          </g>
        ))}
        {lines.map((l, k) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: lines have no identity beyond their position in the list
          <g key={k} onDoubleClick={(e) => remove(e, k)} onContextMenu={(e) => remove(e, k)}>
            <line
              x1={l.a0 + 0.5}
              y1={l.b0 + 0.5}
              x2={l.a1 + 0.5}
              y2={l.b1 + 0.5}
              className="map-line"
              onPointerDown={(e) => e.button === 0 && begin(e, k, "body", l)}
            />
            <rect
              x={l.a0 - 0.5}
              y={l.b0 - 0.5}
              width={2}
              height={2}
              className="map-handle"
              onPointerDown={(e) => e.button === 0 && begin(e, k, "start", l)}
            />
            <rect
              x={l.a1 - 0.5}
              y={l.b1 - 0.5}
              width={2}
              height={2}
              className="map-handle"
              onPointerDown={(e) => e.button === 0 && begin(e, k, "end", l)}
            />
          </g>
        ))}
      </svg>
    </div>
  );
}
