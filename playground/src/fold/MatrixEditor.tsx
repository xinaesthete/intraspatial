// The pair matrix as an editor: FoldSynth's matrix lines, on a canvas the GPU draws from the
// model's own buffers (`pairMatrixRenderer`). Drag on empty space to draw a line, drag an end or
// the body to move it, double-click or right-click a line to remove it.
//
// Built to feel immediate. The canvas redraws every frame from a ref, so a line follows the pointer
// in the frame it moves, whatever React is doing; the page hears about the change at most once per
// frame.
import { useEffect, useRef } from "react";
import { type MatrixLine, SLOT_WORDS, snapEnd } from "../../../src/gpu/sim/springs/index";
import type { SimBuffers } from "../springs/springsLayers";
import { useElementSize } from "../hooks/useElementSize";
import { createPairMatrixRenderer } from "./pairMatrixRenderer";

type Part = "start" | "end" | "body";

interface Drag {
  readonly index: number;
  readonly part: Part;
  /** Pointer position, in cells, when the drag began. */
  readonly origin: readonly [number, number];
  /** The line as it was when the drag began. */
  readonly line: MatrixLine;
}

/** Which line, and which part of it, is at cell position `(a, b)` (fractional), or null. */
function hitLine(lines: readonly MatrixLine[], a: number, b: number): { index: number; part: Part } | null {
  // Topmost first: later lines are drawn over earlier ones.
  for (let k = lines.length - 1; k >= 0; k--) {
    const l = lines[k]!;
    const near = (x: number, y: number) => Math.max(Math.abs(a - (x + 0.5)), Math.abs(b - (y + 0.5))) <= 1.2;
    if (near(l.a1, l.b1)) return { index: k, part: "end" };
    if (near(l.a0, l.b0)) return { index: k, part: "start" };
    const dx = l.a1 - l.a0;
    const dy = l.b1 - l.b0;
    const t = Math.max(0, Math.min(1, ((a - 0.5 - l.a0) * dx + (b - 0.5 - l.b0) * dy) / Math.max(dx * dx + dy * dy, 1e-6)));
    if (Math.hypot(a - 0.5 - (l.a0 + dx * t), b - 0.5 - (l.b0 + dy * t)) <= 0.9) return { index: k, part: "body" };
  }
  return null;
}

export function MatrixEditor(props: {
  /** The device the model runs on, and the model: null while they start. */
  device: GPUDevice | null;
  gpu: SimBuffers | null;
  lines: readonly MatrixLine[];
  onChange: (lines: MatrixLine[]) => void;
  /** The distance drawn as brightest. */
  maxDistance?: number;
}) {
  const { device, gpu, lines, onChange, maxDistance = 12 } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useElementSize(canvasRef);
  const drag = useRef<Drag | null>(null);
  /** What is drawn: ahead of `lines` by at most a frame while dragging. */
  const shown = useRef<{ lines: readonly MatrixLine[]; active: number }>({ lines, active: -1 });
  const pending = useRef<MatrixLine[] | null>(null);
  const latest = useRef({ gpu, onChange, maxDistance });
  useEffect(() => {
    latest.current = { gpu, onChange, maxDistance };
    // The page's lines win unless a drag is ahead of them.
    if (!pending.current) shown.current = { ...shown.current, lines };
  });

  // Canvas pixels follow its size on screen.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const side = Math.max(1, Math.round(size.width * Math.min(devicePixelRatio, 2)));
    canvas.width = side;
    canvas.height = side;
  }, [size.width]);

  // One loop: hand the page any change, then draw.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!device || !canvas) return;
    let stopped = false;
    let raf = 0;
    let renderer: Awaited<ReturnType<typeof createPairMatrixRenderer>> | undefined;
    createPairMatrixRenderer(device, canvas)
      .then((r) => {
        if (stopped) return r.destroy();
        renderer = r;
        const tick = () => {
          if (stopped) return;
          if (pending.current) {
            latest.current.onChange(pending.current);
            pending.current = null;
          }
          const g = latest.current.gpu;
          const positions = g?.positions();
          const slots = typeof g?.slots === "function" ? g.slots() : g?.slots;
          if (g && positions && slots) {
            renderer?.draw({
              positions,
              slots,
              n: g.n,
              K: g.K,
              slotWords: g.slotWords ?? SLOT_WORDS,
              maxDistance: latest.current.maxDistance,
              lines: shown.current.lines,
              active: shown.current.active,
            });
          }
          raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      })
      .catch((e: unknown) => console.error("[pairMatrix]", e));
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      renderer?.destroy();
    };
  }, [device]);

  const n = gpu?.n ?? 1;
  /** Pointer position in cells: fractional, for hit tests. */
  const at = (e: React.PointerEvent | React.MouseEvent): [number, number] => {
    const rect = e.currentTarget.getBoundingClientRect();
    return [((e.clientX - rect.left) / rect.width) * n, ((e.clientY - rect.top) / rect.height) * n];
  };
  const whole = ([a, b]: [number, number]): [number, number] => {
    const clamp = (v: number) => Math.max(0, Math.min(n - 1, Math.round(v - 0.5)));
    return [clamp(a), clamp(b)];
  };
  /** Show a change now and queue it for the page. */
  const change = (next: MatrixLine[], active: number) => {
    shown.current = { lines: next, active };
    pending.current = next;
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const current = shown.current.lines;
    const hit = hitLine(current, ...at(e));
    const [a, b] = whole(at(e));
    if (hit) {
      drag.current = { ...hit, origin: [a, b], line: current[hit.index]! };
      shown.current = { lines: current, active: hit.index };
      return;
    }
    const line = { a0: a, b0: b, a1: a, b1: b };
    drag.current = { index: current.length, part: "end", origin: [a, b], line };
    change([...current, line], current.length);
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    const current = shown.current.lines;
    if (!d) {
      const active = hitLine(current, ...at(e))?.index ?? -1;
      if (active !== shown.current.active) shown.current = { lines: current, active };
      return;
    }
    const [a, b] = whole(at(e));
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
    const before = current[d.index];
    if (before && before.a0 === next.a0 && before.b0 === next.b0 && before.a1 === next.a1 && before.b1 === next.b1) return;
    change(
      current.map((l, k) => (k === d.index ? next : l)),
      d.index,
    );
  };

  const remove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const current = shown.current.lines;
    const hit = hitLine(current, ...at(e));
    if (hit)
      change(
        current.filter((_, k) => k !== hit.index),
        -1,
      );
  };

  return (
    <canvas
      ref={canvasRef}
      className="matrix-editor"
      role="img"
      aria-label="Pair matrix editor"
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={() => {
        drag.current = null;
      }}
      onPointerLeave={() => {
        if (!drag.current) shown.current = { ...shown.current, active: -1 };
      }}
      onDoubleClick={remove}
      onContextMenu={remove}
    />
  );
}
