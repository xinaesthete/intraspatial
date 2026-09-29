// The whole slide at a glance, with the analysis window as a box to drag. Drawn y-down, like the
// image it shows. Click outside the box to move it there.

import { type PointerEvent, useEffect, useRef } from "react";
import type { Rect } from "../../../src/datasource/points";

export interface Overview {
  readonly bitmap: ImageBitmap;
  /** Image pixel → the points' own coordinates, 2×3 row-major. Any affine: registered H&E is often rotated. */
  readonly localFromPixel: ArrayLike<number>;
}

interface Props {
  readonly extent: Rect;
  readonly window: Rect;
  readonly onChange: (w: Rect) => void;
  readonly overview?: Overview;
  /** Tile outlines to show, e.g. the ones the current window needs. */
  readonly tiles?: readonly Rect[];
}

const WIDTH = 1200;

export function WindowPicker({ extent, window, onChange, overview, tiles }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const spanX = extent.maxX - extent.minX;
  const spanY = extent.maxY - extent.minY;
  const s = WIDTH / spanX;
  const height = Math.max(1, Math.round(spanY * s));
  const toPx = (x: number, y: number): [number, number] => [(x - extent.minX) * s, (y - extent.minY) * s];

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, WIDTH, height);
    if (overview) {
      const m = (i: number) => overview.localFromPixel[i] ?? 0;
      // canvas = s · (local − extent.min), local = M · pixel
      ctx.setTransform(s * m(0), s * m(3), s * m(1), s * m(4), s * (m(2) - extent.minX), s * (m(5) - extent.minY));
      ctx.drawImage(overview.bitmap, 0, 0);
      ctx.resetTransform();
    }
    const [wx0, wy0] = toPx(window.minX, window.minY);
    const [wx1, wy1] = toPx(window.maxX, window.maxY);
    // Dim everything outside the window.
    ctx.fillStyle = "rgba(2, 6, 23, 0.55)";
    ctx.fillRect(0, 0, WIDTH, wy0);
    ctx.fillRect(0, wy1, WIDTH, height - wy1);
    ctx.fillRect(0, wy0, wx0, wy1 - wy0);
    ctx.fillRect(wx1, wy0, WIDTH - wx1, wy1 - wy0);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 1;
    for (const t of tiles ?? []) {
      const [a, b] = toPx(t.minX, t.minY);
      const [c, d] = toPx(t.maxX, t.maxY);
      ctx.strokeRect(a + 0.5, b + 0.5, c - a, d - b);
    }
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2;
    ctx.strokeRect(wx0, wy0, wx1 - wx0, wy1 - wy0);
  });

  const local = (e: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [extent.minX + ((e.clientX - r.left) / r.width) * spanX, extent.minY + ((e.clientY - r.top) / r.height) * spanY];
  };
  const place = (cx: number, cy: number): Rect => {
    const w = window.maxX - window.minX;
    const h = window.maxY - window.minY;
    const minX = Math.min(Math.max(cx - w / 2, extent.minX), extent.maxX - w);
    const minY = Math.min(Math.max(cy - h / 2, extent.minY), extent.maxY - h);
    return { minX, minY, maxX: minX + w, maxY: minY + h };
  };
  const inside = (x: number, y: number) => x >= window.minX && x <= window.maxX && y >= window.minY && y <= window.maxY;

  return (
    <canvas
      ref={canvas}
      width={WIDTH}
      height={height}
      className="overview"
      onPointerDown={(e) => {
        const [x, y] = local(e);
        const cx = (window.minX + window.maxX) / 2;
        const cy = (window.minY + window.maxY) / 2;
        if (!inside(x, y)) onChange(place(x, y));
        drag.current = inside(x, y) ? { dx: cx - x, dy: cy - y } : { dx: 0, dy: 0 };
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!drag.current) return;
        const [x, y] = local(e);
        onChange(place(x + drag.current.dx, y + drag.current.dy));
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
    />
  );
}
