// Pointer → world on a deck view, for tools that sample rather than pan: a click picks a point, and
// a shift-drag picks continuously while deck's own pan is held off. The handlers go on an element
// wrapping the view and exactly its size; `view` is the rectangle it shows (`SpatialDeckView`'s
// `onViewport`).
//
// Capture-phase handlers, so a shift-drag is stopped before it reaches deck's canvas: deck never
// sees it, so never pans. A plain click is let through — without movement deck does nothing with it.

import { type PointerEvent, useRef } from "react";
import type { Rect } from "../../../src/datasource/points";

/** The world point under `clientX, clientY` on `el`, which shows `view` (top edge = `view.minY`). */
function worldAt(el: Element, view: Rect, clientX: number, clientY: number): [number, number] {
  const r = el.getBoundingClientRect();
  return [
    view.minX + ((clientX - r.left) / r.width) * (view.maxX - view.minX),
    view.minY + ((clientY - r.top) / r.height) * (view.maxY - view.minY),
  ];
}

/** A press on a control over the map (deck's widgets, a button): it is the control's, not a pick. */
const onControl = (e: PointerEvent<HTMLElement>) =>
  e.target instanceof Element && e.target !== e.currentTarget && e.target.closest("button, a, input, select, .deck-widget") !== null;

/** Further than this, in CSS pixels, and a press is a pan, not a click. */
const CLICK_SLOP = 4;

export function useWorldPick(view: Rect | undefined, onPick: (x: number, y: number) => void) {
  const press = useRef<{ x: number; y: number; drag: boolean } | null>(null);
  const pick = (e: PointerEvent<HTMLElement>) => {
    if (view) onPick(...worldAt(e.currentTarget, view, e.clientX, e.clientY));
  };
  return {
    onPointerDownCapture: (e: PointerEvent<HTMLElement>) => {
      if (e.button !== 0 || onControl(e)) return;
      const drag = e.shiftKey;
      press.current = { x: e.clientX, y: e.clientY, drag };
      if (!drag) return;
      e.stopPropagation();
      e.currentTarget.setPointerCapture(e.pointerId);
      pick(e);
    },
    onPointerMoveCapture: (e: PointerEvent<HTMLElement>) => {
      if (!press.current?.drag) return;
      e.stopPropagation();
      if (e.buttons & 1) pick(e);
    },
    onPointerUpCapture: (e: PointerEvent<HTMLElement>) => {
      const p = press.current;
      press.current = null;
      if (!p) return;
      if (p.drag) {
        e.stopPropagation();
        e.currentTarget.releasePointerCapture(e.pointerId);
      } else if (Math.hypot(e.clientX - p.x, e.clientY - p.y) <= CLICK_SLOP) pick(e);
    },
  };
}
