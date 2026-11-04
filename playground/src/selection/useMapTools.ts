// PROTOTYPE: the map's tools (hand, wand, brush) and the keyboard around them, Photoshop-style: H, W
// and B pick the tool, Shift/Alt/Cmd pick the op, Space pans, [ ] and Cmd/Ctrl+scroll size the tool.

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerInfo } from "../deck/SpatialDeckView";
import type { MapTool } from "./wandTools";
import type { PressOp } from "./worldMask";

/** The op for the keys held: Shift add, Alt subtract, both intersect; Cmd/Ctrl+Alt removes a piece. */
const pressOp = (shift: boolean, alt: boolean, cmd: boolean): PressOp =>
  alt && cmd ? "piece" : shift && alt ? "intersect" : shift ? "add" : alt ? "subtract" : "replace";

/** The brush radius scaled by `f`, within 5 µm and 5 mm. */
const sizeBrush = (r: number, f: number): number => Math.min(5000, Math.max(5, r * f));

/** A key event typed into a field, which tool keys must leave alone; a focused checkbox or slider is
 *  fine (it is what was last touched). */
const typing = (t: EventTarget | null) =>
  (t instanceof HTMLInputElement && !["checkbox", "radio", "range"].includes(t.type)) || t instanceof HTMLTextAreaElement;

/**
 * The tool and its keyboard. Tools only paint while `enabled` (the wand needs the boundary field, and
 * the mask is drawn with it); choosing a painting tool calls `onPaintTool` so the caller can enable
 * them. `resizeWand` maps the wand's σ (in r) to its new value.
 */
export function useMapTools(enabled: boolean, onPaintTool: () => void, resizeWand: (f: (sigma: number) => number) => void) {
  const [tool, setTool] = useState<MapTool>("hand");
  const chooseTool = useCallback(
    (t: MapTool) => {
      setTool(t);
      if (t !== "hand") onPaintTool();
    },
    [onPaintTool],
  );
  const active: MapTool = enabled ? tool : "hand";
  /** The brush's radius, in the transcripts' frame (µm). */
  const [brushR, setBrushR] = useState(100);

  /** Space is held: drags pan, and deck still reports them, so they must not also paint. */
  const panning = useRef(false);
  /** The modifier keys held, from the pointer press and from the keyboard: a drag's start event is its
   *  first move, which need not carry them. */
  const pressedWith = useRef({ shift: false, alt: false });
  /** `size`: Cmd or Ctrl really held (from the keyboard, so a trackpad pinch, which the browser
   *  reports as a Ctrl+wheel, still zooms). */
  const keysHeld = useRef({ shift: false, alt: false, size: false });
  /** The same, as state: the cursor shows the op a press would apply. */
  const [heldOp, setHeldOp] = useState<PressOp>("replace");
  useEffect(() => {
    const press = (e: PointerEvent) => {
      pressedWith.current = { shift: e.shiftKey, alt: e.altKey };
    };
    const key = (e: KeyboardEvent) => {
      keysHeld.current = { shift: e.shiftKey, alt: e.altKey, size: e.metaKey || e.ctrlKey };
      setHeldOp(pressOp(e.shiftKey, e.altKey, e.metaKey || e.ctrlKey));
    };
    globalThis.addEventListener("pointerdown", press, { capture: true });
    globalThis.addEventListener("keydown", key, { capture: true });
    globalThis.addEventListener("keyup", key, { capture: true });
    return () => {
      globalThis.removeEventListener("pointerdown", press, { capture: true });
      globalThis.removeEventListener("keydown", key, { capture: true });
      globalThis.removeEventListener("keyup", key, { capture: true });
    };
  }, []);
  /** The op a press applies. A drag's start event is its first move, so the keys held when the button
   *  went down (or on the keyboard) count too. */
  const opNow = useCallback((event?: PointerInfo): PressOp => {
    const shift = !!event?.srcEvent?.shiftKey || pressedWith.current.shift || keysHeld.current.shift;
    const alt = !!event?.srcEvent?.altKey || pressedWith.current.alt || keysHeld.current.alt;
    return pressOp(shift, alt, keysHeld.current.size);
  }, []);

  const brushOn = active === "brush";
  // [ and ] shrink and grow the patch, through the scale its edge is judged at.
  const handOn = active === "hand";
  useEffect(() => {
    // The hand has no size.
    if (!enabled || handOn) return;
    const on = (e: KeyboardEvent) => {
      if (typing(e.target)) return;
      if (e.key !== "[" && e.key !== "]") return;
      const grow = e.key === "]";
      if (brushOn) setBrushR((r) => sizeBrush(r, grow ? 1.25 : 0.8));
      else if (grow) resizeWand((s) => Math.min(3, Math.round((s + 0.25) * 4) / 4));
      else resizeWand((s) => Math.max(0.25, Math.round((s - 0.25) * 4) / 4));
    };
    globalThis.addEventListener("keydown", on);
    return () => globalThis.removeEventListener("keydown", on);
  }, [enabled, handOn, brushOn, resizeWand]);
  // Cmd/Ctrl + scroll sizes the tool (the brush, or the wand's σ) instead of zooming. A native,
  // non-passive capture listener: it runs before deck's and can stop the zoom.
  const [mapBox, setMapBox] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = mapBox;
    if (!el || active === "hand") return;
    const on = (e: WheelEvent) => {
      if (!keysHeld.current.size) return;
      e.preventDefault();
      e.stopPropagation();
      const f = Math.exp(-e.deltaY * 0.002);
      if (active === "brush") setBrushR((r) => sizeBrush(r, f));
      else resizeWand((s) => Math.min(3, Math.max(0.25, s * f)));
    };
    el.addEventListener("wheel", on, { capture: true, passive: false });
    return () => el.removeEventListener("wheel", on, { capture: true });
  }, [mapBox, active, resizeWand]);
  // Space held: drag pans instead of painting, as Photoshop's temporary hand tool.
  const [spaceHeld, setSpaceHeld] = useState(false);
  useEffect(() => {
    if (active === "hand") return;
    const down = (e: KeyboardEvent) => {
      if (e.key !== " " || typing(e.target)) return;
      e.preventDefault();
      panning.current = true;
      setSpaceHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key !== " ") return;
      panning.current = false;
      setSpaceHeld(false);
    };
    const blur = () => {
      panning.current = false;
      setSpaceHeld(false);
    };
    globalThis.addEventListener("keydown", down);
    globalThis.addEventListener("keyup", up);
    globalThis.addEventListener("blur", blur);
    return () => {
      globalThis.removeEventListener("keydown", down);
      globalThis.removeEventListener("keyup", up);
      globalThis.removeEventListener("blur", blur);
      // Space's release may come while these listeners are gone (under Hand): don't stay panning.
      blur();
    };
  }, [active]);
  // H, W and B pick the tool, as in Photoshop.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
      if (e.key === "h" || e.key === "H") chooseTool("hand");
      else if (e.key === "w" || e.key === "W") chooseTool("wand");
      else if (e.key === "b" || e.key === "B") chooseTool("brush");
    };
    globalThis.addEventListener("keydown", on);
    return () => globalThis.removeEventListener("keydown", on);
  }, [chooseTool]);

  /** Whether a drag pans now (Space held), for event handlers. */
  const isPanning = useCallback(() => panning.current, []);
  /** The map's container, as a callback ref: Cmd/Ctrl+scroll is caught there. */
  const attachMap = setMapBox;
  return { active, chooseTool, brushR, heldOp, opNow, isPanning, spaceHeld, attachMap };
}

export type MapTools = ReturnType<typeof useMapTools>;
