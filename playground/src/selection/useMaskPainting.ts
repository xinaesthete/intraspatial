// PROTOTYPE: painting a world-space mask with the map's tools. Hovering seeds the wand's patch; a click
// or a drag paints it (or the brush's disc) into the mask, combined Photoshop-style by the op held.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Affine2 } from "../../../src/spatial/ngffTransform";
import type { PointerInfo } from "../deck/SpatialDeckView";
import { toolCursor } from "./wandTools";
import type { MapTools } from "./useMapTools";
import {
  apply,
  discOnRaster,
  footprint,
  gridFor,
  islandAt,
  type MaskOp,
  matchingPiece,
  onRaster,
  type Raster,
  smoothOutline,
  sweep,
  type WorldMask,
} from "./worldMask";

/** A press on the map: deck's click, or the start, moves and end of a drag. */
export type PressPhase = "click" | "start" | "move" | "end";

/** What the wand paints at a seed: a 0/1 patch on the window's map, optionally graded. */
export interface SeededPatch {
  readonly mask: Float32Array;
  /** Membership in [0, 1] and where it is above zero, when the patch is graded. */
  readonly weight?: Float32Array;
  readonly support?: Float32Array;
  readonly seed: readonly [number, number];
}

/** The wand's patch at a seed; it may take a while (a graph pull), so it is asynchronous. */
export type PatchAt<P> = (seed: readonly [number, number]) => Promise<P | undefined>;

/**
 * `patchAt(seed)`, kept to the latest seed: one computation at a time, and when the seed moves while
 * one runs, the next starts from wherever it is by then, dropping the positions in between rather
 * than falling behind. Each result is shown as it lands, so a slow patch still follows the pointer.
 */
function useLatest<P>(seed: readonly [number, number] | undefined, patchAt: PatchAt<P> | undefined): P | undefined {
  const [patch, setPatch] = useState<P>();
  const wanted = useRef<{ seed: readonly [number, number]; patchAt: PatchAt<P> }>(undefined);
  const running = useRef(false);
  useEffect(() => {
    wanted.current = seed && patchAt ? { seed, patchAt } : undefined;
    if (!wanted.current || running.current) return;
    running.current = true;
    void (async () => {
      try {
        for (let job = wanted.current; job;) {
          const p = await job.patchAt(job.seed);
          // A result for a seed since left (off the map) is not shown.
          if (wanted.current) setPatch(p);
          job = wanted.current === job ? undefined : wanted.current;
        }
      } catch (err: unknown) {
        console.error("wand patch", err);
      } finally {
        running.current = false;
      }
    })();
  }, [seed, patchAt]);
  // Nothing wanted, nothing shown, whatever last landed.
  return seed && patchAt ? patch : undefined;
}

/** A paint stroke: the union of the patches painted since the button went down. */
interface Stroke {
  readonly op: MaskOp;
  /** Removing whole pieces (Cmd/Ctrl+Alt), not what matches the patch. */
  readonly piece?: boolean;
  /** The mask when the stroke began: the stroke is combined with this, not with itself. */
  readonly base: WorldMask | undefined;
  readonly cell: number;
  readonly keys: Set<number>;
  /** The button is still down. */
  open: boolean;
  /** The seed whose patch is awaited (the `seed` set for it), if any. Not a flag: a move can land
   *  between a render and its effect, and must not be cleared by the previous patch arriving. */
  want: readonly [number, number] | undefined;
  /** The brush's last position, the start of the next swept segment. */
  last?: readonly [number, number];
}

/**
 * The mask, painted on `res`'s map (the window's) with `tools`. `toFrame` takes deck's world position to
 * the transcripts' frame, the mask's. `patchAt` is the wand's patch at a seed, absent while the wand
 * cannot paint. `onPaint` runs as each wand patch lands (the caller locks the metric). `smooth` draws
 * outlines from a smoothed distance field.
 */
export function useMaskPainting<P extends SeededPatch>(
  tools: MapTools,
  res: Raster | undefined,
  toFrame: Affine2 | undefined,
  patchAt: PatchAt<P> | undefined,
  onPaint: () => void,
  smooth: boolean,
) {
  const { active, brushR, heldOp, opNow, isPanning, spaceHeld } = tools;
  const wandOn = active === "wand";
  const brushOn = active === "brush";
  /** The wand's seed: the pointer, in the transcripts' frame. */
  const [seed, setSeed] = useState<readonly [number, number]>();
  const patch = useLatest(seed, patchAt);
  /** Where the brush is over the map. */
  const [brushAt, setBrushAt] = useState<readonly [number, number]>();
  // The persistent mask, on its own grid in the transcripts' frame, and the stroke being painted.
  const [mask, setMask] = useState<WorldMask>();
  const maskNow = useRef<WorldMask>(undefined);
  useEffect(() => {
    maskNow.current = mask;
  }, [mask]);
  const stroke = useRef<Stroke>(undefined);

  // Through refs, so the callbacks handed to deck stay stable.
  const onPressNow = useRef<(phase: PressPhase, x: number | undefined, y: number | undefined, event?: PointerInfo) => void>(undefined);
  /** The pointer over the map, or no position when it has left. */
  const onHoverNow = useRef<(x?: number, y?: number) => void>(undefined);
  /** The point last hovered or painted, to a quarter map pixel, so a still pointer recomputes nothing. */
  const hovered = useRef<string>(undefined);
  useEffect(() => {
    /** The pointer in the transcripts' frame. */
    const frameAt = (x: number, y: number): readonly [number, number] | undefined => {
      if (!toFrame) return undefined;
      const m = toFrame;
      return [m.a * x + m.c * y + m.tx, m.b * x + m.d * y + m.ty];
    };
    /** The pointer in the transcripts' frame, or undefined off the map. Not snapped to a map pixel:
     *  the seed's mixture is interpolated there, so the patch follows the pointer smoothly. */
    const pixelAt = (x: number, y: number): readonly [number, number] | undefined => {
      const f = frameAt(x, y);
      if (!wandOn || !f || !res) return undefined;
      const [minX, minY, maxX, maxY] = res.bbox;
      if (f[0] < minX || f[0] >= maxX || f[1] <= minY || f[1] > maxY) return undefined;
      return f;
    };
    /** Seed the patch at the pointer; the new seed, or undefined when it has not moved a quarter pixel. */
    const enter = (x: number, y: number, force: boolean): readonly [number, number] | undefined => {
      const p = pixelAt(x, y);
      const q = res ? (4 * res.width) / (res.bbox[2] - res.bbox[0]) : 1;
      const k = p ? `${Math.round(p[0] * q)},${Math.round(p[1] * q)}` : undefined;
      if (!p || (!force && k === hovered.current)) return undefined;
      hovered.current = k;
      const s: readonly [number, number] = [p[0], p[1]];
      setSeed(s);
      return s;
    };
    onHoverNow.current = (x, y) => {
      if (brushOn) {
        setBrushAt(x === undefined || y === undefined ? undefined : frameAt(x, y));
        return;
      }
      if (stroke.current?.open) return;
      // Off the map, or out of the canvas: no preview.
      if (x === undefined || y === undefined || !pixelAt(x, y)) {
        if (hovered.current === undefined) return;
        hovered.current = undefined;
        setSeed(undefined);
        return;
      }
      enter(x, y, false);
    };
    onPressNow.current = (phase, x, y, event) => {
      if (phase === "end") {
        const s = stroke.current;
        if (s) s.open = false;
        if (s && !s.want) stroke.current = undefined;
        return;
      }
      if (x === undefined || y === undefined || !res || isPanning()) return;
      if (brushOn) {
        const p = frameAt(x, y);
        if (!p) return;
        setBrushAt(p);
        let s = stroke.current;
        if (phase === "move") {
          if (!s?.open) return;
        } else {
          const base = maskNow.current;
          // The brush's subtract already removes exactly what it touches: "piece" is the same here.
          const pressed = opNow(event);
          const op = pressed === "piece" ? "subtract" : pressed;
          s = { op, base, cell: base?.cell ?? gridFor(res), keys: new Set(), open: phase === "start", want: undefined, last: p };
          stroke.current = s;
        }
        // A swept segment from the last position, so a fast drag leaves no gaps.
        sweep(s.keys, s.last ?? p, p, brushR, s.cell);
        s.last = p;
        setMask(apply(s.base, s.keys, s.cell, s.op));
        if (!s.open) stroke.current = undefined;
        return;
      }
      // Cmd/Ctrl+Alt removes the mask's connected pieces the pointer touches, whatever their mixture:
      // a click on a blob deletes the blob.
      const removing = phase === "move" ? !!stroke.current?.piece : opNow(event) === "piece";
      if (removing) {
        const p = pixelAt(x, y);
        let s = stroke.current;
        if (phase === "move") {
          if (!s?.open) return;
        } else {
          const base = maskNow.current;
          s = {
            op: "subtract",
            piece: true,
            base,
            cell: base?.cell ?? gridFor(res),
            keys: new Set(),
            open: phase === "start",
            want: undefined,
          };
          stroke.current = s;
        }
        if (p) {
          enter(x, y, phase !== "move");
          for (const k of islandAt(s.base, p)) s.keys.add(k);
          setMask(apply(s.base, s.keys, s.cell, s.op));
        }
        if (!s.open) stroke.current = undefined;
        return;
      }
      if (phase === "move") {
        const s = stroke.current;
        const next = s?.open ? enter(x, y, false) : undefined;
        if (s && next) s.want = next;
        return;
      }
      if (!pixelAt(x, y)) return;
      const pressed = opNow(event);
      const op = pressed === "piece" ? "subtract" : pressed;
      const base = maskNow.current;
      const s: Stroke = { op, base, cell: base?.cell ?? gridFor(res), keys: new Set(), open: phase === "start", want: undefined };
      stroke.current = s;
      s.want = enter(x, y, true);
    };
  }, [wandOn, brushOn, brushR, toFrame, res, opNow, isPanning]);
  // Each painted patch lands here once computed: into the stroke's union, then the mask is that union
  // combined with the mask as it was when the stroke began.
  useEffect(() => {
    const s = stroke.current;
    if (!s?.want || !patch || patch.seed !== s.want || !res) return;
    s.want = undefined;
    // Subtract removes only what matches the patch and is connected to the sampled point.
    const touched = s.op === "subtract" ? matchingPiece(s.base, patch.mask, res, patch.seed) : footprint(patch.mask, res, s.cell);
    for (const k of touched) s.keys.add(k);
    setMask(apply(s.base, s.keys, s.cell, s.op));
    if (!s.open) stroke.current = undefined;
    onPaint();
  }, [patch, res, onPaint]);

  // Until the wand has a patch rule it cannot paint, so a drag still pans rather than doing nothing
  // (which reads as broken); the cursor says it is waiting.
  const wandReady = wandOn && !!patchAt;
  // The brush needs only a window, for the mask's grid.
  const painting = (wandReady || (brushOn && !!res)) && !spaceHeld;
  const cursor = wandOn && !wandReady ? "progress" : toolCursor(active, heldOp, spaceHeld);

  /** A 0/1 raster as drawn: with smooth edges, as a smoothed distance field (no pixel staircase). */
  const drawn = useCallback((m: Float32Array, r: Raster) => (smooth ? smoothOutline(m, r.width, r.height) : m), [smooth]);
  const maskOnMap = useMemo(() => (mask && res && mask.keys.size > 0 ? drawn(onRaster(mask, res), res) : null), [mask, res, drawn]);
  /** What a press would paint, on this window's map: the wand's patch, or the brush's disc. One fill for
   *  both, so the two tools preview alike. */
  const preview = useMemo(() => {
    if (!painting || !res) return null;
    // The wand's removals preview what they would take: the whole piece under the pointer, or the part
    // of it that matches the patch and is connected to the pointer.
    const piece =
      wandOn && seed && heldOp === "piece"
        ? islandAt(mask, seed)
        : wandOn && seed && heldOp === "subtract" && patch
          ? matchingPiece(mask, patch.mask, res, seed)
          : undefined;
    // The wand's plain patch with graded necks: its whole support, faded by membership (a press
    // paints where that is at least a half).
    const graded = !piece && wandOn && patch?.support && patch.weight ? patch : undefined;
    const m = piece
      ? onRaster({ cell: mask?.cell ?? 1, keys: piece }, res)
      : graded
        ? graded.support
        : wandOn
          ? patch?.mask
          : brushOn && brushAt
            ? discOnRaster(brushAt, brushR, res)
            : undefined;
    return m ? { cell: drawn(m, res), weight: graded?.weight ?? null } : null;
  }, [painting, res, wandOn, heldOp, seed, mask, patch, brushOn, brushAt, brushR, drawn]);

  const onPress = useCallback(
    (phase: PressPhase, x: number | undefined, y: number | undefined, event?: PointerInfo) => onPressNow.current?.(phase, x, y, event),
    [],
  );
  const onHover = useCallback((x?: number, y?: number) => onHoverNow.current?.(x, y), []);
  const clear = useCallback(() => setMask(undefined), []);
  return { mask, clear, maskOnMap, preview, patch, brushAt, painting, cursor, onPress, onHover };
}
