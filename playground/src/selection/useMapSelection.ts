// PROTOTYPE (docs/wand-metric-and-regions.md): the map's selection tools for a mode-map page, in one
// hook: the boundary field, the wand's patch on it, the Hand/Wand/Brush tools, and the mask they paint.
// It hands back what the page wires in: deck's props (the tool widgets, drag handling), the layers that
// draw the boundaries, preview, mask and brush, the map's click and hover, and the panel's props.
//
// The settings come from `useBoundarySettings`, called before this so a page can act on them first
// (the transcript page computes the broader neighbourhood's Gram from them).

import type { Layer } from "@deck.gl/core";
import { PathLayer } from "@deck.gl/layers";
import type { Matrix4 } from "@math.gl/core";
import { useCallback, useMemo, useState } from "react";
// The tool widgets' own styles, for every page that has the tools.
import "@deck.gl/widgets/stylesheet.css";
import type { ModeBasis } from "../../../src/gpu/spatial/gramModes";
import type { CoLocationModes } from "../../../src/spatial/gram";
import type { Affine2 } from "../../../src/spatial/ngffTransform";
import type { PointerInfo } from "../deck/SpatialDeckView";
import { BoundaryLayer } from "../wand/BoundaryLayer";
import { type GramView, useBoundaryField } from "../wand/useBoundaryField";
import { useWandSettings } from "../wand/useWandSettings";
import { needsExperiments, type Patch, wandPatch, wandPatchOnGraph } from "../wand/wandPatch";
import { useMapTools } from "./useMapTools";
import { type PatchAt, type PressPhase, useMaskPainting } from "./useMaskPainting";
import { toolWidgets } from "./wandTools";
import type { PressOp } from "./worldMask";

/** BoundaryLayer's op code. */
const OP_INDEX: Record<PressOp, number> = { replace: 0, add: 1, subtract: 2, intersect: 3, piece: 2 };

/** A closed circle of 64 segments, for a PathLayer. */
function circle(c: readonly [number, number], r: number): [number, number][] {
  return Array.from({ length: 65 }, (_, i) => {
    const t = (i / 64) * 2 * Math.PI;
    return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)];
  });
}

/** Whether boundaries are shown, and the wand's settings. */
export function useBoundarySettings() {
  const [show, setShow] = useState(false);
  const wand = useWandSettings();
  return { show, setShow, wand };
}

export type BoundarySettings = ReturnType<typeof useBoundarySettings>;

export interface MapSelectionInput {
  /** The window's Gram result. */
  readonly g: GramView | undefined;
  /** The modes the map is painted with (a lock's, or g's own), and the lock's basis if any. */
  readonly modes: CoLocationModes | undefined;
  readonly basis: ModeBasis | undefined;
  /** Whether the page computes a broader neighbourhood; without it the two-scale option is off. */
  readonly twoScale?: boolean;
  /** The broader neighbourhood's result, when the patch is judged at two scales. */
  readonly broad?: GramView | undefined;
  /** Deck's world frame → the frame of `g`'s bbox (the mask's). */
  readonly toFrame: Affine2 | undefined;
  /** Runs as each wand patch lands: the mask is judged by one metric, so the page locks its modes. */
  readonly onPaint: () => void;
  /** The mask's frame units per µm, the brush's unit (default 1: the frame is in µm). */
  readonly unitsPerUm?: number;
  /** Places the layers in deck's world, as the mode map is. */
  readonly modelMatrix?: Matrix4;
}

export function useMapSelection(
  settings: BoundarySettings,
  { g, modes, basis, twoScale = false, broad, toFrame, onPaint, unitsPerUm = 1, modelMatrix }: MapSelectionInput,
) {
  const { show, setShow, wand } = settings;
  // A page with no broader scale judges at one, whatever the settings say.
  const ws = twoScale ? wand.settings : { ...wand.settings, twoScale: false };
  const { setSettings: setWand } = wand;
  // The wand needs the boundary field, and the mask is drawn by the boundary layer, so choosing either
  // painting tool turns boundaries on; with them off the map is the hand's.
  const enableBounds = useCallback(() => setShow(true), [setShow]);
  const resizeWand = useCallback((f: (sigma: number) => number) => setWand((s) => ({ ...s, dSmooth: f(s.dSmooth) })), [setWand]);
  const tools = useMapTools(show, enableBounds, resizeWand);
  const { active, chooseTool, brushR, heldOp, attachMap } = tools;
  const brushOn = active === "brush";
  /** Draw the mask and preview smoothly (quadratic filter, antialiased), or as before, to compare. */
  const [smoothEdges, setSmoothEdges] = useState(true);

  const gb = show && ws.twoScale && g && broad ? broad : undefined;
  const bounds = useBoundaryField(show, g, modes, basis, gb, ws);
  const boundsNow = bounds.main;
  const thetaNow = ws.theta ?? boundsNow?.field.quantiles.p90 ?? 1;
  const patchAt = useMemo<PatchAt<Patch> | undefined>(() => {
    if (!boundsNow || !g) return undefined;
    const two = ws.twoScale && bounds.broad && gb ? { field: bounds.broad, res: gb.res } : undefined;
    // Two scales asked for: wait for the broad field rather than paint with the one-scale patch.
    if (ws.twoScale && !two) return undefined;
    const s = { rule: ws.rule, dSmooth: ws.dSmooth, tol: ws.tol, open: ws.open, graded: ws.graded, slack: ws.slack, theta: thetaNow };
    if (needsExperiments(s, two)) return async (seed) => wandPatch(boundsNow.field, g.res, seed, s, two);
    return (seed) => wandPatchOnGraph(boundsNow.field, g.res, seed, s);
  }, [boundsNow, g, ws.rule, ws.dSmooth, ws.tol, ws.open, ws.graded, ws.slack, ws.twoScale, thetaNow, bounds.broad, gb]);
  // The brush is sized in µm, and painted in the frame's units.
  const brushFrame = brushR * unitsPerUm;
  const paint = useMaskPainting({ ...tools, brushR: brushFrame }, g?.res, toFrame, patchAt, onPaint, smoothEdges);
  const { onPress, onHover: hoverAt, painting, preview, maskOnMap, brushAt } = paint;

  const press = useCallback(
    (phase: PressPhase) =>
      (info: { coordinate?: number[]; viewport?: { unproject: (xy: number[]) => number[] } }, event?: PointerInfo): void => {
        // A drag's info is the press-down pick; its current position is the event's.
        const at =
          phase === "move" && event?.offsetCenter && info.viewport
            ? info.viewport.unproject([event.offsetCenter.x, event.offsetCenter.y])
            : info.coordinate;
        const [x, y] = at ?? [];
        onPress(phase, x, y, event);
      },
    [onPress],
  );
  const onClick = useMemo(() => press("click"), [press]);
  const onHover = useCallback(
    (info: { coordinate?: number[] }) => {
      const [x, y] = info.coordinate ?? [];
      hoverAt(x, y);
    },
    [hoverAt],
  );
  /** For deck: painting takes the drag (scroll still zooms), and the tools are its widgets. */
  const deckProps = useMemo(
    () => ({
      controller: { dragPan: !painting },
      onDragStart: press("start"),
      onDrag: press("move"),
      onDragEnd: press("end"),
      widgets: toolWidgets(active, chooseTool),
    }),
    [painting, press, active, chooseTool],
  );

  const placed = useMemo(() => (modelMatrix ? { modelMatrix } : {}), [modelMatrix]);
  const layers = useMemo<Layer[]>(
    () =>
      g
        ? [
            ...(boundsNow
              ? [
                  new BoundaryLayer({
                    id: "gram-boundary",
                    strength: boundsNow.field.field,
                    bbox: g.res.bbox,
                    width: g.res.width,
                    height: g.res.height,
                    theta: thetaNow,
                    lines: ws.lines,
                    // What a press would paint: none under the hand or while Space makes a drag pan.
                    cell: preview?.cell ?? null,
                    weight: preview?.weight ?? null,
                    op: OP_INDEX[heldOp],
                    smooth: smoothEdges,
                    mask: maskOnMap,
                    ...placed,
                    pickable: false,
                  }),
                ]
              : []),
            // The brush, as a circle on the map: true to size at any zoom, which a cursor cannot be.
            ...(brushOn && brushAt && painting
              ? [
                  new PathLayer<{ path: [number, number][] }>({
                    id: "brush-outline",
                    data: [{ path: circle(brushAt, brushFrame) }],
                    getPath: (d) => d.path,
                    getColor: [255, 255, 255, 230],
                    getWidth: 1.5,
                    widthUnits: "pixels",
                    ...placed,
                    pickable: false,
                  }),
                ]
              : []),
          ]
        : [],
    [g, placed, boundsNow, thetaNow, ws.lines, preview, heldOp, smoothEdges, maskOnMap, painting, brushOn, brushAt, brushFrame],
  );

  return {
    /** The active tool: the similarity wand samples only under the hand. */
    active,
    painting,
    deckProps,
    layers,
    onClick,
    onHover,
    /** Props for the element around the map: the tool's cursor, and where Cmd/Ctrl+scroll is caught. */
    frame: { attachMap, cursor: paint.cursor, onLeave: () => hoverAt() },
    /** Drop the mask, for a new store or element. */
    clear: paint.clear,
    panel: {
      show,
      onShow: setShow,
      wand,
      theta: thetaNow,
      bounds: boundsNow,
      channels: g?.res.labels.length,
      twoScale,
      modes,
      patch: paint.patch,
      brushR,
      mask: paint.mask,
      onClear: paint.clear,
      smoothEdges,
      onSmoothEdges: setSmoothEdges,
    },
  };
}

export type MapSelection = ReturnType<typeof useMapSelection>;
