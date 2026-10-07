// sd.js's headless canvas, as MDV uses it: `useSpatialCanvasRendererFromLayerInputs` turns layer
// configs into deck layers and this component owns the size, the view and the fit. The page adds
// its own deck layers on top (`layers`) and hears where the view is (`onViewport`).
//
// Once framed, deck stays mounted for the life of the component — across stores, coordinate systems
// and the gaps while a store opens. A new Deck would bring a new GPU device, and a page that shares
// deck's device (`useSharedDevice`) can only ever adopt one. So a new store is refitted in place,
// never remounted; don't give this component a `key` that changes with the data.

import type { Layer } from "@deck.gl/core";
import { type SpatialData, viewStateFromBounds } from "@spatialdata/core";
import { layerConfig, SpatialViewer, useSpatialCanvasRendererFromLayerInputs, type ViewState } from "@spatialdata/vis";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Rect } from "../../../src/datasource/points";
import { useElementSize } from "../hooks/useElementSize";

interface Props {
  /** Absent while a store opens: deck stays up, with nothing from sd.js to draw. */
  readonly sdata: SpatialData | undefined;
  readonly coordinateSystem: string;
  /** Image element drawn underneath, by sd.js's own image layer. */
  readonly image?: string;
  /** The page's own layers, drawn above sd.js's. */
  readonly layers: readonly Layer[];
  readonly deckProps?: object;
  /** The visible rectangle in the canvas's world frame, whenever the view or its size changes. */
  readonly onViewport?: (r: Rect) => void;
  /** What to frame when no sd.js layer has bounds — e.g. a store with no image — in the world frame. */
  readonly fallbackBounds?: Rect;
  readonly className?: string;
}

/** What an orthographic view shows, in its world frame: `2^zoom` screen pixels per world unit. */
export function visibleRect(vs: ViewState, width: number, height: number): Rect {
  const scale = 2 ** vs.zoom;
  const [x, y] = vs.target;
  const hw = width / 2 / scale;
  const hh = height / 2 / scale;
  return { minX: x - hw, minY: y - hh, maxX: x + hw, maxY: y + hh };
}

/** One image element as a layer config, kept stable so sd.js does not reload it. */
function useImageInputs(image: string | undefined) {
  return useMemo(() => {
    if (!image) return { layers: {}, layerOrder: [] };
    const id = `image:${image}`;
    return { layers: { [id]: layerConfig("image", { id, elementKey: image, visible: true, opacity: 1 }) }, layerOrder: [id] };
  }, [image]);
}

export function SpatialDeckView({ sdata, coordinateSystem, image, layers, deckProps, onViewport, fallbackBounds, className }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const { width, height } = useElementSize(box);
  const [viewState, setViewState] = useState<ViewState | null>(null);
  /** What the current view was fitted to; a different store or frame is fitted again. */
  const fittedFor = useRef<{ sdata?: SpatialData; coordinateSystem?: string }>({});
  const layerInputs = useImageInputs(image);

  const renderer = useSpatialCanvasRendererFromLayerInputs({
    spatialData: sdata,
    coordinateSystem,
    layerInputs,
    viewState,
    onViewStateChange: setViewState,
    width,
    height,
    externalDeckLayers: layers as Layer[],
    // Fitted below once bounds exist, as MDV does: the built-in fit can commit a degenerate view.
    autoFit: false,
    pickingEnabled: false,
  });

  const { getWorldBoundsForVisibleLayers, hasEnabledLayers, hasLayersDrawn, isBlocking } = renderer;
  useEffect(() => {
    if (!sdata || width <= 0 || height <= 0) return;
    const f = fittedFor.current;
    if (f.sdata === sdata && f.coordinateSystem === coordinateSystem) return;
    // sd.js's own bounds when it has layers to give them, else the caller's; wait for whichever.
    let bounds: Rect | undefined;
    if (hasEnabledLayers) {
      if (isBlocking || !hasLayersDrawn) return;
      bounds = getWorldBoundsForVisibleLayers() ?? undefined;
    } else {
      bounds = fallbackBounds;
    }
    if (!bounds) return;
    fittedFor.current = { sdata, coordinateSystem };
    setViewState(viewStateFromBounds(bounds, width, height));
  }, [
    sdata,
    coordinateSystem,
    hasEnabledLayers,
    isBlocking,
    hasLayersDrawn,
    getWorldBoundsForVisibleLayers,
    width,
    height,
    fallbackBounds,
  ]);

  useEffect(() => {
    if (viewState && width > 0 && height > 0) onViewport?.(visibleRect(viewState, width, height));
  }, [viewState, width, height, onViewport]);

  return (
    <div ref={box} className={className}>
      {/* Mounted once first framed, then never unmounted: a later store is refitted in place. */}
      {viewState ? (
        <SpatialViewer
          width={width}
          height={height}
          viewState={viewState}
          onViewStateChange={setViewState}
          layers={renderer.deckLayers}
          layerOrder={renderer.layerOrder}
          vivLayerProps={renderer.vivLayerProps.length > 0 ? renderer.vivLayerProps : undefined}
          deckProps={deckProps}
        />
      ) : (
        <p className="status">{isBlocking ? "Loading image…" : "Framing view…"}</p>
      )}
    </div>
  );
}
