// The mode map's frame, the same on every page: the deck view (passed as children), the wand's
// pointer handling around it, the wand's readout over it, and how to use it underneath. With the
// map's selection tools, it also carries the tool's cursor; the wand then samples only under the hand,
// since a press with the other tools paints.

import { type CSSProperties, type ReactNode, useCallback } from "react";
import type { Rect } from "../../../src/datasource/points";
import type { CoLocationModes } from "../../../src/spatial/gram";
import { useWorldPick } from "../deck/useWorldPick";
import type { MapSelection } from "../selection/useMapSelection";
import type { Wand } from "./useWand";
import { WandReadout } from "./WandReadout";

/**
 * `viewport` is the rectangle the view shows (`SpatialDeckView`'s `onViewport`); `toPlace` takes a
 * point in it to the frame of the result's bbox, where the two differ. `modes` are the ones the map
 * is painted with.
 */
export function ModeMap({
  wand,
  modes,
  viewport,
  toPlace,
  selection,
  children,
}: {
  wand: Wand;
  modes: CoLocationModes | undefined;
  viewport: Rect | undefined;
  toPlace?: (x: number, y: number) => readonly [number, number] | undefined;
  selection?: MapSelection;
  children: ReactNode;
}) {
  const pick = useWorldPick(viewport, (x, y) => {
    const p = toPlace ? toPlace(x, y) : ([x, y] as const);
    if (p) wand.pick(p[0], p[1]);
  });
  const sampling = !selection || selection.active === "hand";
  const painting = selection?.painting ?? false;
  const cursor = selection?.frame.cursor;
  const onLeave = selection?.frame.onLeave;
  // Called through, not handed over: a value passed as `ref` marks everything it came from as a ref,
  // and then each read of `selection` above would be a ref read during render.
  const attach = selection?.frame.attachMap;
  const attachMap = useCallback((el: HTMLDivElement | null) => attach?.(el), [attach]);
  const map = (
    <div className="deck-wrap" {...(sampling ? pick : {})}>
      {children}
      {wand.sample && modes && <WandReadout wand={wand.sample} modes={modes} />}
    </div>
  );
  return (
    <>
      {selection ? (
        // The tool's cursor, over deck's own (which it rewrites on the canvas every frame).
        <div
          className="tool-cursor"
          ref={attachMap}
          data-tool={painting ? "paint" : "hand"}
          style={{ "--tool-cursor": cursor } as CSSProperties}
          onPointerLeave={onLeave}
        >
          {map}
        </div>
      ) : (
        map
      )}
      <p className="hint">
        Click the map to sample a place · shift-drag to move the sample · drag to pan, wheel to zoom
        {selection ? " (with the hand, top left of the map; the wand and brush paint a mask)" : ""}.
      </p>
    </>
  );
}
