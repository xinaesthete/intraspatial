// The mode map's frame, the same on every page: the deck view (passed as children), the wand's
// pointer handling around it, the wand's readout over it, and how to use it underneath.

import type { ReactNode } from "react";
import type { Rect } from "../../../src/datasource/points";
import type { CoLocationModes } from "../../../src/spatial/gram";
import { useWorldPick } from "../deck/useWorldPick";
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
  children,
}: {
  wand: Wand;
  modes: CoLocationModes | undefined;
  viewport: Rect | undefined;
  toPlace?: (x: number, y: number) => readonly [number, number] | undefined;
  children: ReactNode;
}) {
  const pick = useWorldPick(viewport, (x, y) => {
    const p = toPlace ? toPlace(x, y) : ([x, y] as const);
    if (p) wand.pick(p[0], p[1]);
  });
  return (
    <>
      <div className="deck-wrap" {...pick}>
        {children}
        {wand.sample && modes && <WandReadout wand={wand.sample} modes={modes} />}
      </div>
      <p className="hint">Click the map to sample a place · shift-drag to move the sample · drag to pan, wheel to zoom.</p>
    </>
  );
}
