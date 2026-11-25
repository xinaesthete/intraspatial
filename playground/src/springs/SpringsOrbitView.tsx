// The spring model in 3D on deck.gl: an orbit camera over the simulation's own GPU buffers.
// Deck draws on the page's one device (`useGlobalDevice`), which is the one the simulation runs
// on, so the layers bind the buffers the step kernel writes. Drag to orbit (about the cell under the pointer, if
// there is one), scroll to zoom, shift-drag to pan.
import { OrbitView, type OrbitViewState } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { PivotOrbitController, type PivotPicker } from "@intraspatial/deck";
import { useMemo } from "react";
import { GLOBAL_DEVICE_PROPS, useGlobalDevice } from "../deck/useGlobalDevice";
import { useGpuFrameCost } from "../deck/useGpuFrameCost";
import { type SimBuffers, springsLayers } from "./springsLayers";

const VIEW = new OrbitView({ id: "orbit", orbitAxis: "Y", fovy: 35 });

interface Props {
  /** The running model, or null while it starts. */
  readonly gpu: SimBuffers | null;
  /** Half-width of the scene in model units: sets the starting zoom. */
  readonly extent: number;
  readonly matrix: boolean;
  /** The point to turn the scene about for a pointer position: usually the particle under it. */
  readonly getPivot?: PivotPicker;
  readonly className?: string;
}

export function SpringsOrbitView({ gpu, extent, matrix, getPivot, className }: Props) {
  const { device, status } = useGlobalDevice();
  const cost = useGpuFrameCost(device);
  // 2^zoom pixels per model unit: fit the scene's width into about 600 pixels.
  const initialViewState = useMemo(
    (): OrbitViewState => ({ target: [0, 0, 0], rotationX: 20, rotationOrbit: 30, zoom: Math.log2(300 / extent) }),
    [extent],
  );
  // Turns about the cell under the pointer when there is one, else about the centre.
  const controller = useMemo(() => ({ type: PivotOrbitController, getPivot }), [getPivot]);
  const layers = useMemo(() => (gpu ? springsLayers(gpu, { matrix }) : []), [gpu, matrix]);
  return (
    <div className={className} style={{ position: "relative" }}>
      <DeckGL
        views={VIEW}
        initialViewState={initialViewState}
        controller={controller}
        layers={layers}
        deviceProps={GLOBAL_DEVICE_PROPS}
        // The buffers change under the layers every step, which deck cannot see: draw every frame.
        _animate
        onAfterRender={cost.onAfterRender}
      />
      <div className="view-status">
        {status} · {gpu ? `${gpu.n.toLocaleString()} particles` : "starting…"} · {cost.ms === null ? "–" : cost.ms.toFixed(1)} ms of GPU
        work per frame
      </div>
    </div>
  );
}
