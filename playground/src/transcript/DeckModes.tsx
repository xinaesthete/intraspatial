// Experiment: the transcript co-location modes drawn as a deck.gl layer over sd.js's own image layer, on one
// shared WebGPU device. Deck creates the device (luma 9.4 cannot attach an existing one); this
// library adopts it before any compute, so the layer binds the Gram's rasters on the GPU — no readback.
// Needs sd.js with WebGPU rendering (SpatialData.js#204), linked via `pnpm link:spatialdata`.
//
// The analysis window follows the view: once a pan or zoom settles, the visible rectangle, taken
// into the transcripts' own frame and capped in area, becomes the window the Gram is computed over.
//
// The canvas is sd.js's headless renderer, as MDV uses it: `useSpatialCanvasRendererFromLayerInputs`
// turns layer configs into deck layers, and this page owns the size, the view state and its extra
// layers. No SpatialCanvas chrome, no store.

import type { Device } from "@luma.gl/core";
import { webgpuAdapter } from "@luma.gl/webgpu";
import { Matrix4 } from "@math.gl/core";
import { type SpatialData, viewStateFromBounds } from "@spatialdata/core";
import { layerConfig, SpatialViewer, useSpatialCanvasRendererFromLayerInputs, type ViewState } from "@spatialdata/vis";
import type { Layer } from "deck.gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clampWindow, type Rect } from "../../../src/datasource/points";
import { adoptDevice, getDevice } from "../../../src/gpu/device";
import { invert2, rectThrough2 } from "../../../src/spatial/ngffTransform";
import { listImageElements } from "../datasource/imageContext";
import { listPointsElements } from "../datasource/pointsTileLoader";
import { useAsync, useSettled } from "../hooks/useAsync";
import { useElementSize } from "../hooks/useElementSize";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { buildChannels, NEGATIVE_CONTROLS, STARTER_SETS } from "./channelModel";
import { GramModesLayer } from "./GramModesLayer";
import { ModeLegend } from "./ModeMap";
import { useTranscriptGram } from "./useTranscriptGram";

const STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const COORDINATE_SYSTEM = "global";
const SELECTION = { sets: [...STARTER_SETS.map((s) => s.name), NEGATIVE_CONTROLS], genes: [] };
const GRAM = { radius: 50, qvMin: 20, rasterSide: 0 };
/** Module-level: a new object each render would make deck rebuild its device. */
const DEVICE_PROPS = { type: "webgpu" as const, adapters: [webgpuAdapter], optionalFeatures: ["float32-blendable" as const] };

/** The largest window one Gram is computed over, in µm². About 2× the React page's default
 *  3000×1200: a few hundred ms of tiles and compute. Zoomed further out, the window is the view's
 *  middle rather than all of it. */
const MAX_WINDOW_AREA = 8e6;
/** How long the view must rest before the window moves to it. */
const SETTLE_MS = 300;

const span = (r: Rect): string => `${Math.round(r.maxX - r.minX)}×${Math.round(r.maxY - r.minY)}`;

/** What an orthographic view shows, in its world frame: `2^zoom` screen pixels per world unit. */
function visibleRect(vs: ViewState, width: number, height: number): Rect {
  const scale = 2 ** vs.zoom;
  const [x, y] = vs.target;
  const hw = width / 2 / scale;
  const hh = height / 2 / scale;
  return { minX: x - hw, minY: y - hh, maxX: x + hw, maxY: y + hh };
}

/** One image element as a layer config, kept stable so sd.js does not reload it. */
function useImageInputs(imageEl: string | undefined) {
  return useMemo(() => {
    if (!imageEl) return { layers: {}, layerOrder: [] };
    const id = `image:${imageEl}`;
    return { layers: { [id]: layerConfig("image", { id, elementKey: imageEl, visible: true, opacity: 1 }) }, layerOrder: [id] };
  }, [imageEl]);
}

function Viewer({
  sdata,
  imageEl,
  gramLayers,
  deckProps,
  onViewport,
}: {
  sdata: SpatialData;
  imageEl: string | undefined;
  gramLayers: Layer[];
  deckProps: object;
  /** The visible rectangle in the canvas's world frame, whenever the view or its size changes. */
  onViewport: (r: Rect) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const { width, height } = useElementSize(box);
  const [viewState, setViewState] = useState<ViewState | null>(null);
  const layerInputs = useImageInputs(imageEl);

  const renderer = useSpatialCanvasRendererFromLayerInputs({
    spatialData: sdata,
    coordinateSystem: COORDINATE_SYSTEM,
    layerInputs,
    viewState,
    onViewStateChange: setViewState,
    width,
    height,
    externalDeckLayers: gramLayers,
    // Fitted below once bounds exist, as MDV does: the built-in fit can commit a degenerate view.
    autoFit: false,
    pickingEnabled: false,
  });

  const { getWorldBoundsForVisibleLayers, hasEnabledLayers, hasLayersDrawn, isBlocking } = renderer;
  useEffect(() => {
    if (viewState || !hasEnabledLayers || isBlocking || !hasLayersDrawn || width <= 0 || height <= 0) return;
    const bounds = getWorldBoundsForVisibleLayers();
    if (bounds) setViewState(viewStateFromBounds(bounds, width, height));
  }, [viewState, hasEnabledLayers, isBlocking, hasLayersDrawn, getWorldBoundsForVisibleLayers, width, height]);

  useEffect(() => {
    if (viewState && width > 0 && height > 0) onViewport(visibleRect(viewState, width, height));
  }, [viewState, width, height, onViewport]);

  return (
    <div ref={box} className="deck-canvas">
      {/* Mounted once framed, as MDV does: by then the image props exist, so SpatialViewer
          starts on its Viv path rather than swapping Deck instances — and devices — midway. */}
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

export function DeckModes() {
  const [deviceCheck, setDeviceCheck] = useState<string>("waiting for deck's device…");
  const [ready, setReady] = useState(false);
  const [saturate, setSaturate] = useState(2.5);
  const [chromaWeight, setChromaWeight] = useState(0.5);
  const [opacity, setOpacity] = useState(0.75);

  const onDeviceInitialized = useCallback((device: Device) => {
    const handle = device.handle;
    if (device.type !== "webgpu" || !(handle instanceof GPUDevice)) {
      setDeviceCheck(`deck chose ${device.type}; this page needs WebGPU`);
      return;
    }
    try {
      adoptDevice(handle);
    } catch (e) {
      // A second Deck brings a second device, and this library keeps one per page. In dev that is
      // an HMR remount, so start over. Thrown inside deck's start-up it would stop the render loop.
      if (import.meta.hot) location.reload();
      setDeviceCheck(`NOT shared ✗ — ${(e as Error).message}`);
      return;
    }
    void getDevice().then((ours) =>
      setDeviceCheck(
        `${ours === handle ? "shared ✓" : "NOT shared ✗"} — one GPUDevice for deck and the Gram` +
          ` · float32-blendable ${handle.features.has("float32-blendable") ? "✓" : "✗"}`,
      ),
    );
    setReady(true);
  }, []);

  const sd = useSpatialData(STORE);
  const pointsEl = useMemo(() => (sd.value ? listPointsElements(sd.value).find((n) => n === "transcripts_morton") : undefined), [sd.value]);
  const source = usePointsSource(sd.value, pointsEl, { columns: ["qv"] });
  const imageNames = useAsync(() => (sd.value ? listImageElements(sd.value) : undefined), [sd.value]);
  const imageEl = imageNames.value?.find((n) => /he/i.test(n)) ?? imageNames.value?.[0];

  // The view, once it rests, in the transcripts' frame and capped: the window.
  const [viewport, setViewport] = useState<Rect>();
  const settled = useSettled(viewport, SETTLE_MS);
  const extent = source.value?.grid.bounds;
  const affine = source.value?.space?.affine;
  const toElement = useMemo(() => (affine ? invert2(affine) : undefined), [affine]);
  const window = useMemo(
    () => (settled && extent && toElement ? clampWindow(rectThrough2(toElement, settled), extent, MAX_WINDOW_AREA) : undefined),
    [settled, extent, toElement],
  );
  const tiles = usePointsTiles(source.value, window, GRAM.radius);
  const channels = useMemo(
    () => (ready && source.value ? buildChannels(SELECTION, source.value.features).channels : []),
    [ready, source.value],
  );
  const gram = useTranscriptGram(tiles.value, channels, GRAM);

  // The transcripts' own µm → the global frame the canvas draws in (column-major 4×4).
  const modelMatrix = useMemo(() => {
    const a = source.value?.space?.affine;
    return a ? new Matrix4([a.a, a.b, 0, 0, a.c, a.d, 0, 0, 0, 0, 1, 0, a.tx, a.ty, 0, 1]) : undefined;
  }, [source.value]);

  const g = gram.value;
  const gramLayers = useMemo(
    () =>
      g && modelMatrix
        ? [
            new GramModesLayer({
              id: "gram-modes",
              res: g.res,
              rasters: g.rasters,
              vectors: g.modes.vectors,
              saturate,
              chromaWeight,
              opacity,
              modelMatrix,
              pickable: false,
            }),
          ]
        : [],
    [g, modelMatrix, saturate, chromaWeight, opacity],
  );
  const deckProps = useMemo(() => ({ deviceProps: DEVICE_PROPS, onDeviceInitialized }), [onDeviceInitialized]);

  const error = sd.error ?? source.error ?? tiles.error ?? gram.error;
  return (
    <div className="deck-page">
      <h2 className="page">Transcript modes on deck.gl (WebGPU experiment)</h2>
      <p className="status">
        Device: {deviceCheck}
        {" · "}
        {error ? (
          <span className="error">{error.message}</span>
        ) : tiles.progress ? (
          `loading tiles ${tiles.progress.done}/${tiles.progress.total}…`
        ) : g ? (
          `${g.modes.labels.length} channels · window ${span(g.window)} µm · Gram ${g.ms.gram.toFixed(0)} ms`
        ) : (
          "…"
        )}
      </p>
      <div className="row">
        <label>
          Colour strength{" "}
          <input type="range" min={0.5} max={6} step={0.1} value={saturate} onChange={(e) => setSaturate(Number(e.target.value))} />
        </label>
        <label>
          Hue weight (0 = by importance, 1 = equalised){" "}
          <input type="range" min={0} max={1} step={0.05} value={chromaWeight} onChange={(e) => setChromaWeight(Number(e.target.value))} />
        </label>
        <label>
          Modes over image{" "}
          <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
        </label>
      </div>
      {g && <ModeLegend gram={g} />}
      {sd.value && <Viewer sdata={sd.value} imageEl={imageEl} gramLayers={gramLayers} deckProps={deckProps} onViewport={setViewport} />}
    </div>
  );
}
