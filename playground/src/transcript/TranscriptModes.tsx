// Transcript co-location modes: the Gram form over transcript density, for whatever part of the slide
// is in view (docs/cell-stats.md §4, §12; ADR-0008 points amendment).
//
// The view is sd.js's own (its image layer, its pan and zoom); the modes are a deck layer drawn from
// the Gram's rasters on the GPU device deck and this library share. Once the view rests, the visible
// rectangle — in the transcripts' frame, cut to where there are transcripts — is the window the
// statistics are computed over, unless the window is pinned. A map can only be so many pixels across,
// so zoomed out the neighbourhood radius is raised until the map covers the whole window, and the
// page says so.

import { PathLayer } from "@deck.gl/layers";
import { Matrix4 } from "@math.gl/core";
import { type CSSProperties, useCallback, useMemo, useState } from "react";
import { clampWindow, type PointsTile, type Rect } from "../../../src/datasource/points";
import { type ModeBasis, modeBasis } from "../../../src/gpu/spatial/gramModes";
import { type CoLocationModes, radiusToCover } from "../../../src/spatial/gram";
import { invert2, rectThrough2 } from "../../../src/spatial/ngffTransform";
import { listImageElements } from "../datasource/imageContext";
import { DEFAULT_UNTILED_BUDGET, listPointsElements } from "../datasource/pointsTileLoader";
import { type PointerInfo, SpatialDeckView } from "../deck/SpatialDeckView";
import { SHARED_DEVICE_PROPS, useSharedDevice } from "../deck/useSharedDevice";
import { useAsync, useSettled } from "../hooks/useAsync";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { GramModesLayer } from "../modes/GramModesLayer";
import { MatrixView } from "../modes/MatrixView";
import { ModeLegend } from "../modes/ModeLegend";
import { useMapTools } from "../selection/useMapTools";
import { type PatchAt, type PressPhase, useMaskPainting } from "../selection/useMaskPainting";
import { toolWidgets } from "../selection/wandTools";
import type { PressOp } from "../selection/worldMask";
import { BoundaryLayer } from "../wand/BoundaryLayer";
import { useBoundaryField } from "../wand/useBoundaryField";
import { useWandSettings } from "../wand/useWandSettings";
import { WandPanel } from "../wand/WandPanel";
import { needsExperiments, type Patch, wandPatch, wandPatchOnGraph } from "../wand/wandPatch";
import { ChannelPicker } from "./ChannelPicker";
import { buildChannels, NEGATIVE_CONTROLS, type Selection, STARTER_SETS } from "./channelModel";
import { TranscriptExplainers } from "./TranscriptExplainers";
import { useTranscriptGram } from "./useTranscriptGram";
import { LocalStorePicker } from "../LocalStorePicker";
import "@deck.gl/widgets/stylesheet.css";

/** BoundaryLayer's op code. */
const OP_INDEX: Record<PressOp, number> = { replace: 0, add: 1, subtract: 2, intersect: 3, piece: 2 };

/** A closed circle of 64 segments, for a PathLayer. */
function circle(c: readonly [number, number], r: number): [number, number][] {
  return Array.from({ length: 65 }, (_, i) => {
    const t = (i / 64) * 2 * Math.PI;
    return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)];
  });
}

/** The raster stack's size on the device, in MiB. `bytesPerWordRatio` of 2 reports what the same
 *  stack would cost at f32, which is the comparison the checkbox is making. */
function rasterMiB(
  g: { res: { labels: readonly unknown[]; height: number; resident: { rowWords: number; precision: string } } },
  scale = 1,
) {
  const words = g.res.labels.length * g.res.height * g.res.resident.rowWords;
  return ((words * 4 * scale) / (1024 * 1024)).toFixed(1);
}

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const DEFAULT_SELECTION: Selection = { sets: [...STARTER_SETS.map((s) => s.name), NEGATIVE_CONTROLS], genes: [] };
/** How long the view must rest before the window moves to it. */
const SETTLE_MS = 50;
/** `imageName` for "no image". `undefined` means "pick one". */
const NO_IMAGE = "";

/** Modes fixed from one window, to paint every later window with. Each raster is a density
 *  (molecules per µm², the kernel having unit mass), so the lock stays meaningful when the radius or
 *  quality filter changes — those change how smooth or how full the maps are, not their units. Only
 *  other channels break it. */
interface ModeLock {
  readonly basis: ModeBasis;
  readonly modes: CoLocationModes;
  readonly window: Rect;
}

const sameLabels = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((l, i) => l === b[i]);

const span = (r: Rect): string => `${Math.round(r.maxX - r.minX)}×${Math.round(r.maxY - r.minY)}`;
const outline = (r: Rect): [number, number][] => [
  [r.minX, r.minY],
  [r.maxX, r.minY],
  [r.maxX, r.maxY],
  [r.minX, r.maxY],
  [r.minX, r.minY],
];

export function TranscriptModes() {
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [storeDraft, setStoreDraft] = useState(DEFAULT_STORE);
  const [pointsName, setPointsName] = useState<string>();
  const [imageName, setImageName] = useState<string>();
  const [radius, setRadius] = useState(50);
  const [qvMin, setQvMin] = useState(20);
  const [untiledBudgetM, setUntiledBudgetM] = useState(DEFAULT_UNTILED_BUDGET / 1e6);
  const [exact, setExact] = useState(false);
  const [rasterSide, setRasterSide] = useState(0);
  const [halfRasters, setHalfRasters] = useState(false);
  const [follow, setFollow] = useState(true);
  const [pinned, setPinned] = useState<Rect>();
  const [selection, setSelection] = useState(DEFAULT_SELECTION);
  const [saturate, setSaturate] = useState(2.5);
  const [chromaWeight, setChromaWeight] = useState(0.5);
  const [opacity, setOpacity] = useState(0.75);
  const [lock, setLock] = useState<ModeLock>();

  const device = useSharedDevice();

  const sd = useSpatialData(storeUrl);
  const pointsNames = useMemo(() => (sd.value ? listPointsElements(sd.value) : []), [sd.value]);
  const pointsEl = pointsName ?? pointsNames.find((n) => /morton/.test(n) && !/feature_then/.test(n)) ?? pointsNames[0];
  const source = usePointsSource(sd.value, pointsEl, {
    columns: ["qv"],
    untiledBudget: untiledBudgetM * 1e6,
    tilePointBudget: exact ? Number.POSITIVE_INFINITY : undefined,
  });
  // An element read whole has no qv column to filter on, so its filter is off rather than an error.
  const hasQv = source.value?.columns.includes("qv") ?? true;
  const qvApplied = hasQv ? qvMin : 0;
  const imageNames = useAsync(() => (sd.value ? listImageElements(sd.value) : undefined), [sd.value], sd.value);
  const imageEl = imageName === NO_IMAGE ? undefined : (imageName ?? imageNames.value?.find((n) => /he/i.test(n)) ?? imageNames.value?.[0]);

  // Transcripts' own frame ↔ the frame the canvas draws in.
  const space = source.value?.space;
  const coordinateSystem = space?.system ?? "global";
  const affine = space?.affine;
  const toElement = useMemo(() => (affine ? invert2(affine) : undefined), [affine]);
  const modelMatrix = useMemo(
    () => (affine ? new Matrix4([affine.a, affine.b, 0, 0, affine.c, affine.d, 0, 0, 0, 0, 1, 0, affine.tx, affine.ty, 0, 1]) : undefined),
    [affine],
  );
  const extent = source.value?.grid.bounds;
  const fallbackBounds = useMemo(() => (affine && extent ? rectThrough2(affine, extent) : undefined), [affine, extent]);

  // The view, once it rests, in the transcripts' frame and cut to their extent: the window — or the
  // pinned one.
  const [viewport, setViewport] = useState<Rect>();
  const settled = useSettled(viewport, SETTLE_MS);
  const inView = useMemo(
    () => (settled && extent && toElement ? clampWindow(rectThrough2(toElement, settled), extent, Number.POSITIVE_INFINITY) : undefined),
    [settled, extent, toElement],
  );
  const window = follow ? inView : (pinned ?? inView);
  // The radius the window is analysed at. With an automatic map size, the map's pixels must stay a
  // third of the radius or finer, or molecules between pixel centres drop out; a map can only be so
  // many pixels across, so a wide window raises the radius until the map covers it (rounded up, so
  // the raster is never clamped). With a fixed map size the radius is the one chosen, and `coarse`
  // below warns if its pixels outgrow it.
  const longSide = window ? Math.max(window.maxX - window.minX, window.maxY - window.minY) : 0;
  const rEff = rasterSide > 0 ? radius : Math.max(radius, Math.ceil(radiusToCover(longSide) * 10) / 10);
  const togglePin = useCallback(
    (on: boolean) => {
      setFollow(on);
      setPinned(on ? undefined : window);
    },
    [window],
  );

  const tiles = usePointsTiles(source.value, window, rEff);
  const sampled = sampleNote(tiles.value?.tiles);
  const built = useMemo(
    () => (source.value ? buildChannels(selection, source.value.features) : { channels: [], missing: {} }),
    [selection, source.value],
  );
  // No compute until deck's device is ours: resources made on another device could not be drawn.
  // Memoised by hand (no React Compiler here): a fresh `[]` each render is a new dep for the Gram,
  // which reruns, sets state, renders…
  const channels = useMemo(() => (device.ready ? built.channels : []), [device.ready, built.channels]);
  // Scoped to the source: a result stands in for the next only while it is about the same transcripts.
  const gram = useTranscriptGram(tiles.value, channels, { radius: rEff, qvMin: qvApplied, rasterSide, halfRasters }, source.value);
  const g = gram.value;

  // A lock applies while the channels are the ones it was taken with; otherwise this window's own
  // modes. The checkbox shows whether it applies, not merely whether one is held.
  const locked = lock && g && sameLabels(lock.basis.labels, g.modes.labels) ? lock : undefined;
  const modes = locked?.modes ?? g?.modes;
  const toggleLock = (on: boolean) => setLock(on && g ? { basis: modeBasis(g.res), modes: g.modes, window: g.window } : undefined);

  // PROTOTYPE (docs/wand-metric-and-regions.md): the boundary field, the wand's patch on it, and a mask
  // painted with the map's tools. The wand needs the boundary field, and the mask is drawn by the
  // boundary layer, so choosing either painting tool turns boundaries on; with them off the map is the
  // hand's.
  const [showBounds, setShowBounds] = useState(false);
  const wand = useWandSettings();
  const ws = wand.settings;
  const { setSettings: setWand } = wand;
  const enableBounds = useCallback(() => setShowBounds(true), []);
  const resizeWand = useCallback((f: (sigma: number) => number) => setWand((s) => ({ ...s, dSmooth: f(s.dSmooth) })), [setWand]);
  const tools = useMapTools(showBounds, enableBounds, resizeWand);
  const { active, chooseTool, brushR, heldOp, attachMap } = tools;
  const brushOn = active === "brush";
  /** Draw the mask and preview smoothly (quadratic filter, antialiased), or as before, to compare. */
  const [smoothEdges, setSmoothEdges] = useState(true);

  // The broad neighbourhood: the same window and channels at ws.broad × r, on its own (coarser)
  // raster, computed only while it is asked for.
  const broadOn = showBounds && ws.twoScale;
  const rBroad = Math.round(rEff * ws.broad * 10) / 10;
  const tilesBroad = usePointsTiles(source.value, broadOn ? window : undefined, rBroad);
  const gramBroad = useTranscriptGram(
    broadOn ? tilesBroad.value : undefined,
    channels,
    { radius: rBroad, qvMin: qvApplied, rasterSide: 0, halfRasters },
    source.value,
  );
  const gb = broadOn && g && gramBroad.value && gramBroad.value.window === g.window ? gramBroad.value : undefined;
  const bounds = useBoundaryField(showBounds, g, modes, locked?.basis, gb, ws);
  const boundsNow = bounds.main;
  const thetaNow = ws.theta ?? boundsNow?.field.quantiles.p90 ?? 1;
  const patchAt = useMemo<PatchAt<Patch> | undefined>(() => {
    if (!boundsNow || !g) return undefined;
    const broad = ws.twoScale && bounds.broad && gb ? { field: bounds.broad, res: gb.res } : undefined;
    // Two scales asked for: wait for the broad field rather than paint with the one-scale patch.
    if (ws.twoScale && !broad) return undefined;
    const s = { rule: ws.rule, dSmooth: ws.dSmooth, tol: ws.tol, open: ws.open, graded: ws.graded, slack: ws.slack, theta: thetaNow };
    if (needsExperiments(s, broad)) return async (seed) => wandPatch(boundsNow.field, g.res, seed, s, broad);
    return (seed) => wandPatchOnGraph(boundsNow.field, g.res, seed, s);
  }, [boundsNow, g, ws.rule, ws.dSmooth, ws.tol, ws.open, ws.graded, ws.slack, ws.twoScale, thetaNow, bounds.broad, gb]);
  // The mask is judged by one metric, so the first stroke locks the modes: otherwise every pan would
  // change what "similar" means.
  const lockOnPaint = useCallback(() => {
    if (!lock && g) setLock({ basis: modeBasis(g.res), modes: g.modes, window: g.window });
  }, [lock, g]);
  const paint = useMaskPainting(tools, g?.res, toElement, patchAt, lockOnPaint, smoothEdges);
  const { onPress, onHover: hoverAt } = paint;
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
  const { painting, preview, maskOnMap, brushAt } = paint;
  const deckProps = useMemo(
    () => ({
      deviceProps: SHARED_DEVICE_PROPS,
      onDeviceInitialized: device.onDeviceInitialized,
      // Painting takes the drag; scroll still zooms.
      controller: { dragPan: !painting },
      onDragStart: press("start"),
      onDrag: press("move"),
      onDragEnd: press("end"),
      widgets: toolWidgets(active, chooseTool),
    }),
    [device.onDeviceInitialized, painting, press, active, chooseTool],
  );

  const layers = useMemo(
    () =>
      g && modelMatrix
        ? [
            new GramModesLayer({
              id: "gram-modes",
              res: g.res,
              rasters: g.rasters,
              vectors: (locked?.modes ?? g.modes).vectors,
              basis: locked?.basis ?? null,
              saturate,
              chromaWeight,
              opacity,
              modelMatrix,
              pickable: false,
            }),
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
                    modelMatrix,
                    pickable: false,
                  }),
                ]
              : []),
            // The brush, as a circle on the map: true to size at any zoom, which a cursor cannot be.
            ...(brushOn && brushAt && painting
              ? [
                  new PathLayer<{ path: [number, number][] }>({
                    id: "brush-outline",
                    data: [{ path: circle(brushAt, brushR) }],
                    getPath: (d) => d.path,
                    getColor: [255, 255, 255, 230],
                    getWidth: 1.5,
                    widthUnits: "pixels",
                    modelMatrix,
                    pickable: false,
                  }),
                ]
              : []),
            new PathLayer<{ path: [number, number][] }>({
              id: "gram-window",
              data: [{ path: outline(g.window) }],
              getPath: (d) => d.path,
              getColor: [255, 255, 255, 200],
              getWidth: 1.5,
              widthUnits: "pixels",
              modelMatrix,
              pickable: false,
            }),
            // Where the locked modes were taken from, when that is not the window shown.
            ...(locked && locked.window !== g.window
              ? [
                  new PathLayer<{ path: [number, number][] }>({
                    id: "gram-lock-window",
                    data: [{ path: outline(locked.window) }],
                    getPath: (d) => d.path,
                    getColor: [251, 191, 36, 220],
                    getWidth: 1.5,
                    widthUnits: "pixels",
                    modelMatrix,
                    pickable: false,
                  }),
                ]
              : []),
          ]
        : [],
    [
      g,
      locked,
      modelMatrix,
      saturate,
      chromaWeight,
      opacity,
      boundsNow,
      thetaNow,
      ws.lines,
      preview,
      heldOp,
      smoothEdges,
      maskOnMap,
      painting,
      brushOn,
      brushAt,
      brushR,
    ],
  );

  const error = sd.error ?? source.error ?? tiles.error ?? gram.error;
  const busy = source.loading
    ? "opening the points element…"
    : tiles.progress
      ? `loading tiles ${tiles.progress.done}/${tiles.progress.total}…`
      : gram.loading
        ? "computing…"
        : "";
  // The radius on screen, when the window forced it above the one chosen.
  const raised = g && g.radius > radius ? g.radius : undefined;
  // A fixed map size can make pixels wider than the kernel, and then molecules between pixel centres are lost.
  const coarse = g && rasterSide > 0 ? (g.window.maxX - g.window.minX) / g.raster.width > g.radius : false;

  // What belongs to one store, or one element of it, is dropped with it — by the action that changes
  // it, not an effect after: the pinned window and the mask are in the old element's frame, the lock is
  // the old data's projection, and the last viewport would fetch tiles at the old store's coordinates.
  const openStore = (url: string) => {
    setStoreUrl(url);
    setPointsName(undefined);
    setImageName(undefined);
    setPinned(undefined);
    setLock(undefined);
    setViewport(undefined);
    paint.clear();
  };
  const choosePoints = (name: string) => {
    setPointsName(name);
    setPinned(undefined);
    paint.clear();
  };

  return (
    <div className="page">
      <h2 className="page">Where genes are found together — transcript co-location modes</h2>
      <p className="page">
        Every dot in a Xenium experiment is one detected RNA molecule of a known gene. Instead of first assigning molecules to cells, this
        page looks at where each gene's molecules sit, smoothed over a neighbourhood, and asks which genes turn up in the same places. Pan
        and zoom the slide; when you stop, the part in view is analysed and the colours show which mixtures of genes dominate where.
      </p>

      <div className="layout">
        <aside className="controls">
          <label>
            Store
            <span className="row">
              <input value={storeDraft} onChange={(e) => setStoreDraft(e.target.value)} />
              <button type="button" onClick={() => openStore(storeDraft.trim())}>
                open
              </button>
            </span>
          </label>
          <LocalStorePicker
            onPick={(key) => {
              setStoreDraft(key);
              openStore(key);
            }}
          />
          <label>
            Transcripts
            <select value={pointsEl ?? ""} onChange={(e) => choosePoints(e.target.value)}>
              {pointsNames.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label>
            Untiled points budget (million rows)
            <input
              type="number"
              min={0.5}
              max={100}
              step={0.5}
              value={untiledBudgetM}
              onChange={(e) => setUntiledBudgetM(Math.max(0.5, Number(e.target.value) || 0.5))}
            />
          </label>
          {source.value?.levelFractions && (
            <label className="check">
              <input type="checkbox" checked={exact} onChange={(e) => setExact(e.target.checked)} />
              Every molecule, at any zoom (off: a zoomed-out view analyses a random sample — fast, approximate)
            </label>
          )}
          <label>
            Image
            <select value={imageEl ?? NO_IMAGE} onChange={(e) => setImageName(e.target.value)}>
              {(imageNames.value ?? []).map((n) => (
                <option key={n}>{n}</option>
              ))}
              <option value={NO_IMAGE}>(none)</option>
            </select>
          </label>
          <label>
            <span>
              Neighbourhood radius r: {radius} µm*
              {raised !== undefined && (
                <span className="raised">
                  {" "}
                  — showing {raised} µm* to cover the view; zoom in to reach {radius}
                </span>
              )}
            </span>
            <input type="range" min={1} max={100} step={1} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          </label>
          <label>
            Map resolution (long side, 0 = from r)
            <input
              type="number"
              min={0}
              max={2048}
              step={16}
              value={rasterSide}
              onChange={(e) => setRasterSide(Math.max(0, Number(e.target.value) || 0))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={halfRasters} onChange={(e) => setHalfRasters(e.target.checked)} />
            Half-precision maps (f16){halfRasters && gram.value ? ` — ${rasterMiB(gram.value)} MiB, was ${rasterMiB(gram.value, 2)}` : ""}
          </label>
          <label>
            Minimum molecule quality (qv){hasQv ? "" : " — not available for this element"}
            <input
              type="number"
              min={0}
              max={40}
              value={qvMin}
              disabled={!hasQv}
              onChange={(e) => setQvMin(Math.max(0, Number(e.target.value) || 0))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={follow} onChange={(e) => togglePin(e.target.checked)} />
            Follow the view (off: keep the current window)
          </label>
          <label className="check">
            <input type="checkbox" checked={!!locked} disabled={!g} onChange={(e) => toggleLock(e.target.checked)} />
            Lock the modes (off: recompute them for each window)
          </label>
          <label>
            Colour strength
            <input type="range" min={0.5} max={6} step={0.1} value={saturate} onChange={(e) => setSaturate(Number(e.target.value))} />
          </label>
          <label>
            Hue weight (0 = by importance, 1 = equalised)
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={chromaWeight}
              onChange={(e) => setChromaWeight(Number(e.target.value))}
            />
          </label>
          <label>
            Map over image
            <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          </label>
          <WandPanel
            show={showBounds}
            onShow={setShowBounds}
            wand={wand}
            theta={thetaNow}
            bounds={boundsNow}
            channels={g?.res.labels.length}
            modes={modes}
            broad={gb}
            patch={paint.patch}
            brushR={brushR}
            mask={paint.mask}
            onClear={paint.clear}
            smoothEdges={smoothEdges}
            onSmoothEdges={setSmoothEdges}
          />
          {source.value && (
            <ChannelPicker features={source.value.features} selection={selection} onChange={setSelection} missing={built.missing} />
          )}
          <p className="hint">µm*: the store does not declare a unit; Xenium writes transcript coordinates in micrometres.</p>
        </aside>

        <main className="views">
          {/* Always mounted, never keyed: see SpatialDeckView on why deck must outlive a store. */}
          {/* The tool's cursor, over deck's own (which it rewrites on the canvas every frame). */}
          <div
            className="tool-cursor"
            ref={attachMap}
            data-tool={painting ? "paint" : "hand"}
            style={{ "--tool-cursor": paint.cursor } as CSSProperties}
            onPointerLeave={() => hoverAt()}
          >
            <SpatialDeckView
              className="deck-canvas"
              sdata={sd.value}
              coordinateSystem={coordinateSystem}
              image={imageEl}
              layers={layers}
              deckProps={deckProps}
              onViewport={setViewport}
              onClick={onClick}
              onHover={onHover}
              fallbackBounds={fallbackBounds}
            />
          </div>
          <p className="status">
            {error ? <span className="error">{error.message}</span> : busy}
            {!error && !busy && g && (
              <>
                window {span(g.window)} µm*{follow ? "" : " (kept)"} · r {g.radius} µm*
                {raised !== undefined ? ` (raised from ${radius} to cover the view)` : ""} · {g.modes.labels.length} channels ·{" "}
                {g.stats.perChannel.reduce((s, n) => s + n, 0).toLocaleString()} molecules in them
                {sampled ? ` (${sampled})` : ""} ·{" "}
                {qvApplied > 0 ? `${g.stats.belowMinimum.toLocaleString()} below qv ${qvApplied} · ` : ""}map {g.raster.width}×
                {g.raster.height} ·{" "}
                {source.value?.tiled === false
                  ? `${source.value.totalRows.toLocaleString()} points read whole (no tiled layout)`
                  : `${tiles.value?.fetched ?? 0} tiles fetched, ${(tiles.value?.tiles.length ?? 0) - (tiles.value?.fetched ?? 0)} from cache`}{" "}
                · channels {g.ms.channels.toFixed(0)} ms · Gram {g.ms.gram.toFixed(0)} ms
              </>
            )}
          </p>
          <p className="hint">GPU: {device.status}</p>
          {coarse && (
            <p className="hint error">
              The map's pixels are wider than r, so molecules falling between pixel centres are left out. Raise the map resolution, set it
              to 0, or raise r.
            </p>
          )}
          {g && (
            <>
              {lock && !locked && (
                <p className="hint">
                  The locked modes were for other channels, so these are this window's own.{" "}
                  <button type="button" onClick={() => toggleLock(true)}>
                    lock these instead
                  </button>
                </p>
              )}
              {locked && (
                <p className="hint">
                  Modes locked to the {span(locked.window)} µm* window outlined in amber: a colour means the same mixture everywhere, and
                  the percentages are of that window's variation.
                </p>
              )}
              {modes && <ModeLegend modes={modes} />}
              <h3>How often each pair turns up together, compared with chance</h3>
              <MatrixView
                labels={g.modes.labels}
                values={g.res.g}
                scale="log2"
                describe={(i, j) =>
                  `${g.modes.labels[i]} × ${g.modes.labels[j]}: g = ${(g.res.g[i * g.modes.labels.length + j] ?? 0).toFixed(2)}`
                }
              />
            </>
          )}

          <TranscriptExplainers />
        </main>
      </div>
    </div>
  );
}

/** How much of the element the tiles hold, when any came from a sampled level: "≈25% sample". */
function sampleNote(tiles: readonly PointsTile[] | undefined): string | undefined {
  const fractions = (tiles ?? []).map((t) => t.sampleFraction ?? 1);
  if (!fractions.some((f) => f < 1)) return undefined;
  const pct = (f: number) => `${+(f * 100).toPrecision(2)}%`;
  const lo = Math.min(...fractions);
  const hi = Math.max(...fractions);
  return lo === hi ? `≈${pct(lo)} sample` : `≈${pct(lo)}–${pct(hi)} sample, varying by tile`;
}
