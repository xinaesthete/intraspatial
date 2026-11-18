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
import { useCallback, useMemo, useState } from "react";
import { clampWindow, type PointsTile, type Rect } from "../../../src/datasource/points";
import { type ModeBasis, modeBasis } from "../../../src/gpu/spatial/gramModes";
import { type CoLocationModes, radiusToCover } from "../../../src/spatial/gram";
import { invert2, rectThrough2 } from "../../../src/spatial/ngffTransform";
import { listImageElements } from "../datasource/imageContext";
import { DEFAULT_UNTILED_BUDGET, listPointsElements, type PointsSource, SUBSET_READ_MAX_FEATURES } from "../datasource/pointsTileLoader";
import { SpatialDeckView } from "../deck/SpatialDeckView";
import { SHARED_DEVICE_PROPS, useSharedDevice } from "../deck/useSharedDevice";
import { useAsync, useSettled } from "../hooks/useAsync";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { GramModesLayer } from "../modes/GramModesLayer";
import { MatrixView } from "../modes/MatrixView";
import { ModeCharts } from "../modes/ModeCharts";
import { ModeMap } from "../modes/ModeMap";
import { ImagePicker, MapStyleControls, StoreInput, useImageChoice, useMapStyle } from "../modes/PageControls";
import { useWand, type WandSource, wandLayerProps } from "../modes/useWand";
import { WandPanel } from "../modes/WandPanel";
import { useBoundarySettings, useMapSelection } from "../selection/useMapSelection";
import { BoundaryPanel } from "../wand/WandPanel";
import { ChannelPicker } from "./ChannelPicker";
import { buildChannels, channelCodes, defaultSelection, type Selection } from "./channelModel";
import { TranscriptExplainers } from "./TranscriptExplainers";
import { useGeneModules } from "./useGeneModules";
import { useTranscriptGram } from "./useTranscriptGram";

/** The raster stack's size on the device, in MiB. `bytesPerWordRatio` of 2 reports what the same
 *  stack would cost at f32, which is the comparison the checkbox is making. */
function rasterMiB(
  g: { res: { labels: readonly unknown[]; height: number; resident: { rowWords: number; precision: string } } },
  scale = 1,
) {
  const words = g.res.labels.length * g.res.height * g.res.resident.rowWords;
  return ((words * 4 * scale) / (1024 * 1024)).toFixed(1);
}

/** The SD.js demo's default store (`DEFAULT_DEMO_SPATIALDATA_URL` in @spatialdata/vis, not exported
 *  from its root): Atera whole-transcriptome kidney, 1.19 billion molecules of 24 674 features in a
 *  points-zarr element. `?url=` opens another store, as in the SD.js demo. */
const ATERA_DEMO = "https://oranus.cmd.ox.ac.uk/publicprojects/project/13/spatial/Atera_v1_WTA_Kidney_non_diseased_FFPE.zarr";
const DEFAULT_STORE = new URLSearchParams(location.search).get("url")?.trim() || ATERA_DEMO;
/** Past this many molecules, "every molecule at any zoom" is not offered: zoomed out, it would read
 *  all of them (on Atera, 1.19 billion). Subset reads keep a small panel at full depth anyway. */
const EXACT_MAX_ROWS = 50_000_000;
const DEFAULT_MODULES = 12;
type ChannelMode = "picked" | "modules";
/** How long the view must rest before the window moves to it. */
const SETTLE_MS = 50;
const preferHe = (names: readonly string[]) => names.find((n) => /he/i.test(n)) ?? names[0];

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
  const [pointsName, setPointsName] = useState<string>();
  const [radius, setRadius] = useState(50);
  const [qvMin, setQvMin] = useState(20);
  const [untiledBudgetM, setUntiledBudgetM] = useState(DEFAULT_UNTILED_BUDGET / 1e6);
  const [exact, setExact] = useState(false);
  const [rasterSide, setRasterSide] = useState(0);
  const [halfRasters, setHalfRasters] = useState(false);
  const [follow, setFollow] = useState(true);
  const [pinned, setPinned] = useState<Rect>();
  const [picked, setPicked] = useState<Selection>();
  const [channelMode, setChannelMode] = useState<ChannelMode>("picked");
  const [moduleCount, setModuleCount] = useState(DEFAULT_MODULES);
  const style = useMapStyle({ chromaWeight: 0.5, opacity: 0.75 });
  const { saturate, chromaWeight, opacity } = style;
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
  const images = useImageChoice(imageNames.value ?? [], preferHe);

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

  // Channels: genes and sets picked by hand, or gene modules fitted once to the whole element.
  const features = source.value?.features;
  const selection = useMemo(
    () => picked ?? (features ? defaultSelection(features, SUBSET_READ_MAX_FEATURES) : undefined),
    [picked, features],
  );
  const modules = useGeneModules(source.value, channelMode === "modules", moduleCount);
  const built = useMemo(() => {
    if (channelMode === "modules") return { channels: modules.value?.channels ?? [], missing: {} };
    return features && selection ? buildChannels(selection, features) : { channels: [], missing: {} };
  }, [channelMode, modules.value, selection, features]);
  // A small panel reads only its own genes, each from the finest level its share of the budget
  // allows; a large one (modules: hundreds of genes) reads every gene, which costs far fewer requests.
  const codesKey = channelCodes(built.channels).join(",");
  const readSource = useMemo(() => subsetSource(source.value, codesKey), [source.value, codesKey]);
  const subset = readSource?.featureSubset?.length;

  const tiles = usePointsTiles(readSource, window, rEff);
  const sampled = sampleNote(tiles.value?.tiles);
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

  // The wand's reference is standardised as the map is painted: with the lock's means, when locked.
  const lockedBasis = locked?.basis;
  const vectors = modes?.vectors;
  const wandSource = useMemo<WandSource | undefined>(() => {
    if (!g || !vectors) return undefined;
    const { mean, sd } = lockedBasis ?? g.res.resident;
    return { res: g.res, rasters: g.rasters, mean, sd, vectors, ...(lockedBasis ? { basis: lockedBasis } : {}) };
  }, [g, lockedBasis, vectors]);
  const wand = useWand(wandSource);
  // The canvas's world frame → the transcripts' frame, which the Gram's bbox is in.
  const toPlace = (wx: number, wy: number): readonly [number, number] | undefined => {
    const m = toElement;
    return m ? [m.a * wx + m.c * wy + m.tx, m.b * wx + m.d * wy + m.ty] : undefined;
  };

  // PROTOTYPE (docs/wand-metric-and-regions.md): boundaries, and a mask painted with the map's tools.
  const boundaries = useBoundarySettings();
  const ws = boundaries.wand.settings;
  // The broad neighbourhood: the same window and channels at ws.broad × r, on its own (coarser)
  // raster, computed only while it is asked for.
  const broadOn = boundaries.show && ws.twoScale;
  const rBroad = Math.round(rEff * ws.broad * 10) / 10;
  const tilesBroad = usePointsTiles(readSource, broadOn ? window : undefined, rBroad);
  const gramBroad = useTranscriptGram(
    broadOn ? tilesBroad.value : undefined,
    channels,
    { radius: rBroad, qvMin: qvApplied, rasterSide: 0, halfRasters },
    source.value,
  );
  const gb = broadOn && g && gramBroad.value && gramBroad.value.window === g.window ? gramBroad.value : undefined;
  // The mask is judged by one metric, so the first stroke locks the modes: otherwise every pan would
  // change what "similar" means.
  const lockOnPaint = useCallback(() => {
    if (!lock && g) setLock({ basis: modeBasis(g.res), modes: g.modes, window: g.window });
  }, [lock, g]);
  const tools = useMapSelection(boundaries, {
    g,
    modes,
    basis: lockedBasis,
    twoScale: true,
    broad: gb,
    toFrame: toElement,
    onPaint: lockOnPaint,
    modelMatrix,
  });
  const deckProps = useMemo(
    () => ({ deviceProps: SHARED_DEVICE_PROPS, onDeviceInitialized: device.onDeviceInitialized, ...tools.deckProps }),
    [device.onDeviceInitialized, tools.deckProps],
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
              ...wandLayerProps(wand, g.modes.labels.length),
              modelMatrix,
              pickable: false,
            }),
            ...tools.layers,
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
    [g, locked, modelMatrix, saturate, chromaWeight, opacity, wand, tools.layers],
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
    setPicked(undefined);
    setExact(false);
    images.reset();
    setPinned(undefined);
    setLock(undefined);
    setViewport(undefined);
    tools.clear();
    wand.clear();
  };
  const choosePoints = (name: string) => {
    setPointsName(name);
    setPicked(undefined);
    setExact(false);
    setPinned(undefined);
    tools.clear();
    wand.clear();
  };

  return (
    <div className="page">
      <h2 className="page">Where genes are found together — transcript co-location modes</h2>
      <p className="page">
        Every dot in an imaging spatial transcriptomics experiment (Xenium, Atera…) is one detected RNA molecule of a known gene. Instead of
        first assigning molecules to cells, this page looks at where each gene's molecules sit, smoothed over a neighbourhood, and asks
        which genes turn up in the same places. Pan and zoom the slide; when you stop, the part in view is analysed and the colours show
        which mixtures of genes dominate where.
      </p>

      <div className="layout">
        <aside className="controls">
          <StoreInput initial={DEFAULT_STORE} onOpen={openStore} />
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
          {source.value?.levelFractions && source.value.totalRows > EXACT_MAX_ROWS && (
            <p className="hint">
              {(source.value.totalRows / 1e9).toFixed(2)} billion molecules: zoomed out, tiles are read from a random sample and each
              molecule is weighted to stand for the ones left out.
            </p>
          )}
          {source.value?.levelFractions && source.value.totalRows <= EXACT_MAX_ROWS && (
            <label className="check">
              <input type="checkbox" checked={exact} onChange={(e) => setExact(e.target.checked)} />
              Every molecule, at any zoom (off: a zoomed-out view analyses a random sample — fast, approximate)
            </label>
          )}
          <ImagePicker images={images} />
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
          <MapStyleControls style={style} />
          <WandPanel wand={wand} />
          <BoundaryPanel
            {...tools.panel}
            broad={gb}
            broadError={broadOn ? (tilesBroad.error ?? gramBroad.error)?.message : undefined}
            mixtures="genes"
          />
          <fieldset className="channel-mode">
            <legend>Channels</legend>
            <label className="check">
              <input type="radio" checked={channelMode === "picked"} onChange={() => setChannelMode("picked")} />
              Genes and gene sets I pick
            </label>
            <label className="check">
              <input type="radio" checked={channelMode === "modules"} onChange={() => setChannelMode("modules")} />
              Gene modules from every gene (PCA + e-means)
            </label>
          </fieldset>
          {channelMode === "picked" && source.value && selection && (
            <ChannelPicker features={source.value.features} selection={selection} onChange={setPicked} missing={built.missing} />
          )}
          {channelMode === "picked" && source.value?.forFeatures && !subset && built.channels.length > 0 && (
            <p className="hint">
              These channels cover {codesKey.split(",").length.toLocaleString()} genes, so every gene is read and most tiles come from a
              sample. With at most {SUBSET_READ_MAX_FEATURES} genes (leave out the negative controls), only those genes are read, at far
              greater depth.
            </p>
          )}
          {channelMode === "modules" && (
            <div className="modules">
              <label>
                Modules: {moduleCount}
                <input
                  type="range"
                  min={4}
                  max={24}
                  step={1}
                  value={moduleCount}
                  onChange={(e) => setModuleCount(Number(e.target.value))}
                />
              </label>
              <ModulesNote modules={modules} />
            </div>
          )}
          <p className="hint">µm*: the store does not declare a unit; Xenium writes transcript coordinates in micrometres.</p>
        </aside>

        <main className="views">
          <ModeMap wand={wand} modes={modes} viewport={viewport} toPlace={toPlace} selection={tools}>
            {/* Always mounted, never keyed: see SpatialDeckView on why deck must outlive a store. */}
            <SpatialDeckView
              className="deck-canvas"
              sdata={sd.value}
              coordinateSystem={coordinateSystem}
              image={images.image}
              layers={layers}
              deckProps={deckProps}
              onViewport={setViewport}
              onClick={tools.onClick}
              onHover={tools.onHover}
              fallbackBounds={fallbackBounds}
            />
          </ModeMap>
          <p className="status">
            {error ? <span className="error">{error.message}</span> : busy}
            {!error && !busy && g && (
              <>
                window {span(g.window)} µm*{follow ? "" : " (kept)"} · r {g.radius} µm*
                {raised !== undefined ? ` (raised from ${radius} to cover the view)` : ""} · {g.modes.labels.length} channels ·{" "}
                {subset ? `reading only their ${subset} genes · ` : ""}
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
              {modes && <ModeCharts modes={modes} />}
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

/** The source to read `codesKey`'s genes through: a subset read when the element offers one and the
 *  genes are few enough, else the element itself. */
function subsetSource(source: PointsSource | undefined, codesKey: string): PointsSource | undefined {
  if (!source?.forFeatures || !codesKey) return source;
  const codes = codesKey.split(",").map(Number);
  return codes.length <= SUBSET_READ_MAX_FEATURES ? source.forFeatures(codes) : source;
}

/** How much of the element the tiles hold, when any came from a sampled level: "≈25% sample". A
 *  subset read samples each gene separately, so it reports the range over genes. */
function sampleNote(tiles: readonly PointsTile[] | undefined): string | undefined {
  const all = tiles ?? [];
  const perGene = all.some((t) => t.featureFractions);
  const fractions = all.flatMap((t) => (t.featureFractions ? [...t.featureFractions.values(), 1] : [t.sampleFraction ?? 1]));
  if (!fractions.some((f) => f < 1)) return undefined;
  const pct = (f: number) => `${+(f * 100).toPrecision(2)}%`;
  const lo = Math.min(...fractions);
  const hi = Math.max(...fractions);
  if (perGene) return `abundant genes from ≥${pct(lo)} samples, rare ones whole; each molecule weighted by its sample`;
  return lo === hi ? `≈${pct(lo)} sample, weighted` : `≈${pct(lo)}–${pct(hi)} sample, varying by tile, weighted`;
}

/** What the gene modules were fitted on, and how far along the fit is. */
function ModulesNote({ modules }: { modules: ReturnType<typeof useGeneModules> }) {
  if (modules.sampleError ?? modules.error) return <p className="hint error">{(modules.sampleError ?? modules.error)?.message}</p>;
  if (modules.sampleProgress)
    return (
      <p className="hint">
        Reading a sample of the whole element: tile {modules.sampleProgress.done} of {modules.sampleProgress.total}…
      </p>
    );
  if (modules.loading) return <p className="hint">Embedding every gene and clustering (PCA + e-means)…</p>;
  const m = modules.value;
  if (!m) return null;
  const s = m.summary;
  const pct = (f: number) => `${+(f * 100).toPrecision(2)}%`;
  return (
    <p className="hint">
      Fitted once to the whole element, never to the view, so a module means the same genes everywhere you pan:{" "}
      {s.molecules.toLocaleString()} molecules ({pct(m.sampleFraction)} random sample) in {s.bins.toLocaleString()} bins of{" "}
      {Math.round(s.binSize)} µm*. {s.components} principal components rise above noise ({pct(s.explained)} of the variance).{" "}
      {s.modules.reduce((n, g) => n + g.length, 0).toLocaleString()} genes with spatial structure fall into {s.modules.length} modules;{" "}
      {s.unstructured.toLocaleString()} without it, {s.dropped.rare.toLocaleString()} too rare in the sample and{" "}
      {s.dropped.excluded.toLocaleString()} control probes are left out. {(s.ms / 1000).toFixed(1)} s.
    </p>
  );
}
