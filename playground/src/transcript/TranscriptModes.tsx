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
import { clampWindow, type Rect } from "../../../src/datasource/points";
import { type ModeBasis, modeBasis } from "../../../src/gpu/spatial/gramModes";
import { type CoLocationModes, radiusToCover } from "../../../src/spatial/gram";
import { invert2, rectThrough2 } from "../../../src/spatial/ngffTransform";
import { listImageElements } from "../datasource/imageContext";
import { listPointsElements } from "../datasource/pointsTileLoader";
import { SpatialDeckView } from "../deck/SpatialDeckView";
import { SHARED_DEVICE_PROPS, useSharedDevice } from "../deck/useSharedDevice";
import { useAsync, useSettled } from "../hooks/useAsync";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { ChannelPicker } from "./ChannelPicker";
import { buildChannels, NEGATIVE_CONTROLS, type Selection, STARTER_SETS } from "./channelModel";
import { Explain } from "./Explain";
import { GramModesLayer } from "./GramModesLayer";
import { MatrixView } from "./MatrixView";
import { ModeLegend } from "./ModeLegend";
import { useTranscriptGram } from "./useTranscriptGram";

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
  const [rasterSide, setRasterSide] = useState(0);
  const [follow, setFollow] = useState(true);
  const [pinned, setPinned] = useState<Rect>();
  const [selection, setSelection] = useState(DEFAULT_SELECTION);
  const [saturate, setSaturate] = useState(2.5);
  const [chromaWeight, setChromaWeight] = useState(0.5);
  const [opacity, setOpacity] = useState(0.75);
  const [lock, setLock] = useState<ModeLock>();

  const device = useSharedDevice();
  const deckProps = useMemo(
    () => ({ deviceProps: SHARED_DEVICE_PROPS, onDeviceInitialized: device.onDeviceInitialized }),
    [device.onDeviceInitialized],
  );

  const sd = useSpatialData(storeUrl);
  const pointsNames = useMemo(() => (sd.value ? listPointsElements(sd.value) : []), [sd.value]);
  const pointsEl = pointsName ?? pointsNames.find((n) => /morton/.test(n) && !/feature_then/.test(n)) ?? pointsNames[0];
  const source = usePointsSource(sd.value, pointsEl, { columns: ["qv"] });
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
  const built = useMemo(
    () => (source.value ? buildChannels(selection, source.value.features) : { channels: [], missing: {} }),
    [selection, source.value],
  );
  // No compute until deck's device is ours: resources made on another device could not be drawn.
  const channels = device.ready ? built.channels : [];
  // Scoped to the source: a result stands in for the next only while it is about the same transcripts.
  const gram = useTranscriptGram(tiles.value, channels, { radius: rEff, qvMin, rasterSide }, source.value);
  const g = gram.value;

  // A lock applies while the channels are the ones it was taken with; otherwise this window's own
  // modes. The checkbox shows whether it applies, not merely whether one is held.
  const locked = lock && g && sameLabels(lock.basis.labels, g.modes.labels) ? lock : undefined;
  const modes = locked?.modes ?? g?.modes;
  const toggleLock = (on: boolean) => setLock(on && g ? { basis: modeBasis(g.res), modes: g.modes, window: g.window } : undefined);

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
    [g, locked, modelMatrix, saturate, chromaWeight, opacity],
  );

  const error = sd.error ?? source.error ?? tiles.error ?? gram.error;
  const busy = tiles.progress ? `loading tiles ${tiles.progress.done}/${tiles.progress.total}…` : gram.loading ? "computing…" : "";
  // The radius on screen, when the window forced it above the one chosen.
  const raised = g && g.radius > radius ? g.radius : undefined;
  // A fixed map size can make pixels wider than the kernel, and then molecules between pixel centres are lost.
  const coarse = g && rasterSide > 0 ? (g.window.maxX - g.window.minX) / g.raster.width > g.radius : false;

  // What belongs to one store, or one element of it, is dropped with it — by the action that changes
  // it, not an effect after: the pinned window is in the old element's frame, the lock is the old
  // data's projection, and the last viewport would fetch tiles at the old store's coordinates.
  const openStore = (url: string) => {
    setStoreUrl(url);
    setPointsName(undefined);
    setImageName(undefined);
    setPinned(undefined);
    setLock(undefined);
    setViewport(undefined);
  };
  const choosePoints = (name: string) => {
    setPointsName(name);
    setPinned(undefined);
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
          <label>
            Transcripts
            <select value={pointsEl ?? ""} onChange={(e) => choosePoints(e.target.value)}>
              {pointsNames.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
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
          <label>
            Minimum molecule quality (qv)
            <input type="number" min={0} max={40} value={qvMin} onChange={(e) => setQvMin(Math.max(0, Number(e.target.value) || 0))} />
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
          {source.value && (
            <ChannelPicker features={source.value.features} selection={selection} onChange={setSelection} missing={built.missing} />
          )}
          <p className="hint">µm*: the store does not declare a unit; Xenium writes transcript coordinates in micrometres.</p>
        </aside>

        <main className="views">
          {/* Always mounted, never keyed: see SpatialDeckView on why deck must outlive a store. */}
          <SpatialDeckView
            className="deck-canvas"
            sdata={sd.value}
            coordinateSystem={coordinateSystem}
            image={imageEl}
            layers={layers}
            deckProps={deckProps}
            onViewport={setViewport}
            fallbackBounds={fallbackBounds}
          />
          <p className="status">
            {error ? <span className="error">{error.message}</span> : busy}
            {!error && !busy && g && (
              <>
                window {span(g.window)} µm*{follow ? "" : " (kept)"} · r {g.radius} µm*
                {raised !== undefined ? ` (raised from ${radius} to cover the view)` : ""} · {g.modes.labels.length} channels ·{" "}
                {g.stats.perChannel.reduce((s, n) => s + n, 0).toLocaleString()} molecules in them · {g.stats.belowMinimum.toLocaleString()}{" "}
                below qv {qvMin} · map {g.raster.width}×{g.raster.height} · {tiles.value?.fetched ?? 0} tiles fetched,{" "}
                {(tiles.value?.tiles.length ?? 0) - (tiles.value?.fetched ?? 0)} from cache · channels {g.ms.channels.toFixed(0)} ms · Gram{" "}
                {g.ms.gram.toFixed(0)} ms
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
              <MatrixView gram={g} />
            </>
          )}

          <section className="explainers">
            <Explain title="What is a channel?">
              A channel is one gene, or a group of genes counted together — "T cells" adds up the molecules of several T-cell marker genes.
              A gene can only be in one channel at a time. <b>Negative controls</b> are probes that match no gene: they show what random
              background looks like, so a real pattern should look different from them.
            </Explain>
            <Explain
              title="What does the matrix show?"
              math={String.raw`g_{ab} = \frac{|A|\,\big(C_{ab} - S_{ab}\big)}{W_a W_b}, \qquad C = M M^\top`}
            >
              For each pair of channels, <b>g</b> compares how often their molecules lie within about 2r of each other with what you would
              expect if they were scattered at random over the window. 1 means "no more than chance", 2 means twice as often, below 1 means
              they keep apart. Molecules all sit inside tissue, so almost every pair comes out above 1 — compare pairs with each other, and
              with the negative controls, rather than with 1. In the formula, M holds each channel's smoothed map, W its total, |A| the
              window area and S the self-pairs, which are removed.
            </Explain>
            <Explain
              title="What do the colours mean?"
              math={String.raw`\mathrm{corr} = V \Lambda V^\top, \qquad \text{colour}(x) = \mathrm{OKLab}\big(z(x)\!\cdot\! v_1,\ z(x)\!\cdot\! v_2,\ z(x)\!\cdot\! v_3\big)`}
            >
              The channels' maps are summarised by a few <b>modes</b>: patterns of channels that rise and fall together across the window.
              Mode 1 sets brightness and modes 2 and 3 set hue, so places with similar colours have similar mixtures of genes. The legend
              says which channels push each mode each way, and how much of the variation it carries.
            </Explain>
            <Explain title="Locking the modes">
              Normally each window gets its own modes, and each place is compared with the rest of its window: the same colour in two
              windows can mean different mixtures, and the strongest contrast in a window always gets the strongest colours. Locked, every
              window is measured against the window where you locked — same averages, same modes, same scale — so a colour means the same
              mixture wherever you look, and a window with little going on looks muted. Each map counts molecules per µm², so the lock also
              holds when you change the radius or the quality filter: a larger radius smooths the map, and its colours calm down, but a
              colour still means the same mixture. It lapses only if the channels change.
            </Explain>
            <Explain title="Why does the radius set the resolution?">
              r is the neighbourhood each molecule is smoothed over. Anything smaller than r is blurred away, so the map is computed at
              about three pixels per radius — a finer grid would cost more without showing more. A large r shows tissue-scale structure; a
              small r gets closer to single cells, at more cost. A map can only be so many pixels across (2048), so one map covers at most
              about 680 µm* at r = 1, or 6.8 mm* at r = 10. Zoomed out past that, the radius is raised until the map covers the whole view,
              and the radius label says so: every molecule still counts, but the neighbourhood is coarser. Zoom in, and the radius you chose
              comes back.
            </Explain>
            <Explain title="Which part of the slide is analysed?">
              The part in view, once you stop moving: the white outline marks it. Zoomed far out, that can be the whole slide, and the first
              time every molecule on it has to be fetched, which takes a while. Molecules are fetched in tiles and kept, so going back over
              ground already seen is fast. Turn off "follow the view" to keep one window while you look around it. Each mode's sign is kept
              consistent between windows so colours do not flip for no reason — but if two modes swap order, the colours change, and that
              change is real.
            </Explain>
            <Explain title="What is the quality filter?">
              Every molecule has a quality score, qv. At 20, the chance that it was read as the wrong gene is about 1 in 100. Molecules
              below the threshold are dropped, as in Xenium's own per-cell counts.
            </Explain>
          </section>
        </main>
      </div>
    </div>
  );
}
