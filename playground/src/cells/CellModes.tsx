// Cell co-location modes: the Gram matrix over a set of per-cell channels, and its OKLab
// eigen-projection (docs/cell-stats.md §4), on the deck/sd.js stack `transcriptmodes` uses.
//
// Channels come from either a categorical `obs` column or chosen vars of `X` — a cell type is the
// one-hot case of a per-cell mark, so both take the identical path. Where `cellstats` selects a PAIR,
// here the whole K×K matrix is decomposed and the leading modes painted at once: the map answers
// "what co-location structure is there, and where" without a pair being nominated first.
//
// The view is sd.js's (its image layer, its pan and zoom); the mode map is a deck layer drawn from
// the Gram's rasters on the device deck and this library share. The wand samples a pixel's profile
// and outlines everywhere within a whitened distance of it, in the same layer.

import type { Layer } from "@deck.gl/core";
import { PathLayer } from "@deck.gl/layers";
import { useCallback, useMemo, useState } from "react";
import type { Rect } from "../../../src/datasource/points";
import { envelopeExcursions, type SpectrumEnvelopeResult, spectrumEnvelope } from "../../../src/gpu/spatial/gramEnvelope";
import { type ModeBasis, modeBasis } from "../../../src/gpu/spatial/gramModes";
import { apronCoverage, type CoLocationModes } from "../../../src/spatial/gram";
import { IDENTITY2 } from "../../../src/spatial/ngffTransform";
import { equivalentRadius, KERNELS, kernelLabel } from "../../../src/spatial/kernels";
import { listImageElements } from "../datasource/imageContext";
import { SpatialDeckView } from "../deck/SpatialDeckView";
import { SHARED_DEVICE_PROPS, useSharedDevice } from "../deck/useSharedDevice";
import { useAsync } from "../hooks/useAsync";
import { defaultSystem, useCellTable, useCellTables, useVarCatalog } from "../hooks/useCellTable";
import { useSpatialData } from "../hooks/useSpatialData";
import { GramModesLayer } from "../modes/GramModesLayer";
import { onGpu } from "../modes/gramSnapshots";
import { MatrixView } from "../modes/MatrixView";
import { ColoursExplain, ModeCharts } from "../modes/ModeCharts";
import { ModeMap } from "../modes/ModeMap";
import { ImagePicker, MapStyleControls, StoreInput, useImageChoice, useMapStyle } from "../modes/PageControls";
import { useWand, type WandSource, wandLayerProps } from "../modes/useWand";
import { WandExplain, WandPanel } from "../modes/WandPanel";
import { MaskStats } from "../selection/MaskStats";
import { useBoundarySettings, useMapSelection } from "../selection/useMapSelection";
import { BoundariesExplain, BoundaryPanel } from "../wand/WandPanel";
import { Explain } from "../transcript/Explain";
import { buildCellChannels, type ChannelSpec, tableBounds, viewDims, WINDOW_MODES, type WindowMode, windowBbox } from "./cellChannels";
import { ChannelSource, type SourceKind, usableObsColumns } from "./ChannelSource";
import { useCellGram } from "./useCellGram";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const RASTER_SIDES = [256, 384, 512];
const sameLabels = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((l, i) => l === b[i]);

const outline = ([x0, y0, x1, y1]: readonly number[]): [number, number][] => [
  [x0!, y0!],
  [x1!, y0!],
  [x1!, y1!],
  [x0!, y1!],
  [x0!, y0!],
];

export function CellModes() {
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [tableName, setTableName] = useState<string>();
  const [kindChoice, setKind] = useState<SourceKind>("obs");
  const [obsChoice, setObsColumn] = useState<string>();
  const [matrix, setMatrix] = useState("X");
  const [vars, setVars] = useState<number[]>([]);
  const [kernelIdx, setKernelIdx] = useState(1); // Epanechnikov: kernels.ts measures it as the best-behaved member
  const [rangeChoice, setRangeUm] = useState<number>();
  const [umPerUnit, setUmPerUnit] = useState(1);
  const [rasterSide, setRasterSide] = useState(384);
  const [windowMode, setWindowMode] = useState<WindowMode>("apron");
  const style = useMapStyle();
  const { saturate, chromaWeight, opacity } = style;
  const [systemChoice, setSystem] = useState<string>();
  const [sims, setSims] = useState(99);
  const [envelope, setEnvelope] = useState<{ generation: number; result: SpectrumEnvelopeResult }>();
  const [envProgress, setEnvProgress] = useState<string>();
  const [viewport, setViewport] = useState<Rect>();

  const device = useSharedDevice();
  // Store → table → channels.
  const sd = useSpatialData(storeUrl);
  const tables = useCellTables(sd.value);
  const usable = useMemo(() => (tables.value ?? []).filter((t) => t.hasCentroids), [tables.value]);
  const tableEl = tableName ?? usable[0]?.name;
  const tableInfo = usable.find((t) => t.name === tableEl);
  // A store with no `global` (mibitof: one system per field of view) needs the system chosen, and
  // only the images in it offered — sd.js draws nothing for an image outside the view's system.
  const systems = sd.value?.coordinateSystems ?? [];
  const system = systemChoice ?? defaultSystem(sd.value);
  const base = useCellTable(sd.value, tableEl, system);
  const catalog = useVarCatalog(sd.value, tableEl);
  const allImages = useAsync(() => (sd.value ? listImageElements(sd.value) : undefined), [sd.value], sd.value);
  const imageNames = useMemo(
    () => (allImages.value ?? []).filter((n) => !system || (sd.value?.images?.[n]?.coordinateSystems.includes(system) ?? false)),
    [allImages.value, sd.value, system],
  );
  const images = useImageChoice(imageNames);

  const obsColumns = useMemo(() => usableObsColumns(tableInfo?.columns), [tableInfo]);
  // A store with no annotation has only X to offer, whatever tab was last chosen.
  const kind: SourceKind = obsColumns.length ? kindChoice : "x";
  const obsColumn = obsChoice ?? tableInfo?.suggested ?? obsColumns[0]?.name;
  const spec = useMemo<ChannelSpec | undefined>(
    () => (kind === "obs" ? (obsColumn ? { kind, column: obsColumn } : undefined) : vars.length >= 2 ? { kind, vars, matrix } : undefined),
    [kind, obsColumn, vars, matrix],
  );
  // The catalogue only for X: obs channels never read it, and its late arrival must not rebuild them.
  const xCatalog = spec?.kind === "x" ? catalog.value : undefined;
  const built = useAsync(
    () => (sd.value && base.value && spec ? buildCellChannels(sd.value, base.value, spec, xCatalog) : undefined),
    [sd.value, base.value, spec, xCatalog],
    base.value,
  );

  // Lengths: the range box is the EFFECTIVE pair-kernel support (J⊛J, twice the splat radius), so it
  // is halved for the splat, then rescaled per kernel so switching kernel holds the probed scale fixed.
  const toWorld = (um: number) => um / umPerUnit;
  const full = useMemo(() => (base.value ? tableBounds(base.value) : undefined), [base.value]);
  const defaultRange = full ? Number(((Math.max(full[2] - full[0], full[3] - full[1]) / 40) * umPerUnit).toPrecision(3)) : 0;
  const rangeUm = rangeChoice ?? defaultRange;
  const kernel = KERNELS[kernelIdx] ?? KERNELS[1]!;
  const radius = equivalentRadius(kernel, toWorld(rangeUm) / 2);
  // The window depends on the radius, and the raster on the window — so this order is forced.
  const win = useMemo(() => (full && radius > 0 ? windowBbox(full, radius, windowMode) : undefined), [full, radius, windowMode]);
  const params = useMemo(
    () => (win ? { bbox: win.bbox, ...viewDims(win.bbox, rasterSide), radius, kernel } : undefined),
    [win, rasterSide, radius, kernel],
  );

  // No compute until deck's device is ours: resources made on another device could not be drawn.
  const gram = useCellGram(device.ready ? built.value?.channels : undefined, params, base.value);
  const g = gram.value;
  const wandSource = useMemo<WandSource | undefined>(
    () => (g ? { res: g.res, rasters: g.rasters, mean: g.res.resident.mean, sd: g.res.resident.sd, vectors: g.modes.vectors } : undefined),
    [g],
  );
  const wand = useWand(wandSource);
  const env = envelope && g && envelope.generation === g.generation ? envelope.result : undefined;

  const K = g?.modes.labels.length ?? 0;

  // PROTOTYPE (docs/wand-metric-and-regions.md): boundaries, and a mask painted with the map's tools.
  // The table's frame is deck's world, so no transform; no broader scale is computed here.
  const view = useMemo(() => (g ? { ...g, radius: g.params.radius } : undefined), [g]);
  const boundaries = useBoundarySettings();
  // The mask is judged by one metric, so the first stroke fixes the modes and standardisation the
  // wand measures with, for as long as the channels stay the same: a new range, raster or window
  // otherwise changes what "similar" means under a mask already painted.
  const [paintLock, setPaintLock] = useState<{ modes: CoLocationModes; basis: ModeBasis }>();
  const lockNow = paintLock && g && sameLabels(paintLock.modes.labels, g.modes.labels) ? paintLock : undefined;
  const lockOnPaint = useCallback(() => {
    if (!paintLock && g) setPaintLock({ modes: g.modes, basis: modeBasis(g.res) });
  }, [paintLock, g]);
  const tools = useMapSelection(boundaries, {
    g: view,
    modes: lockNow?.modes ?? g?.modes,
    basis: lockNow?.basis,
    toFrame: IDENTITY2,
    // The brush is sized in µm; the table's units are the µm-per-unit box's.
    unitsPerUm: 1 / umPerUnit,
    onPaint: lockOnPaint,
  });
  const deckProps = useMemo(
    () => ({ deviceProps: SHARED_DEVICE_PROPS, onDeviceInitialized: device.onDeviceInitialized, ...tools.deckProps }),
    [device.onDeviceInitialized, tools.deckProps],
  );
  const layers = useMemo<Layer[]>(
    () =>
      g
        ? [
            new GramModesLayer({
              id: "cell-modes",
              res: g.res,
              rasters: g.rasters,
              vectors: g.modes.vectors,
              saturate,
              chromaWeight,
              opacity,
              ...wandLayerProps(wand, K),
              pickable: false,
            }),
            ...tools.layers,
            new PathLayer<{ path: [number, number][] }>({
              id: "cell-window",
              data: [{ path: outline(g.res.bbox) }],
              getPath: (d) => d.path,
              getColor: [255, 255, 255, 200],
              getWidth: 1.5,
              widthUnits: "pixels",
              pickable: false,
            }),
          ]
        : [],
    [g, wand, saturate, chromaWeight, opacity, K, tools.layers],
  );

  /** Simulate the null spectrum. Re-splats this result's own channels and window, so the band is
   *  about the spectrum on screen; a later compute drops it rather than banding a different one. */
  const runEnvelope = () => {
    if (!g) return;
    setEnvProgress("starting…");
    const at = g.generation;
    void onGpu(() =>
      spectrumEnvelope(g.channels, g.params, {
        simulations: sims,
        onProgress: async (done, total) => {
          setEnvProgress(`${done}/${total} realisations…`);
          // Yield so the progress actually paints.
          await new Promise((r) => setTimeout(r, 0));
        },
      }),
    ).then(
      (result) => {
        setEnvelope({ generation: at, result });
        setEnvProgress(undefined);
      },
      (err: unknown) => setEnvProgress(`failed: ${err instanceof Error ? err.message : String(err)}`),
    );
  };

  // What is in one table's frame, or measured with one set of channels, is dropped with it — by the
  // action that changes it: the wand's place, the painted mask and the metric it was painted with.
  const dropSelection = () => {
    wand.clear();
    tools.clear();
    setPaintLock(undefined);
  };
  /** A change of channels keeps the frame but not the metric. */
  const newChannels = () => {
    tools.clear();
    setPaintLock(undefined);
  };
  const openStore = (url: string) => {
    setStoreUrl(url);
    setTableName(undefined);
    setObsColumn(undefined);
    setVars([]);
    setMatrix("X");
    setRangeUm(undefined);
    images.reset();
    setSystem(undefined);
    dropSelection();
    setViewport(undefined);
  };
  const chooseTable = (name: string) => {
    setTableName(name);
    setObsColumn(undefined);
    setVars([]);
    setMatrix("X");
    setRangeUm(undefined);
    dropSelection();
  };

  const error = sd.error ?? tables.error ?? base.error ?? built.error ?? gram.error;
  const busy = tables.loading
    ? "inspecting the store…"
    : base.loading
      ? "reading centroids…"
      : built.loading
        ? kind === "x"
          ? "reading expression…"
          : "splitting by category…"
        : gram.loading
          ? "computing…"
          : !spec && kind === "x"
            ? "pick at least two vars and press “use these vars”"
            : "";
  const unit = base.value?.units.micrometres !== undefined ? "µm" : "µm*";
  const pxPerRadius = g ? g.params.radius / ((g.res.bbox[2] - g.res.bbox[0]) / g.res.width) : 0;
  const coverage = useMemo(() => {
    if (!g) return 0;
    const cov = apronCoverage(g.res, g.params.radius);
    return cov.length ? cov.reduce((a, b) => a + b, 0) / cov.length : 0;
  }, [g]);

  return (
    <div className="page">
      <h2 className="page">Which cell types live together — co-location modes</h2>
      <p className="page">
        Every cell is smoothed over a neighbourhood, one map per channel — a cell type, or one gene's expression. The page then asks which
        channels rise and fall together across the tissue, and paints the strongest of those patterns as colour: places with similar colours
        have similar mixtures of cells. Click the map to pick a place; everywhere with a similar mixture is outlined.
      </p>

      <div className="layout">
        <aside className="controls">
          <StoreInput initial={DEFAULT_STORE} onOpen={openStore} />
          <label>
            Table
            <select value={tableEl ?? ""} onChange={(e) => chooseTable(e.target.value)}>
              {(tables.value ?? []).map((t) => (
                <option key={t.name} value={t.name} disabled={!t.hasCentroids}>
                  {t.error ? `${t.name} — unreadable` : `${t.name} — ${t.nRows} cells${t.hasCentroids ? "" : ", no obsm/spatial"}`}
                </option>
              ))}
            </select>
          </label>
          <ChannelSource
            key={tableEl}
            kind={kind}
            onKind={(k) => {
              setKind(k);
              newChannels();
            }}
            obsColumns={obsColumns}
            obsColumn={obsColumn}
            onObsColumn={(c) => {
              setObsColumn(c);
              newChannels();
            }}
            catalog={catalog.value}
            matrix={matrix}
            onMatrix={(m) => {
              setMatrix(m);
              newChannels();
            }}
            applied={vars}
            onApply={(v) => {
              setVars(v);
              newChannels();
            }}
          />
          {systems.length > 1 && (
            <label>
              Coordinate system
              <select
                value={system ?? ""}
                onChange={(e) => {
                  setSystem(e.target.value);
                  images.reset();
                  dropSelection();
                }}
              >
                {systems.map((n) => (
                  <option key={n}>{n}</option>
                ))}
              </select>
            </label>
          )}
          <ImagePicker images={images} />
          <label>
            Kernel
            <select value={kernelIdx} onChange={(e) => setKernelIdx(Number(e.target.value))}>
              {KERNELS.map((k, i) => (
                <option key={kernelLabel(k)} value={i}>
                  {kernelLabel(k)}
                  {i === 1 ? " (recommended)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Interaction range ({unit})
            <input
              type="number"
              min={0}
              value={rangeUm || ""}
              onChange={(e) => setRangeUm(Number(e.target.value) > 0 ? Number(e.target.value) : undefined)}
            />
          </label>
          <label>
            µm per unit
            <input
              type="number"
              min={0}
              step="any"
              value={umPerUnit}
              onChange={(e) => setUmPerUnit(Number(e.target.value) > 0 ? Number(e.target.value) : 1)}
            />
          </label>
          <label>
            Map resolution
            <select value={rasterSide} onChange={(e) => setRasterSide(Number(e.target.value))}>
              {RASTER_SIDES.map((s) => (
                <option key={s} value={s}>
                  {s}²
                </option>
              ))}
            </select>
          </label>
          <label>
            Analysis window
            <select value={windowMode} onChange={(e) => setWindowMode(e.target.value as WindowMode)}>
              {WINDOW_MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <MapStyleControls style={style} />
          <WandPanel wand={wand} />
          <BoundaryPanel {...tools.panel} mixtures="cells" />
          <p className="hint">
            {unit === "µm*" ? "µm*: the store states no unit; the µm-per-unit box is your declaration. " : ""}⚠ The store must be
            CORS-enabled. The mode terrain (3-D) is still on the <a href="./cellmodes-legacy.html">legacy page</a>.
          </p>
        </aside>

        <main className="views">
          {/* Always mounted, never keyed: see SpatialDeckView on why deck must outlive a store. */}
          <ModeMap wand={wand} modes={g?.modes} viewport={viewport} selection={tools}>
            <SpatialDeckView
              className="deck-canvas"
              sdata={sd.value}
              coordinateSystem={system ?? "global"}
              image={images.image}
              layers={layers}
              deckProps={deckProps}
              onViewport={setViewport}
              onClick={tools.onClick}
              onHover={tools.onHover}
              fallbackBounds={full ? { minX: full[0], minY: full[1], maxX: full[2], maxY: full[3] } : undefined}
            />
          </ModeMap>
          <p className="status">
            {error ? <span className="error">{error.message}</span> : busy}
            {!error && !busy && g && (
              <>
                {built.value?.note} · {kernelLabel(kernel)}, range {rangeUm.toPrecision(3)} {unit} · map {g.res.width}×{g.res.height},
                kernel {pxPerRadius.toFixed(1)} px
                {pxPerRadius < 2 ? <span className="raised"> — under-resolved: widen the range or coarsen the map</span> : ""} · window{" "}
                {win?.label},{" "}
                {windowMode === "full" ? (
                  <span className="raised">apron empty — g carries the edge deficit</span>
                ) : (
                  <>
                    apron coverage {coverage.toFixed(2)}
                    {coverage < 0.5 ? <span className="raised"> (thin — window is near the tissue edge)</span> : ""}
                  </>
                )}{" "}
                · PSD defect {g.modes.psdDefect.toExponential(1)} · {g.ms.toFixed(0)} ms
              </>
            )}
          </p>
          <p className="hint">GPU: {device.status}</p>

          {g && (
            <>
              <ModeCharts
                modes={g.modes}
                {...(env ? { envelope: env } : {})}
                nullControls={
                  <>
                    <span className="row">
                      <select value={sims} onChange={(e) => setSims(Number(e.target.value))}>
                        {[39, 99, 199].map((n) => (
                          <option key={n} value={n}>
                            {n} permutations
                          </option>
                        ))}
                      </select>
                      <button type="button" disabled={envProgress !== undefined && !envProgress.startsWith("failed")} onClick={runEnvelope}>
                        run null
                      </button>
                      <span className="hint">{envProgress}</span>
                    </span>
                    {env && <EnvelopeVerdict env={env} />}
                  </>
                }
              />
              <h3>Which channels are found in the same places</h3>
              <MatrixView
                labels={g.modes.labels}
                values={g.res.corr}
                scale="signed"
                describe={(a, b) => {
                  const r = g.res.corr[a * K + b] ?? 0;
                  const gg = g.res.g[a * K + b] ?? 0;
                  const how = r > 0.2 ? "found in the same places" : r < -0.2 ? "spatially exclusive" : "little spatial relation";
                  return `${g.modes.labels[a]} × ${g.modes.labels[b]}: r = ${r.toFixed(3)} (${how}) · g = ${gg.toFixed(3)} (1 = random)`;
                }}
              />
              <MaskStats
                channels={built.value?.channels}
                mask={tools.panel.mask}
                window={g.res.bbox}
                rMax={toWorld(rangeUm)}
                umPerUnit={umPerUnit}
                unit={unit}
              />
            </>
          )}

          <section className="explainers">
            <Explain title="What is a channel?">
              A channel is one set of cells, or one per-cell quantity: a cell type from an annotation column, or one gene's expression from
              X. Most raw SpatialData stores have no cell-type column at all — a Xenium export carries cell ids, areas and counts and
              nothing else — but they all have X, so the expression tab works on any of them. A cell type is just the case where each cell's
              weight is 1 or 0, so both take exactly the same path.
            </Explain>
            <ColoursExplain mixtures="cells" />
            <Explain title="What is the interaction range?">
              The range is how far apart two cells can be and still count as "together": the reach of the smoothing applied to each cell,
              taken twice (once for each cell of the pair), so the box means what it says. A large range shows tissue-scale structure; a
              small one gets closer to neighbouring cells. The map needs about two pixels per smoothing radius; below that the status line
              warns that it is under-resolved.
            </Explain>
            <Explain title="What is the analysis window, and the apron?">
              The window (white outline) is what gets measured and drawn. Every cell is smoothed regardless, so cells just outside the
              window — the apron — still count towards the cells inside it, and correct the counts near its edge. On the full extent there
              is nothing outside to correct with, so pairs near the edge are undercounted. Insetting by one range is all it takes, which is
              the default; the status line reports how well populated the apron is.
            </Explain>
            <WandExplain />
            <BoundariesExplain mixtures="cells" />
            <Explain title="How much variation does each mode carry — and is it real?">
              The bars are each mode's share of the total variation; they add up to 1. But 45% means nothing on its own: some structure is
              inevitable just because cells are not spread evenly. "Run null" shuffles which cell carries which mark, many times, keeping
              every cell in place (and each cell's marks together), and recomputes the spectrum each time. The grey band is the range those
              shuffles produce; a bar outside it, marked with a dot, is structure the shuffles cannot explain.
            </Explain>
            <Explain title="The loadings and the matrix">
              Each row of the loadings is one mode; each bar a channel's weight in it. Channels that lean the same way are found in the same
              places; opposite signs mean they avoid each other. The matrix shows every pair's spatial correlation directly, from −1
              (exclusive) through 0 to 1 (always together); hover a cell for it, and for g, the pair's co-location compared with random
              placement.
            </Explain>
          </section>
        </main>
      </div>
    </div>
  );
}

function EnvelopeVerdict({ env }: { env: SpectrumEnvelopeResult }) {
  const out = envelopeExcursions(env);
  return (
    <p className="hint">
      {((1 - env.envelope.alpha) * 100).toFixed(0)}% global envelope over {env.envelope.simulations} shuffles · p ={" "}
      {env.envelope.p.toFixed(3)} ·{" "}
      {env.envelope.exits ? (
        <b className="good">
          outside the null band:{" "}
          {out
            .map(
              (e) => `mode ${e.mode} ${e.above ? "above" : "below"} (${(e.observed * 100).toFixed(0)}% vs ${(e.bound * 100).toFixed(0)}%)`,
            )
            .join(", ")}
        </b>
      ) : (
        <b className="raised">inside the null band — random relabelling of the same cells gives this spectrum</b>
      )}
    </p>
  );
}
