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
import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { BoundaryLayer } from "./BoundaryLayer";
import {
  type BoundaryField,
  boundaryStrength,
  focusCell,
  laplacianCell,
  openCell,
  ppcaMetric,
  ridgeCell,
  ridges,
} from "./boundaryStrength";
import { ChannelPicker } from "./ChannelPicker";
import { buildChannels, NEGATIVE_CONTROLS, type Selection, STARTER_SETS } from "./channelModel";
import { Explain } from "./Explain";
import { GramModesLayer } from "../modes/GramModesLayer";
import { MatrixView } from "../modes/MatrixView";
import { ModeLegend } from "../modes/ModeLegend";
import { useTranscriptGram } from "./useTranscriptGram";
import { type MapTool, toolCursor, toolWidgets } from "./wandTools";
import {
  apply,
  discOnRaster,
  footprint,
  gridFor,
  islandAt,
  type MaskOp,
  matchingPiece,
  type PressOp,
  maskArea,
  onRaster,
  smoothOutline,
  sweep,
  type WorldMask,
} from "./worldMask";
import { gradedCell } from "./gradedNecks";
import "@deck.gl/widgets/stylesheet.css";

/** A press on the map: deck's click, or the start, moves and end of a drag. */
type PressPhase = "click" | "start" | "move" | "end";

/** A paint stroke: the union of the patches painted since the button went down. */
interface Stroke {
  readonly op: MaskOp;
  /** Removing whole pieces (Cmd/Ctrl+Alt), not what matches the patch. */
  readonly piece?: boolean;
  /** The mask when the stroke began: the stroke is combined with this, not with itself. */
  readonly base: WorldMask | undefined;
  readonly cell: number;
  readonly keys: Set<number>;
  /** The button is still down. */
  open: boolean;
  /** The seed whose patch is awaited (the `cellSeed` set for it), if any. Not a flag: a move can land
   *  between a render and its effect, and must not be cleared by the previous patch arriving. */
  want: readonly [number, number] | undefined;
  /** The brush's last position, the start of the next swept segment. */
  last?: readonly [number, number];
}

/** The op for the keys held: Shift add, Alt subtract, both intersect; Cmd/Ctrl+Alt removes a piece. */
const pressOp = (shift: boolean, alt: boolean, cmd: boolean): PressOp =>
  alt && cmd ? "piece" : shift && alt ? "intersect" : shift ? "add" : alt ? "subtract" : "replace";

/** BoundaryLayer's op code. */
const OP_INDEX: Record<PressOp, number> = { replace: 0, add: 1, subtract: 2, intersect: 3, piece: 2 };

/** The brush radius scaled by `f`, within 5 µm and 5 mm. */
const sizeBrush = (r: number, f: number): number => Math.min(5000, Math.max(5, r * f));

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

/** PROTOTYPE: the boundary panel's recommended settings (docs/wand-metric-and-regions.md §4). Each is
 *  a complete setting of the fine-tune controls, so the panel can say which one is in effect. */
interface BoundaryValues {
  q: number;
  w: number;
  residualSmooth: number;
  lines: boolean;
  rule: "laplacian" | "threshold" | "ridges";
  dSmooth: number;
  tol: number;
  open: number;
  /** Judge the patch at a second, broader radius too: the broad cell picks it, this one places its edge. */
  twoScale: boolean;
  /** The broad radius, as a multiple of r. */
  broad: number;
  /** How far, in r, the edge may sit outside the broad cell. */
  slack: number;
}
type BoundaryPresetName = "patch" | "tissue" | "fine";
const ONE_SCALE = { twoScale: false, broad: 3, slack: 1 } as const;
const BOUNDARY_PRESETS: Record<BoundaryPresetName, { label: string; hint: string; values: BoundaryValues }> = {
  patch: {
    label: "The patch the click is in (recommended)",
    hint: "a compact region of the click's mixture; thin branches dropped",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 1, tol: 0, open: 0.75, ...ONE_SCALE },
  },
  tissue: {
    label: "Follow the tissue",
    hint: "everywhere connected to the click with the same mixture, branches and all",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 1, tol: 0, open: 0, ...ONE_SCALE },
  },
  fine: {
    label: "Fine detail",
    hint: "edges judged at half the radius: smaller patches, closer to the boundary",
    values: { q: 3, w: 0, residualSmooth: 0, lines: true, rule: "laplacian", dSmooth: 0.5, tol: 0, open: 0.5, ...ONE_SCALE },
  },
};

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
  // PROTOTYPE: hovering seeds the patch; a click or a drag paints it into the mask. Through refs, so
  // deck's props stay stable.
  const onMapPress = useRef<(phase: PressPhase, x: number | undefined, y: number | undefined, event?: PointerInfo) => void>(undefined);
  /** The pointer over the map, or no position when it has left. */
  const onMapHover = useRef<(x?: number, y?: number) => void>(undefined);
  const press = useCallback(
    (phase: PressPhase) =>
      (info: { coordinate?: number[]; viewport?: { unproject: (xy: number[]) => number[] } }, event?: PointerInfo): void => {
        // A drag's info is the press-down pick; its current position is the event's.
        const at =
          phase === "move" && event?.offsetCenter && info.viewport
            ? info.viewport.unproject([event.offsetCenter.x, event.offsetCenter.y])
            : info.coordinate;
        const [x, y] = at ?? [];
        onMapPress.current?.(phase, x, y, event);
      },
    [],
  );
  const onClick = useMemo(() => press("click"), [press]);
  const onHover = useCallback((info: { coordinate?: number[] }) => {
    const [x, y] = info.coordinate ?? [];
    onMapHover.current?.(x, y);
  }, []);

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

  // PROTOTYPE (docs/wand-metric-and-regions.md, experiment 1): boundary strength in the PPCA metric.
  const [showBounds, setShowBounds] = useState(false);
  /** The map's tool. The wand needs the boundary field, and the mask is drawn by the boundary layer,
   *  so choosing either painting tool turns boundaries on; with them off the map is the hand's. */
  const [tool, setTool] = useState<MapTool>("hand");
  const chooseTool = useCallback((t: MapTool) => {
    setTool(t);
    if (t !== "hand") setShowBounds(true);
  }, []);
  const active: MapTool = showBounds ? tool : "hand";
  const wandOn = active === "wand";
  const brushOn = active === "brush";
  /** The brush's radius, in the transcripts' frame (µm), and where it is over the map. */
  const [brushR, setBrushR] = useState(100);
  /** Draw the mask and preview smoothly (quadratic filter, antialiased), or as before, to compare. */
  const [smoothEdges, setSmoothEdges] = useState(true);
  const [brushAt, setBrushAt] = useState<readonly [number, number]>();
  const [mainModes, setMainModes] = useState(3);
  const [residualWeight, setResidualWeight] = useState(0);
  /** σ, in kernel radii, of the blur the modes after q are differentiated through. */
  const [residualSmooth, setResidualSmooth] = useState(0);
  /** Undefined = follow the window's 90th percentile. */
  const [theta, setTheta] = useState<number>();
  const [bounds, setBounds] = useState<{ field: BoundaryField; sigma2: number; q: number; version: number; generation: number }>();
  const boundsChain = useRef(Promise.resolve());
  const boundsVersion = useRef(0);
  useEffect(() => {
    if (!showBounds || !g || !modes) return;
    const K = g.res.labels.length;
    const q = Math.min(mainModes, K);
    const { A, sigma2 } = ppcaMetric(modes.vectors, modes.values, K, q, residualWeight);
    const sd = locked?.basis.sd ?? g.res.resident.sd;
    const mean = locked?.basis.mean ?? g.res.resident.mean;
    // One at a time: the strength buffers are shared between calls.
    boundsChain.current = boundsChain.current
      .then(() => boundaryStrength(g.res, g.rasters, mean, sd, A, g.radius, q, residualSmooth))
      .then((field) => setBounds({ field, sigma2, q, version: ++boundsVersion.current, generation: g.generation }))
      .catch((err: unknown) => console.error("boundary strength", err));
  }, [showBounds, g, modes, locked, mainModes, residualWeight, residualSmooth]);
  const boundsNow = showBounds && bounds && g && bounds.generation === g.generation ? bounds : undefined;
  const thetaNow = theta ?? boundsNow?.field.quantiles.p90 ?? 1;
  const [boundLines, setBoundLines] = useState(true);
  /** A click on the map, in the transcripts' frame: the seed of the ridge-bounded cell. */
  const [cellSeed, setCellSeed] = useState<readonly [number, number]>();
  /** How the clicked cell is bounded: the Laplacian of the wand distance, or the ridge lines. */
  const [cellRule, setCellRule] = useState<"laplacian" | "threshold" | "ridges">("laplacian");
  /** σ, in kernel radii, of the smoothing on d before its Laplacian (Marr–Hildreth's σ, on top of r). */
  const [dSmooth, setDSmooth] = useState(1);
  /** Also stop where d exceeds this; 0 = off. */
  const [cellTol, setCellTol] = useState(0);
  /** Opening radius, in kernel radii: parts of the cell narrower than this are dropped; 0 = off. */
  const [cellOpen, setCellOpen] = useState(0.75);
  /** The opening graded by neck width (a lobe fades as its neck narrows), or the binary opening.
   *  Off by default: on real data the fill behind a small patch has often percolated over most of the
   *  map, and grading shows that cluster faintly, which swings more than the binary pops (E8). */
  const [gradedNecks, setGradedNecks] = useState(false);
  const [twoScale, setTwoScale] = useState(false);
  const [broadFactor, setBroadFactor] = useState(3);
  const [focusSlack, setFocusSlack] = useState(1);

  // The broad neighbourhood: the same window and channels at broadFactor × r, on its own (coarser)
  // raster, computed only while it is asked for.
  const broadOn = showBounds && twoScale;
  const rBroad = Math.round(rEff * broadFactor * 10) / 10;
  const tilesBroad = usePointsTiles(source.value, broadOn ? window : undefined, rBroad);
  const gramBroad = useTranscriptGram(
    broadOn ? tilesBroad.value : undefined,
    channels,
    { radius: rBroad, qvMin: qvApplied, rasterSide: 0, halfRasters },
    source.value,
  );
  const gb = broadOn && g && gramBroad.value && gramBroad.value.window === g.window ? gramBroad.value : undefined;
  const [boundsB, setBoundsB] = useState<{ field: BoundaryField; generation: number }>();
  useEffect(() => {
    if (!gb) return;
    const K = gb.res.labels.length;
    const q = Math.min(mainModes, K);
    const { A } = ppcaMetric(gb.modes.vectors, gb.modes.values, K, q, residualWeight);
    boundsChain.current = boundsChain.current
      .then(() => boundaryStrength(gb.res, gb.rasters, gb.res.resident.mean, gb.res.resident.sd, A, gb.radius, q, 0, "broad"))
      .then((field) => setBoundsB({ field, generation: gb.generation }))
      .catch((err: unknown) => console.error("boundary strength (broad)", err));
  }, [gb, mainModes, residualWeight]);
  const boundsBroad = gb && boundsB && boundsB.generation === gb.generation ? boundsB : undefined;
  const boundaryValues: BoundaryValues = {
    q: mainModes,
    w: residualWeight,
    residualSmooth,
    lines: boundLines,
    rule: cellRule,
    dSmooth,
    tol: cellTol,
    open: cellOpen,
    twoScale,
    broad: broadFactor,
    slack: focusSlack,
  };
  /** The preset every fine-tune control matches, if any; θ is left out, as it follows the window. */
  const activePreset = (Object.keys(BOUNDARY_PRESETS) as BoundaryPresetName[]).find((name) => {
    const v = BOUNDARY_PRESETS[name].values;
    return (Object.keys(v) as (keyof BoundaryValues)[]).every((k) => v[k] === boundaryValues[k]);
  });
  const applyPreset = (name: BoundaryPresetName) => {
    const v = BOUNDARY_PRESETS[name].values;
    setMainModes(v.q);
    setResidualWeight(v.w);
    setResidualSmooth(v.residualSmooth);
    setBoundLines(v.lines);
    setCellRule(v.rule);
    setDSmooth(v.dSmooth);
    setCellTol(v.tol);
    setCellOpen(v.open);
    setTwoScale(v.twoScale);
    setBroadFactor(v.broad);
    setFocusSlack(v.slack);
    setTheta(undefined);
  };
  const cell = useMemo(() => {
    if (!boundsNow || !g || !cellSeed) return undefined;
    const { width, height, bbox } = g.res;
    const [minX, minY, maxX, maxY] = bbox;
    const col = ((cellSeed[0] - minX) / (maxX - minX)) * width - 0.5;
    const row = ((maxY - cellSeed[1]) / (maxY - minY)) * height - 0.5;
    if (col < -0.5 || row < -0.5 || col > width - 0.5 || row > height - 0.5) return undefined;
    let inside: Uint8Array;
    let dEdge: number | undefined;
    let gated: number | undefined;
    let moved = false;
    let weight: Float32Array | undefined;
    let support: Float32Array | undefined;
    let scales: { broad: number; narrow: number } | undefined;
    if (cellRule !== "ridges") {
      const lap = cellRule === "laplacian";
      ({ mask: inside, dEdge, gated } = laplacianCell(boundsNow.field, width, height, { col, row }, dSmooth, cellTol, lap));
      if (twoScale && boundsBroad && gb) {
        // The same click on the broad raster, then the broad cell at the same σ (in its own r).
        const { width: bw, height: bh, bbox: bb } = gb.res;
        const bcol = ((cellSeed[0] - bb[0]) / (bb[2] - bb[0])) * bw - 0.5;
        const brow = ((bb[3] - cellSeed[1]) / (bb[3] - bb[1])) * bh - 0.5;
        const broad = laplacianCell(boundsBroad.field, bw, bh, { col: bcol, row: brow }, dSmooth, cellTol, lap).mask;
        const share = (m: Uint8Array) => m.reduce((a, v) => a + v, 0) / m.length;
        scales = { broad: share(broad), narrow: share(inside) };
        inside = focusCell(inside, width, height, broad, bw, bh, { col, row }, focusSlack * boundsNow.field.perR);
      }
      const rho = cellOpen * boundsNow.field.perR;
      if (gradedNecks && rho >= 0.5) {
        const start = Math.max(0, Math.min(height - 1, Math.round(row))) * width + Math.max(0, Math.min(width - 1, Math.round(col)));
        ({ weight, support, mask: inside } = gradedCell(inside, width, height, { col, row }, rho));
        moved = !inside[start];
      } else ({ mask: inside, moved } = openCell(inside, width, height, { col, row }, rho));
    } else {
      inside = ridgeCell(ridges(boundsNow.field.field, width, height, thetaNow), width, height, { col, row });
    }
    let count = 0;
    for (const v of inside) count += v;
    return { mask: Float32Array.from(inside), weight, support, seed: cellSeed, count, total: width * height, dEdge, gated, moved, scales };
  }, [boundsNow, g, cellSeed, thetaNow, cellRule, dSmooth, cellTol, cellOpen, gradedNecks, twoScale, boundsBroad, gb, focusSlack]);
  // The persistent mask, on its own grid in the transcripts' frame, and the stroke being painted.
  const [mask, setMask] = useState<WorldMask>();
  const maskNow = useRef<WorldMask>(undefined);
  useEffect(() => {
    maskNow.current = mask;
  }, [mask]);
  const stroke = useRef<Stroke>(undefined);
  /** Space is held: drags pan, and deck still reports them, so they must not also paint. */
  const panning = useRef(false);
  /** The modifier keys held, from the pointer press and from the keyboard: a drag's start event is its
   *  first move, which need not carry them. */
  const pressedWith = useRef({ shift: false, alt: false });
  /** `size`: Cmd or Ctrl really held (from the keyboard, so a trackpad pinch, which the browser
   *  reports as a Ctrl+wheel, still zooms). */
  const keysHeld = useRef({ shift: false, alt: false, size: false });
  /** The same, as state: the cursor shows the op a press would apply. */
  const [heldOp, setHeldOp] = useState<PressOp>("replace");
  useEffect(() => {
    const press = (e: PointerEvent) => {
      pressedWith.current = { shift: e.shiftKey, alt: e.altKey };
    };
    const key = (e: KeyboardEvent) => {
      keysHeld.current = { shift: e.shiftKey, alt: e.altKey, size: e.metaKey || e.ctrlKey };
      setHeldOp(pressOp(e.shiftKey, e.altKey, e.metaKey || e.ctrlKey));
    };
    globalThis.addEventListener("pointerdown", press, { capture: true });
    globalThis.addEventListener("keydown", key, { capture: true });
    globalThis.addEventListener("keyup", key, { capture: true });
    return () => {
      globalThis.removeEventListener("pointerdown", press, { capture: true });
      globalThis.removeEventListener("keydown", key, { capture: true });
      globalThis.removeEventListener("keyup", key, { capture: true });
    };
  }, []);
  /** The op a press applies. A drag's start event is its first move, so the keys held when the button
   *  went down (or on the keyboard) count too. */
  const opNow = (event?: PointerInfo): PressOp => {
    const shift = !!event?.srcEvent?.shiftKey || pressedWith.current.shift || keysHeld.current.shift;
    const alt = !!event?.srcEvent?.altKey || pressedWith.current.alt || keysHeld.current.alt;
    return pressOp(shift, alt, keysHeld.current.size);
  };
  /** The point last hovered or painted, to a quarter map pixel, so a still pointer recomputes nothing. */
  const hovered = useRef<string>(undefined);
  useEffect(() => {
    /** The pointer in the transcripts' frame. */
    const toFrame = (x: number, y: number): readonly [number, number] | undefined => {
      if (!toElement) return undefined;
      const m = toElement;
      return [m.a * x + m.c * y + m.tx, m.b * x + m.d * y + m.ty];
    };
    /** The pointer in the transcripts' frame, or undefined off the map. Not snapped to a map pixel:
     *  the seed's mixture is interpolated there, so the patch follows the pointer smoothly. */
    const pixelAt = (x: number, y: number): readonly [number, number] | undefined => {
      const f = toFrame(x, y);
      if (!wandOn || !f || !g) return undefined;
      const [minX, minY, maxX, maxY] = g.res.bbox;
      if (f[0] < minX || f[0] >= maxX || f[1] <= minY || f[1] > maxY) return undefined;
      return f;
    };
    /** Seed the patch at the pointer; the new seed, or undefined when it has not moved a quarter pixel. */
    const enter = (x: number, y: number, force: boolean): readonly [number, number] | undefined => {
      const p = pixelAt(x, y);
      const q = g ? (4 * g.res.width) / (g.res.bbox[2] - g.res.bbox[0]) : 1;
      const k = p ? `${Math.round(p[0] * q)},${Math.round(p[1] * q)}` : undefined;
      if (!p || (!force && k === hovered.current)) return undefined;
      hovered.current = k;
      const seed: readonly [number, number] = [p[0], p[1]];
      setCellSeed(seed);
      return seed;
    };
    onMapHover.current = (x, y) => {
      if (brushOn) {
        setBrushAt(x === undefined || y === undefined ? undefined : toFrame(x, y));
        return;
      }
      if (stroke.current?.open) return;
      // Off the map, or out of the canvas: no preview.
      if (x === undefined || y === undefined || !pixelAt(x, y)) {
        if (hovered.current === undefined) return;
        hovered.current = undefined;
        setCellSeed(undefined);
        return;
      }
      enter(x, y, false);
    };
    onMapPress.current = (phase, x, y, event) => {
      if (phase === "end") {
        const s = stroke.current;
        if (s) s.open = false;
        if (s && !s.want) stroke.current = undefined;
        return;
      }
      if (x === undefined || y === undefined || !g || panning.current) return;
      if (brushOn) {
        const p = toFrame(x, y);
        if (!p) return;
        setBrushAt(p);
        let s = stroke.current;
        if (phase === "move") {
          if (!s?.open) return;
        } else {
          const base = maskNow.current;
          // The brush's subtract already removes exactly what it touches: "piece" is the same here.
          const pressed = opNow(event);
          const op = pressed === "piece" ? "subtract" : pressed;
          s = { op, base, cell: base?.cell ?? gridFor(g.res), keys: new Set(), open: phase === "start", want: undefined, last: p };
          stroke.current = s;
        }
        // A swept segment from the last position, so a fast drag leaves no gaps.
        sweep(s.keys, s.last ?? p, p, brushR, s.cell);
        s.last = p;
        setMask(apply(s.base, s.keys, s.cell, s.op));
        if (!s.open) stroke.current = undefined;
        return;
      }
      // Cmd/Ctrl+Alt removes the mask's connected pieces the pointer touches, whatever their mixture:
      // a click on a blob deletes the blob.
      const removing = phase === "move" ? !!stroke.current?.piece : opNow(event) === "piece";
      if (removing) {
        const p = pixelAt(x, y);
        let s = stroke.current;
        if (phase === "move") {
          if (!s?.open) return;
        } else {
          const base = maskNow.current;
          s = {
            op: "subtract",
            piece: true,
            base,
            cell: base?.cell ?? gridFor(g.res),
            keys: new Set(),
            open: phase === "start",
            want: undefined,
          };
          stroke.current = s;
        }
        if (p) {
          enter(x, y, phase !== "move");
          for (const k of islandAt(s.base, p)) s.keys.add(k);
          setMask(apply(s.base, s.keys, s.cell, s.op));
        }
        if (!s.open) stroke.current = undefined;
        return;
      }
      if (phase === "move") {
        const s = stroke.current;
        const seed = s?.open ? enter(x, y, false) : undefined;
        if (s && seed) s.want = seed;
        return;
      }
      if (!pixelAt(x, y)) return;
      const pressed = opNow(event);
      const op = pressed === "piece" ? "subtract" : pressed;
      const base = maskNow.current;
      const s: Stroke = { op, base, cell: base?.cell ?? gridFor(g.res), keys: new Set(), open: phase === "start", want: undefined };
      stroke.current = s;
      s.want = enter(x, y, true);
    };
  }, [wandOn, brushOn, brushR, toElement, g]);
  // Each painted patch lands here once computed: into the stroke's union, then the mask is that union
  // combined with the mask as it was when the stroke began. The mask is judged by one metric, so the
  // first stroke locks the modes: otherwise every pan would change what "similar" means.
  useEffect(() => {
    const s = stroke.current;
    if (!s?.want || !cell || cell.seed !== s.want || !g) return;
    s.want = undefined;
    // Subtract removes only what matches the patch and is connected to the sampled point.
    const touched =
      s.op === "subtract" ? matchingPiece(s.base, cell.mask, g.res, cell.seed ?? [0, 0]) : footprint(cell.mask, g.res, s.cell);
    for (const k of touched) s.keys.add(k);
    setMask(apply(s.base, s.keys, s.cell, s.op));
    if (!s.open) stroke.current = undefined;
    if (!lock) setLock({ basis: modeBasis(g.res), modes: g.modes, window: g.window });
  }, [cell, g, lock]);
  // [ and ] shrink and grow the patch, through the scale its edge is judged at.
  useEffect(() => {
    if (!showBounds) return;
    const on = (e: KeyboardEvent) => {
      const t = e.target;
      // Not while typing; a focused checkbox or slider is fine (it is what was last touched).
      const typing = t instanceof HTMLInputElement && !["checkbox", "radio", "range"].includes(t.type);
      if (typing || t instanceof HTMLTextAreaElement) return;
      if (e.key !== "[" && e.key !== "]") return;
      const grow = e.key === "]";
      if (brushOn) setBrushR((r) => sizeBrush(r, grow ? 1.25 : 0.8));
      else if (grow) setDSmooth((s) => Math.min(3, Math.round((s + 0.25) * 4) / 4));
      else setDSmooth((s) => Math.max(0.25, Math.round((s - 0.25) * 4) / 4));
    };
    globalThis.addEventListener("keydown", on);
    return () => globalThis.removeEventListener("keydown", on);
  }, [showBounds, brushOn]);
  // Cmd/Ctrl + scroll sizes the tool (the brush, or the wand's σ) instead of zooming. A native,
  // non-passive capture listener: it runs before deck's and can stop the zoom.
  const mapBox = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = mapBox.current;
    if (!el || active === "hand") return;
    const on = (e: WheelEvent) => {
      if (!keysHeld.current.size) return;
      e.preventDefault();
      e.stopPropagation();
      const f = Math.exp(-e.deltaY * 0.002);
      if (active === "brush") setBrushR((r) => sizeBrush(r, f));
      else setDSmooth((s) => Math.min(3, Math.max(0.25, s * f)));
    };
    el.addEventListener("wheel", on, { capture: true, passive: false });
    return () => el.removeEventListener("wheel", on, { capture: true });
  }, [active]);
  // Space held: drag pans instead of painting, as Photoshop's temporary hand tool.
  const [spaceHeld, setSpaceHeld] = useState(false);
  useEffect(() => {
    if (active === "hand") return;
    const typing = (t: EventTarget | null) =>
      (t instanceof HTMLInputElement && !["checkbox", "radio", "range"].includes(t.type)) || t instanceof HTMLTextAreaElement;
    const down = (e: KeyboardEvent) => {
      if (e.key !== " " || typing(e.target)) return;
      e.preventDefault();
      panning.current = true;
      setSpaceHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key !== " ") return;
      panning.current = false;
      setSpaceHeld(false);
    };
    const blur = () => {
      panning.current = false;
      setSpaceHeld(false);
    };
    globalThis.addEventListener("keydown", down);
    globalThis.addEventListener("keyup", up);
    globalThis.addEventListener("blur", blur);
    return () => {
      globalThis.removeEventListener("keydown", down);
      globalThis.removeEventListener("keyup", up);
      globalThis.removeEventListener("blur", blur);
    };
  }, [active]);
  // H, W and B pick the tool, as in Photoshop.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey || t instanceof HTMLTextAreaElement) return;
      if (t instanceof HTMLInputElement && !["checkbox", "radio", "range"].includes(t.type)) return;
      if (e.key === "h" || e.key === "H") chooseTool("hand");
      else if (e.key === "w" || e.key === "W") chooseTool("wand");
      else if (e.key === "b" || e.key === "B") chooseTool("brush");
    };
    globalThis.addEventListener("keydown", on);
    return () => globalThis.removeEventListener("keydown", on);
  }, [chooseTool]);
  // Until this window's field is ready the wand cannot paint, so a drag still pans rather than doing
  // nothing (which reads as broken); the cursor says it is waiting.
  const wandReady = wandOn && !!boundsNow;
  // The brush needs only a window, for the mask's grid.
  const painting = (wandReady || (brushOn && !!g)) && !spaceHeld;
  const cursor = wandOn && !wandReady ? "progress" : toolCursor(active, heldOp, spaceHeld);
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
  /** A 0/1 raster as drawn: with smooth edges, as a smoothed distance field (no pixel staircase). */
  const drawn = useCallback(
    (m: Float32Array, res: { width: number; height: number }) => (smoothEdges ? smoothOutline(m, res.width, res.height) : m),
    [smoothEdges],
  );
  const maskOnMap = useMemo(() => (mask && g && mask.keys.size > 0 ? drawn(onRaster(mask, g.res), g.res) : null), [mask, g, drawn]);
  /** What a press would paint, on this window's map: the wand's patch, or the brush's disc. One fill for
   *  both, so the two tools preview alike. */
  const preview = useMemo(() => {
    if (!painting || !g) return null;
    // The wand's removals preview what they would take: the whole piece under the pointer, or the part
    // of it that matches the patch and is connected to the pointer.
    const piece =
      wandOn && cellSeed && heldOp === "piece"
        ? islandAt(mask, cellSeed)
        : wandOn && cellSeed && heldOp === "subtract" && cell
          ? matchingPiece(mask, cell.mask, g.res, cellSeed)
          : undefined;
    // The wand's plain patch with graded necks: its whole support, faded by membership (a press
    // paints where that is at least a half).
    const graded = !piece && wandOn && cell?.support && cell.weight ? cell : undefined;
    const m = piece
      ? onRaster({ cell: mask?.cell ?? 1, keys: piece }, g.res)
      : graded
        ? graded.support
        : wandOn
          ? cell?.mask
          : brushOn && brushAt
            ? discOnRaster(brushAt, brushR, g.res)
            : undefined;
    return m ? { cell: drawn(m, g.res), weight: graded?.weight ?? null } : null;
  }, [painting, g, wandOn, heldOp, cellSeed, mask, cell, brushOn, brushAt, brushR, drawn]);

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
                    strength: boundsNow.field.buffer,
                    version: boundsNow.version,
                    bbox: g.res.bbox,
                    width: g.res.width,
                    height: g.res.height,
                    theta: thetaNow,
                    lines: boundLines,
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
      boundLines,
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
          <div className="picker">
            <div className="picker-head">Boundaries (prototype): where the mixture of genes changes fastest</div>
            <label className="check">
              <input type="checkbox" checked={showBounds} onChange={(e) => setShowBounds(e.target.checked)} />
              Show boundaries (the wand, top left of the map, selects by them)
            </label>
            <label>
              Settings
              <select
                value={activePreset ?? "custom"}
                onChange={(e) => {
                  if (e.target.value in BOUNDARY_PRESETS) applyPreset(e.target.value as BoundaryPresetName);
                }}
              >
                {(Object.keys(BOUNDARY_PRESETS) as BoundaryPresetName[]).map((name) => (
                  <option key={name} value={name}>
                    {BOUNDARY_PRESETS[name].label}
                  </option>
                ))}
                {!activePreset && <option value="custom">Custom (fine-tuned below)</option>}
              </select>
            </label>
            <p className="hint">{activePreset ? BOUNDARY_PRESETS[activePreset].hint : "Your own settings: pick a preset to go back."}</p>
            <p className="hint">
              Pick a tool on the map. Wand (W): hover to preview the patch (cyan), click or drag to paint. Brush (B): drag to paint a disc.
              Shift: add · Alt: subtract what matches · Cmd/Ctrl+Alt: remove the piece · Shift+Alt: intersect · Space: pan · H: back to the
              hand · Cmd/Ctrl+scroll or <kbd>[</kbd> <kbd>]</kbd>: size (wand σ {dSmooth.toFixed(2)}r, brush radius {Math.round(brushR)}{" "}
              µm).
              {mask && mask.keys.size > 0 ? ` Mask: ${(maskArea(mask) / 1e6).toFixed(3)} mm² (amber).` : ""}
            </p>
            {mask && mask.keys.size > 0 && (
              <button type="button" onClick={() => setMask(undefined)}>
                clear the mask
              </button>
            )}
            <details>
              <summary>Fine-tune</summary>
              <label className="check">
                <input type="checkbox" checked={gradedNecks} onChange={(e) => setGradedNecks(e.target.checked)} />
                Graded necks (a part joined by a neck near the opening's width fades with the neck; a press paints what is at least half
                shown. Off: the binary opening, whose parts appear and vanish whole)
              </label>
              <label className="check">
                <input type="checkbox" checked={smoothEdges} onChange={(e) => setSmoothEdges(e.target.checked)} />
                Smooth edges (outlines from a smoothed distance field, quadratic filter, antialiased; off: the pixel staircase, bilinear, as
                before)
              </label>
              <label className="check">
                <input type="checkbox" checked={boundLines} onChange={(e) => setBoundLines(e.target.checked)} />
                As thin lines (off: every place above θ)
              </label>
              <label>
                Bound the clicked cell by
                <select
                  value={cellRule}
                  onChange={(e) =>
                    setCellRule(e.target.value === "ridges" ? "ridges" : e.target.value === "threshold" ? "threshold" : "laplacian")
                  }
                >
                  <option value="laplacian">where the difference from the click stops speeding up (∇²d = 0)</option>
                  <option value="threshold">a plain tolerance on the difference (d ≤ tol)</option>
                  <option value="ridges">the boundary lines above θ</option>
                </select>
              </label>
              <label>
                Smooth the difference first (σ, in r): {dSmooth.toFixed(2)}
                <input type="range" min={0} max={3} step={0.1} value={dSmooth} onChange={(e) => setDSmooth(Number(e.target.value))} />
              </label>
              <label>
                …and no further than this difference (tolerance): {cellTol > 0 ? cellTol.toFixed(2) : "off"}
                <input type="range" min={0} max={6} step={0.1} value={cellTol} onChange={(e) => setCellTol(Number(e.target.value))} />
              </label>
              <label>
                Drop parts narrower than (in r): {cellOpen > 0 ? cellOpen.toFixed(2) : "off"}
                <input type="range" min={0} max={4} step={0.1} value={cellOpen} onChange={(e) => setCellOpen(Number(e.target.value))} />
              </label>
              <label className="check">
                <input type="checkbox" checked={twoScale} onChange={(e) => setTwoScale(e.target.checked)} />
                Judge the patch at a broader neighbourhood too
                {twoScale && (gb ? ` (r ${gb.radius} µm*, map ${gb.raster.width}×${gb.raster.height})` : " (computing…)")}
              </label>
              {twoScale && (
                <>
                  <label>
                    Broader neighbourhood: {broadFactor.toFixed(1)}× r
                    <input
                      type="range"
                      min={1.5}
                      max={6}
                      step={0.5}
                      value={broadFactor}
                      onChange={(e) => setBroadFactor(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    How far the edge may sit outside it (in r): {focusSlack.toFixed(2)}
                    <input
                      type="range"
                      min={0}
                      max={4}
                      step={0.25}
                      value={focusSlack}
                      onChange={(e) => setFocusSlack(Number(e.target.value))}
                    />
                  </label>
                </>
              )}
              <label>
                Main modes (q): {Math.min(mainModes, g?.res.labels.length ?? mainModes)}
                {g && mainModes >= g.res.labels.length ? " = every mode" : ""}
                <input
                  type="range"
                  min={1}
                  max={g?.res.labels.length ?? 32}
                  step={1}
                  value={mainModes}
                  onChange={(e) => setMainModes(Number(e.target.value))}
                />
              </label>
              <label>
                Count what the main modes leave out (w): {residualWeight.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={residualWeight}
                  onChange={(e) => setResidualWeight(Number(e.target.value))}
                />
              </label>
              <label>
                Smooth what they leave out first (σ, in r): {residualSmooth.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={3}
                  step={0.1}
                  value={residualSmooth}
                  onChange={(e) => setResidualSmooth(Number(e.target.value))}
                />
              </label>
              <label>
                Boundary strength (θ): {thetaNow.toPrecision(3)}
                {theta === undefined ? " (this window's 90th percentile)" : ""}
                <input
                  type="range"
                  min={0}
                  max={boundsNow ? boundsNow.field.quantiles.max : 10}
                  step={(boundsNow?.field.quantiles.max ?? 10) / 200}
                  value={thetaNow}
                  onChange={(e) => setTheta(Number(e.target.value))}
                />
              </label>
              {theta !== undefined && (
                <button type="button" onClick={() => setTheta(undefined)}>
                  follow the 90th percentile
                </button>
              )}
            </details>
            {boundsNow && (
              <p className="hint">
                Strength per radius r: median {boundsNow.field.quantiles.p50.toPrecision(3)} · 90%{" "}
                {boundsNow.field.quantiles.p90.toPrecision(3)} · 99% {boundsNow.field.quantiles.p99.toPrecision(3)} · max{" "}
                {boundsNow.field.quantiles.max.toPrecision(3)}
                {cell
                  ? ` · cell: ${cell.count.toLocaleString()} of ${cell.total.toLocaleString()} map pixels (${((100 * cell.count) / cell.total).toFixed(1)}%)${cell.dEdge !== undefined ? `, d at its edge ≈ ${cell.dEdge.toFixed(2)}${cell.gated ? `, ${(100 * cell.gated).toFixed(0)}% of the edge set by the tolerance` : ""}${cell.moved ? ", the pointer is on a branch narrower than the opening: nothing to pick" : ""}${cell.scales ? ` (broad cell ${(100 * cell.scales.broad).toFixed(1)}%, narrow ${(100 * cell.scales.narrow).toFixed(1)}%)` : ""}` : ""}`
                  : ""}
                {boundsNow.q < (g?.res.labels.length ?? 0)
                  ? ` · noise floor σ² ${boundsNow.sigma2.toPrecision(3)} (λ₁ ${modes?.values[0]?.toPrecision(3)})`
                  : ""}
              </p>
            )}
          </div>
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
            ref={mapBox}
            data-tool={painting ? "paint" : "hand"}
            style={{ "--tool-cursor": cursor } as CSSProperties}
            onPointerLeave={() => onMapHover.current?.()}
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
            <Explain
              title="What do the boundaries show? (prototype)"
              math={String.raw`u = A z,\quad A_k = \begin{cases} v_k^\top/\sqrt{\lambda_k} & k \le q \\ \sqrt{w}\, v_k^\top/\sigma & k > q \end{cases},\quad \sigma^2 = \overline{\lambda_{k>q}},\qquad \text{strength} = r\,\sqrt{\lambda_{\max}\!\big(\textstyle\sum_k \nabla u_k \nabla u_k^\top\big)}`}
            >
              How fast the mixture of genes changes at each place, in the direction it changes fastest, measured the way the wand measures
              difference. 1 means "one typical spread of difference across one radius". The <b>main modes</b> count in their own spread. The
              rest are counted at the noise level, scaled by <b>w</b>: at 0 they are ignored (only what the colours show), at 1 they count
              as much as noise does. Setting q to every mode is the old "full Mahalanobis" metric, where the faintest modes count most.
              Amber marks places at or above θ, where a wand patch bounded by edges would stop. White shows weaker change.
            </Explain>
            <Explain title="What is the quality filter?">
              Every molecule has a quality score, qv. At 20, the chance that it was read as the wrong gene is about 1 in 100. Molecules
              below the threshold are dropped, as in Xenium's own per-cell counts. Points stored without a tiled layout are read whole, and
              that read brings only positions and genes, so for them the filter is off.
            </Explain>
          </section>
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
