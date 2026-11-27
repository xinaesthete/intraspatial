// PROTOTYPE: SIMBA's patch grid (R4, `src/spatial/patchGrid.ts`) laid live over a support mask, to
// see what changes at smaller patch areas. The mask's runs are built once per mask and threshold;
// every slider step re-runs the layout and the support rule on the host (a few ms here, 16 ms on a
// 62 Mpx canvas) and redraws on the page's WebGPU device: the footprint as a bit-packed tint, every
// candidate faint, kept patches outlined, kept edge patches dashed.

import { OrthographicView, type OrthographicViewState } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type PixelBox,
  componentBoxes,
  type MaskRuns,
  maskRuns,
  type PatchGrid,
  patchGrid,
  runPixelCount,
  toPixels,
} from "../../../src/spatial/patchGrid";
import { GLOBAL_DEVICE_PROPS, useGlobalDevice } from "../deck/useGlobalDevice";
import { useElementSize } from "../hooks/useElementSize";
import { MathTex } from "../Math";
import { decodeMask, fetchMask, type MaskImage, syntheticMask } from "./loadMask";
import { MaskLayer, packRuns } from "./MaskLayer";
import { EDGE, KEPT, RectLayer } from "./RectLayer";

const VIEW = new OrthographicView({ id: "map" });

/** Past this many candidates a layout is refused: the page would stall drawing them. */
const MAX_CANDIDATES = 2_000_000;

/** An upper bound on the candidates a size and stride give, from the pieces' boxes alone. */
function candidateBound(boxes: readonly PixelBox[], sizePx: number, stridePx: number): number {
  const along = (span: number) => (span <= sizePx ? 1 : Math.floor((span - sizePx) / stridePx) + 2);
  return boxes.reduce((n, b) => n + along(b.x1 - b.x0) * along(b.y1 - b.y0), 0);
}

const now = () => performance.now();

interface Layout {
  readonly grid: PatchGrid;
  readonly ms: number;
  readonly rects: Float32Array;
  readonly flags: Uint32Array;
  readonly edgeKept: number;
}

function layout(runs: MaskRuns, size: number, stride: number, pixelSize: number, minFraction: number): Layout {
  const t0 = now();
  const grid = patchGrid({
    footprint: runs,
    size,
    stride,
    units: "um",
    pixelSize,
    rules: [{ kind: "fraction", name: "support fraction", mask: runs, min: minFraction }],
  });
  const ms = now() - t0;
  const n = grid.candidates.length;
  const rects = new Float32Array(n * 4);
  const flags = new Uint32Array(n);
  let edgeKept = 0;
  grid.candidates.forEach((p, i) => {
    rects.set([p.x0, p.y0, p.x1, p.y1], i * 4);
    flags[i] = (p.kept ? KEPT : 0) | (p.edge ? EDGE : 0);
    if (p.kept && p.edge) edgeKept++;
  });
  return { grid, ms, rects, flags, edgeKept };
}

interface SweepPoint {
  readonly size: number;
  readonly candidates: number;
  readonly kept: number;
}

/** Kept and candidate counts over a range of sizes, a few per frame so the page stays live. */
function useSweep(
  runs: MaskRuns | null,
  boxes: readonly PixelBox[],
  sizes: readonly number[],
  strideFor: (size: number) => number,
  pixelSize: number,
  minFraction: number,
): SweepPoint[] {
  // Tagged with the inputs they came from, so a stale sweep reads as empty without a reset.
  const inputs = useMemo(
    () => ({ runs, boxes, sizes, strideFor, pixelSize, minFraction }),
    [runs, boxes, sizes, strideFor, pixelSize, minFraction],
  );
  const [points, setPoints] = useState<{ inputs: object | null; list: SweepPoint[] }>({ inputs: null, list: [] });
  useEffect(() => {
    const { runs, boxes, sizes, strideFor, pixelSize, minFraction } = inputs;
    if (!runs) return;
    let i = 0;
    let timer = 0;
    const out: SweepPoint[] = [];
    const step = () => {
      const t0 = now();
      while (i < sizes.length && now() - t0 < 12) {
        const size = sizes[i++]!;
        const stride = strideFor(size);
        if (candidateBound(boxes, toPixels(size, "um", pixelSize), toPixels(stride, "um", pixelSize)) > MAX_CANDIDATES) continue;
        const g = patchGrid({
          footprint: runs,
          size,
          stride,
          units: "um",
          pixelSize,
          rules: [{ kind: "fraction", name: "f", mask: runs, min: minFraction }],
        });
        out.push({ size, candidates: g.candidates.length, kept: g.patches.length });
      }
      setPoints({ inputs, list: [...out] });
      if (i < sizes.length) timer = window.setTimeout(step, 0);
    };
    timer = window.setTimeout(step, 0);
    return () => window.clearTimeout(timer);
  }, [inputs]);
  return points.inputs === inputs ? points.list : [];
}

/** Counts against patch size, log y, with the current size marked. */
function SweepChart({
  points,
  current,
  minSize,
  maxSize,
}: {
  points: readonly SweepPoint[];
  current: number;
  minSize: number;
  maxSize: number;
}) {
  const W = 300;
  const H = 150;
  const pad = { l: 40, r: 8, t: 22, b: 24 };
  const top = Math.max(10, ...points.map((p) => p.candidates));
  const lyHi = Math.ceil(Math.log10(top));
  // Log on both axes: the counts change most at the small sizes.
  const px = (x: number) => pad.l + (Math.log(x / minSize) / Math.max(1e-9, Math.log(maxSize / minSize))) * (W - pad.l - pad.r);
  const py = (y: number) => H - pad.b - (Math.log10(Math.max(1, y)) / lyHi) * (H - pad.t - pad.b);
  const line = (key: "candidates" | "kept") => points.map((p) => `${px(p.size)},${py(p[key])}`).join(" ");
  const decades = Array.from({ length: lyHi + 1 }, (_, k) => k);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="patches against patch size">
      <text x={pad.l} y={12} className="chart-title">
        patches vs patch size (log–log)
      </text>
      {decades.map((k) => (
        <g key={k}>
          <line x1={pad.l} x2={W - pad.r} y1={py(10 ** k)} y2={py(10 ** k)} className="chart-guide" />
          <text x={pad.l - 4} y={py(10 ** k) + 3} className="chart-tick" textAnchor="end">
            {k === 0 ? "1" : `1e${k}`}
          </text>
        </g>
      ))}
      <line x1={px(current)} x2={px(current)} y1={pad.t} y2={H - pad.b} className="chart-marker" />
      <polyline fill="none" stroke="#94a3b8" strokeWidth={1.5} points={line("candidates")} />
      <polyline fill="none" stroke="#fbbf24" strokeWidth={1.5} points={line("kept")} />
      <text x={pad.l} y={H - 6} className="chart-tick">
        {minSize.toFixed(0)} µm
      </text>
      <text x={W - pad.r} y={H - 6} className="chart-tick" textAnchor="end">
        {maxSize.toFixed(0)} µm
      </text>
      <text x={W - pad.r} y={pad.t + 8} className="chart-tick" textAnchor="end">
        <tspan fill="#94a3b8">candidates</tspan> <tspan fill="#fbbf24">kept</tspan>
      </text>
    </svg>
  );
}

function Explainers() {
  return (
    <div className="explainers">
      <details className="explain" open>
        <summary>One grid per piece of tissue</summary>
        <div className="explain-body">
          The footprint is split into separate pieces: pixels that touch by a side belong together. Each piece gets its own grid, starting
          at the top-left corner of the box around it. There is no single lattice over the whole slide, so two fragments' patches don't line
          up. Along each axis, a piece spanning <MathTex display={false} tex="[\ell, h)" /> gets starts every stride{" "}
          <MathTex display={false} tex="t" /> from its near edge, and one more patch ending exactly at its far edge if the stride missed it.
          That last one is an <b>edge patch</b> (dashed cyan).
          <div className="explain-math">
            <MathTex tex="s_k = \ell + k\,t,\quad k = 0, 1, \dots, \left\lfloor \tfrac{h - \ell - S}{t} \right\rfloor, \qquad s_\text{edge} = h - S" />
          </div>
          A piece no wider than a patch gets a single patch, ending at its far edge.
        </div>
      </details>
      <details className="explain">
        <summary>Which patches are kept</summary>
        <div className="explain-body">
          A patch is kept when at least the minimum fraction of its pixels are support. Every candidate is a full square, so a patch hanging
          over the edge of the tissue is measured over its whole area and needs enough tissue inside it.
          <div className="explain-math">
            <MathTex tex="f(P) = \frac{|M \cap P|}{|P|} \ \ge\ f_\text{min}" />
          </div>
          SIMBA keeps a patch at <MathTex display={false} tex="f \ge 0.5" /> (half tissue). Support is counted exactly from the mask's row
          runs, not sampled.
        </div>
      </details>
      <details className="explain">
        <summary>What changes as patches get smaller</summary>
        <div className="explain-body">
          The count grows with the inverse square of the stride: a piece <MathTex display={false} tex="W \times H" /> holds about
          <div className="explain-math">
            <MathTex tex="\left(\tfrac{W - S}{t} + 1\right)\left(\tfrac{H - S}{t} + 1\right)" />
          </div>
          candidates, so halving the size (stride half the size) gives about four times as many. Each pixel inside the tissue lies in{" "}
          <MathTex display={false} tex="(S/t)^2" /> patches, four at half stride, so neighbouring patches share most of their pixels and
          their medians are not independent. Smaller patches follow the tissue's outline more closely: narrow strips and small fragments
          that a 300 µm patch can't fill at half support start to pass.
        </div>
      </details>
      <details className="explain">
        <summary>Units</summary>
        <div className="explain-body">
          Sizes are in µm and rounded to whole pixels:
          <MathTex display={false} tex="S_\text{px} = \operatorname{round}(S_\mu / \text{pixel size})" />. SIMBA works at 1 µm/px. The
          1/8-resolution mock rasters are 8 µm/px, so 300 µm is 38 px (37.5 rounded), and the grid differs slightly from the full-resolution
          one.
        </div>
      </details>
    </div>
  );
}

/** Shared, so an empty readout keeps one identity between renders. */
const NO_HITS: readonly Hit[] = [];

interface Hit {
  readonly index: number;
  readonly id: number;
  readonly box: PixelBox;
  readonly fraction: number;
  readonly kept: boolean;
  readonly edge: boolean;
}

export function PatchGridDemo() {
  const gpu = useGlobalDevice();
  const [image, setImage] = useState<MaskImage>(() => syntheticMask());
  const [pixelSize, setPixelSize] = useState(4);
  const [threshold, setThreshold] = useState(0);
  const [size, setSize] = useState(300);
  const [linked, setLinked] = useState(true);
  const [strideFixed, setStrideFixed] = useState(150);
  const [minFraction, setMinFraction] = useState(0.5);
  const [showCandidates, setShowCandidates] = useState(true);
  const [showKept, setShowKept] = useState(true);
  const [url, setUrl] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [picked, setPicked] = useState<{ grid: Layout | null; hits: readonly Hit[] }>({ grid: null, hits: [] });

  const stride = linked ? size / 2 : Math.min(strideFixed, size * 4);

  // Built once per mask and threshold: every slider step reuses the runs.
  const mask = useMemo(() => {
    const t0 = now();
    const runs = maskRuns(image.values, image.width, image.height, (v) => v > threshold);
    const ms = now() - t0;
    const boxes = componentBoxes(runs);
    return { runs, ms, boxes, pixels: runPixelCount(runs), bits: packRuns(runs) };
  }, [image, threshold]);

  const sizePx = Math.max(1, Math.round(size / pixelSize));
  const stridePx = Math.max(1, Math.round(stride / pixelSize));
  const bound = candidateBound(mask.boxes, sizePx, stridePx);
  const result = useMemo((): Layout | string => {
    if (bound > MAX_CANDIDATES) return `about ${bound.toLocaleString()} candidates: too many to draw; make the patch or stride larger.`;
    try {
      return layout(mask.runs, size, stride, pixelSize, minFraction);
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, [mask, size, stride, pixelSize, minFraction, bound]);
  const grid = typeof result === "string" ? null : result;

  // Slider range: four pixels up to 1 mm or the canvas's short side.
  const minSize = 4 * pixelSize;
  const maxSize = Math.max(minSize + pixelSize, Math.min(1000, Math.min(image.width, image.height) * pixelSize));
  const sweepSizes = useMemo(() => {
    const n = 48;
    return Array.from({ length: n }, (_, k) => minSize * (maxSize / minSize) ** (k / (n - 1)));
  }, [minSize, maxSize]);
  const strideFor = useCallback((s: number) => (linked ? s / 2 : Math.min(strideFixed, s * 4)), [linked, strideFixed]);
  const sweep = useSweep(mask.runs, mask.boxes, sweepSizes, strideFor, pixelSize, minFraction);

  // Fitted to each new mask, to a resized canvas, and on demand; panning and zooming move it from
  // that fit. Controlled rather than deck's `initialViewState`, which deck compares deeply: a second
  // fit to the same mask would not move the view.
  const box = useRef<HTMLDivElement>(null);
  const view = useElementSize(box);
  const [fits, setFits] = useState(0);
  const fitted = useMemo((): OrthographicViewState | null => {
    void fits; // a new fit per press of "fit view"
    if (view.width <= 0 || view.height <= 0) return null;
    const zoom = Math.log2(0.95 * Math.min(view.width / image.width, view.height / image.height));
    return { target: [image.width / 2, image.height / 2, 0], zoom, maxZoom: 6, minZoom: zoom - 3 };
  }, [image, view.width, view.height, fits]);
  const [moved, setMoved] = useState<{ from: OrthographicViewState | null; view: OrthographicViewState } | null>(null);
  const viewState = moved?.from === fitted ? moved.view : fitted;

  // Patches under the last click, on the grid they were read from: a new grid clears them.
  const hits = picked.grid === grid ? picked.hits : NO_HITS;

  const layers = useMemo(
    () => [
      new MaskLayer({ id: "footprint", bits: mask.bits, width: image.width, height: image.height, colour: [0.32, 0.55, 0.62, 0.6] }),
      new RectLayer({
        id: "patches",
        rects: grid?.rects ?? null,
        flags: grid?.flags ?? null,
        showCandidates,
        showKept,
        selected: hits[0]?.index ?? -1,
      }),
    ],
    [mask, image, grid, showCandidates, showKept, hits],
  );

  const onClick = useCallback(
    (info: { coordinate?: number[] }) => {
      const c = info.coordinate;
      if (!c || !grid) return;
      const [x, y] = c as [number, number];
      const found: Hit[] = [];
      grid.grid.candidates.forEach((p, index) => {
        if (x >= p.x0 && x < p.x1 && y >= p.y0 && y < p.y1)
          found.push({ index, id: p.id, box: p, fraction: p.support[0]!, kept: p.kept, edge: p.edge });
      });
      found.sort((a, b) => Number(b.kept) - Number(a.kept));
      setPicked({ grid, hits: found });
    },
    [grid],
  );

  const load = async (p: Promise<MaskImage>) => {
    setLoadError(null);
    try {
      setImage(await p);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  };
  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void load(decodeMask(file, file.name));
  };

  const um = (px: number) => `${(px * pixelSize).toFixed(0)} µm`;
  const areaMm2 = (mask.pixels * pixelSize * pixelSize) / 1e6;

  return (
    <>
      <h2 className="page">
        Patch grid <span className="badge">prototype</span>
      </h2>
      <p className="page">
        SIMBA tiles each specimen with overlapping square patches (300 µm, stride 150 µm), keeps those that are at least half tissue, and
        correlates per-patch medians. This page lays the same grid (<code>patchGrid</code>, which matches SIMBA's patch tables exactly) over
        a support mask so you can see how it changes with the patch size. Click a patch to read its support.
      </p>
      <div className="layout">
        <div className="controls">
          <h3>Mask</h3>
          <div className="row">
            <button type="button" onClick={() => void load(Promise.resolve(syntheticMask())).then(() => setPixelSize(4))}>
              made-up section
            </button>
            <label className="file">
              upload PNG
              <input type="file" accept="image/png,image/*" onChange={onFile} />
            </label>
          </div>
          <div className="row">
            <input type="text" placeholder="or a PNG url" value={url} onChange={(e) => setUrl(e.target.value)} />
            <button type="button" disabled={!url} onClick={() => void load(fetchMask(url))}>
              load
            </button>
          </div>
          {loadError && <p className="error">{loadError}</p>}
          <p className="hint">
            Any non-zero pixel is support. The Leap034 mocks (<code>collagen.png</code>, <code>cu.png</code>) are 1/8 resolution: set 8
            µm/px.
          </p>
          <label>
            pixel size (µm/px)
            <input
              type="number"
              min={0.05}
              step={0.05}
              value={pixelSize}
              onChange={(e) => setPixelSize(Math.max(0.05, Number(e.target.value) || 1))}
            />
          </label>
          <label>
            support = value &gt; {threshold}
            <input type="range" min={0} max={254} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          </label>

          <h3>Grid</h3>
          <label>
            patch size: {size.toFixed(0)} µm ({sizePx} px)
            <input
              type="range"
              min={minSize}
              max={maxSize}
              step={pixelSize}
              value={Math.min(Math.max(size, minSize), maxSize)}
              onChange={(e) => setSize(Number(e.target.value))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} />
            stride = half the size
          </label>
          <label className={linked ? "disabled" : undefined}>
            stride: {stride.toFixed(0)} µm ({stridePx} px)
            <input
              type="range"
              disabled={linked}
              min={pixelSize}
              max={maxSize}
              step={pixelSize}
              value={linked ? stride : strideFixed}
              onChange={(e) => setStrideFixed(Number(e.target.value))}
            />
          </label>
          <label>
            minimum support fraction: {minFraction.toFixed(2)}
            <input type="range" min={0} max={1} step={0.05} value={minFraction} onChange={(e) => setMinFraction(Number(e.target.value))} />
          </label>
          <div className="row">
            <label className="check">
              <input type="checkbox" checked={showCandidates} onChange={(e) => setShowCandidates(e.target.checked)} />
              candidates
            </label>
            <label className="check">
              <input type="checkbox" checked={showKept} onChange={(e) => setShowKept(e.target.checked)} />
              kept
            </label>
          </div>

          <table className="stats">
            <tbody>
              <tr>
                <th>mask</th>
                <td>
                  {image.name}, {image.width.toLocaleString()} × {image.height.toLocaleString()} px
                </td>
              </tr>
              <tr>
                <th>support</th>
                <td>
                  {mask.pixels.toLocaleString()} px ({areaMm2.toFixed(2)} mm²), {mask.boxes.length} piece
                  {mask.boxes.length === 1 ? "" : "s"}
                </td>
              </tr>
              <tr>
                <th>candidates</th>
                <td>{grid ? grid.grid.candidates.length.toLocaleString() : "–"}</td>
              </tr>
              <tr>
                <th>kept</th>
                <td className="kept">
                  {grid
                    ? `${grid.grid.patches.length.toLocaleString()} (${grid.edgeKept.toLocaleString()} edge) · ${(
                        (100 * grid.grid.patches.length) /
                        Math.max(1, grid.grid.candidates.length)
                      ).toFixed(0)}%`
                    : "–"}
                </td>
              </tr>
              <tr>
                <th>time</th>
                <td>
                  runs {mask.ms.toFixed(1)} ms (once per mask) · layout {grid ? `${grid.ms.toFixed(1)} ms` : "–"}
                </td>
              </tr>
            </tbody>
          </table>
          {typeof result === "string" && <p className="error">{result}</p>}
          <SweepChart points={sweep} current={size} minSize={minSize} maxSize={maxSize} />
          <p className="hint">
            The same rule at {sweepSizes.length} sizes from {minSize.toFixed(0)} to {maxSize.toFixed(0)} µm,{" "}
            {linked ? "stride half the size" : `stride ${strideFixed.toFixed(0)} µm`}.
          </p>
        </div>
        <div className="views">
          <div className="legend">
            <span className="legend-row">
              <span className="swatch" style={{ background: "rgba(82,140,158,0.6)" }} /> support
            </span>
            <span className="legend-row">
              <span className="swatch" style={{ background: "rgba(217,230,255,0.08)", borderColor: "rgba(217,230,255,0.35)" }} />
              candidate
            </span>
            <span className="legend-row">
              <span className="swatch" style={{ borderColor: "#fbbf24", borderWidth: 2 }} /> kept
            </span>
            <span className="legend-row">
              <span className="swatch" style={{ borderColor: "#40e6ff", borderWidth: 2, borderStyle: "dashed" }} /> kept edge patch
            </span>
            <button type="button" onClick={() => setFits((n) => n + 1)}>
              fit view
            </button>
          </div>
          <div ref={box} className="deck-canvas">
            {viewState && (
              <DeckGL
                views={VIEW}
                viewState={viewState}
                onViewStateChange={(e) => setMoved({ from: fitted, view: e.viewState as OrthographicViewState })}
                controller
                layers={layers}
                deviceProps={GLOBAL_DEVICE_PROPS}
                onClick={onClick}
              />
            )}
          </div>
          <div className="status">{gpu.status}</div>
          {hits.length > 0 && (
            <table className="hits">
              <thead>
                <tr>
                  <th>patch</th>
                  <th>bounds (px, half-open)</th>
                  <th>support</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {hits.slice(0, 8).map((h) => (
                  <tr key={h.index} className={h.kept ? "kept" : undefined}>
                    <td>patch_{String(h.id).padStart(5, "0")}</td>
                    <td>
                      x {h.box.x0}–{h.box.x1}, y {h.box.y0}–{h.box.y1} ({um(h.box.x0)}, {um(h.box.y0)})
                    </td>
                    <td>{h.fraction.toFixed(3)}</td>
                    <td>
                      {h.kept ? "kept" : "dropped"}
                      {h.edge ? " · edge" : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <Explainers />
        </div>
      </div>
    </>
  );
}
