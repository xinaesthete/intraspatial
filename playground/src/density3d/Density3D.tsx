// A 3D density of measured transcripts, drawn as surfaces: `splatVolume` over a window of a
// SpatialData points element, one field per chosen gene, each drawn by `IsoSurfaceLayer`.
//
// The molecules of the window are put on the device once. Each gene's field is the same points with
// a weight per point (1 for that gene's molecules, 0 for the rest), which is also how a density of
// cell expression would be asked for: cell centres, weighted by a gene's expression.

import { type Layer, OrbitView, type OrbitViewState } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { type Color, IsoSurfaceLayer } from "@intraspatial/deck";
import { useCallback, useMemo, useState } from "react";
import { type PointsTile, pointFraction, type Rect } from "../../../src/datasource/points";
import { listPointsElements } from "../datasource/pointsTileLoader";
import { GLOBAL_DEVICE_PROPS, useGlobalDevice } from "../deck/useGlobalDevice";
import { useSettled } from "../hooks/useAsync";
import { usePointsSource, usePointsTiles, useSpatialData } from "../hooks/useSpatialData";
import { StoreInput } from "../modes/PageControls";
import { Explain } from "../transcript/Explain";
import { type FieldBox, useDensityFields, useGpuBuffer } from "./useVolumeDensity";

const DEFAULT_STORE =
  new URLSearchParams(location.search).get("url")?.trim() || "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const SLOTS = 3;
const SLOT_COLOURS: readonly Color[] = [
  [56, 189, 248, 255],
  [251, 146, 60, 255],
  [163, 230, 53, 255],
];
/** Category of a molecule that belongs to no chosen gene. */
const HIDDEN = SLOTS;
/** Probes that are not genes: left out of the starting choice, though they can still be picked. */
const CONTROL = /^(blank|negcontrol|unassigned|deprecated|intergenic|genomiccontrol)/i;
/** Depth for the third dimension; quality to filter on, where the element has it. */
const POINTS_OPTIONS = { columns: ["qv"], depth: true };
const VIEW = new OrbitView({ id: "orbit", orbitAxis: "Z", fovy: 35 });

interface Cloud {
  readonly n: number;
  /** x, y, z, 0 per molecule: centred on the window, y flipped to match the flat views, z stretched. */
  readonly positions: Float32Array;
  readonly codes: Int32Array;
  /** How many molecules each one stands for: 1, or more where a tile was read from a sample. */
  readonly scale: Float32Array;
  readonly window: Rect;
  /** The molecules' depth range, unstretched. 0 when the element has no depth. */
  readonly depth: number;
  readonly hasDepth: boolean;
  /** Molecules per feature code in the window. */
  readonly counts: ReadonlyMap<number, number>;
}

/** The molecules of `tiles` that are inside `window` and pass the quality filter. */
function gather(tiles: readonly PointsTile[], window: Rect, qvMin: number, stretch: number): Cloud {
  const keep = (t: PointsTile, i: number) => {
    const x = t.xs[i]!;
    const y = t.ys[i]!;
    const qv = t.columns.qv;
    return x >= window.minX && x < window.maxX && y >= window.minY && y < window.maxY && (!qv || qv[i]! >= qvMin);
  };
  let n = 0;
  let zLo = Number.POSITIVE_INFINITY;
  let zHi = Number.NEGATIVE_INFINITY;
  let hasDepth = true;
  for (const t of tiles) {
    const zs = t.columns.z;
    if (!zs) hasDepth = false;
    for (let i = 0; i < t.count; i++) {
      if (!keep(t, i)) continue;
      n++;
      const z = zs?.[i] ?? 0;
      zLo = Math.min(zLo, z);
      zHi = Math.max(zHi, z);
    }
  }
  const positions = new Float32Array(n * 4);
  const codes = new Int32Array(n);
  const scale = new Float32Array(n);
  const counts = new Map<number, number>();
  const cx = (window.minX + window.maxX) / 2;
  const cy = (window.minY + window.maxY) / 2;
  const cz = n ? (zLo + zHi) / 2 : 0;
  let at = 0;
  for (const t of tiles) {
    const zs = hasDepth ? t.columns.z : undefined;
    for (let i = 0; i < t.count; i++) {
      if (!keep(t, i)) continue;
      const code = t.codes[i]!;
      const stands = 1 / pointFraction(t, code);
      positions.set([t.xs[i]! - cx, cy - t.ys[i]!, ((zs?.[i] ?? cz) - cz) * stretch, 0], at * 4);
      codes[at] = code;
      scale[at] = stands;
      counts.set(code, (counts.get(code) ?? 0) + stands);
      at++;
    }
  }
  return { n, positions, codes, scale, window, depth: n && hasDepth ? zHi - zLo : 0, hasDepth, counts };
}

/** A window `side` across, centred at the fractions `fx`, `fy` of `extent` and kept inside it. */
function windowAt(extent: Rect, side: number, fx: number, fy: number): Rect {
  const w = Math.min(side, extent.maxX - extent.minX);
  const h = Math.min(side, extent.maxY - extent.minY);
  const minX = extent.minX + (extent.maxX - extent.minX - w) * fx;
  const minY = extent.minY + (extent.maxY - extent.minY - h) * fy;
  return { minX, minY, maxX: minX + w, maxY: minY + h };
}

/** Each molecule's weight in one slot's field: what it stands for if it is that slot's gene, else 0.
 *  Null when no molecule is. */
function slotWeights(cloud: Cloud | null, categories: Uint32Array | null, slot: number): Float32Array | null {
  if (!cloud || !categories) return null;
  const w = new Float32Array(cloud.n);
  let any = false;
  for (let i = 0; i < cloud.n; i++) {
    if (categories[i] !== slot) continue;
    w[i] = cloud.scale[i]!;
    any = true;
  }
  return any ? w : null;
}

const css = (c: Color) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

function Range(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  /** A log slider: `value` runs from `min` to `max` in equal ratios. */
  log?: boolean;
  onChange: (v: number) => void;
}) {
  const to = props.log ? Math.log : (v: number) => v;
  const from = props.log ? Math.exp : (v: number) => v;
  return (
    <label>
      {props.label}
      <input
        type="range"
        min={to(props.min)}
        max={to(props.max)}
        step={props.log ? 0.01 : props.step}
        value={to(props.value)}
        onChange={(e) => props.onChange(from(Number(e.target.value)))}
      />
    </label>
  );
}

export function Density3D() {
  const { device, status } = useGlobalDevice();
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [pointsName, setPointsName] = useState<string>();
  const [side, setSide] = useState(400);
  const [across, setAcross] = useState(0.5);
  const [down, setDown] = useState(0.5);
  const [qvMin, setQvMin] = useState(20);
  const [stretch, setStretch] = useState(4);
  const [radius, setRadius] = useState(12);
  const [detail, setDetail] = useState(160);
  const [levelMode, setLevelMode] = useState<"relative" | "absolute">("relative");
  const [relative, setRelative] = useState(4);
  const [absolute, setAbsolute] = useState(1);

  const sd = useSpatialData(storeUrl);
  const pointsNames = useMemo(() => (sd.value ? listPointsElements(sd.value) : []), [sd.value]);
  const pointsEl = pointsName ?? pointsNames.find((n) => /morton/.test(n) && !/feature_then/.test(n)) ?? pointsNames[0];
  const source = usePointsSource(sd.value, pointsEl, POINTS_OPTIONS);
  const extent = source.value?.grid.bounds;
  const wanted = useMemo(() => (extent ? windowAt(extent, side, across, down) : undefined), [extent, side, across, down]);
  const rect = useSettled(wanted, 250);
  const tiles = usePointsTiles(source.value, rect, 0);

  const loaded = tiles.value;
  const cloud = useMemo(() => (loaded ? gather(loaded.tiles, loaded.window, qvMin, stretch) : null), [loaded, qvMin, stretch]);

  // Genes, most molecules in the window first.
  const names = useMemo(() => new Map((source.value?.features ?? []).map((f) => [f.code, f.name])), [source.value]);
  const ranked = useMemo(
    () =>
      cloud
        ? [...cloud.counts].sort((a, b) => b[1] - a[1]).map(([code, count]) => ({ code, count, name: names.get(code) ?? `#${code}` }))
        : [],
    [cloud, names],
  );
  // The choice belongs to one element. It starts as the three commonest genes of the first window.
  const [choice, setChoice] = useState<{ source: object; slots: readonly (number | null)[] }>();
  if (source.value && choice?.source !== source.value && ranked.length > 0) {
    const first = ranked.filter((g) => !CONTROL.test(g.name)).slice(0, SLOTS);
    setChoice({ source: source.value, slots: Array.from({ length: SLOTS }, (_, i) => first[i]?.code ?? null) });
  }
  const slots = useMemo(
    () => (choice && choice.source === source.value ? choice.slots : Array.from({ length: SLOTS }, () => null)),
    [choice, source.value],
  );

  // Per molecule, which slot draws it; and per slot, each molecule's weight in that slot's field.
  const categories = useMemo(() => {
    if (!cloud) return null;
    const slotOf = new Map(slots.flatMap((code, i) => (code === null ? [] : [[code, i] as const])));
    const out = new Uint32Array(cloud.n);
    for (let i = 0; i < cloud.n; i++) out[i] = slotOf.get(cloud.codes[i]!) ?? HIDDEN;
    return out;
  }, [cloud, slots]);
  const weights0 = useMemo(() => slotWeights(cloud, categories, 0), [cloud, categories]);
  const weights1 = useMemo(() => slotWeights(cloud, categories, 1), [cloud, categories]);
  const weights2 = useMemo(() => slotWeights(cloud, categories, 2), [cloud, categories]);

  // The box of voxels: the window and the molecules' depth, with room for a molecule's reach.
  const box = useMemo((): FieldBox | null => {
    if (!cloud || cloud.n === 0) return null;
    const w = cloud.window.maxX - cloud.window.minX;
    const h = cloud.window.maxY - cloud.window.minY;
    const d = cloud.depth * stretch;
    const size = [w + 2 * radius, h + 2 * radius, d + 2 * radius] as const;
    const voxel = Math.max(...size) / detail;
    const cells = (s: number) => Math.max(1, Math.ceil(s / voxel));
    return { dims: [cells(size[0]), cells(size[1]), cells(size[2])], origin: [-size[0] / 2, -size[1] / 2, -size[2] / 2], voxel };
  }, [cloud, stretch, radius, detail]);

  const points = useGpuBuffer(device, cloud?.positions ?? null, "density3d:points");
  const gpuWeights = [
    useGpuBuffer(device, weights0, "density3d:weights-0"),
    useGpuBuffer(device, weights1, "density3d:weights-1"),
    useGpuBuffer(device, weights2, "density3d:weights-2"),
  ] as const;
  const density = useDensityFields({ device, points, n: cloud?.n ?? 0, weights: gpuWeights, box, radius });

  // A level as a field value: the value in the middle of an even layer of molecules of that density,
  // as deep as these are. A layer thinner than twice the reach fills only part of a molecule's
  // sphere of influence, which is the `u` below; stretching depth thins the molecules by the stretch.
  const levelOf = (slot: number): number => {
    const code = slots[slot];
    if (!cloud || code === null || code === undefined) return Number.POSITIVE_INFINITY;
    const area = (cloud.window.maxX - cloud.window.minX) * (cloud.window.maxY - cloud.window.minY);
    const mean = (cloud.counts.get(code) ?? 0) / (area * (cloud.hasDepth ? Math.max(cloud.depth, 1e-6) : 1));
    const wantedDensity = levelMode === "relative" ? relative * mean : absolute / 1000;
    if (!cloud.hasDepth) return wantedDensity * (Math.PI / 3) * radius ** 2;
    const u = Math.min(1, (cloud.depth * stretch) / 2 / radius);
    const spread = ((2 * Math.PI) / 3) * radius ** 3 * (u - u ** 3 + (3 * u ** 5) / 5 - u ** 7 / 7);
    return (wantedDensity * spread) / stretch;
  };

  // The camera: reset for a new window size, otherwise the user's.
  const startView = useMemo(
    (): OrbitViewState => ({ target: [0, 0, 0], rotationX: 40, rotationOrbit: -25, zoom: Math.log2(620 / side) }),
    [side],
  );
  const [view, setView] = useState<{ for: OrbitViewState; now: OrbitViewState }>({ for: startView, now: startView });
  if (view.for !== startView) setView({ for: startView, now: startView });
  const viewState = view.for === startView ? view.now : startView;
  const onViewStateChange = useCallback(({ viewState: next }: { viewState: unknown }) => {
    setView((v) => ({ ...v, now: next as OrbitViewState }));
  }, []);

  const layers: Layer[] = [];
  const fields = density.value;
  if (fields) {
    const maxSteps = Math.ceil(2 * Math.hypot(...fields.box.dims)) + 8;
    fields.fields.forEach((field, slot) => {
      if (!field) return;
      layers.push(
        new IsoSurfaceLayer({
          id: `surface-${slot}`,
          field,
          ...fields.box,
          threshold: levelOf(slot),
          color: SLOT_COLOURS[slot]!,
          maxSteps,
        }),
      );
    });
  }

  const progress = tiles.progress && tiles.progress.total > 0 ? ` (${tiles.progress.done}/${tiles.progress.total} tiles)` : "";
  const error = sd.error ?? source.error ?? tiles.error;
  const busy = sd.loading
    ? "opening the store…"
    : source.loading
      ? "opening the molecules…"
      : tiles.loading
        ? `reading molecules${progress}…`
        : "";
  const thin = box && radius < box.voxel * 1.5;
  const fmt = (v: number) => Math.round(v).toLocaleString();

  return (
    <>
      <h2 className="page">Density in 3D (WebGPU)</h2>
      <p className="page prototype">
        <strong>Prototype.</strong> A first try of the 3D density kernel and the surface drawing on measured molecules. One window of one
        slide, up to three genes. It needs a store whose molecules carry a depth (a "z" column); the page says so when they do not.
      </p>
      <p className="page">
        Where are a gene's molecules crowded together, in three dimensions? Each chosen gene gets a surface wrapped around the places where
        its molecules are denser than the level you set. Drag to turn, scroll to zoom, right-drag to slide.
      </p>
      <div className="layout">
        <div className="controls">
          <StoreInput initial={storeUrl} onOpen={setStoreUrl} />
          <label>
            Molecules
            <select value={pointsEl ?? ""} onChange={(e) => setPointsName(e.target.value)}>
              {pointsNames.map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label>
            Window
            <select value={side} onChange={(e) => setSide(Number(e.target.value))}>
              {[200, 400, 800, 1600].map((s) => (
                <option key={s} value={s}>
                  {s} units across (µm on Xenium)
                </option>
              ))}
            </select>
          </label>
          <Range
            label={`Window position, across: ${(across * 100).toFixed(0)}%`}
            value={across}
            min={0}
            max={1}
            step={0.01}
            onChange={setAcross}
          />
          <Range label={`Window position, down: ${(down * 100).toFixed(0)}%`} value={down} min={0} max={1} step={0.01} onChange={setDown} />
          {slots.map((code, slot) => (
            <label key={slot}>
              <span>
                <span className="swatch" style={{ background: css(SLOT_COLOURS[slot]!) }} /> Gene {slot + 1}
              </span>
              <select
                value={code ?? ""}
                onChange={(e) => {
                  if (!source.value) return;
                  const next = slots.map((c, i) => (i === slot ? (e.target.value === "" ? null : Number(e.target.value)) : c));
                  setChoice({ source: source.value, slots: next });
                }}
              >
                <option value="">none</option>
                {code !== null && !ranked.some((g) => g.code === code) && <option value={code}>{names.get(code) ?? code} (0)</option>}
                {ranked.map((g) => (
                  <option key={g.code} value={g.code}>
                    {g.name} ({fmt(g.count)})
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label>
            Level
            <select value={levelMode} onChange={(e) => setLevelMode(e.target.value as typeof levelMode)}>
              <option value="relative">times each gene's own average here</option>
              <option value="absolute">molecules per 1000 cubic units, same for all</option>
            </select>
          </label>
          {levelMode === "relative" ? (
            <Range
              label={`Surface at ${relative.toFixed(1)}× the average`}
              value={relative}
              min={0.5}
              max={60}
              step={0}
              log
              onChange={setRelative}
            />
          ) : (
            <Range
              label={`Surface at ${absolute.toPrecision(2)} per 1000 cubic units`}
              value={absolute}
              min={0.01}
              max={100}
              step={0}
              log
              onChange={setAbsolute}
            />
          )}
          <Range
            label={`Reach of a molecule: ${radius.toFixed(1)} units`}
            value={radius}
            min={2}
            max={60}
            step={0}
            log
            onChange={setRadius}
          />
          {thin && (
            <div className="hint raised">
              The reach is close to one small cube's width: the surface will look blocky. Raise detail or reach.
            </div>
          )}
          <label>
            Detail
            <select value={detail} onChange={(e) => setDetail(Number(e.target.value))}>
              {[96, 160, 256, 384].map((d) => (
                <option key={d} value={d}>
                  {d} small cubes along the window
                </option>
              ))}
            </select>
          </label>
          <Range label={`Depth stretched ${stretch.toFixed(1)}×`} value={stretch} min={1} max={12} step={0.5} onChange={setStretch} />
          <Range label={`Leave out molecules of quality below ${qvMin}`} value={qvMin} min={0} max={40} step={1} onChange={setQvMin} />
          <div className="status">
            {busy}
            {error && <span className="error">{error.message}</span>}
            {density.error && <span className="error">{density.error}</span>}
          </div>
          {cloud && !cloud.hasDepth && cloud.n > 0 && (
            <div className="hint raised">
              These molecules have no depth here, so they all sit on one flat sheet and the surfaces are flat too.
            </div>
          )}
        </div>
        <div className="views">
          <div className="deck-canvas">
            <DeckGL
              views={VIEW}
              viewState={viewState}
              onViewStateChange={onViewStateChange}
              controller
              layers={layers}
              deviceProps={GLOBAL_DEVICE_PROPS}
            />
            <div className="view-status">
              {device ? "WebGPU" : status}
              {cloud && ` · ${fmt(cloud.n)} molecules in the window, ${cloud.depth.toFixed(1)} units deep`}
              {fields && ` · ${fields.box.dims.join("×")} small cubes · density worked out in ${fields.ms.toFixed(0)} ms`}
            </div>
          </div>
          <div className="explainers">
            <Explain
              title="What the surfaces are"
              math={String.raw`F(p) = \sum_i w_i \left(1 - \frac{\lVert p - p_i\rVert^2}{R^2}\right)^2 \quad \text{for } \lVert p - p_i\rVert < R`}
            >
              Every molecule spreads a soft influence around itself, strongest where it sits and gone at the distance "reach". The space is
              cut into small cubes, and in each one the influences of the gene's molecules are added up. That total is high where molecules
              are crowded. The surface is the skin where the total equals the level: inside it the gene is denser than the level, outside it
              sparser. Raising the level shrinks the surfaces onto the most crowded places.
            </Explain>
            <Explain title="The two kinds of level">
              "Times each gene's own average" compares a gene with itself: at 4×, the surface encloses places where that gene is four times
              as dense as it is over the whole window on average. Every gene shows its own hot spots however common or rare it is.
              "Molecules per 1000 cubic units" is one number for all genes, so a common gene has large surfaces and a rare one may have
              none; the surfaces can then be compared between genes and between windows.
            </Explain>
            <Explain title="Why depth is stretched">
              A tissue section is far thinner than it is wide: tens of units deep against thousands across. Drawn to scale it is a sheet of
              paper. Stretching depth makes the layering visible. The reach is not stretched with it, so in real units a molecule reaches
              less far in depth than sideways, by the stretch.
            </Explain>
            <Explain title="Cells instead of molecules">
              The same sum works for cells: put a point at each cell's centre and let its weight be how much of a gene the cell expresses.
              This page weights each molecule 1 for its own gene and 0 for the others, through that same weight input. The cell tables in
              this store have no depth, so cells are not shown here.
            </Explain>
          </div>
        </div>
      </div>
    </>
  );
}
