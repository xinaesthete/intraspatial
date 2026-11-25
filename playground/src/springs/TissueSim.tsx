// Data beside simulation, on one Deck: a map of measured cell positions (left) and a 3D orbit view
// (right) of a spring-model tissue started from a patch of those cells. The simulated cells are
// also drawn back on the map, where they were measured.
//
// One Deck, two deck views, one WebGPU device: deck makes it, the library adopts it
// (`useSharedDevice`), and the simulation and the cell positions live on it as plain buffers that
// `@intraspatial/deck`'s layers bind. Nothing is read back to draw.
import { type Layer, OrbitView, type OrbitViewState, OrthographicView, type OrthographicViewState } from "@deck.gl/core";
import DeckGL from "@deck.gl/react";
import { type Color, ParticleLayer } from "@intraspatial/deck";
import type { Device } from "@luma.gl/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getDevice, writeView } from "../../../src/gpu/device";
import {
  createSpheroidGpuRunner,
  type SpheroidScenario,
  type SpringsGpu,
  spheroidDefaults,
  spheroidScenario,
} from "../../../src/gpu/sim/springs/index";
import { SHARED_DEVICE_PROPS, useSharedDevice } from "../deck/useSharedDevice";
import { useAsync } from "../hooks/useAsync";
import { defaultSystem, useCellTable, useCellTables } from "../hooks/useCellTable";
import { useElementSize } from "../hooks/useElementSize";
import { useSpatialData } from "../hooks/useSpatialData";
import { StoreInput } from "../modes/PageControls";
import { BALL_PRESETS, BALL_STRUCTURE } from "./SpringsDemo";
import { modelToTable, nearestCells, patchToModel, seedCells } from "./seedFromCells";
import { springsLayers } from "./springsLayers";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const DATA_COLOUR: readonly Color[] = [[100, 116, 139, 255]];
const VIEWS = [
  new OrthographicView({ id: "map", x: 0, width: "50%", controller: true }),
  new OrbitView({ id: "orbit", x: "50%", width: "50%", orbitAxis: "Y", fovy: 35, controller: true }),
];
/** Each layer belongs to the view its id starts with. */
const layerFilter = ({ layer, viewport }: { layer: Layer; viewport: { id: string } }) => layer.id.startsWith(viewport.id);

interface Cells {
  readonly xs: Float32Array;
  readonly ys: Float32Array;
  readonly label: string;
}

/** A made-up section for when no store is open: a few clumps of cells and a sparse background. */
function syntheticCells(n = 6000): Cells {
  let s = 0x2545f491;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const clumps = [
    [500, 500, 90],
    [1100, 700, 140],
    [700, 1300, 70],
    [1500, 1400, 110],
  ] as const;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const clump = rnd() < 0.8 ? clumps[Math.floor(rnd() * clumps.length)] : undefined;
    xs[i] = clump ? clump[0] + gauss() * clump[2] : rnd() * 2000;
    ys[i] = clump ? clump[1] + gauss() * clump[2] : rnd() * 2000;
  }
  return { xs, ys, label: `${n.toLocaleString()} made-up cells` };
}

/** `[x, y, 0, 0]` per cell in a storage buffer on the shared device, for ParticleLayer to bind. */
function useCellBuffer(ready: boolean, cells: Cells): GPUBuffer | undefined {
  const made = useAsync(async () => {
    if (!ready) return undefined;
    const device = await getDevice();
    const n = cells.xs.length;
    const data = new Float32Array(Math.max(1, n) * 4);
    for (let i = 0; i < n; i++) data.set([cells.xs[i]!, cells.ys[i]!], i * 4);
    const buffer = device.createBuffer({
      label: "tissuesim:cells",
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    writeView(device.queue, buffer, data);
    return buffer;
  }, [ready, cells]);
  // The previous table's buffer is freed once its replacement is on screen.
  const buffer = made.value;
  useEffect(() => () => buffer?.destroy(), [buffer]);
  return buffer;
}

/** Runs a spheroid on the shared device and hands back its device object for drawing. */
function useSpheroidRunner(scenario: SpheroidScenario | undefined, ready: boolean, playing: boolean, stepsPerFrame: number) {
  const [live, setLive] = useState<{ scenario: SpheroidScenario; gpu: SpringsGpu }>();
  const [error, setError] = useState<string>();
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (!scenario || !ready) return;
    let stopped = false;
    let busy = false;
    let raf = 0;
    const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
    // No device given: the library's, which is deck's once adopted. The view draws the buffers, so
    // the host copy is only refreshed when cells divide (`hostCurrent: false`).
    createSpheroidGpuRunner(scenario, { hostCurrent: false })
      .then((runner) => {
        if (stopped) return;
        setLive({ scenario, gpu: runner.gpu });
        const tick = () => {
          if (stopped) return;
          if (playing && !busy) {
            busy = true;
            runner
              .advance(stepsPerFrame)
              .then(() => setStep(runner.gpu.stepIndex))
              .catch(report)
              .finally(() => {
                busy = false;
              });
          }
          raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      })
      .catch(report);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [scenario, ready, playing, stepsPerFrame]);
  return { gpu: live && live.scenario === scenario ? live.gpu : undefined, error, step };
}

export function TissueSim() {
  const shared = useSharedDevice();
  const [storeUrl, setStoreUrl] = useState(DEFAULT_STORE);
  const [count, setCount] = useState(150);
  const [pick, setPick] = useState<readonly [number, number]>();
  const [playing, setPlaying] = useState(true);
  const [stepsPerFrame, setStepsPerFrame] = useState(4);
  const [showGel, setShowGel] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Store → first table with centroids → its cells. Anything missing: the made-up section.
  const sd = useSpatialData(storeUrl);
  const tables = useCellTables(sd.value);
  const tableName = tables.value?.find((t) => t.hasCentroids)?.name;
  const table = useCellTable(sd.value, tableName, defaultSystem(sd.value));
  const synthetic = useMemo(() => syntheticCells(), []);
  const cells = useMemo((): Cells => {
    const t = table.value;
    if (!t || t.totalCells === 0) return synthetic;
    const xs = new Float32Array(t.totalCells);
    const ys = new Float32Array(t.totalCells);
    let at = 0;
    for (const cloud of t.types) {
      xs.set(cloud.xs, at);
      ys.set(cloud.ys, at);
      at += cloud.n;
    }
    return { xs, ys, label: `${t.totalCells.toLocaleString()} cells from ${t.tableName}` };
  }, [table.value, synthetic]);
  const loading = sd.loading || tables.loading || table.loading;
  const storeError = sd.error ?? tables.error ?? table.error;

  const bounds = useMemo(() => {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < cells.xs.length; i++) {
      minX = Math.min(minX, cells.xs[i]!);
      maxX = Math.max(maxX, cells.xs[i]!);
      minY = Math.min(minY, cells.ys[i]!);
      maxY = Math.max(maxY, cells.ys[i]!);
    }
    return { minX, minY, maxX, maxY };
  }, [cells]);
  // A new table starts from its own middle.
  const [pickedFor, setPickedFor] = useState(cells);
  if (pickedFor !== cells) {
    setPickedFor(cells);
    setPick(undefined);
  }
  const [cx, cy] = pick ?? [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2];
  const patch = useMemo(() => nearestCells(cells.xs, cells.ys, cx, cy, count), [cells, cx, cy, count]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `nonce` restarts the same patch
  const scenario = useMemo(() => {
    const sc = spheroidScenario({ ...BALL_STRUCTURE, cellRadius: 0, seed: 1 + nonce });
    // As the springs page does: the 3D defaults, then the preset over them.
    sc.configure({ ...spheroidDefaults(3), ...BALL_PRESETS[0]?.config });
    seedCells(sc, patchToModel(cells.xs, cells.ys, patch));
    return sc;
  }, [cells, patch, nonce]);
  const run = useSpheroidRunner(scenario, shared.ready, playing, stepsPerFrame);
  const cellBuffer = useCellBuffer(shared.ready, cells);

  const box = useRef<HTMLDivElement>(null);
  const size = useElementSize(box);
  const sized = size.width > 0 && size.height > 0;
  const initialViewState = useMemo(() => {
    // Each view has half the width. 2^zoom is pixels per unit: fit the table, with a small margin.
    const paneWidth = size.width / 2;
    const fit = Math.min(paneWidth / Math.max(bounds.maxX - bounds.minX, 1), size.height / Math.max(bounds.maxY - bounds.minY, 1));
    const gel = BALL_STRUCTURE.matrixRadius ?? 11;
    const map: OrthographicViewState = {
      target: [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, 0],
      zoom: Math.log2(Math.max(fit, 1e-6) * 0.92),
    };
    const orbit: OrbitViewState = {
      target: [0, 0, 0],
      rotationX: 35,
      rotationOrbit: 20,
      zoom: Math.log2(Math.min(paneWidth, size.height) / (2.2 * gel)),
    };
    return { map, orbit };
    // A resize refits too: simple, and the views are small.
  }, [bounds, size.width, size.height]);

  const layers = useMemo(() => {
    const out: Layer[] = [];
    if (cellBuffer) {
      out.push(
        new ParticleLayer({
          id: "map-data",
          positions: cellBuffer,
          count: cells.xs.length,
          // About a third of the gap cells would have if they filled the table's extent evenly.
          radius: 0.35 * Math.sqrt(((bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY)) / Math.max(1, cells.xs.length)),
          radiusMinPixels: 0.75,
          colors: DATA_COLOUR,
          shading: 0,
        }),
      );
    }
    if (run.gpu) {
      out.push(
        ...springsLayers(run.gpu, {
          id: "map-sim",
          shading: 0,
          modelMatrix: modelToTable(patch),
          scale: patch.spacing,
          radiusMinPixels: 1.5,
          onTop: true,
        }),
        ...springsLayers(run.gpu, { id: "orbit-sim", matrix: showGel }),
      );
    }
    return out;
  }, [cellBuffer, cells, bounds, patch, run.gpu, showGel]);

  const onClick = useCallback((info: { coordinate?: number[]; viewport?: { id: string } }) => {
    const [x, y] = info.coordinate ?? [];
    if (info.viewport?.id === "map" && x !== undefined && y !== undefined) setPick([x, y]);
  }, []);

  return (
    <>
      <h2 className="page">Tissue from data: a map beside a simulation</h2>
      <p className="page">
        Left: where cells were measured in a tissue section. Right: a simulated tissue grown from a patch of those cells, in 3D. Click the
        map to choose the patch. The simulated cells are also drawn on the map (teal, leaders red), over the measured ones (grey).
      </p>
      <p className="page prototype">
        <strong>First prototype.</strong> This page tests whether one drawing surface can show data and a simulation together. The
        simulation takes only the cells' positions from the data: it knows nothing about cell types or the real tissue around them, and its
        settings are in arbitrary units. Treat it as a sketch, not a prediction.
      </p>
      <div className="layout">
        <div className="controls">
          <StoreInput initial={DEFAULT_STORE} onOpen={setStoreUrl} />
          <p className="hint">
            Needs a SpatialData store with a table of cell positions. Without one (or while it loads) the page uses made-up cells.
          </p>
          <div className="row">
            <button type="button" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={() => setNonce((v) => v + 1)}>
              Restart
            </button>
          </div>
          <label>
            Cells in the patch: {patch.indices.length}
            <input type="range" min={20} max={600} step={10} value={count} onChange={(e) => setCount(Number(e.target.value))} />
          </label>
          <label>
            Steps per frame: {stepsPerFrame}
            <input type="range" min={1} max={40} value={stepsPerFrame} onChange={(e) => setStepsPerFrame(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={showGel} onChange={(e) => setShowGel(e.target.checked)} />
            Show the gel in the 3D view
          </label>
          <p className="hint">
            The patch is the {patch.indices.length} cells nearest the point you click. One simulation unit is set to the typical gap between
            neighbouring cells there ({patch.spacing.toFixed(1)} in the data's units), so cells start about one diameter apart. They are
            placed on a flat plane inside a ball of gel and then left to move, divide and push into it.
          </p>
          <div className="status">
            {loading ? "opening store… (made-up cells meanwhile)" : cells.label} · step {run.step.toLocaleString()}
          </div>
          <div className="status">{shared.status}</div>
          {storeError && <div className="status error">Store: {storeError.message}. Showing made-up cells.</div>}
          {run.error && <div className="status error">{run.error}</div>}
        </div>
        <div className="views">
          <div className="deck-canvas" ref={box}>
            {sized && (
              <DeckGL
                views={VIEWS}
                initialViewState={initialViewState}
                layers={layers}
                layerFilter={layerFilter}
                deviceProps={SHARED_DEVICE_PROPS}
                onDeviceInitialized={shared.onDeviceInitialized as (d: Device) => void}
                onClick={onClick}
                // The simulation's buffers change under the layers every step: draw every frame.
                _animate
              />
            )}
            <div className="pane-label left">Measured cells (click to choose a patch)</div>
            <div className="pane-label right">Simulated tissue (drag to orbit)</div>
          </div>
        </div>
      </div>
    </>
  );
}
