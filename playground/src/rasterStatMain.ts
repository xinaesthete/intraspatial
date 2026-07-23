// B1c — the raster demonstration (docs/stream-b-bridge-plan.md).
//
// End-to-end: pull ONE decoded level tile of a real SpatialData morphology channel through the
// existing `Loader.getChunk`, convert it with the B1a converter (`tileToField`, stamping the B1b
// placement + channel facets via `imageToGraph`), then run the registered `getisOrd` op on it IN
// THE GRAPH and read back the Gi* hot-spot z-scores. Everything runs on `navigator.gpu` (the
// browser backend), exactly the op the GPU vitest suite exercises — so a green readout here is the
// same statistic, now on live HTJ2K bytes.
//
// A "synthetic fixture" path runs the identical graph on a dep-free Mandelbrot plane tile, so the
// page demonstrates the pipeline (and its CPU-golden agreement) even with no store reachable.

import { chunkCounts, syntheticPlane, tileToField } from "../../src/datasource";
import { Graph, pull, registerBuiltinOps } from "../../src/gpu/graph";
import { browserBackend } from "../../src/gpu/graph/backend.browser";
import type { FieldValue, GpuField } from "../../src/gpu/graph/handle";
import { imageTileSource } from "./datasource/imageToGraph";
import { openSpatialDataImage } from "./datasource/spatialDataLoader";

/** The live HTJ2K SpatialData store (zarr v3). Configurable — not hard-coded in library code. */
const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr";
const RADIUS = 3;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const storeInput = $<HTMLInputElement>("store");
const elementInput = $<HTMLInputElement>("element");
const runBtn = $<HTMLButtonElement>("run");
const fixtureBtn = $<HTMLButtonElement>("fixture");
const statusEl = $<HTMLDivElement>("status");
const readoutEl = $<HTMLDivElement>("readout");
const inCanvas = $<HTMLCanvasElement>("in");
const zCanvas = $<HTMLCanvasElement>("z");

storeInput.value = DEFAULT_STORE;

interface GridValue {
  width: number;
  height: number;
  data: Float32Array;
}

function asGrid(v: FieldValue): GridValue {
  if (v.shape.kind !== "grid") throw new Error("expected a grid");
  return { width: v.shape.width, height: v.shape.height, data: v.data as Float32Array };
}

/** Grayscale render of an intensity grid, auto-scaled to [min,max]. */
function drawIntensity(c: HTMLCanvasElement, g: GridValue): void {
  c.width = g.width;
  c.height = g.height;
  let lo = Infinity, hi = -Infinity;
  for (const v of g.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1;
  const img = new ImageData(g.width, g.height);
  for (let i = 0; i < g.data.length; i++) {
    const t = Math.round(((g.data[i]! - lo) / span) * 255);
    img.data[i * 4] = t;
    img.data[i * 4 + 1] = t;
    img.data[i * 4 + 2] = t;
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
}

/** Diverging blue–white–red render of a z-score grid, symmetric about 0. */
function drawZ(c: HTMLCanvasElement, g: GridValue): { min: number; max: number; mean: number; hot: [number, number]; hotZ: number } {
  c.width = g.width;
  c.height = g.height;
  let lo = Infinity, hi = -Infinity, sum = 0, hotZ = -Infinity, hotI = 0;
  for (let i = 0; i < g.data.length; i++) {
    const v = g.data[i]!;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    sum += v;
    if (v > hotZ) { hotZ = v; hotI = i; }
  }
  const scale = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const img = new ImageData(g.width, g.height);
  for (let i = 0; i < g.data.length; i++) {
    const t = g.data[i]! / scale; // [-1,1]
    let r: number, gg: number, b: number;
    if (t >= 0) { r = 255; gg = Math.round(255 * (1 - t)); b = Math.round(255 * (1 - t)); }
    else { b = 255; r = Math.round(255 * (1 + t)); gg = Math.round(255 * (1 + t)); }
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = gg;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
  return { min: lo, max: hi, mean: sum / g.data.length, hot: [hotI % g.width, Math.floor(hotI / g.width)], hotZ };
}

function setStatus(msg: string, err = false): void {
  statusEl.textContent = msg;
  statusEl.style.color = err ? "#fca5a5" : "#94a3b8";
}

/** Run `getisOrd` on a grid source and render the input + z-score, returning the facet/stat readout. */
async function runGraph(g: Graph, src: GpuField, inputGrid: GridValue, provenance: string): Promise<void> {
  const z = g.op1("getisOrd", { grid: src }, { radius: RADIUS });
  const zVal = await pull(g, z, { ctx: { backend: browserBackend } });
  const zGrid = asGrid(zVal);
  drawIntensity(inCanvas, inputGrid);
  const stats = drawZ(zCanvas, zGrid);

  const placed = src.placement ? `system="${src.placement.system}"` : "absent (array space)";
  const zPlaced = z.placement ? `system="${z.placement.system}"` : "absent";
  readoutEl.innerHTML =
    `<b>${provenance}</b><br>` +
    `grid: ${inputGrid.width}×${inputGrid.height} (${inputGrid.width * inputGrid.height} cells), radius ${RADIUS}<br>` +
    `Gi* z-score: min ${stats.min.toFixed(2)}, max ${stats.max.toFixed(2)}, mean ${stats.mean.toFixed(3)}<br>` +
    `hottest cell: (${stats.hot[0]}, ${stats.hot[1]}) z=${stats.hotZ.toFixed(2)}<br>` +
    `source placement: ${placed}<br>` +
    `getisOrd output placement (facet propagated): ${zPlaced}`;
  setStatus("done — Getis-Ord Gi* computed on the graph.");
}

async function runLive(): Promise<void> {
  const url = storeInput.value.trim();
  const element = elementInput.value.trim() || "morphology_focus";
  setStatus(`opening store ${url} …`);
  try {
    const img = await openSpatialDataImage(url, element, { maxChannels: 1 });
    setStatus(`decoding a level tile of "${element}" …`);
    // Coarsest level → a small, whole-image tile (fast, and the hot-spot map is legible).
    const level = img.ms.levelCount - 1;
    const [nx, ny] = chunkCounts(img.ms, level);
    // Prefer an INTERIOR chunk. Only the trailing chunk in each axis can be a thin partial
    // sliver (the level's dims aren't a multiple of the chunk size), so the naive "centre"
    // index `floor(n/2)` can land on e.g. a 1024×44 edge strip. Clamping away from the last
    // index guarantees a full, square-ish chunk.
    const interior = (n: number) => Math.min(Math.floor(n / 2), Math.max(0, n - 2));
    const cx = interior(nx), cy = interior(ny);
    const tile = await img.loader.getChunk({ level, x: cx, y: cy, z: 0 });

    const g = new Graph();
    registerBuiltinOps();
    const src = imageTileSource(g, img, tile, { channel: 0 });
    const inputGrid = asGrid(tileToField(tile, { channel: 0 }));
    await runGraph(g, src, inputGrid, `LIVE: ${element} · level ${level} · chunk (${cx},${cy}) · ${img.label}`);
  } catch (e) {
    setStatus(`live store failed: ${(e as Error).message}\nFalling back is available via "run fixture".`, true);
  }
}

async function runFixture(): Promise<void> {
  setStatus("running the synthetic Mandelbrot-plane fixture …");
  registerBuiltinOps();
  const { loader } = syntheticPlane({ width: 128, height: 128, chunk: 128, levelCount: 2 });
  const tile = await loader.getChunk({ level: 0, x: 0, y: 0, z: 0 });
  const g = new Graph();
  const src = tileToField(tile);
  const handle = g.source(src, "tile");
  await runGraph(g, handle, asGrid(src), "FIXTURE: synthetic Mandelbrot plane 128×128 (no server)");
}

runBtn.addEventListener("click", () => void runLive());
fixtureBtn.addEventListener("click", () => void runFixture());
// Auto-attempt the live store on load; the fixture button is always available offline.
void runLive();
