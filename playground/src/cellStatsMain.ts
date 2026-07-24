// B2 demo — cell-centroid ingestion → per-cell-type KDE splat (docs/stream-b-bridge-plan.md).
//
// End-to-end: read a real SpatialData regions table (`cellTable.ts`, zarrita via zarrextra), group
// the centroids into per-`cell_type_id` clouds (ADR-0018 — never merged), render ALL 49750
// centroids as a scatter coloured by type, then run the registered `splatDensity` op on the
// SELECTED type's cloud IN THE GRAPH (browserBackend, pull) and render its KDE density grid. A
// synthetic 2-type fixture runs the identical path offline.
//
// This is the spec-independent prerequisite for the cell-type spatial statistics (TCM, cross-PCF)
// that follow — it proves per-type centroid clouds flow into the op-graph, splat at real scale, and
// carry their placement facet through to the output grid.

import { Graph, pull, registerBuiltinOps } from "../../src/gpu/graph";
import { browserBackend } from "../../src/gpu/graph/backend.browser";
import type { FieldValue, GpuField } from "../../src/gpu/graph/handle";
import { computeTcm } from "../../src/spatial/tcm";
import { type CellTable, DEFAULT_CELL_TABLE, readCellTable, syntheticCellTable } from "./datasource/cellTable";

/** The live Leap034 SpatialData store (zarr v3, CORS `*`). Configurable — not hard-coded downstream. */
const DEFAULT_STORE = "http://localhost:5055/project/289/spatial/leap034_layers.zarr/";
const SPLAT = { width: 384, height: 384, sigma: 3, radiusSigma: 4 } as const;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const storeInput = $<HTMLInputElement>("store");
const tableInput = $<HTMLInputElement>("table");
const runBtn = $<HTMLButtonElement>("run");
const fixtureBtn = $<HTMLButtonElement>("fixture");
const typeSelect = $<HTMLSelectElement>("type");
const statusEl = $<HTMLDivElement>("status");
const readoutEl = $<HTMLDivElement>("readout");
const scatterCanvas = $<HTMLCanvasElement>("scatter");
const kdeCanvas = $<HTMLCanvasElement>("kde");
const typeSelectB = $<HTMLSelectElement>("typeB");
const radiusInput = $<HTMLInputElement>("radius");
const sigmaInput = $<HTMLInputElement>("sigma");
const tcmBtn = $<HTMLButtonElement>("tcmBtn");
const tcmCanvas = $<HTMLCanvasElement>("tcm");
const tcmReadoutEl = $<HTMLDivElement>("tcmReadout");

/** TCM grid resolution + world-unit defaults (α=5 is fixed; radius:σ ≈ 2:1 as in the paper). */
const TCM_GRID = 384;
const spanOf = (b: [number, number, number, number]) => Math.max(b[2] - b[0], b[3] - b[1], 1);
const defaultRadius = (b: [number, number, number, number]) => spanOf(b) / 50;
const defaultSigma = (b: [number, number, number, number]) => spanOf(b) / 100;

storeInput.value = DEFAULT_STORE;
tableInput.value = DEFAULT_CELL_TABLE;

let current: CellTable | null = null;

/** A distinct-ish colour per integer id (golden-angle hue wheel). */
function idColor(id: number): [number, number, number] {
  const h = (id * 137.508) % 360;
  return hslToRgb(h / 360, 0.65, 0.55);
}
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

function setStatus(msg: string, err = false): void {
  statusEl.textContent = msg;
  statusEl.style.color = err ? "#fca5a5" : "#94a3b8";
}

interface GridValue {
  width: number;
  height: number;
  data: Float32Array;
}
function asGrid(v: FieldValue): GridValue {
  if (v.shape.kind !== "grid") throw new Error("expected a grid");
  return { width: v.shape.width, height: v.shape.height, data: v.data as Float32Array };
}

/** World-bounds over ALL types (so scatter + every KDE share one frame). */
function tableBounds(t: CellTable): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const ty of t.types) {
    for (let i = 0; i < ty.xs.length; i++) {
      const x = ty.xs[i]!, y = ty.ys[i]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  const pad = 0.04 * Math.max(maxX - minX, maxY - minY, 1);
  return [minX - pad, minY - pad, maxX + pad, maxY + pad];
}

/** Scatter of every centroid, coloured by cell_type_id; the selected type is drawn brighter/larger. */
function drawScatter(t: CellTable, bbox: [number, number, number, number], selectedId: number | null): void {
  const W = 900, H = 900;
  scatterCanvas.width = W;
  scatterCanvas.height = H;
  const ctx = scatterCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const [minX, minY, maxX, maxY] = bbox;
  const sx = W / (maxX - minX || 1);
  const sy = H / (maxY - minY || 1);
  for (const ty of t.types) {
    const [r, g, b] = idColor(ty.id);
    const sel = ty.id === selectedId;
    ctx.fillStyle = sel ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},0.45)`;
    const rad = sel ? 1.7 : 1.0;
    for (let i = 0; i < ty.xs.length; i++) {
      const px = (ty.xs[i]! - minX) * sx;
      const py = (ty.ys[i]! - minY) * sy;
      ctx.fillRect(px - rad, py - rad, rad * 2, rad * 2);
    }
  }
}

/** Viridis-ish render of a KDE density grid, auto-scaled to [0,max]. */
function drawKde(c: HTMLCanvasElement, g: GridValue): { max: number; sum: number } {
  c.width = g.width;
  c.height = g.height;
  let hi = 0, sum = 0;
  for (const v of g.data) {
    if (v > hi) hi = v;
    sum += v;
  }
  const span = hi || 1;
  const img = new ImageData(g.width, g.height);
  for (let i = 0; i < g.data.length; i++) {
    const t = g.data[i]! / span; // [0,1]
    // simple magma-ish ramp: black → purple → orange → white
    const r = Math.round(255 * Math.min(1, t * 1.6));
    const gg = Math.round(255 * Math.max(0, t * t));
    const b = Math.round(255 * Math.max(0, 0.5 * t + 0.4 * (1 - Math.abs(t - 0.4) * 2)));
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = gg;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
  return { max: hi, sum };
}

/** Run splatDensity on the selected type's cloud IN THE GRAPH, render + report. */
async function splatSelected(): Promise<void> {
  const t = current;
  if (!t) return;
  const id = Number(typeSelect.value);
  const ty = t.types.find((x) => x.id === id);
  if (!ty) return;
  const bbox = tableBounds(t);
  drawScatter(t, bbox, id);
  setStatus(`splatting type ${id} (${ty.n} cells) on the GPU …`);
  try {
    const g = new Graph();
    registerBuiltinOps();
    const src: GpuField = ty.source(g);
    const density = g.op1("splatDensity", { points: src }, { ...SPLAT, bbox });
    const dv = await pull(g, density, { ctx: { backend: browserBackend } });
    const stats = drawKde(kdeCanvas, asGrid(dv));

    const srcPlaced = src.placement ? `system="${src.placement.system}"` : "absent (array space)";
    const outPlaced = density.placement ? `system="${density.placement.system}"` : "absent";
    const prov = src.provenance;
    readoutEl.innerHTML =
      `<b>${t.label}</b><br>` +
      `total cells: ${t.totalCells} &nbsp;·&nbsp; cell types: ${t.types.length}<br>` +
      `selected type id ${id}: <b>${ty.n}</b> cells<br>` +
      `placement system: <b>${t.system}</b> &nbsp;(region "${t.provenance.region || "—"}", instance_key "${t.provenance.instanceKey}")<br>` +
      `source cloud placement: ${srcPlaced}<br>` +
      `splatDensity output placement (facet propagated): ${outPlaced}<br>` +
      `provenance carried: cellTypeId=${prov?.cellTypeId ?? "—"}, region=${prov?.region ?? "—"}<br>` +
      `KDE grid: ${SPLAT.width}×${SPLAT.height}, σ=${SPLAT.sigma} — peak density ${stats.max.toFixed(3)}`;
    setStatus(`done — type ${id} splatted through the op-graph.`);
  } catch (e) {
    setStatus(`splat failed: ${(e as Error).message}`, true);
  }
}

/** Diverging blue–white–red render of a TCM Γ grid, symmetric about 0. */
function drawTcm(c: HTMLCanvasElement, g: GridValue): { min: number; max: number } {
  c.width = g.width;
  c.height = g.height;
  let lo = Infinity, hi = -Infinity;
  for (const v of g.data) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const scale = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const img = new ImageData(g.width, g.height);
  for (let i = 0; i < g.data.length; i++) {
    const t = g.data[i]! / scale; // [-1,1]
    let r: number, gg: number, b: number;
    if (t >= 0) {
      r = 255;
      gg = Math.round(255 * (1 - t));
      b = Math.round(255 * (1 - t));
    } else {
      b = 255;
      r = Math.round(255 * (1 + t));
      gg = Math.round(255 * (1 + t));
    }
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = gg;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
  return { min: lo, max: hi };
}

/** Compute the TCM Γ_ab(x) for the selected (A, B) pair on the CPU (Mode 1, against the tested
 *  reference oracle) and render it. Fast enough at Leap034 scale via the bucket-grid path. */
function computeTcmMap(): void {
  const t = current;
  if (!t) return;
  const idA = Number(typeSelect.value);
  const idB = Number(typeSelectB.value);
  const A = t.types.find((x) => x.id === idA);
  const B = t.types.find((x) => x.id === idB);
  if (!A || !B) return;
  const bbox = tableBounds(t);
  const radius = Number(radiusInput.value) || defaultRadius(bbox);
  const sigma = Number(sigmaInput.value) || defaultSigma(bbox);
  setStatus(`computing TCM Γ(${idA}→${idB}) …`);
  // Yield a frame so the status paints before the (synchronous) compute.
  requestAnimationFrame(() => {
    const t0 = performance.now();
    const grid = computeTcm(
      { xs: A.xs, ys: A.ys },
      { xs: B.xs, ys: B.ys },
      { width: TCM_GRID, height: TCM_GRID, bbox, radius, sigma, alpha: 5 },
    );
    const ms = performance.now() - t0;
    const stats = drawTcm(tcmCanvas, { width: TCM_GRID, height: TCM_GRID, data: grid });
    tcmReadoutEl.innerHTML =
      `<b>Γ<sub>AB</sub></b> — A = type ${idA} (${A.n} cells), B = type ${idB} (${B.n} cells)<br>` +
      `radius ${radius.toPrecision(3)}, σ ${sigma.toPrecision(3)} (world units), α=5, grid ${TCM_GRID}² · ${ms.toFixed(0)} ms (CPU)<br>` +
      `Γ range [${stats.min.toFixed(3)}, ${stats.max.toFixed(3)}] · red = A clusters around B, blue = A excludes B`;
    setStatus(`done — TCM Γ(${idA}→${idB}) computed (Mode 1, faithful; validated against the reference oracle).`);
  });
}

/** Populate the type dropdown + counts and draw the full scatter. */
function present(t: CellTable): void {
  current = t;
  typeSelect.innerHTML = "";
  typeSelectB.innerHTML = "";
  for (const ty of t.types) {
    const label = `id ${ty.id} — ${ty.n} cells`;
    for (const sel of [typeSelect, typeSelectB]) {
      const opt = document.createElement("option");
      opt.value = String(ty.id);
      opt.textContent = label;
      sel.appendChild(opt);
    }
  }
  // Default A = most populous, B = second most populous (skipping degenerate n<=1 ids), for a
  // legible first splat and a meaningful first TCM pair.
  const ranked = [...t.types].filter((x) => x.n > 1).sort((a, b) => b.n - a.n);
  const biggest = ranked[0] ?? t.types[0];
  const second = ranked[1] ?? biggest;
  if (biggest) typeSelect.value = String(biggest.id);
  if (second) typeSelectB.value = String(second.id);
  const bbox = tableBounds(t);
  radiusInput.value = defaultRadius(bbox).toPrecision(3);
  sigmaInput.value = defaultSigma(bbox).toPrecision(3);
  drawScatter(t, bbox, biggest ? biggest.id : null);
  setStatus(`read ${t.totalCells} cells in ${t.types.length} types — select types; A splats, A→B gives the TCM.`);
  void splatSelected();
  computeTcmMap();
}

async function runLive(): Promise<void> {
  const url = storeInput.value.trim();
  const table = tableInput.value.trim() || DEFAULT_CELL_TABLE;
  setStatus(`reading table "${table}" from ${url} …`);
  try {
    const t = await readCellTable(url, { table });
    present(t);
  } catch (e) {
    setStatus(`live store failed: ${(e as Error).message}\nUse "run fixture" to demo offline.`, true);
  }
}

function runFixture(): void {
  setStatus("building the synthetic 2-type fixture …");
  present(syntheticCellTable());
}

runBtn.addEventListener("click", () => void runLive());
fixtureBtn.addEventListener("click", () => runFixture());
typeSelect.addEventListener("change", () => {
  void splatSelected();
  computeTcmMap();
});
typeSelectB.addEventListener("change", () => computeTcmMap());
tcmBtn.addEventListener("click", () => computeTcmMap());
radiusInput.addEventListener("change", () => computeTcmMap());
sigmaInput.addEventListener("change", () => computeTcmMap());
// Auto-attempt the live store on load; the fixture button is always available offline.
void runLive();
