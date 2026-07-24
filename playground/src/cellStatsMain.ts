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
import { crossPCF, crossPCFMatrix, type LabelledCells } from "../../src/spatial/pcf";
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
const pcfCanvas = $<HTMLCanvasElement>("pcf");
const pcfReadoutEl = $<HTMLDivElement>("pcfReadout");
const matrixCanvas = $<HTMLCanvasElement>("matrix");
const matrixReadoutEl = $<HTMLDivElement>("matrixReadout");
const PCF_BINS = 30;

// Coordinated-view state: the last N×N matrix (for hover lookup), the hovered cell, and a debounce
// timer so the expensive TCM only recomputes when the mouse settles on a pair.
let lastMatrix: { types: number[]; counts: number[]; g: Float64Array } | null = null;
let hoverCell: { a: number; b: number } | null = null;
let tcmTimer: ReturnType<typeof setTimeout> | undefined;
/** Fixed emphasis colours for the hovered pair in the scatter. */
const A_COLOR: [number, number, number] = [34, 211, 238]; // cyan
const B_COLOR: [number, number, number] = [245, 158, 11]; // amber

/** Flatten the per-type clouds back into one labelled cell set for the N-way pass. */
function allCells(t: CellTable): LabelledCells {
  const xs: number[] = [];
  const ys: number[] = [];
  const typeId: number[] = [];
  for (const ty of t.types) {
    for (let i = 0; i < ty.xs.length; i++) {
      xs.push(ty.xs[i]!);
      ys.push(ty.ys[i]!);
      typeId.push(ty.id);
    }
  }
  return { xs, ys, typeId };
}

/** TCM grid resolution + world-unit defaults (α=5 is fixed; radius:σ ≈ 2:1 as in the paper).
 *  256² keeps the smooth Γ field visually identical to 384² while staying interactive on hover. */
const TCM_GRID = 256;
const spanOf = (b: [number, number, number, number]) => Math.max(b[2] - b[0], b[3] - b[1], 1);
const defaultRadius = (b: [number, number, number, number]) => spanOf(b) / 50;
const defaultSigma = (b: [number, number, number, number]) => spanOf(b) / 100;

storeInput.value = DEFAULT_STORE;
tableInput.value = DEFAULT_CELL_TABLE;

let current: CellTable | null = null;

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

/** Scatter of every centroid, highlighting the current pair: type A in cyan, type B in amber, all
 *  other cells faint grey — so the linked pair pops against the tissue as you hover the matrix. */
function drawScatterPair(t: CellTable, bbox: [number, number, number, number], idA: number, idB: number): void {
  const W = 900, H = 900;
  scatterCanvas.width = W;
  scatterCanvas.height = H;
  const ctx = scatterCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const [minX, minY, maxX, maxY] = bbox;
  const sx = W / (maxX - minX || 1);
  const sy = H / (maxY - minY || 1);
  const draw = (xs: readonly number[], ys: readonly number[], color: [number, number, number], alpha: number, rad: number) => {
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
    for (let i = 0; i < xs.length; i++) {
      const px = (xs[i]! - minX) * sx;
      const py = (ys[i]! - minY) * sy;
      ctx.fillRect(px - rad, py - rad, rad * 2, rad * 2);
    }
  };
  for (const ty of t.types) {
    if (ty.id === idA || ty.id === idB) continue;
    draw(ty.xs, ty.ys, [148, 163, 184], 0.16, 0.8); // context: faint grey
  }
  const A = t.types.find((x) => x.id === idA);
  const B = t.types.find((x) => x.id === idB);
  if (B && idB !== idA) draw(B.xs, B.ys, B_COLOR, 0.9, 1.6);
  if (A) draw(A.xs, A.ys, A_COLOR, 0.95, 1.8);
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

/** Line plot of g(r) vs r with a dashed g=1 (CSR) reference. */
function drawPcfCurve(c: HTMLCanvasElement, r: number[], g: number[]): void {
  const W = 760;
  const H = 260;
  const padL = 46;
  const padB = 26;
  const padT = 12;
  const padR = 12;
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const rMax = r[r.length - 1] ?? 1;
  let gMax = 1.2;
  for (const v of g) if (v > gMax) gMax = v;
  gMax = Math.ceil(gMax * 1.1);
  const px = (rv: number) => padL + (rv / rMax) * (W - padL - padR);
  const py = (gv: number) => H - padB - (gv / gMax) * (H - padB - padT);
  ctx.strokeStyle = "#475569";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, py(1));
  ctx.lineTo(W - padR, py(1));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#334155";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < r.length; i++) {
    const X = px(r[i]!);
    const Y = py(g[i]!);
    if (i === 0) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  }
  ctx.stroke();
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("g=1", padL + 4, py(1) - 3);
  ctx.fillText(`g (0…${gMax})`, 4, padT + 8);
  ctx.fillText(`r → ${rMax.toPrecision(3)} (world units)`, W - padR - 150, H - 8);
}

/** Compute + render the cross-PCF g_ab(r) curve for the selected pair (CPU, Mode 1). */
function computePcf(): void {
  const t = current;
  if (!t) return;
  const idA = Number(typeSelect.value);
  const idB = Number(typeSelectB.value);
  const A = t.types.find((x) => x.id === idA);
  const B = t.types.find((x) => x.id === idB);
  if (!A || !B) return;
  const bbox = tableBounds(t);
  const rMax = spanOf(bbox) / 8;
  const res = crossPCF({ xs: A.xs, ys: A.ys }, { xs: B.xs, ys: B.ys }, { bbox, rMax, nBins: PCF_BINS });
  drawPcfCurve(pcfCanvas, res.r, res.g);
  pcfReadoutEl.innerHTML =
    `<b>g<sub>AB</sub>(r)</b> — A = type ${idA} (${A.n} cells), B = type ${idB} (${B.n} cells)<br>` +
    `rMax ${rMax.toPrecision(3)} (world units), ${PCF_BINS} bins · g(r→0) = ${(res.g[0] ?? 0).toFixed(2)} · g&gt;1 clustering, g&lt;1 exclusion`;
}

const MATRIX_CELL = 22;

/** N×N diverging heatmap of the cross-PCF matrix: log₂(g) mapped red (clustering) ↔ blue (exclusion),
 *  with an optional hovered-cell outline + row/col guides for the coordinated view. */
function drawMatrix(c: HTMLCanvasElement, res: { types: number[]; g: Float64Array }, hover: { a: number; b: number } | null): void {
  const N = res.types.length;
  const cell = MATRIX_CELL;
  c.width = N * cell;
  c.height = N * cell;
  const ctx = c.getContext("2d")!;
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const g = res.g[a * N + b]!;
      const l = g > 0 ? Math.max(-2, Math.min(2, Math.log2(g))) / 2 : -1; // [-1,1], 0 = CSR
      let r: number, gg: number, bl: number;
      if (l >= 0) {
        r = 255;
        gg = Math.round(255 * (1 - l));
        bl = Math.round(255 * (1 - l));
      } else {
        bl = 255;
        r = Math.round(255 * (1 + l));
        gg = Math.round(255 * (1 + l));
      }
      ctx.fillStyle = `rgb(${r},${gg},${bl})`;
      ctx.fillRect(b * cell, a * cell, cell, cell);
    }
  }
  if (hover) {
    ctx.fillStyle = "rgba(226,232,240,0.14)";
    ctx.fillRect(0, hover.a * cell, N * cell, cell); // row A
    ctx.fillRect(hover.b * cell, 0, cell, N * cell); // col B
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
    ctx.strokeRect(hover.b * cell + 1, hover.a * cell + 1, cell - 2, cell - 2);
  }
}

/** Compute + render the N-way cross-PCF association matrix for all cell types (one batched pass). */
function computeMatrix(): void {
  const t = current;
  if (!t) return;
  const bbox = tableBounds(t);
  const radius = defaultRadius(bbox);
  const t0 = performance.now();
  const res = crossPCFMatrix(allCells(t), { bbox, radius });
  const ms = performance.now() - t0;
  lastMatrix = res;
  drawMatrix(matrixCanvas, res, hoverCell);
  matrixReadoutEl.innerHTML =
    `<b>N-way cross-PCF</b> — ${res.types.length}×${res.types.length} (all ordered pairs), contact radius ${radius.toPrecision(3)} (world units) · ${ms.toFixed(0)} ms (one batched pass) · ` +
    `<b>hover a cell</b> to link the scatter · cross-PCF · Γ below (Γ on settle/click).`;
}

/** Point the linked views (scatter pair, cross-PCF, and optionally the expensive TCM) at a pair. */
function setPair(idA: number, idB: number, doTcm: boolean): void {
  typeSelect.value = String(idA);
  typeSelectB.value = String(idB);
  const t = current;
  if (!t) return;
  drawScatterPair(t, tableBounds(t), idA, idB);
  computePcf();
  if (doTcm) {
    void splatSelected();
    computeTcmMap();
  }
}

/** Debounce the expensive views (KDE-of-A splat + TCM) so they recompute only when the mouse
 *  settles on a matrix cell. The cheap ones (scatter, cross-PCF) already updated on hover. */
function scheduleTcm(): void {
  if (tcmTimer) clearTimeout(tcmTimer);
  tcmTimer = setTimeout(() => {
    void splatSelected();
    computeTcmMap();
  }, 300);
}

/** Map a mouse event on the matrix canvas to a `(row a, col b)` cell, or null if outside. */
function matrixCellFromEvent(e: MouseEvent): { a: number; b: number } | null {
  const res = lastMatrix;
  if (!res) return null;
  const N = res.types.length;
  const rect = matrixCanvas.getBoundingClientRect();
  const b = Math.floor(((e.clientX - rect.left) / rect.width) * N);
  const a = Math.floor(((e.clientY - rect.top) / rect.height) * N);
  if (a < 0 || a >= N || b < 0 || b >= N) return null;
  return { a, b };
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
  setStatus(`read ${t.totalCells} cells in ${t.types.length} types — hover the N-way matrix to link the scatter · cross-PCF · Γ for any pair.`);
  computeMatrix();
  const idA = biggest ? biggest.id : (t.types[0]?.id ?? 0);
  const idB = second ? second.id : idA;
  setPair(idA, idB, true);
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
typeSelect.addEventListener("change", () => setPair(Number(typeSelect.value), Number(typeSelectB.value), true));
typeSelectB.addEventListener("change", () => setPair(Number(typeSelect.value), Number(typeSelectB.value), true));
tcmBtn.addEventListener("click", () => computeTcmMap());
radiusInput.addEventListener("change", () => computeTcmMap());
sigmaInput.addEventListener("change", () => computeTcmMap());

// --- coordinated view: hover the N-way matrix to link the scatter, cross-PCF, and (on settle) Γ ---
matrixCanvas.addEventListener("mousemove", (e) => {
  const res = lastMatrix;
  if (!res) return;
  const cell = matrixCellFromEvent(e);
  if (!cell) return;
  if (hoverCell && hoverCell.a === cell.a && hoverCell.b === cell.b) return;
  hoverCell = cell;
  const N = res.types.length;
  const idA = res.types[cell.a]!;
  const idB = res.types[cell.b]!;
  const g = res.g[cell.a * N + cell.b]!;
  drawMatrix(matrixCanvas, res, hoverCell);
  const rel = g > 1 ? `clustering (${g.toFixed(2)}×)` : g < 1 && g > 0 ? `exclusion (${g.toFixed(2)}×)` : "no co-location";
  matrixReadoutEl.innerHTML =
    `<b>A = type ${idA}</b> (${res.counts[cell.a]} cells, <span style="color:#22d3ee">cyan</span>) → ` +
    `<b>B = type ${idB}</b> (${res.counts[cell.b]} cells, <span style="color:#f59e0b">amber</span>) · ` +
    `g = ${g.toFixed(3)} — ${rel} · scatter + cross-PCF live; Γ on settle (or click)`;
  setPair(idA, idB, false);
  scheduleTcm();
});
matrixCanvas.addEventListener("mouseleave", () => {
  hoverCell = null;
  if (lastMatrix) drawMatrix(matrixCanvas, lastMatrix, null);
});
matrixCanvas.addEventListener("click", (e) => {
  const cell = matrixCellFromEvent(e);
  if (!cell || !lastMatrix) return;
  if (tcmTimer) clearTimeout(tcmTimer);
  setPair(lastMatrix.types[cell.a]!, lastMatrix.types[cell.b]!, true);
});
matrixCanvas.style.cursor = "crosshair";
// Auto-attempt the live store on load; the fixture button is always available offline.
void runLive();
