// Co-location modes — the Gram matrix over a set of channels, and its OKLab eigen-projection.
//
// The premise this page is built on, and the reason it is separate from `cellstats.html`: **most
// wild-type SpatialData stores carry no cell-type annotation at all.** A raw Xenium export has
// `cell_id`, areas and counts in `obs` and nothing else, so a demo that requires a cell-type column
// cannot open it. What such a store does have is `X`. Since `gram.ts` takes per-cell *weights* and
// a cell type is only the one-hot case of one, both sources take the identical path — which is why
// the channel source is a tab rather than two pipelines.
//
// The other difference from `cellstats.html` is the interaction. That page selects a PAIR and shows
// Γ_AB for it, which does not scale past a handful of channels and asks the user to already know
// which pair matters. Here the whole K×K matrix is decomposed and the leading modes are painted at
// once: the map answers "what co-location structure is there, and where" without a pair being
// nominated first.

import { oklabToSrgb } from "../../src/color/oklab";
import { cssRgb, deg, diverging, oklchToSrgbMapped, rgbBytes, type Srgb } from "../../src/color/ramps";
import { gramMatrixGpu } from "../../src/gpu/spatial/gramMatrix";
import { modeSwatch, paintGramModes } from "../../src/gpu/spatial/gramModes";
import { type ChannelCloud, channelsFromExpression, coLocationModes } from "../../src/spatial/gram";
import { equivalentRadius, KERNELS, type KernelSpec, kernelLabel } from "../../src/spatial/kernels";
import {
  type CellTable,
  type ColumnInfo,
  listCellTables,
  NO_TYPE_COLUMN,
  readCellTable,
  type TableInfo,
  TooManyTypesError,
} from "./datasource/cellTable";
import { listVars, readVarColumns, selectionCost, type VarCatalog } from "./datasource/varMatrix";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr/";
const MAX_CHANNELS = 32;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const storeInput = $<HTMLInputElement>("store");
const inspectBtn = $<HTMLButtonElement>("inspect");
const tableSelect = $<HTMLSelectElement>("tableSel");
const loadBtn = $<HTMLButtonElement>("load");
const tabObs = $<HTMLButtonElement>("tabObs");
const tabX = $<HTMLButtonElement>("tabX");
const paneObs = $<HTMLDivElement>("paneObs");
const paneX = $<HTMLDivElement>("paneX");
const obsColSelect = $<HTMLSelectElement>("obsCol");
const obsNoteEl = $<HTMLSpanElement>("obsNote");
const varMatrixSel = $<HTMLSelectElement>("varMatrix");
const varSearch = $<HTMLInputElement>("varSearch");
const varList = $<HTMLSelectElement>("varList");
const varChosen = $<HTMLDivElement>("varChosen");
const varAddBtn = $<HTMLButtonElement>("varAdd");
const varClearBtn = $<HTMLButtonElement>("varClear");
const varNoteEl = $<HTMLParagraphElement>("varNote");
const kernelSelect = $<HTMLSelectElement>("kernel");
const radiusInput = $<HTMLInputElement>("radius");
const scaleInput = $<HTMLInputElement>("scale");
const resSelect = $<HTMLSelectElement>("res");
const satInput = $<HTMLInputElement>("sat");
const computeBtn = $<HTMLButtonElement>("compute");
const statusEl = $<HTMLDivElement>("status");
const modeCanvas = $<HTMLCanvasElement>("modemap");
const corrCanvas = $<HTMLCanvasElement>("corr");
const screeCanvas = $<HTMLCanvasElement>("scree");
const loadingsCanvas = $<HTMLCanvasElement>("loadings");
const modeReadoutEl = $<HTMLDivElement>("modeReadout");
const corrReadoutEl = $<HTMLDivElement>("corrReadout");
const modeLegendEl = $<HTMLDivElement>("modeLegend");

storeInput.value = DEFAULT_STORE;

// ---- state -------------------------------------------------------------------------------------

let storeTables: TableInfo[] = [];
/** The untyped base read: every cell as one cloud, plus `rowOrder` — the row index that `obs` and
 *  `X` columns are aligned to. Read once per table, whichever channel source is used. */
let base: CellTable | null = null;
let varCatalog: VarCatalog | null = null;
let chosenVars: number[] = [];
let source: "obs" | "x" = "obs";
let lastCorr: { labels: string[]; corr: Float64Array } | null = null;

const setStatus = (msg: string, err = false): void => {
  statusEl.textContent = msg;
  statusEl.style.color = err ? "#f87171" : "#94a3b8";
};
const kernelOf = (): KernelSpec => KERNELS[Number(kernelSelect.value)] ?? KERNELS[0]!;
const umPerUnit = (): number => {
  const declared = Number(scaleInput.value);
  return Number.isFinite(declared) && declared > 0 ? declared : 1;
};
const toWorld = (um: number) => um / umPerUnit();
const toUm = (world: number) => world * umPerUnit();
const unitSuffix = () => (base?.units.micrometres !== undefined ? " µm" : " µm*");

const DIVERGING_LUT = ((): Uint8Array => {
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = rgbBytes(diverging((2 * i) / 255 - 1, { centreL: 0.19, endL: 0.8, endC: 0.17 }));
    out[i * 3] = r;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = b;
  }
  return out;
})();

function tableBounds(t: CellTable): [number, number, number, number] {
  const xs = t.rowOrder?.xs ?? [];
  const ys = t.rowOrder?.ys ?? [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const m = t.placement.worldFromArray;
  for (let i = 0; i < xs.length; i++) {
    // Apply the placement on the host for the BOUNDS only; the points themselves are carried and
    // transformed on the GPU (ADR-0018), so this must match what splatDensity will do.
    const x = m.origin[0] + m.axes[0][0] * xs[i]! + m.axes[1][0] * ys[i]!;
    const y = m.origin[1] + m.axes[0][1] * xs[i]! + m.axes[1][1] * ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  const padX = (maxX - minX) * 0.02 || 1;
  const padY = (maxY - minY) * 0.02 || 1;
  return [minX - padX, minY - padY, maxX + padX, maxY + padY];
}

/** Raster dims on an AREA budget, so a non-square ROI is not stretched and world cells stay square. */
function viewDims(b: [number, number, number, number], target: number): { width: number; height: number } {
  const w = Math.max(b[2] - b[0], 1e-9);
  const h = Math.max(b[3] - b[1], 1e-9);
  const scale = Math.sqrt((target * target) / (w * h));
  return { width: Math.max(16, Math.round(w * scale)), height: Math.max(16, Math.round(h * scale)) };
}

// ---- channel sources ---------------------------------------------------------------------------

function selectTab(which: "obs" | "x"): void {
  source = which;
  tabObs.setAttribute("aria-selected", String(which === "obs"));
  tabX.setAttribute("aria-selected", String(which === "x"));
  paneObs.hidden = which !== "obs";
  paneX.hidden = which !== "x";
}

/** Categorical `obs` columns are the only ones that can define channels: a numeric column has no
 *  categories to split on, and one with as many distinct values as cells is an identifier. */
function fillObsColumns(info: TableInfo | undefined): void {
  obsColSelect.innerHTML = "";
  const cats = (info?.columns ?? []).filter((c: ColumnInfo) => c.kind === "categorical" && (c.nCategories ?? 0) >= 2);
  for (const col of cats) {
    const opt = document.createElement("option");
    opt.value = col.name;
    opt.textContent = `${col.name} — ${col.nCategories} categories`;
    obsColSelect.appendChild(opt);
  }
  obsNoteEl.textContent = cats.length
    ? `${cats.length} usable categorical column${cats.length > 1 ? "s" : ""}.`
    : "no categorical obs column with ≥2 categories — this store has no annotation, so use the X tab.";
  if (!cats.length) selectTab("x");
}

function renderVarList(): void {
  const q = varSearch.value.trim().toLowerCase();
  const names = varCatalog?.names ?? [];
  varList.innerHTML = "";
  let shown = 0;
  let total = 0;
  for (let i = 0; i < names.length; i++) {
    if (q && !names[i]!.toLowerCase().includes(q)) continue;
    total++;
    if (shown >= 400) continue;
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = chosenVars.includes(i) ? `✓ ${names[i]}` : names[i]!;
    varList.appendChild(opt);
    shown++;
  }
  if (total > shown) {
    const opt = document.createElement("option");
    opt.disabled = true;
    opt.textContent = `… ${total - shown} more — narrow the filter`;
    varList.appendChild(opt);
  }
}

function renderChips(): void {
  varChosen.innerHTML = "";
  if (chosenVars.length === 0) {
    varChosen.textContent = "no vars selected";
    return;
  }
  for (const v of chosenVars) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = varCatalog?.names[v] ?? `var${v}`;
    chip.title = "click to remove";
    chip.addEventListener("click", () => {
      chosenVars = chosenVars.filter((x) => x !== v);
      renderChips();
      renderVarList();
    });
    varChosen.appendChild(chip);
  }
}

async function refreshVarCatalog(t: CellTable): Promise<void> {
  chosenVars = [];
  varCatalog = null;
  renderChips();
  renderVarList();
  varNoteEl.textContent = "reading var names …";
  const cat = await listVars(storeInput.value.trim(), t.tableName);
  if (cat.error || cat.nVars === 0) {
    varNoteEl.textContent = `no readable X on '${t.tableName}'${cat.error ? ` — ${cat.error}` : ""}.`;
    return;
  }
  varCatalog = cat;
  varMatrixSel.innerHTML = "";
  for (const m of cat.matrices) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m === "X" ? "X" : `layers/${m}`;
    varMatrixSel.appendChild(opt);
  }
  renderVarList();
  const how =
    cat.encoding === "csr"
      ? "stored CSR — a gene's entries are scattered across every cell's row, so ANY selection reads the whole matrix"
      : cat.encoding === "csc"
        ? "stored CSC — a selection reads only its own slices"
        : "dense";
  varNoteEl.textContent = `${cat.nVars} vars × ${cat.nCells} cells; ${how}.`;
}

/** Build the channel set for whichever source is active. */
async function buildChannels(t: CellTable): Promise<{ channels: ChannelCloud[]; note: string }> {
  if (source === "obs") {
    const col = obsColSelect.value;
    if (!col) throw new Error("no categorical obs column selected");
    const typed = await readCellTable(storeInput.value.trim(), { table: t.tableName, typeColumn: col });
    const channels = typed.types
      .filter((ty) => ty.n > 0)
      .sort((a, b) => b.n - a.n)
      .slice(0, MAX_CHANNELS)
      .map((ty) => ({ label: ty.label ?? `type ${ty.id}`, xs: ty.xs as number[], ys: ty.ys as number[] }));
    const dropped = typed.types.length - channels.length;
    return {
      channels,
      note: `${channels.length} categories of '${col}'${dropped > 0 ? ` (largest ${MAX_CHANNELS}; ${dropped} dropped)` : ""}`,
    };
  }
  if (!varCatalog) throw new Error("no X catalogue — inspect and load first");
  if (chosenVars.length < 2) throw new Error("pick at least two vars — a 1×1 matrix has nothing to correlate");
  if (!t.rowOrder) throw new Error("this source has no table row order to join X against");
  const names = chosenVars.map((v) => varCatalog!.names[v] ?? `var${v}`);
  const cost = selectionCost(varCatalog, chosenVars.length);
  setStatus(`reading ${chosenVars.length} vars (~${(cost * 100).toFixed(0)}% of the matrix) …`);
  const cols = await readVarColumns(storeInput.value.trim(), {
    table: t.tableName,
    vars: chosenVars,
    matrix: varMatrixSel.value || "X",
    names,
  });
  if (cols.nCells !== t.rowOrder.xs.length) {
    throw new Error(`X has ${cols.nCells} rows but the table has ${t.rowOrder.xs.length} centroids`);
  }
  const thin = cols.stats.filter((s) => s.nonZero < 20).map((s) => `${s.label} (${s.nonZero})`);
  return {
    channels: channelsFromExpression(t.rowOrder.xs, t.rowOrder.ys, cols.values, cols.names),
    note: `${cols.names.length} vars from ${cols.matrix}${thin.length ? ` · sparse: ${thin.join(", ")}` : ""}`,
  };
}

// ---- charts ------------------------------------------------------------------------------------

const CELL = 20;

function drawCorr(labels: string[], corr: Float64Array, hover: { a: number; b: number } | null): void {
  const N = labels.length;
  const gut = 104;
  corrCanvas.width = gut + N * CELL;
  corrCanvas.height = gut + N * CELL;
  const ctx = corrCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, corrCanvas.width, corrCanvas.height);
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const k = Math.max(0, Math.min(255, Math.round(((corr[a * N + b]! + 1) / 2) * 255))) * 3;
      ctx.fillStyle = `rgb(${DIVERGING_LUT[k]},${DIVERGING_LUT[k + 1]},${DIVERGING_LUT[k + 2]})`;
      ctx.fillRect(gut + b * CELL, gut + a * CELL, CELL, CELL);
    }
  }
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (let a = 0; a < N; a++) {
    ctx.textAlign = "right";
    ctx.fillText(labels[a]!, gut - 5, gut + a * CELL + CELL / 2, gut - 8);
    ctx.save();
    ctx.translate(gut + a * CELL + CELL / 2, gut - 5);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "left";
    ctx.fillText(labels[a]!, 0, 0, gut - 8);
    ctx.restore();
  }
  if (hover) {
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
    ctx.strokeRect(gut + hover.b * CELL + 1, gut + hover.a * CELL + 1, CELL - 2, CELL - 2);
  }
}

function drawScree(explained: Float64Array): void {
  const n = Math.min(explained.length, 24);
  const W = 460;
  const H = 200;
  screeCanvas.width = W;
  screeCanvas.height = H;
  const ctx = screeCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const pad = 28;
  const bw = (W - 2 * pad) / n;
  const peak = Math.max(...Array.from({ length: n }, (_, i) => explained[i]!), 1e-9);
  for (let i = 0; i < n; i++) {
    const h = ((H - 2 * pad) * Math.max(0, explained[i]!)) / peak;
    // Modes 1-3 are the ones the map uses; colour them as their OKLab axis so the chart and the
    // map are legible together rather than needing a mental key.
    const swatch: Srgb = i < 3 ? oklabToSrgb(modeSwatch([i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0])) : [0.35, 0.42, 0.53];
    ctx.fillStyle = cssRgb(swatch);
    ctx.fillRect(pad + i * bw + 1, H - pad - h, bw - 2, h);
  }
  ctx.strokeStyle = "#334155";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad);
  ctx.stroke();
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i < Math.min(n, 12); i++) ctx.fillText(String(i + 1), pad + i * bw + bw / 2, H - pad + 12);
  ctx.textAlign = "left";
  ctx.fillText(`top mode ${(explained[0]! * 100).toFixed(1)}% of variance`, pad, 16);
}

function drawLoadings(labels: string[], vectors: Float64Array): void {
  const K = labels.length;
  const rows = Math.min(3, K);
  const W = 460;
  const rowH = 54;
  const H = rows * rowH + 26;
  loadingsCanvas.width = W;
  loadingsCanvas.height = H;
  const ctx = loadingsCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const pad = 46;
  const bw = (W - pad - 10) / K;
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  for (let k = 0; k < rows; k++) {
    const mid = 20 + k * rowH + rowH / 2;
    ctx.strokeStyle = "#334155";
    ctx.beginPath();
    ctx.moveTo(pad, mid);
    ctx.lineTo(W - 10, mid);
    ctx.stroke();
    ctx.fillStyle = cssRgb(oklabToSrgb(modeSwatch([k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0])));
    ctx.textAlign = "right";
    ctx.fillText(`mode ${k + 1}`, pad - 6, mid + 3);
    for (let a = 0; a < K; a++) {
      const v = vectors[k * K + a]!;
      const h = v * (rowH / 2 - 4);
      ctx.fillRect(pad + a * bw + 1, mid - Math.max(h, 0), Math.max(bw - 2, 1), Math.abs(h));
    }
  }
  ctx.fillStyle = "#64748b";
  ctx.textAlign = "left";
  ctx.fillText(`${K} channels, left to right in matrix order`, pad, H - 6);
}

// ---- the computation ---------------------------------------------------------------------------

async function computeModes(): Promise<void> {
  const t = base;
  if (!t) {
    setStatus("load a table first.", true);
    return;
  }
  try {
    const { channels, note } = await buildChannels(t);
    if (channels.length < 2) throw new Error("need at least two channels");
    const bbox = tableBounds(t);
    const target = Number(resSelect.value) || 384;
    const { width, height } = viewDims(bbox, target);
    // The box is the EFFECTIVE range (support of J⊛J = 2·splat radius), so halve it for the splat,
    // then rescale per kernel so switching kernel holds the probed scale fixed rather than the
    // nominal radius (kernels.ts: μ₂ = r²/(n+2) shrinks with order).
    const rangeUm = Number(radiusInput.value) || toUm(Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 40);
    radiusInput.value = rangeUm.toPrecision(3);
    const radius = equivalentRadius(kernelOf(), toWorld(rangeUm) / 2);

    setStatus(`splatting ${channels.length} channels at ${width}×${height} …`);
    const t0 = performance.now();
    const res = await gramMatrixGpu(channels, { bbox, width, height, radius, kernel: kernelOf() });
    const modes = coLocationModes(res);
    const info = await paintGramModes(modeCanvas, res, {
      vectors: modes.vectors,
      saturate: Number(satInput.value) || 2.5,
    });
    const ms = performance.now() - t0;

    lastCorr = { labels: [...res.labels], corr: res.corr };
    drawCorr(lastCorr.labels, lastCorr.corr, null);
    drawScree(modes.explained);
    drawLoadings(res.labels, modes.vectors);

    const top = Array.from({ length: Math.min(3, res.labels.length) }, (_, k) => {
      const loads = res.labels
        .map((lab, a) => ({ lab, v: modes.vectors[k * res.labels.length + a]! }))
        .sort((x, y) => Math.abs(y.v) - Math.abs(x.v))
        .slice(0, 3)
        .map((x) => `${x.v >= 0 ? "+" : "−"}${x.lab}`)
        .join(" ");
      return `mode ${k + 1} (${(modes.explained[k]! * 100).toFixed(0)}%): ${loads}`;
    }).join(" · ");

    setStatus(`done — ${note}, ${ms.toFixed(0)} ms (submission).`);
    modeReadoutEl.innerHTML =
      `<b>${res.labels.length} channels</b> · ${kernelLabel(kernelOf())}, range ${rangeUm.toPrecision(3)}${unitSuffix()} · ` +
      `raster ${width}×${height} · <b>PSD defect ${modes.psdDefect.toExponential(1)}</b> (0 = the eigenvalues are variances) · ${top}`;
    modeLegendEl.textContent =
      `L = mode 1, a = mode 2, b = mode 3, each saturating at ±${(Number(satInput.value) || 2.5).toFixed(1)}σ ` +
      `(σ = ${info.sigmas.map((s) => s.toFixed(2)).join(", ")}). Equal perceived colour distance ≈ equal distance ` +
      `between co-location profiles — that is what OKLab buys over three RGB ramps.`;
  } catch (err) {
    setStatus(`failed: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

// ---- load / inspect ------------------------------------------------------------------------------

async function inspectStore(autoLoad: boolean): Promise<void> {
  const url = storeInput.value.trim();
  setStatus(`inspecting ${url} …`);
  try {
    storeTables = await listCellTables(url);
    const usable = storeTables.filter((t) => t.hasCentroids);
    tableSelect.innerHTML = "";
    for (const info of storeTables) {
      const opt = document.createElement("option");
      opt.value = info.name;
      opt.textContent = info.error
        ? `${info.name} — unreadable (${info.error})`
        : `${info.name} — ${info.nRows} cells${info.hasCentroids ? "" : ", no obsm/spatial"}`;
      opt.disabled = !info.hasCentroids;
      tableSelect.appendChild(opt);
    }
    if (usable.length === 0) {
      setStatus(`no table in this store has obsm/spatial centroids (found ${storeTables.length}).`, true);
      return;
    }
    tableSelect.value = usable[0]!.name;
    fillObsColumns(usable[0]);
    setStatus(`${usable.length} usable table${usable.length > 1 ? "s" : ""} — ${usable[0]!.name} selected.`);
    if (autoLoad) await loadTable();
  } catch (err) {
    setStatus(`inspect failed: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

/**
 * Read the centroids ONCE, untyped.
 *
 * The base read deliberately does not split on any column: it exists to provide `rowOrder`, the row
 * index that both `obs` and `X` are aligned to, and it must succeed on a store with no annotation
 * at all. Splitting for the obs source is a second, cheap read.
 */
async function loadTable(): Promise<void> {
  const url = storeInput.value.trim();
  const table = tableSelect.value;
  setStatus(`reading centroids from "${table}" …`);
  try {
    base = await readCellTable(url, { table, typeColumn: NO_TYPE_COLUMN });
    const bbox = tableBounds(base);
    radiusInput.value = toUm(Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 40).toPrecision(3);
    setStatus(
      `${base.totalCells} cells · ${base.units.raw ? `unit "${base.units.raw}"` : "no unit stated"}` +
        `${base.units.micrometres === undefined ? " — lengths are marked µm* and the µm/unit box is your declaration" : ""}.`,
    );
    await refreshVarCatalog(base);
  } catch (err) {
    if (err instanceof TooManyTypesError) setStatus(`read failed: ${err.message}`, true);
    else setStatus(`read failed: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

// ---- wiring --------------------------------------------------------------------------------------

for (const [i, k] of KERNELS.entries()) {
  const opt = document.createElement("option");
  opt.value = String(i);
  opt.textContent = kernelLabel(k) + (i === 1 ? " (recommended)" : "");
  kernelSelect.appendChild(opt);
}
kernelSelect.value = "1"; // Epanechnikov: kernels.ts measures it as the best-behaved member

tabObs.addEventListener("click", () => selectTab("obs"));
tabX.addEventListener("click", () => selectTab("x"));
inspectBtn.addEventListener("click", () => void inspectStore(false));
loadBtn.addEventListener("click", () => void loadTable());
tableSelect.addEventListener("change", () => fillObsColumns(storeTables.find((t) => t.name === tableSelect.value)));
computeBtn.addEventListener("click", () => void computeModes());
varSearch.addEventListener("input", renderVarList);
varList.addEventListener("change", () => {
  const v = Number(varList.value);
  if (!Number.isFinite(v)) return;
  if (chosenVars.includes(v)) chosenVars = chosenVars.filter((x) => x !== v);
  else if (chosenVars.length >= MAX_CHANNELS) {
    varNoteEl.textContent = `${MAX_CHANNELS} channels is the cap — the matrix is O(K²·P) per view.`;
    return;
  } else chosenVars.push(v);
  renderChips();
  renderVarList();
});
varAddBtn.addEventListener("click", () => {
  const q = varSearch.value.trim().toLowerCase();
  const names = varCatalog?.names ?? [];
  for (let i = 0; i < names.length && chosenVars.length < MAX_CHANNELS; i++) {
    if (q && !names[i]!.toLowerCase().includes(q)) continue;
    if (!chosenVars.includes(i)) chosenVars.push(i);
  }
  renderChips();
  renderVarList();
});
varClearBtn.addEventListener("click", () => {
  chosenVars = [];
  renderChips();
  renderVarList();
});
corrCanvas.addEventListener("mousemove", (e) => {
  if (!lastCorr) return;
  const N = lastCorr.labels.length;
  const rect = corrCanvas.getBoundingClientRect();
  const s = corrCanvas.width / rect.width;
  const a = Math.floor(((e.clientY - rect.top) * s - 104) / CELL);
  const b = Math.floor(((e.clientX - rect.left) * s - 104) / CELL);
  if (a < 0 || b < 0 || a >= N || b >= N) return;
  drawCorr(lastCorr.labels, lastCorr.corr, { a, b });
  const r = lastCorr.corr[a * N + b]!;
  corrReadoutEl.innerHTML =
    `<b>${lastCorr.labels[a]}</b> vs <b>${lastCorr.labels[b]}</b> · spatial correlation r = ${r.toFixed(3)} — ` +
    (r > 0.2 ? "found in the same places" : r < -0.2 ? "spatially exclusive" : "little spatial relation");
});

// A swatch strip so the OKLab axes are readable without guessing.
modeLegendEl.style.borderLeft = `3px solid ${cssRgb(oklchToSrgbMapped([0.7, 0.12, deg(250)]))}`;
modeLegendEl.style.paddingLeft = "6px";

void inspectStore(true);
