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
import { envelopeExcursions, type SpectrumEnvelopeResult, spectrumEnvelope } from "../../src/gpu/spatial/gramEnvelope";
import type { GramMatrixGpuResult } from "../../src/gpu/spatial/gramMatrix";
import { gramMatrixGpu } from "../../src/gpu/spatial/gramMatrix";
import { type ModePaintInfo, modeSwatch, paintGramModes } from "../../src/gpu/spatial/gramModes";
import {
  type ColourBy,
  type HeightSource,
  type OrbitCamera,
  paintGramTerrain,
  probeChannels,
  similaritySwatch,
  standardise,
} from "../../src/gpu/spatial/gramTerrain";
import type { ImageOverlay } from "../../src/gpu/spatial/imageOverlayWgsl";
import { apronCoverage, type ChannelCloud, channelsFromExpression, coLocationModes } from "../../src/spatial/gram";
import { equivalentRadius, KERNELS, type KernelSpec, kernelLabel } from "../../src/spatial/kernels";
import {
  type CellTable,
  type ColumnInfo,
  listCellTables,
  NO_TYPE_COLUMN,
  openSpatialData,
  readCellTable,
  type TableInfo,
  TooManyTypesError,
} from "./datasource/cellTable";
import { type ContextImage, listImageElements, loadContextImage, uvFromWorld } from "./datasource/imageContext";
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
const chromaWInput = $<HTMLInputElement>("chromaW");
const windowSel = $<HTMLSelectElement>("windowSel");
const heightSel = $<HTMLSelectElement>("height");
const colourBySel = $<HTMLSelectElement>("colourBy");
const hscaleInput = $<HTMLInputElement>("hscale");
const modesUsedSel = $<HTMLSelectElement>("modesUsed");
const dspanInput = $<HTMLInputElement>("dspan");
const stepSel = $<HTMLSelectElement>("stepSel");
const resetCamBtn = $<HTMLButtonElement>("resetCam");
const terrainCanvas = $<HTMLCanvasElement>("terrain");
const terrainLegendEl = $<HTMLDivElement>("terrainLegend");
const wandReadoutEl = $<HTMLDivElement>("wandReadout");
const wandOverlayEl = $<HTMLDivElement>("wandOverlay");
const imageSel = $<HTMLSelectElement>("imageSel");
const imageLoadBtn = $<HTMLButtonElement>("imageLoad");
const imageMixInput = $<HTMLInputElement>("imageMix");
const imageNoteEl = $<HTMLSpanElement>("imageNote");
const simsSelect = $<HTMLSelectElement>("sims");
const runEnvelopeBtn = $<HTMLButtonElement>("runEnvelope");
const screeLegendEl = $<HTMLDivElement>("screeLegend");
const computeBtn = $<HTMLButtonElement>("compute");
const statusEl = $<HTMLDivElement>("status");
const modeCanvas = $<HTMLCanvasElement>("modemap");
const corrCanvas = $<HTMLCanvasElement>("corr");
const screeCanvas = $<HTMLCanvasElement>("scree");
const loadingsCanvas = $<HTMLCanvasElement>("loadings");
const loadingsLegendEl = $<HTMLDivElement>("loadingsLegend");
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
/** Both matrices, because they answer different questions and the hover shows both: `corr` is the
 *  standardised one the modes come from, `g` the association normalised so CSR reads 1 — and `g` is
 *  the one the viewport apron actually moves, since the apron changes `mass`, not the rasters. */
let lastCorr: { labels: string[]; corr: Float64Array; g: Float64Array } | null = null;

/** The live Gram result, kept so the terrain can redraw on a camera move without re-splatting.
 *  Its `resident` rasters are valid only until the next `gramMatrixGpu` — which is exactly why the
 *  terrain redraws from this handle rather than caching anything of its own. */
let live: { res: GramMatrixGpuResult; vectors: Float64Array } | null = null;
/** The wand: a standardised channel vector sampled from a pixel, and where it came from. */
let wand: { z: Float64Array; col: number; row: number; label: string } | null = null;

const DEFAULT_CAM: OrbitCamera = { azimuth: 0, elevation: 0.55, distance: 1.35, target: [0, 0, 0] };
let cam: OrbitCamera = DEFAULT_CAM;

/** The loaded context image and its world→UV affine. Held together because an image whose placement
 *  could not be inverted must not be drawn at all — see `imageOverlay`. */
let ctxImage: { img: ContextImage; uvFromWorld: Float64Array } | null = null;

/** The last simulated null, held so the scree chart can keep its band across repaints. Cleared by a
 *  recompute: a band drawn against a different channel set or window would be a false claim. */
let lastEnvelope: SpectrumEnvelopeResult | null = null;

/** What both views pass to their shaders. One source, so the flat map and the terrain cannot
 *  disagree about where the image is or how much of it is showing. */
function imageOverlay(): ImageOverlay | undefined {
  const mix = Number(imageMixInput.value);
  if (!ctxImage || !(mix > 0)) return undefined;
  return { texture: ctxImage.img.texture, uvFromWorld: ctxImage.uvFromWorld, mix };
}

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

/**
 * Apply the table's array→world placement to a cloud, on the host.
 *
 * `gramMatrixGpu` has **no placement facet**: it splats raw `xs`/`ys` straight against the `bbox`,
 * so points and bbox must already be in one space. (`splatDensity` does carry a placement and
 * applies it on the GPU per ADR-0018 — the Gram splat does not, and conflating the two is exactly
 * the bug this function exists to prevent: a world bbox with array-space points put the whole
 * tissue in a corner covering 1/scale of each axis.)
 *
 * World is the right space to land in rather than array: the NGFF scale is real information, and
 * every length the statistic takes is a world length.
 */
function placeCloud(t: CellTable, xs: readonly number[], ys: readonly number[]): { xs: number[]; ys: number[] } {
  const m = t.placement.worldFromArray;
  const n = xs.length;
  const wx = new Array<number>(n);
  const wy = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    wx[i] = m.origin[0] + m.axes[0][0] * xs[i]! + m.axes[1][0] * ys[i]!;
    wy[i] = m.origin[1] + m.axes[0][1] * xs[i]! + m.axes[1][1] * ys[i]!;
  }
  return { xs: wx, ys: wy };
}

/** TIGHT bounds — deliberately unpadded. A cosmetic margin would be indistinguishable from tissue
 *  that happens to be empty, and worse, it would silently swallow the window inset below: pad by 2%
 *  and then inset by a radius smaller than that, and the "inset" window still contains every cell,
 *  so the apron comes out empty while looking like it worked. */
function boundsOf(xs: readonly number[], ys: readonly number[]): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i]! < minX) minX = xs[i]!;
    if (xs[i]! > maxX) maxX = xs[i]!;
    if (ys[i]! < minY) minY = ys[i]!;
    if (ys[i]! > maxY) maxY = ys[i]!;
  }
  if (!Number.isFinite(minX)) return [0, 0, 1, 1];
  return [minX, minY, maxX, maxY];
}

function tableBounds(t: CellTable): [number, number, number, number] {
  const w = placeCloud(t, t.rowOrder?.xs ?? [], t.rowOrder?.ys ?? []);
  return boundsOf(w.xs, w.ys);
}

/**
 * The analysis window — a sub-rectangle of the data extent, with the cells outside it as the apron.
 *
 * Insetting is not cosmetic. `gramMatrixGpu` splats every cell it is handed but counts mass only
 * inside the `bbox`, so a window strictly inside the data gets an edge-corrected `g`; a window set
 * to the data's own extent has nothing outside to correct with and carries the full deficit
 * (0.97 at r=5 down to 0.82 at r=25 under CSR — `gram.test.ts`). An inset of one splat radius is
 * all it takes, which is why that is the default.
 *
 * This is also the shape the interactive camera wants: the window becomes the viewport, and the
 * apron becomes the cells just off-screen.
 */
function windowBbox(full: [number, number, number, number], radius: number): { bbox: [number, number, number, number]; label: string } {
  const [x0, y0, x1, y1] = full;
  const spanX = x1 - x0;
  const spanY = y1 - y0;
  const shrink = (fx: number, fy: number, label: string) => ({
    bbox: [x0 + spanX * fx, y0 + spanY * fy, x1 - spanX * fx, y1 - spanY * fy] as [number, number, number, number],
    label,
  });
  switch (windowSel.value) {
    case "full":
      return { bbox: full, label: "full extent (no apron)" };
    case "half":
      return shrink(0.25, 0.25, "centre 50%");
    case "quarter":
      return shrink(0.375, 0.375, "centre 25%");
    default: {
      // Inset by the splat radius on each side — the exact width of the halo the window's own
      // pixels can reach. Clamped so a huge range on a small tissue cannot invert the rectangle.
      const dx = Math.min(radius, spanX * 0.4);
      const dy = Math.min(radius, spanY * 0.4);
      return { bbox: [x0 + dx, y0 + dy, x1 - dx, y1 - dy], label: "inset by range" };
    }
  }
}

/**
 * Map a pointer position to raster `(col, row)`, honouring the canvas's `object-fit: contain`.
 *
 * The element box does NOT share the raster's aspect: a tall raster in the wide map tile hits the
 * tile's `max-height` and is then letterboxed — scaled to fit and CENTRED, with dead margins on the
 * long sides. The naive `(clientX − left) / width × resW` folds those margins into the image and
 * mislocates every sample (measured: a 316×466 raster drawn as a 420 px column inside a 1510 px box,
 * so X was off by up to ~3×). Aspect comes from the raster dims, which the canvas is drawn at, so
 * this is correct at any device-pixel ratio. Clamped rather than rejected, so a drag past the edge
 * samples the edge instead of jumping.
 */
function pointerToRaster(
  canvas: HTMLCanvasElement,
  resW: number,
  resH: number,
  clientX: number,
  clientY: number,
): { col: number; row: number; inside: boolean } {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / resW, rect.height / resH); // contain: the limiting axis wins
  const x = (clientX - rect.left - (rect.width - resW * scale) / 2) / scale;
  const y = (clientY - rect.top - (rect.height - resH * scale) / 2) / scale;
  const inside = x >= 0 && x <= resW && y >= 0 && y <= resH;
  return { col: Math.max(0, Math.min(resW, x)), row: Math.max(0, Math.min(resH, y)), inside };
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
  const cat = await listVars(await openSpatialData(storeInput.value.trim()), t.tableName);
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
    const typed = await readCellTable(await openSpatialData(storeInput.value.trim()), { table: t.tableName, typeColumn: col });
    const channels = typed.types
      .filter((ty) => ty.n > 0)
      .sort((a, b) => b.n - a.n)
      .slice(0, MAX_CHANNELS)
      .map((ty) => ({ label: ty.label ?? `type ${ty.id}`, ...placeCloud(t, ty.xs, ty.ys) }));
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
  const cols = await readVarColumns(await openSpatialData(storeInput.value.trim()), {
    table: t.tableName,
    vars: chosenVars,
    matrix: varMatrixSel.value || "X",
    names,
  });
  if (cols.nCells !== t.rowOrder.xs.length) {
    throw new Error(`X has ${cols.nCells} rows but the table has ${t.rowOrder.xs.length} centroids`);
  }
  const thin = cols.stats.filter((s) => s.nonZero < 20).map((s) => `${s.label} (${s.nonZero})`);
  const w = placeCloud(t, t.rowOrder.xs, t.rowOrder.ys);
  return {
    channels: channelsFromExpression(w.xs, w.ys, cols.values, cols.names),
    note: `${cols.names.length} vars from ${cols.matrix}${thin.length ? ` · sparse: ${thin.join(", ")}` : ""}`,
  };
}

// ---- the terrain and the wand --------------------------------------------------------------------

/** Redraw the displaced surface from rasters already on the device. No re-splat, no readback — so
 *  this is what every camera move, slider drag and colour change calls, at frame rate. */
function drawTerrain(): void {
  if (!live) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = terrainCanvas.clientWidth || 900;
  const cssH = Math.round((cssW * 9) / 16);
  terrainCanvas.width = Math.round(cssW * dpr);
  terrainCanvas.height = Math.round(cssH * dpr);
  const K = live.res.labels.length;
  void paintGramTerrain(terrainCanvas, live.res, {
    vectors: live.vectors,
    camera: cam,
    heightSource: heightSel.value as HeightSource,
    colourBy: colourBySel.value as ColourBy,
    heightScale: Number(hscaleInput.value) || 0,
    saturate: Number(satInput.value) || 2.5,
    chromaWeight: Number(chromaWInput.value),
    step: Number(stepSel.value) || 1,
    reference: wand?.z,
    // "all" is stored as 0 so the option does not have to know K at authoring time.
    modesUsed: Number(modesUsedSel.value) || K,
    distanceSpan: Number(dspanInput.value) || 1.2,
    marker: wand ? { col: wand.col, row: wand.row } : undefined,
    image: imageOverlay(),
  }).then((info) => {
    const m = Number(modesUsedSel.value) || K;
    terrainLegendEl.innerHTML =
      `drag to orbit · shift-drag to pan · wheel to dolly · ${info.gridW}×${info.gridH} grid, ` +
      `${(info.triangles / 1000).toFixed(0)}k triangles from the resident rasters — no re-splat. ` +
      (wand
        ? `Similarity is the <b>whitened distance</b> |Λ<sup>−½</sup>Vᵀ Δz| over ${m === K ? `all ${K}` : m} mode${m === 1 ? "" : "s"} — ` +
          `${m === K ? "full Mahalanobis in channel space" : "distance in exactly the space the colour shows"}, ` +
          `saturating at ${(Number(dspanInput.value) || 1.2).toFixed(1)}.`
        : `<span style="color:#fbbf24">No wand reference yet — click or drag on the mode map above; until then every similarity reads 0.</span>`);
  });
}

/**
 * Simulate the null spectrum and band the scree chart with it.
 *
 * Needs the CHANNELS, not the finished matrix — a permutation changes which cell carries which
 * mark, so every realisation is a fresh splat. They are rebuilt here rather than cached from the
 * last compute because the var selection may have moved since, and silently enveloping a spectrum
 * against a different channel set would be worse than refusing.
 */
async function runEnvelope(): Promise<void> {
  const t = base;
  if (!t || !live) {
    setStatus("compute modes first.", true);
    return;
  }
  const sims = Number(simsSelect.value) || 99;
  runEnvelopeBtn.disabled = true;
  try {
    const { channels } = await buildChannels(t);
    const full = tableBounds(t);
    const rangeUm = Number(radiusInput.value) || 1;
    const radius = equivalentRadius(kernelOf(), toWorld(rangeUm) / 2);
    const { bbox } = windowBbox(full, radius);
    const { width, height } = viewDims(bbox, Number(resSelect.value) || 384);

    const res = await spectrumEnvelope(
      channels,
      { bbox, width, height, radius, kernel: kernelOf() },
      {
        simulations: sims,
        onProgress: async (done, total) => {
          setStatus(`null spectrum — ${done}/${total} realisations …`);
          // Yield so the status actually paints; without this the page freezes for the whole run.
          await new Promise((r) => setTimeout(r, 0));
        },
      },
    );
    lastEnvelope = res;
    drawScree(res.observed, res);

    const out = envelopeExcursions(res);
    const verdict = res.envelope.exits
      ? `<b style="color:#86efac">outside the null band</b> — ${out
          .map((e) => `mode ${e.mode} ${e.above ? "above" : "below"} (${(e.observed * 100).toFixed(0)}% vs ${(e.bound * 100).toFixed(0)}%)`)
          .join(", ")}`
      : `<b style="color:#fbbf24">inside the null band</b> — this spectrum is what random labelling of the same cells produces`;
    screeLegendEl.innerHTML =
      `Grey band: the ${((1 - res.envelope.alpha) * 100).toFixed(0)}% global rank envelope over ${sims} random ` +
      `labellings — every cell kept in place, marks shuffled between them, one shuffle shared across all ` +
      `channels so within-cell co-expression survives and only the geography is destroyed. ` +
      `<b>p = ${res.envelope.p.toFixed(3)}</b> (smallest attainable ${(1 / (sims + 1)).toFixed(3)}) · ${verdict}.`;
    setStatus(`null spectrum done — ${sims} permutations in ${(res.elapsedMs / 1000).toFixed(1)} s.`);
  } catch (err) {
    setStatus(`null spectrum failed: ${err instanceof Error ? err.message : String(err)}`, true);
  } finally {
    runEnvelopeBtn.disabled = false;
  }
}

/** Repaint the flat mode map, carrying the crosshair. Cheap — one fullscreen fragment pass over
 *  rasters already on the device — which is what makes redrawing it on every drag step affordable. */
async function paintMap(): Promise<ModePaintInfo | null> {
  if (!live) return null;
  return paintGramModes(modeCanvas, live.res, {
    vectors: live.vectors,
    saturate: Number(satInput.value) || 2.5,
    chromaWeight: Number(chromaWInput.value),
    marker: wand ? { col: wand.col, row: wand.row } : undefined,
    image: imageOverlay(),
    // The selection outline needs the same reference and metric the terrain's similarity uses, or
    // the outline would enclose a different region than the colour shows.
    reference: wand?.z,
    modesUsed: Number(modesUsedSel.value) || live.res.labels.length,
    tolerance: Number(dspanInput.value) || 1.2,
  });
}

/**
 * Fetch one pyramid level of a store image and hold it as the context layer.
 *
 * The overlay is refused outright when the element carries no stored transform, or when the
 * transform's 2×2 cannot be inverted. Both cases would otherwise draw the tissue somewhere
 * plausible-looking and wrong, which is worse than drawing nothing — the whole value of the overlay
 * is that you can trust what lines up with what.
 */
async function loadImage(): Promise<void> {
  const element = imageSel.value;
  if (!element) return;
  imageNoteEl.textContent = `loading ${element} …`;
  try {
    const img = await loadContextImage(await openSpatialData(storeInput.value.trim()), element);
    const uv = uvFromWorld(img);
    if (!uv) {
      ctxImage = null;
      imageNoteEl.innerHTML = img.aligned
        ? `<span style="color:#f87171">${element}: placement is not invertible in XY — cannot align.</span>`
        : `<span style="color:#f87171">${element} carries no stored transform, so it cannot be put in the ` +
          `same coordinate system as the centroids. Not overlaid.</span>`;
    } else {
      ctxImage = { img, uvFromWorld: uv };
      imageNoteEl.textContent = `${img.label} · aligned via the store's own transform`;
    }
  } catch (err) {
    ctxImage = null;
    imageNoteEl.innerHTML = `<span style="color:#f87171">image failed: ${err instanceof Error ? err.message : String(err)}</span>`;
  }
  void paintMap();
  drawTerrain();
}

/** Turn a pointer position over the 2-D mode map into a wand reference: read the K channel
 *  densities at that pixel off the device and standardise them the same way `corr` was built. */
async function sampleWandAt(clientX: number, clientY: number): Promise<void> {
  if (!live) return;
  const { col, row } = pointerToRaster(modeCanvas, live.res.width, live.res.height, clientX, clientY);
  const raw = await probeChannels(live.res, col, row);
  const z = standardise(raw, live.res.resident.mean, live.res.resident.sd);
  wand = { z, col: Math.round(col), row: Math.round(row), label: `${Math.round(col)}, ${Math.round(row)}` };
  const K = live.res.labels.length;

  // Adaptive precision: a smoothed density for a gene detected in a few percent of cells is genuinely
  // ~1e-3, and a fixed 2 dp would print every one as "0.00" — which is exactly the density reading
  // the sample is meant to show. Exponential below a hundredth, plain above it.
  const fmtDens = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toExponential(1));

  // What the sampled profile IS, in the channels' own terms — the four channels furthest from
  // typical, signed, each with its raw kernel-smoothed density alongside the standardised z so the
  // reading is anchored to a real quantity and not only a count of σ.
  const top = live.res.labels
    .map((lab, a) => ({ lab, z: z[a]!, dens: raw[a]! }))
    .sort((p, q) => Math.abs(q.z) - Math.abs(p.z))
    .slice(0, 4)
    .map((p) => `${p.z >= 0 ? "+" : "−"}${p.lab} ${Math.abs(p.z).toFixed(1)}σ <span style="color:#64748b">(ρ ${fmtDens(p.dens)})</span>`)
    .join(" · ");

  // The eigen-projection: the sample's coordinate on each mode, yₖ = Σ_a zₐ V_{k,a}. This is what the
  // map's colour at that pixel literally is — mode 1 → L, modes 2–3 → the two chroma axes — so
  // showing it reads the pixel back in the very basis the picture is painted in.
  const proj = Array.from({ length: Math.min(3, K) }, (_, k) => {
    let y = 0;
    for (let a = 0; a < K; a++) y += z[a]! * live!.vectors[k * K + a]!;
    return `<span style="color:${modeColour(k)}">mode ${k + 1} ${y >= 0 ? "+" : "−"}${Math.abs(y).toFixed(2)}</span>`;
  }).join(" · ");

  const swatch = cssRgb(oklabToSrgb(similaritySwatch(1)));
  const html =
    `<b style="color:${swatch}">wand @ px ${wand.label}</b> — ${top}<br>` +
    `<span style="color:#94a3b8">eigen-projection</span> ${proj} — the pixel's colour, in the mode basis. ` +
    `<span style="color:#64748b">ρ = kernel-smoothed density; drag to move the sample, rule lines mark it in both views.</span>`;
  // Written to both the footer readout and the pinned overlay, from one string, so they cannot drift.
  wandReadoutEl.innerHTML = html;
  wandOverlayEl.innerHTML = html;
  wandOverlayEl.hidden = false;
  void paintMap();
  drawTerrain();
}

/**
 * Drag the sample around the map, live.
 *
 * Sampling is a GPU dispatch plus a buffer map, so it cannot keep up with pointer events one-to-one
 * and must not be allowed to queue: fifty pending readbacks would land in order and repaint the
 * terrain fifty times for positions the pointer left long ago. Instead the latest position is held
 * and one sample runs at a time — the classic coalescing loop, which drops intermediate positions
 * rather than falling behind. The final pointer position is always sampled, because the loop
 * re-checks after each readback.
 */
function attachWand(): void {
  let queued: { x: number; y: number } | null = null;
  let busy = false;
  const pump = async (): Promise<void> => {
    if (busy || !queued) return;
    busy = true;
    while (queued) {
      const at = queued;
      queued = null;
      await sampleWandAt(at.x, at.y);
    }
    busy = false;
  };
  const track = (e: PointerEvent) => {
    queued = { x: e.clientX, y: e.clientY };
    void pump();
  };
  modeCanvas.addEventListener("pointerdown", (e) => {
    modeCanvas.setPointerCapture(e.pointerId);
    track(e);
  });
  modeCanvas.addEventListener("pointermove", (e) => {
    // `buttons` rather than a flag of our own: pointer capture means we still see moves after the
    // button is released outside the canvas, and this is the state that says whether it is a drag.
    if (e.buttons & 1) track(e);
  });
  modeCanvas.addEventListener("pointerup", (e) => modeCanvas.releasePointerCapture(e.pointerId));
}

/** Orbit / pan / dolly. Elevation is clamped short of the poles: at exactly ±π/2 the view direction
 *  is parallel to the up vector and the basis degenerates. */
function attachCamera(): void {
  let dragging: "orbit" | "pan" | null = null;
  let lastX = 0;
  let lastY = 0;
  terrainCanvas.addEventListener("pointerdown", (e) => {
    dragging = e.shiftKey || e.button === 1 ? "pan" : "orbit";
    lastX = e.clientX;
    lastY = e.clientY;
    terrainCanvas.setPointerCapture(e.pointerId);
    terrainCanvas.style.cursor = "grabbing";
  });
  terrainCanvas.addEventListener("pointerup", (e) => {
    dragging = null;
    terrainCanvas.releasePointerCapture(e.pointerId);
    terrainCanvas.style.cursor = "grab";
  });
  terrainCanvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (dragging === "orbit") {
      const lim = Math.PI / 2 - 0.02;
      cam = {
        ...cam,
        azimuth: cam.azimuth + dx * 0.008,
        elevation: Math.max(-lim, Math.min(lim, cam.elevation - dy * 0.008)),
      };
    } else {
      // Pan in the camera's screen plane, scaled by distance so the grab point tracks the cursor at
      // any zoom. Only the horizontal component rotates with azimuth; vertical pan is along model Z
      // when looking down, which is what feels right on a terrain.
      const s = cam.distance * 0.0016;
      const ca = Math.cos(cam.azimuth);
      const sa = Math.sin(cam.azimuth);
      cam = {
        ...cam,
        target: [
          cam.target[0] - dx * s * ca - dy * s * sa * Math.sin(cam.elevation),
          cam.target[1] - dx * s * sa + dy * s * ca * Math.sin(cam.elevation),
          cam.target[2] + dy * s * Math.cos(cam.elevation),
        ],
      };
    }
    drawTerrain();
  });
  terrainCanvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      cam = { ...cam, distance: Math.max(0.3, Math.min(12, cam.distance * Math.exp(e.deltaY * 0.0012))) };
      drawTerrain();
    },
    { passive: false },
  );
}

// ---- charts ------------------------------------------------------------------------------------

const CELL = 20;

/** The OKLab axis colour for mode k (0,1,2) — the same swatch the map and scree use, so a loading
 *  row, its scree bar and its screen colour are one visual key rather than three. */
const modeColour = (k: number): string => cssRgb(oklabToSrgb(modeSwatch([k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0])));

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

function drawScree(explained: Float64Array, env?: SpectrumEnvelopeResult | null): void {
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
  // The band has to share the bar scale, or "outside the envelope" would be a claim about the
  // drawing rather than about the numbers.
  const peak = Math.max(
    ...Array.from({ length: n }, (_, i) => explained[i]!),
    ...(env ? Array.from({ length: n }, (_, i) => env.envelope.hi[i] ?? 0) : []),
    1e-9,
  );
  const yOf = (v: number) => H - pad - ((H - 2 * pad) * Math.max(0, v)) / peak;

  // Null band first, so the observed bars read as sitting against it.
  if (env) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.22)";
    for (let i = 0; i < n; i++) {
      const lo = yOf(env.envelope.lo[i] ?? 0);
      const hi = yOf(env.envelope.hi[i] ?? 0);
      ctx.fillRect(pad + i * bw + 1, hi, bw - 2, Math.max(lo - hi, 1));
    }
  }
  for (let i = 0; i < n; i++) {
    const y = yOf(explained[i]!);
    // Modes 1-3 are the ones the map uses; colour them as their OKLab axis so the chart and the
    // map are legible together rather than needing a mental key.
    const swatch: Srgb = i < 3 ? oklabToSrgb(modeSwatch([i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0])) : [0.35, 0.42, 0.53];
    ctx.fillStyle = cssRgb(swatch);
    // Bars drawn narrow when there is a band, so the grey shows on both sides of each.
    const inset = env ? 4 : 1;
    ctx.fillRect(pad + i * bw + inset, y, bw - 2 * inset, H - pad - y);
    // A mode outside the band is the finding; mark it rather than leaving it to be eyeballed.
    if (env && (explained[i]! > (env.envelope.hi[i] ?? 0) || explained[i]! < (env.envelope.lo[i] ?? 0))) {
      ctx.fillStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.arc(pad + i * bw + bw / 2, y - 6, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
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
  if (env) {
    ctx.fillStyle = "#cbd5e1";
    ctx.textAlign = "right";
    ctx.fillText(`null band ${((1 - env.envelope.alpha) * 100).toFixed(0)}% · p = ${env.envelope.p.toFixed(3)}`, W - pad, 16);
  }
}

/** The loadings chart's geometry, shared by the painter and the hover hit-test so the two cannot
 *  drift. `labelH` is the rotated channel-label gutter at the bottom. */
const LOAD = { W: 460, rowH: 54, pad: 46, right: 10, top: 20, labelH: 40 } as const;
const loadCol = (clientX: number, K: number): number => {
  const rect = loadingsCanvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * LOAD.W;
  const bw = (LOAD.W - LOAD.pad - LOAD.right) / K;
  return Math.floor((x - LOAD.pad) / bw);
};

function drawLoadings(labels: string[], vectors: Float64Array, hover: number | null): void {
  const K = labels.length;
  const rows = Math.min(3, K);
  const { W, rowH, pad, right, top, labelH } = LOAD;
  const H = top + rows * rowH + labelH;
  loadingsCanvas.width = W;
  loadingsCanvas.height = H;
  const ctx = loadingsCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const bw = (W - pad - right) / K;

  // The hovered channel's column, behind the bars, so a single gene is legible straight across all
  // three modes — "how does THIS gene load?" is the question the flat rows cannot answer on their own.
  if (hover !== null && hover >= 0 && hover < K) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.14)";
    ctx.fillRect(pad + hover * bw, top - 4, bw, rows * rowH + 4);
  }

  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  for (let k = 0; k < rows; k++) {
    const mid = top + k * rowH + rowH / 2;
    ctx.strokeStyle = "#334155";
    ctx.beginPath();
    ctx.moveTo(pad, mid);
    ctx.lineTo(W - right, mid);
    ctx.stroke();
    ctx.fillStyle = modeColour(k);
    ctx.textAlign = "right";
    ctx.fillText(`mode ${k + 1}`, pad - 6, mid + 3);
    for (let a = 0; a < K; a++) {
      const v = vectors[k * K + a]!;
      const h = v * (rowH / 2 - 4);
      ctx.fillRect(pad + a * bw + 1, mid - Math.max(h, 0), Math.max(bw - 2, 1), Math.abs(h));
    }
  }

  // Channel labels along the bottom, rotated so gene names fit. Thinned when the bars are too narrow
  // to carry every name without collision; the hover readout covers whatever is dropped.
  const everyN = bw >= 34 ? 1 : bw >= 16 ? 2 : Math.ceil(11 / bw);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const axisY = top + rows * rowH + 4;
  for (let a = 0; a < K; a++) {
    if (hover !== a && a % everyN !== 0) continue;
    ctx.save();
    ctx.translate(pad + a * bw + bw / 2, axisY);
    ctx.rotate(-Math.PI / 2.6);
    ctx.fillStyle = hover === a ? "#e2e8f0" : "#64748b";
    ctx.fillText(labels[a]!, 0, 0, labelH - 2);
    ctx.restore();
  }
  ctx.textBaseline = "alphabetic";
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
    const full = tableBounds(t);
    // The box is the EFFECTIVE range (support of J⊛J = 2·splat radius), so halve it for the splat,
    // then rescale per kernel so switching kernel holds the probed scale fixed rather than the
    // nominal radius (kernels.ts: μ₂ = r²/(n+2) shrinks with order).
    const rangeUm = Number(radiusInput.value) || toUm(Math.max(full[2] - full[0], full[3] - full[1]) / 40);
    radiusInput.value = rangeUm.toPrecision(3);
    const radius = equivalentRadius(kernelOf(), toWorld(rangeUm) / 2);
    // The window depends on the radius, and the raster on the window — so this order is forced.
    const { bbox, label: winLabel } = windowBbox(full, radius);
    const target = Number(resSelect.value) || 384;
    const { width, height } = viewDims(bbox, target);

    setStatus(`splatting ${channels.length} channels at ${width}×${height} …`);
    const t0 = performance.now();
    const res = await gramMatrixGpu(channels, { bbox, width, height, radius, kernel: kernelOf() });
    const modes = coLocationModes(res);

    // A new splat invalidates the old wand: `z` was standardised against the previous window's
    // means, and the pixel it came from may not even exist now. Dropping it is the honest move —
    // silently reusing it would compare against a reference that no longer means what it says.
    //
    // `live` is set BEFORE painting, and the paint goes through `paintMap` rather than calling
    // `paintGramModes` here. A second call site is how the image blend went missing on this path:
    // every option added to the map afterwards had to be remembered in two places, and this one was
    // not. One painter, one set of options.
    live = { res, vectors: modes.vectors };
    wand = null;
    wandReadoutEl.textContent = "no sample — click or drag on the mode map to set the wand reference";
    wandOverlayEl.hidden = true;
    const info = await paintMap();
    const ms = performance.now() - t0;
    drawTerrain();

    lastCorr = { labels: [...res.labels], corr: res.corr, g: res.g };
    drawCorr(lastCorr.labels, lastCorr.corr, null);
    lastEnvelope = null;
    drawScree(modes.explained, null);
    drawLoadings(res.labels, modes.vectors, null);

    const top = Array.from({ length: Math.min(3, res.labels.length) }, (_, k) => {
      const loads = res.labels
        .map((lab, a) => ({ lab, v: modes.vectors[k * res.labels.length + a]! }))
        .sort((x, y) => Math.abs(y.v) - Math.abs(x.v))
        .slice(0, 3)
        .map((x) => `${x.v >= 0 ? "+" : "−"}${x.lab}`)
        .join(" ");
      return `mode ${k + 1} (${(modes.explained[k]! * 100).toFixed(0)}%): ${loads}`;
    }).join(" · ");

    // Apron coverage: the ring density relative to the window's own. ~1 means the edge correction
    // had real cells to work with; 0 means the window is at the tissue boundary and g is biased.
    const cov = apronCoverage(res, radius);
    const covMean = cov.length ? [...cov].reduce((a, b) => a + b, 0) / cov.length : 0;
    const covNote =
      windowSel.value === "full"
        ? `<span style="color:#fbbf24">apron empty — g carries the ROI edge deficit</span>`
        : `apron coverage ${covMean.toFixed(2)}` +
          (covMean < 0.5 ? ` <span style="color:#fbbf24">(thin — window is near the tissue edge)</span>` : "");

    // Pixels per splat radius. Below ~2 the kernel is narrower than the sampling grid, so the map
    // and the terrain show aliasing rather than the smoothed field — a real and easily-missed
    // failure mode, since the statistic itself stays perfectly well defined.
    const pxPerRadius = radius / ((bbox[2] - bbox[0]) / width);
    const resNote =
      pxPerRadius < 2
        ? ` <span style="color:#fbbf24">kernel is ${pxPerRadius.toFixed(1)} px — under-resolved, widen the range or coarsen the raster</span>`
        : ` kernel ${pxPerRadius.toFixed(1)} px`;

    setStatus(`done — ${note}, ${ms.toFixed(0)} ms (submission).`);
    modeReadoutEl.innerHTML =
      `<b>${res.labels.length} channels</b> · ${kernelLabel(kernelOf())}, range ${rangeUm.toPrecision(3)}${unitSuffix()} · ` +
      `raster ${width}×${height},${resNote} · window ${winLabel}, ${covNote} · ` +
      `<b>PSD defect ${modes.psdDefect.toExponential(1)}</b> (0 = the eigenvalues are variances) · ${top}`;
    modeLegendEl.textContent =
      `L = mode 1, a = mode 2, b = mode 3, each saturating at ±${(Number(satInput.value) || 2.5).toFixed(1)}σ ` +
      `(σ = ${(info?.sigmas ?? [0, 0, 0]).map((s: number) => s.toFixed(2)).join(", ")}). Equal perceived colour ` +
      `distance ≈ equal distance between co-location profiles — that is what OKLab buys over three RGB ramps.`;
  } catch (err) {
    setStatus(`failed: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

// ---- load / inspect ------------------------------------------------------------------------------

async function inspectStore(autoLoad: boolean): Promise<void> {
  const url = storeInput.value.trim();
  setStatus(`inspecting ${url} …`);
  try {
    const sdata = await openSpatialData(url);
    storeTables = await listCellTables(sdata);
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
    // Image elements are listed but NOT fetched: naming them is metadata, and a level costs a
    // dozen chunk requests, so it waits for an explicit click.
    void listImageElements(sdata)
      .then((names) => {
        imageSel.innerHTML = "";
        for (const n of names) {
          const opt = document.createElement("option");
          opt.value = n;
          opt.textContent = n;
          imageSel.appendChild(opt);
        }
        imageNoteEl.textContent = names.length ? `${names.length} image element(s) — pick one and load` : "store has no images";
      })
      .catch((err) => {
        imageNoteEl.textContent = `image list failed: ${err instanceof Error ? err.message : String(err)}`;
      });
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
    base = await readCellTable(await openSpatialData(url), { table, typeColumn: NO_TYPE_COLUMN });
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
  const g = lastCorr.g[a * N + b]!;
  corrReadoutEl.innerHTML =
    `<b>${lastCorr.labels[a]}</b> vs <b>${lastCorr.labels[b]}</b> · spatial correlation r = ${r.toFixed(3)} — ` +
    (r > 0.2 ? "found in the same places" : r < -0.2 ? "spatially exclusive" : "little spatial relation") +
    ` · association g = ${g.toFixed(3)} (1 = complete spatial randomness) — <i>this</i> is the number the ` +
    `window's apron corrects; r and the modes never divide by mass, so they do not move with it.`;
});

// Loadings hover: light up one channel's column across all three modes and spell out its signed
// loading per mode, in the mode colours. This is what turns the bar chart from "some weights" into
// "gene X drives mode 1 and opposes mode 2", which is the reading the chart is for.
const LOADINGS_LEGEND = loadingsLegendEl.innerHTML;
loadingsCanvas.addEventListener("mousemove", (e) => {
  if (!live) return;
  const K = live.res.labels.length;
  const a = loadCol(e.clientX, K);
  if (a < 0 || a >= K) {
    drawLoadings(live.res.labels, live.vectors, null);
    loadingsLegendEl.innerHTML = LOADINGS_LEGEND;
    return;
  }
  drawLoadings(live.res.labels, live.vectors, a);
  const rows = Math.min(3, K);
  const parts = Array.from({ length: rows }, (_, k) => {
    const v = live!.vectors[k * K + a]!;
    return `<span style="color:${modeColour(k)}">mode ${k + 1} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}</span>`;
  });
  loadingsLegendEl.innerHTML = `<b>${live.res.labels[a]}</b> — ${parts.join(" · ")} (unit eigenvector components; sign is the co-variation direction).`;
});
loadingsCanvas.addEventListener("mouseleave", () => {
  if (live) drawLoadings(live.res.labels, live.vectors, null);
  loadingsLegendEl.innerHTML = LOADINGS_LEGEND;
});

attachWand();
runEnvelopeBtn.addEventListener("click", () => void runEnvelope());
imageLoadBtn.addEventListener("click", () => void loadImage());
imageMixInput.addEventListener("input", () => {
  // One control, both views — the blend is the shared state, not a per-view setting.
  void paintMap();
  drawTerrain();
});
// The metric controls drive BOTH views: they set the terrain's similarity shading and the map's
// selection outline, which are the same level set seen two ways. Anything that only changes the
// terrain's geometry stays on drawTerrain alone.
const redrawBoth = () => {
  void paintMap();
  drawTerrain();
};
for (const el of [heightSel, colourBySel, stepSel]) el.addEventListener("change", drawTerrain);
hscaleInput.addEventListener("input", drawTerrain);
modesUsedSel.addEventListener("change", redrawBoth);
dspanInput.addEventListener("input", redrawBoth);
// Chroma weighting shapes the colour of BOTH views, so it redraws both — the map's chroma axes and
// the terrain's colour mute together, exactly as saturate would if it were live.
chromaWInput.addEventListener("input", redrawBoth);
resetCamBtn.addEventListener("click", () => {
  cam = DEFAULT_CAM;
  drawTerrain();
});
window.addEventListener("resize", drawTerrain);
attachCamera();

// A swatch strip so the OKLab axes are readable without guessing.
modeLegendEl.style.borderLeft = `3px solid ${cssRgb(oklchToSrgbMapped([0.7, 0.12, deg(250)]))}`;
modeLegendEl.style.paddingLeft = "6px";

void inspectStore(true);
