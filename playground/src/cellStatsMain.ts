// Cell-type spatial statistics — a coordinated view over one cell table
// (docs/cell-stats.md, docs/muspan-cell-stats-plan.md).
//
// SOURCES. Four, all producing the same `CellTable`, so nothing downstream knows the difference:
// a SpatialData store (inspected first — tables and candidate type columns are enumerated, not
// hard-coded), a converted MDV project, a CSV export, or a synthetic fixture for offline use.
//
// The MDV path is the one that differs in kind rather than in format, and the difference is the ROI
// picker. An MDV cell table holds MANY regions — the covid project is 545,400 cells across 32 — and
// every statistic here is per-ROI: ρ_B is a density over one ROI, and the edge correction is against
// one ROI's boundary. Pooling the regions would not be a larger dataset, it would be a wrong one,
// measuring the gaps between tissue sections. See `datasource/mdvCellTable.ts`.
//
// STATISTICS. Every view is driven by ONE cell-type pair (A, B): the centroid scatter, a KDE of each
// type through the registered `splatDensity` op in a shared world frame, the TCM Γ_AB(x), the
// cross-PCF g_AB(r), and the N-way cross-PCF matrix. The matrix is the driver — hover to link every
// view, click to pin.
//
// Γ runs on the GPU by the RENDER formulation (`computeTcmRender`): B splatted through a radial
// kernel, then A splatted as Gaussians whose per-point weight is the transformed mark fetched from
// that first pass in the vertex stage. No neighbour search. The CPU oracle is one button away, and
// reporting the agreement between them is part of the demo rather than a hidden test.
//
// KERNEL. The paper's hard disk is n = 0 of a smoothness ladder (src/spatial/kernels.ts). Switching
// kernel holds the SPATIAL SCALE fixed via `equivalentRadius`, because μ₂ = r²/(n+2) shrinks with
// order — comparing at equal r would silently confound "smoother" with "more local".
//
// ORIENTATION. Every panel here draws world y DOWNWARD, because these are imaging coordinates: a
// cell's y is a row index in the section image, so minY belongs at the top and anything else shows
// the tissue upside down relative to its own source. The scatter does that naturally (canvas y grows
// down); the GPU field panels need `flipY`, because the raster convention is row 0 = maxY and
// `paintFieldTexture` would otherwise put row 0 at the top and render a y-UP plot. Without it the
// scatter and the KDE/Γ panels are mirror images of each other — measured on
// COVID_SAMPLE_16_ROI_3, the Fibroblast centre of mass sat at 0.536 down the scatter and 0.461 down
// the KDE, which is the same number reflected.
//
// COLOUR. Ramps are built in OKLCh (src/color/ramps.ts). Γ is signed and is read by comparing its
// arms, so the diverging ramp derives lightness from |Γ| alone and lets only the hue carry the sign:
// equal clustering and exclusion are then equally prominent by construction. An sRGB blue→white→red
// ramp cannot do that — its arms differ by >0.15 in perceived lightness.

import { cssRgb, deg, diverging, oklchToSrgbMapped, rgbBytes, type Srgb, sequential } from "../../src/color/ramps";
import { Graph, pullResident, registerBuiltinOps } from "../../src/gpu/graph";
import { browserBackend } from "../../src/gpu/graph/backend.browser";
import type { FieldValue, GpuField } from "../../src/gpu/graph/handle";
import { paintFieldTexture } from "../../src/gpu/spatial/paintField";
import { computeTcmRender, renderTcm, type TcmRenderParams } from "../../src/gpu/spatial/tcmRender";
import { equivalentRadius, KERNELS, type KernelSpec, kernelLabel } from "../../src/spatial/kernels";
import { crossPCF, crossPCFMatrix, type LabelledCells } from "../../src/spatial/pcf";
import { crossPCFEnvelope, type PcfNullModel } from "../../src/spatial/pcfEnvelope";
import { tcmKernelField } from "../../src/spatial/tcmKernel";
import { csvToCellTable, inspectCsv, parseCsv } from "./datasource/cellCsv";
import {
  type CellTable,
  type CellTypeCloud,
  DEFAULT_CELL_TABLE,
  listCellTables,
  openSpatialData,
  readCellTable,
  syntheticCellTable,
  type TableInfo,
} from "./datasource/cellTable";
import { inspectMdvTable, MdvStore, type MdvTableInfo, readMdvCellTable, spatialDatasources } from "./datasource/mdvCellTable";

/** The live Leap034 SpatialData store (zarr v3, CORS `*`). Configurable — not hard-coded downstream. */
const DEFAULT_STORE = "http://localhost:5055/project/289/spatial/leap034_layers.zarr/";
/** A converted MDV project (`pnpm mdv:zarr <project-dir> <out.zarr>`), served CORS-enabled. Like
 *  DEFAULT_STORE this is a convenience default, not a dependency — nothing downstream knows it. */
const DEFAULT_MDV_STORE = "http://localhost:5056/covid.mdv.zarr";
/** Raster sizes are given for the LONGER axis; the other follows the world box's aspect, so world
 *  cells stay square. A fixed square raster over a non-square ROI stretches every spatial view and
 *  — worse than ugly — makes the mark raster anisotropic, so the kernel is sampled at different
 *  resolutions in x and y. */
// 256, not 384: the tiles display at ~330 px wide under a 440 px height cap, so a 384-budget raster
// (181×814 on Leap034) is about twice the resolution anything can show — and every one of those
// pixels is paid for twice, in the de-pad loop and again in the ImageData write.
const SPLAT_TARGET = 256;
const SPLAT_RADIUS_SIGMA = 4;
const TCM_TARGET = 256;
/** The B-density raster the mark is sampled from. Finer than Γ on purpose: this is the one
 *  approximation the render formulation makes, and it is the knob that closes it. */
const MARK_TARGET = 512;
const SCATTER_TARGET = 900;
const PCF_BINS = 30;
const MATRIX_CELL = 22;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const storeInput = $<HTMLInputElement>("store");
const inspectBtn = $<HTMLButtonElement>("inspect");
const tableSelect = $<HTMLSelectElement>("tableSel");
const typeColSelect = $<HTMLSelectElement>("typeCol");
const runBtn = $<HTMLButtonElement>("run");
const fixtureBtn = $<HTMLButtonElement>("fixture");
const csvInput = $<HTMLInputElement>("csvFile");
const mdvStoreInput = $<HTMLInputElement>("mdvStore");
const mdvInspectBtn = $<HTMLButtonElement>("mdvInspect");
const mdvRegionSelect = $<HTMLSelectElement>("mdvRegion");
const mdvRunBtn = $<HTMLButtonElement>("mdvRun");
const typeSelect = $<HTMLSelectElement>("type");
const typeSelectB = $<HTMLSelectElement>("typeB");
const kernelSelect = $<HTMLSelectElement>("kernel");
const radiusInput = $<HTMLInputElement>("radius");
const sigmaInput = $<HTMLInputElement>("sigma");
const scaleInput = $<HTMLInputElement>("scale");
const unitsEl = $<HTMLSpanElement>("units");
const tcmBtn = $<HTMLButtonElement>("tcmBtn");
const oracleBtn = $<HTMLButtonElement>("oracleBtn");
const envToggle = $<HTMLInputElement>("envToggle");
const envNullSelect = $<HTMLSelectElement>("envNull");
const envSimsInput = $<HTMLInputElement>("envSims");
const envAlphaInput = $<HTMLInputElement>("envAlpha");
const statusEl = $<HTMLDivElement>("status");
const readoutEl = $<HTMLDivElement>("readout");
const scatterCanvas = $<HTMLCanvasElement>("scatter");
const kdeACanvas = $<HTMLCanvasElement>("kdeA");
const kdeBCanvas = $<HTMLCanvasElement>("kdeB");
const tcmCanvas = $<HTMLCanvasElement>("tcm");
const tcmReadoutEl = $<HTMLDivElement>("tcmReadout");
const pcfCanvas = $<HTMLCanvasElement>("pcf");
const pcfReadoutEl = $<HTMLDivElement>("pcfReadout");
const matrixCanvas = $<HTMLCanvasElement>("matrix");
const matrixReadoutEl = $<HTMLDivElement>("matrixReadout");

// ---- colour ------------------------------------------------------------------------------------

// A and B share a lightness so neither type looks more important than the other; only the hue
// differs. Same principle as the diverging ramp, applied to a categorical pair.
const A_HUE = deg(220);
const B_HUE = deg(70);
const A_CSS = cssRgb(oklchToSrgbMapped([0.74, 0.13, A_HUE]));
const B_CSS = cssRgb(oklchToSrgbMapped([0.74, 0.13, B_HUE]));
const CONTEXT_CSS = "rgba(148,163,184,0.16)";

/** Bake a ramp into a 256-entry byte LUT. Per-pixel OKLCh (cube roots, and a gamut bisection when
 *  the request is out of range) is far too slow for a 256² field every hover; the ramp is 1-D, so
 *  one table costs nothing and the colour is identical. */
function rampLut(f: (t: number) => Srgb, n = 256): Uint8Array {
  const lut = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = rgbBytes(f(i / (n - 1)));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}
/** Signed ramp: index 0 → t = −1, index 255 → t = +1.
 *
 *  Inverted against the library default: on a dark page the NEUTRAL should recede, so lightness
 *  rises with |Γ| instead of falling. The property that matters is untouched — L still depends on
 *  |Γ| alone and only the hue carries the sign, so the two arms stay equally prominent. */
const DIVERGING_LUT = rampLut((u) => diverging(2 * u - 1, { centreL: 0.19, endL: 0.8, endC: 0.17 }));
const SEQ_A_LUT = rampLut((u) => sequential(u, A_HUE));
const SEQ_B_LUT = rampLut((u) => sequential(u, B_HUE));

// ---- state -------------------------------------------------------------------------------------

let current: CellTable | null = null;
let lastMatrix: { types: number[]; counts: number[]; g: Float64Array } | null = null;
let hoverCell: { a: number; b: number } | null = null;
/** The last CLICKED cell — the view returns to this pair when the mouse leaves the matrix. */
let pinnedCell: { a: number; b: number } | null = null;
/** Wall time of the last KDE pair splat, for the throughput readout. */
let lastKdeMs = 0;
/** Discovery result for the current store, so picking a table can repopulate the column list. */
let storeTables: TableInfo[] = [];
/** Left/top gutter used by the last matrix draw (0 when there are no names to show). */
let matrixGutter = 0;
/** The most recent Γ, kept so the CPU-oracle check can compare against exactly what is on screen. */
let lastGamma: { idA: number; idB: number; ms: number; dims: { width: number; height: number }; params: TcmRenderParams } | null = null;

const kernelOf = (): KernelSpec => KERNELS[Number(kernelSelect.value)] ?? KERNELS[0]!;

/** What to call a cell type: its category name when the store gave one, else the bare id. */
const typeLabel = (ty: CellTypeCloud | undefined): string => (ty ? (ty.label ?? `type ${ty.id}`) : "—");
const labelOfId = (id: number): string => typeLabel(current?.types.find((t) => t.id === id));
/** True when at least one type carries a real name — drives whether the matrix gets label gutters. */
const hasNames = (t: CellTable | null): boolean => !!t?.types.some((x) => x.label !== undefined);

function setStatus(msg: string, err = false): void {
  statusEl.textContent = msg;
  statusEl.style.color = err ? "#fca5a5" : "#94a3b8";
}

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

const spanOf = (b: [number, number, number, number]) => Math.max(b[2] - b[0], b[3] - b[1], 1);
const defaultRadius = (b: [number, number, number, number]) => spanOf(b) / 50;
const defaultSigma = (b: [number, number, number, number]) => spanOf(b) / 100;
/** KDE bandwidth, scale-adaptive: a fixed small σ is a delta function against a ~10⁴-unit tissue
 *  extent, which renders as sparse dots rather than a density. */
const kdeSigma = (b: [number, number, number, number]) => spanOf(b) / 150;

/** Pixel dimensions whose aspect matches the world box, on an AREA budget of `target²`.
 *
 *  Square world cells, and canvases that carry their own aspect so CSS `width:100%` cannot stretch
 *  them. The budget is on area rather than the longer axis because Leap034's ROI is 0.22 aspect:
 *  sizing the long axis to 256 leaves 57 px across the short one, whereas the same pixel count
 *  spent proportionally gives 121×542. Clamped so a pathological aspect cannot ask for a texture
 *  the device will refuse. */
function viewDims(b: [number, number, number, number], target: number): { width: number; height: number } {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  if (!(w > 0) || !(h > 0)) return { width: target, height: target };
  const root = Math.sqrt(w / h);
  const clamp = (v: number) => Math.max(16, Math.min(4096, Math.round(v)));
  return { width: clamp(target * root), height: clamp(target / root) };
}

// ---- physical units ------------------------------------------------------------------------------

/** Micrometres per world unit currently in force. The store's value when it stated one; otherwise
 *  whatever the user has declared, defaulting to 1 and flagged in the HUD as an assumption. */
function umPerUnit(): number {
  const v = Number(scaleInput.value);
  return Number.isFinite(v) && v > 0 ? v : 1;
}
/** Lengths are ENTERED in µm (the paper speaks in µm: r = 100, σ = 50) and used in world units. */
const toWorld = (um: number) => um / umPerUnit();
const toUm = (world: number) => world * umPerUnit();

/** " µm" when the scale is trustworthy, " µm*" when it is the user's declaration rather than the
 *  store's. The asterisk is not decoration: a length reported in µm off an assumed scale is exactly
 *  the error this whole path exists to avoid, so it stays visible at every readout. */
const unitSuffix = () => (current?.units.micrometres !== undefined ? " µm" : " µm*");

/** Describe what the store did and did not say about scale, and set up the µm/unit box. */
function applyUnits(t: CellTable): void {
  const u = t.units;
  if (u.micrometres !== undefined) {
    scaleInput.value = String(u.micrometres);
    unitsEl.innerHTML = `<b>${u.raw}</b> — ${u.micrometres} µm per unit, from <code>${u.via}</code>. Lengths below are µm.`;
    unitsEl.style.color = "#86efac";
  } else {
    // Do NOT silently default to 1 µm/unit: "unknown scale" and "one micrometre" are different
    // states, and the second is a claim the store never made.
    scaleInput.value = scaleInput.value || "1";
    const said = u.raw ? `axes say "<b>${u.raw}</b>"` : "no axis metadata found";
    const via = u.via ? ` (<code>${u.via}</code>)` : "";
    unitsEl.innerHTML =
      `${said}${via} — <b>no physical scale stated</b>. Lengths are marked µm* and use the declared ` +
      `${scaleInput.value} µm/unit, which is an assumption, not metadata.`;
    unitsEl.style.color = "#fcd34d";
  }
  if (u.unsupported?.length) {
    unitsEl.innerHTML += ` ⚠ unhandled transform(s): ${u.unsupported.join(", ")} — placement approximate.`;
    unitsEl.style.color = "#fca5a5";
  }
}

/** World-bounds over ALL types (so scatter + every KDE + Γ share one frame). */
function tableBounds(t: CellTable): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const ty of t.types) {
    for (let i = 0; i < ty.xs.length; i++) {
      const x = ty.xs[i]!;
      const y = ty.ys[i]!;
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

/** The mark radius actually used: the input is a TOP-HAT-equivalent radius, rescaled per kernel so
 *  the probed spatial scale (μ₂) is held fixed. Without this, switching to a smoother kernel would
 *  quietly shrink the neighbourhood and look like a change in the biology. */
function markRadius(bbox: [number, number, number, number]): { baseUm: number; base: number; effective: number } {
  const baseUm = Number(radiusInput.value) || toUm(defaultRadius(bbox));
  const base = toWorld(baseUm);
  return { baseUm, base, effective: equivalentRadius(kernelOf(), base) };
}
/** Γ bandwidth, entered in µm. */
const sigmaUm = (bbox: [number, number, number, number]) => Number(sigmaInput.value) || toUm(defaultSigma(bbox));

// ---- views -------------------------------------------------------------------------------------

/** Scatter of every centroid, highlighting the current pair; all other cells faint. */
function drawScatterPair(t: CellTable, bbox: [number, number, number, number], idA: number, idB: number): void {
  const { width: W, height: H } = viewDims(bbox, SCATTER_TARGET);
  scatterCanvas.width = W;
  scatterCanvas.height = H;
  const ctx = scatterCanvas.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  const [minX, minY, maxX, maxY] = bbox;
  const sx = W / (maxX - minX || 1);
  const sy = H / (maxY - minY || 1);
  const draw = (xs: readonly number[], ys: readonly number[], style: string, rad: number) => {
    ctx.fillStyle = style;
    for (let i = 0; i < xs.length; i++) {
      const px = (xs[i]! - minX) * sx;
      const py = (ys[i]! - minY) * sy;
      ctx.fillRect(px - rad, py - rad, rad * 2, rad * 2);
    }
  };
  for (const ty of t.types) {
    if (ty.id === idA || ty.id === idB) continue;
    draw(ty.xs, ty.ys, CONTEXT_CSS, 0.8);
  }
  const A = t.types.find((x) => x.id === idA);
  const B = t.types.find((x) => x.id === idB);
  if (B && idB !== idA) draw(B.xs, B.ys, B_CSS, 1.6);
  if (A) draw(A.xs, A.ys, A_CSS, 1.8);
}

/** Line plot of g(r) vs r with a dashed g=1 (CSR) reference. */
function drawPcfCurve(c: HTMLCanvasElement, r: number[], g: number[], band?: { lo: Float64Array; hi: Float64Array }): void {
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
  // The band has to fit too, or a curve that exits upward would be clipped out of the picture —
  // exactly the case the envelope exists to show.
  if (band) for (const v of band.hi) if (Number.isFinite(v) && v > gMax) gMax = v;
  gMax = Math.ceil(gMax * 1.1);
  const px = (rv: number) => padL + (rv / rMax) * (W - padL - padR);
  const py = (gv: number) => H - padB - (gv / gMax) * (H - padB - padT);

  // Band first, so the observed curve is drawn over it and stays legible where it crosses.
  if (band) {
    ctx.fillStyle = "rgba(148,163,184,0.22)";
    ctx.beginPath();
    for (let i = 0; i < r.length; i++) ctx[i === 0 ? "moveTo" : "lineTo"](px(r[i]!), py(band.hi[i]!));
    for (let i = r.length - 1; i >= 0; i--) ctx.lineTo(px(r[i]!), py(band.lo[i]!));
    ctx.closePath();
    ctx.fill();
    // Points where the observed curve is outside get a marker: the test is "does it leave the band
    // anywhere", so where it leaves is the finding and should not need squinting.
    ctx.fillStyle = "#f87171";
    for (let i = 0; i < r.length; i++) {
      const v = g[i]!;
      if (v < band.lo[i]! || v > band.hi[i]!) {
        ctx.beginPath();
        ctx.arc(px(r[i]!), py(v), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

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
  ctx.strokeStyle = A_CSS;
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

/** N×N diverging heatmap of the cross-PCF matrix: log₂(g) through the OKLCh ramp, with hover/pin
 *  markers and — when the store gave us names — labelled axes. */
function drawMatrix(
  c: HTMLCanvasElement,
  res: { types: number[]; g: Float64Array },
  hover: { a: number; b: number } | null,
  pinned: { a: number; b: number } | null,
): void {
  const N = res.types.length;
  const cell = MATRIX_CELL;
  const names = hasNames(current) ? res.types.map(labelOfId) : null;
  const gut = names ? 96 : 0;
  matrixGutter = gut;
  c.width = gut + N * cell;
  c.height = gut + N * cell;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, c.width, c.height);
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const g = res.g[a * N + b]!;
      // log₂(g) clipped to ±2, i.e. a 4-fold enrichment or depletion saturates the ramp.
      const l = g > 0 ? Math.max(-2, Math.min(2, Math.log2(g))) / 2 : -1;
      const k = Math.max(0, Math.min(255, Math.round(((l + 1) / 2) * 255))) * 3;
      ctx.fillStyle = `rgb(${DIVERGING_LUT[k]},${DIVERGING_LUT[k + 1]},${DIVERGING_LUT[k + 2]})`;
      ctx.fillRect(gut + b * cell, gut + a * cell, cell, cell);
    }
  }
  if (names) {
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (let a = 0; a < N; a++) {
      ctx.textAlign = "right";
      ctx.fillText(names[a]!, gut - 5, gut + a * cell + cell / 2, gut - 8);
    }
    for (let b = 0; b < N; b++) {
      ctx.save();
      ctx.translate(gut + b * cell + cell / 2, gut - 5);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "left";
      ctx.fillText(names[b]!, 0, 0, gut - 8);
      ctx.restore();
    }
  }
  if (pinned) {
    ctx.strokeStyle = "#a3e635"; // lime dashed = pinned (where the view returns on mouse-leave)
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(gut + pinned.b * cell + 1, gut + pinned.a * cell + 1, cell - 2, cell - 2);
    ctx.setLineDash([]);
  }
  if (hover) {
    ctx.fillStyle = "rgba(226,232,240,0.14)";
    ctx.fillRect(gut, gut + hover.a * cell, N * cell, cell); // row A
    ctx.fillRect(gut + hover.b * cell, gut, cell, N * cell); // col B
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
    ctx.strokeRect(gut + hover.b * cell + 1, gut + hover.a * cell + 1, cell - 2, cell - 2);
  }
}

// ---- computations ------------------------------------------------------------------------------

/** Splat ONE type's cloud through the graph into its canvas. */
/** One graph for the whole session. Nothing is EXECUTED through it here — it exists so the demo can
 *  read back `splatDensity`'s inferred output placement, which is build-time information. */
const graph = new Graph();
registerBuiltinOps();
/** Points sources are per (type, table) and immutable — build each at most once. */
const sourceCache = new Map<string, GpuField>();

async function splatOne(
  t: CellTable,
  id: number,
  canvas: HTMLCanvasElement,
  lut: Uint8Array,
  bbox: [number, number, number, number],
): Promise<{ n: number; outSystem?: string } | null> {
  const ty = t.types.find((x) => x.id === id);
  if (!ty) return null;
  const g = graph;
  const key = `${t.tableName}/${t.typeColumn}/${id}`;
  let src = sourceCache.get(key);
  if (!src) {
    src = ty.source(g);
    sourceCache.set(key, src);
  }
  const dims = viewDims(bbox, SPLAT_TARGET);
  const sigma = kdeSigma(bbox);
  const density = g.op1("splatDensity", { points: src }, { ...dims, radiusSigma: SPLAT_RADIUS_SIGMA, sigma, bbox });
  // The graph produces a TEXTURE-resident value, which is exactly what the paint pass consumes —
  // so the op-graph feeds the display path with no copy, no readback and no adapter in between.
  const v = await pullResident(g, density, { ctx: { backend: browserBackend } });
  if (!v.texture) throw new Error("splatDensity did not return a texture-resident value");
  await paintFieldTexture(canvas, v.texture.texture, v.texture.width, v.texture.height, { lut, flipY: true });
  // The lease is the caller's (ADR-0017). Returning it AFTER submitting the paint is safe: queue
  // order means the paint reads the texture before anything the pool hands it to next writes it.
  browserBackend.releaseTexture(v.texture);
  return { n: ty.n, outSystem: density.placement?.system };
}

/** Splat BOTH types of the current pair — same op, same world frame, so A and B are directly
 *  comparable side by side (and share a lightness ramp, so brightness means density, not identity). */
async function splatPair(idA: number, idB: number): Promise<number> {
  const t = current;
  if (!t) return 0;
  const bbox = tableBounds(t);
  const t0 = performance.now();
  try {
    const A = await splatOne(t, idA, kdeACanvas, SEQ_A_LUT, bbox); // sequential: one GPU pull at a time
    const B = await splatOne(t, idB, kdeBCanvas, SEQ_B_LUT, bbox);
    readoutEl.innerHTML =
      `<b>${t.label}</b><br>` +
      `<span class="chipA">A = ${labelOfId(idA)}</span>: ${A?.n ?? "—"} cells &nbsp;·&nbsp; ` +
      `<span class="chipB">B = ${labelOfId(idB)}</span>: ${B?.n ?? "—"} cells · each panel auto-scaled by a GPU max reduction<br>` +
      `placement <b>${t.system}</b> (region "${t.provenance.region || "—"}", instance_key "${t.provenance.instanceKey}") · ` +
      `splatDensity output: ${A?.outSystem ? `system="${A.outSystem}"` : "absent"} (facet propagated) · KDE ${viewDims(bbox, SPLAT_TARGET).width}×${viewDims(bbox, SPLAT_TARGET).height}, σ=${toUm(kdeSigma(bbox)).toPrecision(3)}${unitSuffix()}`;
  } catch (e) {
    setStatus(`splat failed: ${(e as Error).message}`, true);
  }
  return performance.now() - t0;
}

/** Compute Γ_AB by the two-pass GPU render formulation and draw it. */
async function computeTcmMap(idA = Number(typeSelect.value), idB = Number(typeSelectB.value)): Promise<void> {
  const t = current;
  if (!t) return;
  const A = t.types.find((x) => x.id === idA);
  const B = t.types.find((x) => x.id === idB);
  if (!A || !B) return;
  const bbox = tableBounds(t);
  const { baseUm, effective } = markRadius(bbox);
  const sigma = toWorld(sigmaUm(bbox));
  const kernel = kernelOf();
  const dims = viewDims(bbox, TCM_TARGET);
  const mark = viewDims(bbox, MARK_TARGET);
  setStatus(`rendering Γ(${labelOfId(idA)} → ${labelOfId(idB)}) …`);
  try {
    const params = {
      ...dims,
      bbox,
      radius: effective,
      sigma,
      alpha: 5,
      kernel,
      markWidth: mark.width,
      markHeight: mark.height,
    };
    const t0 = performance.now();
    // Two render passes, then a paint straight from the target texture — Γ never crosses to the
    // host on the interactive path. The oracle button downloads on demand.
    const tex = await renderTcm({ xs: A.xs, ys: A.ys }, { xs: B.xs, ys: B.ys }, params);
    await paintFieldTexture(tcmCanvas, tex.tex, dims.width, dims.height, { lut: DIVERGING_LUT, signed: true, flipY: true });
    const ms = performance.now() - t0;
    lastGamma = { idA, idB, ms, dims, params };
    const u = unitSuffix();
    const scaleNote =
      kernel.kind === "poly" && kernel.order === 0
        ? `radius ${baseUm.toPrecision(3)}${u}`
        : `radius ${baseUm.toPrecision(3)}${u} → ${toUm(effective).toPrecision(3)}${u} (matched μ₂)`;
    tcmReadoutEl.innerHTML =
      `<b>Γ<sub>AB</sub></b> — A = ${labelOfId(idA)} (${A.n} cells), B = ${labelOfId(idB)} (${B.n} cells)<br>` +
      `${kernelLabel(kernel)} kernel, ${scaleNote}, σ ${sigmaUm(bbox).toPrecision(3)}${u}, α=5 · Γ ${dims.width}×${dims.height}, mark ${mark.width}×${mark.height} · 2 GPU render passes<br>` +
      `red = A associated with B here, blue = excluded; symmetric about 0, auto-scaled on the GPU` +
      `<br>live on hover: the whole pair — 2 KDE splats + Γ — <b>submitted in ${(lastKdeMs + ms).toFixed(1)} ms</b> of main-thread time. ` +
      `Nothing is read back, so this is submission, not GPU completion: the point is that the UI never blocks on the GPU. ` +
      `No debounce; requests coalesce.`;
    setStatus(`done — Γ rendered in ${ms.toFixed(0)} ms.`);
  } catch (e) {
    setStatus(`Γ render failed: ${(e as Error).message}`, true);
  }
}

/** Recompute the same Γ on the CPU, exactly (continuous marks, f64), and report the agreement.
 *
 *  This is the parity claim made visible: the render path approximates the mark by sampling a
 *  raster, and this says by how much, on the data actually on screen — rather than asking anyone to
 *  take a test's word for it. */
function checkAgainstOracle(): void {
  const t = current;
  const got = lastGamma;
  if (!t || !got) {
    setStatus("compute Γ first, then check it against the oracle.", true);
    return;
  }
  const A = t.types.find((x) => x.id === got.idA);
  const B = t.types.find((x) => x.id === got.idB);
  if (!A || !B) return;
  setStatus("computing the exact CPU oracle …");
  void (async () => {
    // The interactive path never downloads Γ, so the comparison re-runs the render with the SAME
    // params and takes the download here — the one place the numbers are actually consumed.
    // Time the DOWNLOADING variant here: it is the like-for-like comparison, since both sides have
    // to produce a host-side grid for the difference to be computable at all.
    const gpu0 = performance.now();
    const grid = await computeTcmRender({ xs: A.xs, ys: A.ys }, { xs: B.xs, ys: B.ys }, got.params);
    const gpuMs = performance.now() - gpu0;
    await new Promise((r) => requestAnimationFrame(() => r(null))); // let the status paint
    const t0 = performance.now();
    const { gamma } = tcmKernelField({ xs: A.xs, ys: A.ys }, { xs: B.xs, ys: B.ys }, { ...got.params, radiusSigma: 4 });
    const ms = performance.now() - t0;
    let peak = 0;
    let err = 0;
    for (let i = 0; i < gamma.length; i++) {
      peak = Math.max(peak, Math.abs(gamma[i]!));
      err = Math.max(err, Math.abs(grid[i]! - gamma[i]!));
    }
    const rel = err / Math.max(peak, 1e-30);
    tcmReadoutEl.innerHTML +=
      `<br><b>oracle check</b>: CPU exact ${ms.toFixed(0)} ms vs GPU ${gpuMs.toFixed(0)} ms including the download ` +
      `(<b>${(ms / Math.max(gpuMs, 0.01)).toFixed(1)}×</b>) · max |Δ| / peak = <b>${(100 * rel).toFixed(3)}%</b> ` +
      "— the render path samples the mark from a raster; the oracle evaluates it at each cell in f64.";
    setStatus(`oracle agreement ${(100 * rel).toFixed(3)}% of peak (CPU ${ms.toFixed(0)} ms, GPU ${got.ms.toFixed(0)} ms).`);
  })();
}

/** What each null holds fixed, for the readout — so the number on screen always says what it tested. */
const NULL_BLURB: Record<PcfNullModel, string> = {
  shift:
    "random toroidal shift of B — each pattern keeps its own clustering, only their relative position is destroyed, so this tests ASSOCIATION.",
  label:
    "random labelling within A∪B — positions fixed, A/B shuffled between them with n<sub>A</sub>, n<sub>B</sub> held. Tests whether the split is arranged; right for nested subsets, but two distinct self-clustered types fail it trivially.",
};

/** Newest envelope request. A pair hovered past while a simulation is running must not have its
 *  band painted onto the pair that replaced it, and there is no cancelling a synchronous loop — so
 *  the token is checked after it returns and a stale result is dropped. */
let envelopeToken = 0;

/** Compute + render the cross-PCF g_ab(r) curve for the selected pair (CPU, Mode 1).
 *
 *  `withEnvelope` is gated the same way Γ is, and for the same reason: the curve is a few
 *  milliseconds and the envelope is hundreds (199 relabellings of ~800k pairs at the panel's default
 *  270 µm range), so simulating on every mousemove would make the coordinated view stop feeling
 *  coordinated. The curve is drawn immediately either way and the band arrives after a yield. */
function computePcf(withEnvelope: boolean): void {
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
  const head =
    `<b>g<sub>AB</sub>(r)</b> — A = ${labelOfId(idA)} (${A.n} cells), B = ${labelOfId(idB)} (${B.n} cells)<br>` +
    `rMax ${rMax.toPrecision(3)} (world units), ${PCF_BINS} bins · g(r→0) = ${(res.g[0] ?? 0).toFixed(2)} · g&gt;1 clustering, g&lt;1 exclusion`;
  pcfReadoutEl.innerHTML = head;

  const token = ++envelopeToken;
  if (!withEnvelope || !envToggle.checked) return;
  if (idA === idB) {
    pcfReadoutEl.innerHTML = `${head}<br><span class="warn">no envelope for a self-pair — neither null has content when A and B are the same population.</span>`;
    return;
  }

  pcfReadoutEl.innerHTML = `${head}<br>simulating the random-labelling null …`;
  // Yield once so the curve paints before the simulation blocks the thread.
  setTimeout(() => {
    if (token !== envelopeToken) return;
    const sims = Math.max(19, Math.min(999, Math.round(Number(envSimsInput.value) || 199)));
    const alpha = Math.min(0.5, Math.max(0.001, Number(envAlphaInput.value) || 0.05));
    const nullModel = envNullSelect.value as PcfNullModel;
    try {
      const env = crossPCFEnvelope(
        { xs: A.xs, ys: A.ys },
        { xs: B.xs, ys: B.ys },
        { bbox, rMax, nBins: PCF_BINS, nullModel, simulations: sims, alpha, seed: 0x5eed },
      );
      if (token !== envelopeToken) return; // a newer pair won while this ran
      drawPcfCurve(pcfCanvas, res.r, res.g, env.envelope);
      const e = env.envelope;
      const floor = 1 / (sims + 1);
      const pStr = e.p <= floor + 1e-12 ? `≤ ${floor.toFixed(3)} (the floor at ${sims} simulations)` : e.p.toFixed(3);
      pcfReadoutEl.innerHTML =
        `${head}<br><b>global rank envelope</b> — p = ${pStr}, ${e.exits ? "<b>curve exits the band</b>" : "curve stays inside"} ` +
        `at α=${e.alpha} · ${sims} relabellings of ${env.pairs.toLocaleString()} pairs in ${env.simulateMs.toFixed(0)} ms<br>` +
        `<span class="note">Null: ${NULL_BLURB[nullModel]} ` +
        `The band is a whole-curve test, so leaving it <i>anywhere</i> is significant at α; a pointwise band drawn over the same simulations would not be.</span>`;
    } catch (err) {
      if (token !== envelopeToken) return;
      pcfReadoutEl.innerHTML = `${head}<br><span class="warn">envelope failed: ${(err as Error).message}</span>`;
    }
  }, 0);
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
  drawMatrix(matrixCanvas, res, hoverCell, pinnedCell);
  matrixReadoutEl.innerHTML =
    `<b>N-way cross-PCF</b> — ${res.types.length}×${res.types.length} (all ordered pairs), contact radius ${radius.toPrecision(3)} (world units) · ${ms.toFixed(0)} ms (one batched pass) · ` +
    `<b>hover a cell</b> to link the scatter · cross-PCF · Γ below (Γ on settle/click).`;
}

/** Point the linked views at a pair. `doTcm` gates the expensive ones. */
function setPair(idA: number, idB: number, doTcm: boolean): void {
  typeSelect.value = String(idA);
  typeSelectB.value = String(idB);
  const t = current;
  if (!t) return;
  drawScatterPair(t, tableBounds(t), idA, idB);
  // `doTcm` also gates the envelope: both are the "settled on this pair" work.
  computePcf(doTcm);
  if (doTcm) requestGpuViews(idA, idB);
}

/**
 * Drive the GPU views (KDE splats + Γ) at hover speed, without a debounce.
 *
 * The debounce this replaces existed because Γ used to be CPU work — seconds for the dominant cell
 * type. The render formulation put it at single-digit milliseconds, so waiting for the mouse to
 * settle now costs more than the compute does, and the coordinated view stops feeling coordinated.
 *
 * What replaces it is COALESCING rather than queueing: `wantedPair` holds the newest request, and
 * the drain loop repeats until what has been drawn is what was last asked for. Mousemove fires far
 * faster than a GPU round-trip completes, so a queue would spend all its time rendering pairs the
 * mouse left long ago; dropping the intermediates is both faster and more correct. WebGPU offers no
 * cancellation, so a request already in flight is allowed to finish — at worst one stale frame.
 */
let wantedPair: { idA: number; idB: number } | null = null;
let draining = false;

function requestGpuViews(idA: number, idB: number): void {
  wantedPair = { idA, idB };
  if (!draining) void drainGpuViews();
}

async function drainGpuViews(): Promise<void> {
  draining = true;
  try {
    while (wantedPair) {
      const { idA, idB } = wantedPair;
      wantedPair = null; // anything arriving from here on schedules another lap
      const kdeMs = await splatPair(idA, idB);
      await computeTcmMap(idA, idB);
      lastKdeMs = kdeMs;
    }
  } finally {
    draining = false;
  }
}

/** Map a mouse event on the matrix canvas to a `(row a, col b)` cell, or null if outside.
 *  The canvas is CSS-scaled and may carry a label gutter, so both have to be undone. */
function matrixCellFromEvent(e: MouseEvent): { a: number; b: number } | null {
  const res = lastMatrix;
  if (!res) return null;
  const N = res.types.length;
  const rect = matrixCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (matrixCanvas.width / rect.width) - matrixGutter;
  const y = (e.clientY - rect.top) * (matrixCanvas.height / rect.height) - matrixGutter;
  const b = Math.floor(x / MATRIX_CELL);
  const a = Math.floor(y / MATRIX_CELL);
  if (a < 0 || a >= N || b < 0 || b >= N) return null;
  return { a, b };
}

// ---- sources -----------------------------------------------------------------------------------

/** Fill the type A/B dropdowns and pick a sensible opening pair. */
function fillTypeSelects(t: CellTable): { idA: number; idB: number } {
  typeSelect.innerHTML = "";
  typeSelectB.innerHTML = "";
  for (const ty of t.types) {
    const label = `${typeLabel(ty)} — ${ty.n} cells`;
    for (const sel of [typeSelect, typeSelectB]) {
      const opt = document.createElement("option");
      opt.value = String(ty.id);
      opt.textContent = label;
      sel.appendChild(opt);
    }
  }
  // Default A = most populous, B = second (skipping degenerate n<=1 ids), for a legible first splat.
  const ranked = [...t.types].filter((x) => x.n > 1).sort((a, b) => b.n - a.n);
  const idA = (ranked[0] ?? t.types[0])?.id ?? 0;
  const idB = (ranked[1] ?? ranked[0] ?? t.types[0])?.id ?? idA;
  typeSelect.value = String(idA);
  typeSelectB.value = String(idB);
  return { idA, idB };
}

/** Take a freshly loaded table and drive every view from it. */
function present(t: CellTable): void {
  current = t;
  lastGamma = null;
  sourceCache.clear(); // type ids are only unique within a table
  const { idA, idB } = fillTypeSelects(t);
  applyUnits(t);
  const bbox = tableBounds(t);
  radiusInput.value = toUm(defaultRadius(bbox)).toPrecision(3);
  sigmaInput.value = toUm(defaultSigma(bbox)).toPrecision(3);
  setStatus(`read ${t.totalCells} cells in ${t.types.length} types — hover the N-way matrix to link every view.`);
  computeMatrix();
  // Pin the initial pair so leaving the matrix always has a home to return to.
  if (lastMatrix) {
    const a = lastMatrix.types.indexOf(idA);
    const b = lastMatrix.types.indexOf(idB);
    if (a >= 0 && b >= 0) {
      pinnedCell = { a, b };
      drawMatrix(matrixCanvas, lastMatrix, null, pinnedCell);
    }
  }
  setPair(idA, idB, true);
}

/** Populate the type-column dropdown for whichever table is selected. */
function fillColumnSelect(info: TableInfo | undefined): void {
  typeColSelect.innerHTML = "";
  if (!info) return;
  for (const col of info.columns) {
    const opt = document.createElement("option");
    opt.value = col.name;
    const kind = col.kind === "categorical" ? `${col.nCategories ?? col.categories?.length ?? "?"} categories` : "numeric";
    opt.textContent = `${col.name} — ${kind}`;
    // Columns the heuristic ruled out stay in the list but are visibly deprioritised: the guess is
    // only a default, and the user knows their data better than the regex does.
    if (col.score <= 0) opt.textContent += " (unlikely)";
    typeColSelect.appendChild(opt);
  }
  if (info.suggested) typeColSelect.value = info.suggested;
}

/**
 * Inspect the store: enumerate its tables and each one's candidate cell-type columns.
 *
 * Auto-selects when there is only one table (and its suggested column), because being made to pick
 * from a list of one is pure friction. With several, the list is shown and the best guess
 * pre-selected — the cell-type column is not standardised, so guessing-and-showing beats either
 * hard-coding a name or interrogating the user.
 */
async function inspectStore(autoLoad: boolean): Promise<void> {
  const url = storeInput.value.trim();
  setStatus(`inspecting ${url} …`);
  try {
    storeTables = await listCellTables(await openSpatialData(url));
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
    const chosen = usable[0]!;
    tableSelect.value = chosen.name;
    fillColumnSelect(chosen);
    const note =
      usable.length === 1
        ? `one table (${chosen.name}), selected automatically`
        : `${usable.length} usable tables — ${chosen.name} selected`;
    setStatus(`${note}; type column: ${chosen.suggested ?? "(none suggested — pick one)"}.`);
    if (autoLoad) await loadSelected();
  } catch (e) {
    setStatus(`inspect failed: ${(e as Error).message}\nUse "run fixture" or open a CSV to work offline.`, true);
  }
}

async function loadSelected(): Promise<void> {
  const url = storeInput.value.trim();
  const table = tableSelect.value || DEFAULT_CELL_TABLE;
  const typeColumn = typeColSelect.value || undefined;
  setStatus(`reading "${table}" (${typeColumn ?? "default column"}) …`);
  try {
    present(await readCellTable(await openSpatialData(url), { table, typeColumn }));
  } catch (e) {
    setStatus(`read failed: ${(e as Error).message}`, true);
  }
}

/** Open a CSV: parse, guess x/y/type from the headers (falling back to the shape of the data), and
 *  load it through the same path as a store. */
async function loadCsv(file: File): Promise<void> {
  setStatus(`parsing ${file.name} …`);
  try {
    const rows = parseCsv(await file.text());
    const schema = inspectCsv(rows);
    if (!schema.suggestedX || !schema.suggestedY || !schema.suggestedType) {
      setStatus(`could not identify x / y / type columns in ${file.name}. Headers: ${schema.headers.join(", ")}`, true);
      return;
    }
    const t = csvToCellTable(rows, {
      xColumn: schema.suggestedX,
      yColumn: schema.suggestedY,
      typeColumn: schema.suggestedType,
      system: file.name,
      label: file.name,
    });
    // Reflect the CSV's own columns in the pickers, so what was chosen is visible and not magic.
    tableSelect.innerHTML = `<option>${file.name}</option>`;
    typeColSelect.innerHTML = "";
    for (const c of schema.columns) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} — ${c.distinct} distinct`;
      typeColSelect.appendChild(opt);
    }
    typeColSelect.value = schema.suggestedType;
    present(t);
    setStatus(
      `${file.name}: x=${schema.suggestedX}, y=${schema.suggestedY}, type=${schema.suggestedType} — ${t.totalCells} cells in ${t.types.length} types.`,
    );
  } catch (e) {
    setStatus(`CSV failed: ${(e as Error).message}`, true);
  }
}

// ---- MDV projects ------------------------------------------------------------------------------

/** The opened MDV store and the spatial table chosen from it, held between inspect and load so the
 *  region dropdown can be populated without re-reading. */
let mdvStore: MdvStore | undefined;
let mdvTable: MdvTableInfo | undefined;

/** Open an MDV zarr store and report what it holds: spatial tables, their type columns, their ROIs.
 *  Metadata only — no column data — so this is fast even on the 545,400-row covid table. */
async function inspectMdv(): Promise<void> {
  const url = mdvStoreInput.value.trim();
  if (!url) return;
  setStatus(`opening ${url} …`);
  mdvRegionSelect.innerHTML = "";
  try {
    const store = await MdvStore.open(url);
    const spatial = spatialDatasources(store);
    if (spatial.length === 0) {
      setStatus(`${url}: no datasource declares position fields — nothing spatial to load.`, true);
      return;
    }
    const info = inspectMdvTable(spatial[0]!);
    mdvStore = store;
    mdvTable = info;

    // Reflect the MDV project's own axes in the shared pickers, so the choice is visible rather than
    // magic — same contract as the CSV path.
    tableSelect.innerHTML = "";
    for (const d of spatial) {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} — ${d.rows} rows`;
      tableSelect.appendChild(opt);
    }
    tableSelect.value = info.name;
    typeColSelect.innerHTML = "";
    for (const c of info.typeColumns) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} — ${c.nCategories} categories${c.score <= 0 ? " (unlikely)" : ""}`;
      typeColSelect.appendChild(opt);
    }
    if (info.suggested) typeColSelect.value = info.suggested;

    for (const r of info.regions) {
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r;
      mdvRegionSelect.appendChild(opt);
    }
    const unit = info.micrometres !== undefined ? `${info.micrometres} µm/unit (from regions.scale)` : "no physical scale declared";
    setStatus(
      `${info.name}: ${info.rows} rows · ${info.regions.length} ROIs on '${info.regionColumn}' · ` +
        `type column '${info.suggested ?? info.typeColumns[0]?.name}' · ${unit}. Pick an ROI and load.`,
    );
  } catch (e) {
    setStatus(`MDV open failed: ${(e as Error).message}`, true);
  }
}

/** Load the selected ROI of the selected MDV table through the same `present` path as every other
 *  source. */
async function loadMdvRegion(): Promise<void> {
  if (!mdvStore || !mdvTable) {
    setStatus("inspect an MDV store first.", true);
    return;
  }
  const region = mdvRegionSelect.value;
  const typeColumn = typeColSelect.value || mdvTable.suggested || mdvTable.typeColumns[0]?.name;
  if (!region || !typeColumn) {
    setStatus("no ROI or cell-type column selected.", true);
    return;
  }
  setStatus(`reading ${region} …`);
  try {
    const t = await readMdvCellTable(mdvStore, { datasource: tableSelect.value || mdvTable.name, typeColumn, region });
    present(t);
    setStatus(`${t.label} — hover the N-way matrix to link every view.`);
  } catch (e) {
    setStatus(`MDV read failed: ${(e as Error).message}`, true);
  }
}

// ---- wiring ------------------------------------------------------------------------------------

storeInput.value = DEFAULT_STORE;
mdvStoreInput.value = DEFAULT_MDV_STORE;
for (const [i, k] of KERNELS.entries()) {
  const opt = document.createElement("option");
  opt.value = String(i);
  opt.textContent = kernelLabel(k) + (i === 0 ? " (the paper)" : "");
  kernelSelect.appendChild(opt);
}
kernelSelect.value = "0";

inspectBtn.addEventListener("click", () => void inspectStore(false));
runBtn.addEventListener("click", () => void loadSelected());
fixtureBtn.addEventListener("click", () => {
  setStatus("building the synthetic 2-type fixture …");
  present(syntheticCellTable());
});
csvInput.addEventListener("change", () => {
  const f = csvInput.files?.[0];
  if (f) void loadCsv(f);
});
// Changing any envelope control re-runs it for the pair already on screen — otherwise the band
// shown and the settings displayed would disagree, which is the one thing a significance readout
// must never do.
for (const el of [envToggle, envNullSelect, envSimsInput, envAlphaInput]) {
  el.addEventListener("change", () => computePcf(true));
}
mdvInspectBtn.addEventListener("click", () => void inspectMdv());
mdvRunBtn.addEventListener("click", () => void loadMdvRegion());
// An ROI is one click, and re-reading is cheap (the columns are already fetched), so changing the
// ROI loads it rather than arming a second button.
mdvRegionSelect.addEventListener("change", () => void loadMdvRegion());
tableSelect.addEventListener("change", () => fillColumnSelect(storeTables.find((t) => t.name === tableSelect.value)));
typeSelect.addEventListener("change", () => setPair(Number(typeSelect.value), Number(typeSelectB.value), true));
typeSelectB.addEventListener("change", () => setPair(Number(typeSelect.value), Number(typeSelectB.value), true));
tcmBtn.addEventListener("click", () => void computeTcmMap());
oracleBtn.addEventListener("click", () => checkAgainstOracle());
kernelSelect.addEventListener("change", () => void computeTcmMap());
radiusInput.addEventListener("change", () => void computeTcmMap());
sigmaInput.addEventListener("change", () => void computeTcmMap());
scaleInput.addEventListener("change", () => {
  // Re-declaring the scale re-labels every length; the world-unit values are unchanged, so only
  // the entered µm figures need rescaling.
  const t = current;
  if (!t) return;
  applyUnits(t);
  const bbox = tableBounds(t);
  radiusInput.value = toUm(defaultRadius(bbox)).toPrecision(3);
  sigmaInput.value = toUm(defaultSigma(bbox)).toPrecision(3);
  void computeTcmMap();
});

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
  drawMatrix(matrixCanvas, res, hoverCell, pinnedCell);
  const rel = g > 1 ? `clustering (${g.toFixed(2)}×)` : g < 1 && g > 0 ? `exclusion (${g.toFixed(2)}×)` : "no co-location";
  matrixReadoutEl.innerHTML =
    `<span class="chipA"><b>A = ${labelOfId(idA)}</b></span> (${res.counts[cell.a]} cells) → ` +
    `<span class="chipB"><b>B = ${labelOfId(idB)}</b></span> (${res.counts[cell.b]} cells) · ` +
    `g = ${g.toFixed(3)} — ${rel} · scatter + cross-PCF live · Γ on settle · click to pin`;
  setPair(idA, idB, true);
});
matrixCanvas.addEventListener("mouseleave", () => {
  hoverCell = null;
  const res = lastMatrix;
  if (!res) return;
  drawMatrix(matrixCanvas, res, null, pinnedCell);
  if (!pinnedCell) return;
  // Snap the linked views back to the PINNED pair, so leaving the matrix doesn't strand you on
  // whatever cell the mouse happened to exit over.
  const N = res.types.length;
  const idA = res.types[pinnedCell.a]!;
  const idB = res.types[pinnedCell.b]!;
  const g = res.g[pinnedCell.a * N + pinnedCell.b]!;
  matrixReadoutEl.innerHTML =
    `<b>pinned</b> — <span class="chipA">A = ${labelOfId(idA)}</span> (${res.counts[pinnedCell.a]} cells) → ` +
    `<span class="chipB">B = ${labelOfId(idB)}</span> (${res.counts[pinnedCell.b]} cells) · g = ${g.toFixed(3)} · hover to explore, click to re-pin`;
  setPair(idA, idB, true);
});
matrixCanvas.addEventListener("click", (e) => {
  const cell = matrixCellFromEvent(e);
  if (!cell || !lastMatrix) return;
  pinnedCell = cell;
  drawMatrix(matrixCanvas, lastMatrix, hoverCell, pinnedCell);
  setPair(lastMatrix.types[cell.a]!, lastMatrix.types[cell.b]!, true);
});
matrixCanvas.style.cursor = "crosshair";

// Auto-inspect the default store on load and open its (single) table; the fixture and CSV paths are
// always available offline.
void inspectStore(true);
