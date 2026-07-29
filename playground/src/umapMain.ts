// UMAP as a live optimisation you can watch — the interactive half of
// `docs/umap-on-anndata.md` §4.
//
// The point of this page is NOT to show a finished embedding. It is to show the
// embedding *forming*, and to make visible how the layout relates to the graph driving
// it. Three things follow from that:
//
//   • **One embedding, continuously stepped.** Changing the feature subset rebuilds the
//     fuzzy graph and hands the optimiser the coordinates it already had, so the layout
//     relaxes into the new structure. Two independent UMAP runs differ by an arbitrary
//     rotation and reflection, so cross-fading between them would show a great deal of
//     motion that means nothing. Here every pixel of movement is the optimiser.
//   • **Speed is a first-class channel.** Colouring by per-point speed shows which
//     regions have settled and which are still negotiating — that is the "nature of the
//     change", and it is invisible in a static plot.
//   • **Stress relates layout back to graph.** For each point, the mean length of its
//     graph edges: where the 2-D projection cannot honour the high-dimensional
//     neighbourhood, stress stays high no matter how long it runs. That is the honest
//     answer to "how much should I trust this picture here?".
//
// **The generator matters as much as the algorithm.** The first version of this page drew
// isotropic Gaussian blobs, which is the one case where any projection works — the
// embedding settled into coloured dots and there was nothing to watch. The generators in
// `src/spatial/syntheticManifolds.ts` are shapes chosen so the embedding has something to
// get right or wrong: a rolled sheet to unroll, linked rings it cannot lay flat without
// tearing, a rare population it can lose, and a uniform-noise null control to calibrate
// the rest against. Real AnnData tables load through the same interface, with principal
// components standing in for gene programmes.
//
// Rendering is canvas 2-D on purpose. The GPU carries the k-NN (`knnGpu`) and the layout
// (`GpuUmapLayout`), which are the parts that are actually expensive; a 2-D context keeps
// the page readable as a demonstration of the algorithm rather than of a renderer. The
// heavy path is shared with the offline CLI — same `umapGraphFor`.

import { categoricalHues, deg, oklchToSrgbMapped, rgbBytes } from "../../src/color/ramps";
import { knnGpu } from "../../src/gpu/spatial/knn";
import { GpuUmapLayout } from "../../src/gpu/spatial/umapLayoutGpu";
import { knnDescentCpu } from "../../src/spatial/knnDescent";
import { MANIFOLDS } from "../../src/spatial/syntheticManifolds";
import { subsetColumns, umapGraphFor } from "../../src/spatial/umap";
import type { FuzzyGraph, KnnResult } from "../../src/spatial/umapGraph";
import { fitAB, initLayout, mulberry32, optimizeLayoutStep, reheatLayout } from "../../src/spatial/umapLayout";
import { inspectStore, labelColumnsOf, type StoreCatalog, storeDataset, syntheticDataset, type UmapDataset } from "./datasource/umapSource";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>("view");
const ctx2d = canvas.getContext("2d")!;
const banner = $<HTMLDivElement>("banner");

/** The layout, whichever backend is driving it.
 *
 *  Both keep ONE embedding and step it — the difference is only where the arithmetic
 *  happens. The GPU path keeps the coordinates resident and is ~80x faster per epoch at
 *  a few thousand cells (measured: 16.55 ms -> 0.20 ms at n=4000, 75k edges), which is
 *  what makes "several epochs per frame" affordable. The host path stays selectable
 *  because it is the reproducible one, and being able to flip between them on the same
 *  graph is the cheapest way to see that the racy kernel really does agree. */
interface LayoutDriver {
  step(epochs: number, opts: { initialAlpha: number; minDist: number }): void;
  /** Latest coordinates on the host. For the GPU driver this is the last readback. */
  coords(): Float32Array;
  /** Pull the device's current state into `coords()`. No-op on the host driver. */
  sync(): Promise<void>;
  readonly epoch: number;
  alphaNow(initialAlpha: number): number;
  reheat(epoch: number): void;
  dispose(): void;
}

let data: UmapDataset;
let active: boolean[] = [];
let graph: FuzzyGraph | undefined;
let knn: KnnResult | undefined;
let driver: LayoutDriver | undefined;
let running = true;
let rebuildToken = 0;
let busy = false;

/** Previous frame's coordinates, for the per-point speed channel. */
let previous: Float32Array | undefined;
let speed: Float32Array | undefined;
let stress: Float32Array | undefined;
/** True once the canvas holds a frame worth fading into. Cleared on resize and on a
 *  cold restart so trails never smear across a discontinuity. */
let canFade = false;
/** Rolling mean of the per-frame step cost, for the throughput readout. */
let stepMs = 0;

let catalog: StoreCatalog | undefined;

function showBanner(text: string, ms = 1600): void {
  banner.textContent = text;
  banner.classList.add("show");
  window.setTimeout(() => banner.classList.remove("show"), ms);
}

function setStatus(text: string): void {
  $<HTMLDivElement>("status").textContent = text;
}

// --- colour ---------------------------------------------------------------------------

/** Categorical colours generated rather than listed: a real `obs` column can have forty
 *  levels, and a fixed palette of eight would silently alias them onto each other — two
 *  different cell types drawn in the same colour is a wrong picture, not a cosmetic
 *  shortfall. Hues are spread in OKLCh at constant lightness and chroma, so no level looks
 *  more important than another. */
type Rgb = [number, number, number];
let paletteCache: { count: number; rgb: Rgb[] } | undefined;
function palette(count: number): Rgb[] {
  if (paletteCache?.count === count) return paletteCache.rgb;
  const rgb = categoricalHues(Math.max(1, count), deg(200)).map((h) => rgbBytes(oklchToSrgbMapped([0.74, 0.13, h])));
  paletteCache = { count, rgb };
  return rgb;
}

/** Blue → amber → red, for the scalar channels. */
function heat(t: number): Rgb {
  const u = Math.max(0, Math.min(1, t));
  return [
    Math.round(255 * Math.min(1, u * 2)),
    Math.round(255 * Math.min(1, Math.max(0, 1.4 - Math.abs(u - 0.5) * 2.2))),
    Math.round(255 * Math.max(0, 1 - u * 2)),
  ];
}

/** A wrapping ramp for the cyclic ground truths (a cycle phase, an angle round a ring).
 *  A linear ramp on an angle puts a hard colour discontinuity at an arbitrary place and
 *  invents a boundary that is not in the data. */
function cyclic(t: number): Rgb {
  return rgbBytes(oklchToSrgbMapped([0.72, 0.14, t * Math.PI * 2]));
}

// --- graph construction ---------------------------------------------------------------

/** Columns of the currently-enabled feature blocks. Order is stable so a toggle changes
 *  the feature set without permuting what stays. */
function activeColumns(): number[] {
  const cols: number[] = [];
  data.blocks.forEach((b, i) => {
    if (active[i]) cols.push(...b.columns);
  });
  return cols;
}

const LAYOUT_EPOCHS = 400;

/** `fitAB` is a 100-iteration least-squares fit; memoise it or it runs every frame. */
let abCache: { minDist: number; ab: { a: number; b: number } } | undefined;
function abFor(minDist: number): { a: number; b: number } {
  if (!abCache || abCache.minDist !== minDist) abCache = { minDist, ab: fitAB(minDist, 1) };
  return abCache.ab;
}

function hostDriver(g: FuzzyGraph, carried?: Float32Array): LayoutDriver {
  const state = initLayout(g, { dim: 2, nEpochs: LAYOUT_EPOCHS, seed: 7 }, carried);
  return {
    step(epochs, opts) {
      for (let i = 0; i < epochs; i++) {
        optimizeLayoutStep(state, g, { nEpochs: LAYOUT_EPOCHS, initialAlpha: opts.initialAlpha, seed: 7, ab: abFor(opts.minDist) });
      }
    },
    coords: () => state.embedding,
    sync: async () => {},
    get epoch() {
      return state.epoch;
    },
    alphaNow: (initialAlpha) => initialAlpha * Math.max(0, 1 - state.epoch / LAYOUT_EPOCHS),
    reheat: (epoch) => reheatLayout(state, epoch),
    dispose: () => {},
  };
}

async function gpuDriver(g: FuzzyGraph, carried?: Float32Array): Promise<LayoutDriver> {
  const gl = await GpuUmapLayout.create(g, { dim: 2, nEpochs: LAYOUT_EPOCHS, seed: 7 }, carried);
  // The device holds the truth; this is the host mirror the renderer draws from, refreshed
  // by `sync()` once per frame rather than once per epoch.
  let mirror: Float32Array<ArrayBufferLike> = carried ? Float32Array.from(carried) : new Float32Array(g.n * 2);
  return {
    step(epochs) {
      // `initialAlpha`/`minDist` are baked in at create time for this driver — changing
      // them mid-run rebuilds it (see `applyLayoutBackend`), which keeps the kernel's
      // uniform simple and the schedule consistent.
      gl.step(epochs);
    },
    coords: () => mirror,
    async sync() {
      mirror = await gl.read();
    },
    get epoch() {
      return gl.epoch;
    },
    alphaNow: () => gl.alphaAt(gl.epoch),
    reheat: (epoch) => gl.reheat(epoch),
    dispose: () => gl.destroy(),
  };
}

function gpuLayoutSelected(): boolean {
  return $<HTMLSelectElement>("layoutBackend").value === "gpu";
}

/**
 * The k-NN the graph build will use.
 *
 * Exact on the GPU is the default and stays correct to tens of thousands of points — it
 * tiles its dispatches, so the OS watchdog cannot silently truncate it. It is still
 * O(n²·D), and the page exposes the approximate path because past ~30k the quadratic term
 * is what you are waiting for, not the constant.
 */
function knnFn(): { fn: (d: ArrayLike<number>, n: number, dim: number, k: number) => Promise<KnnResult> | KnnResult; label: string } {
  const choice = $<HTMLSelectElement>("knnBackend").value;
  // "auto" is the default because the wrong choice is not a slowdown you shrug at: an
  // exact GPU search over the 162k-cell Xenium table takes over two minutes in the browser,
  // during which the page looks hung. The threshold is where the exact path still returns
  // in a few seconds here, well below the ~100k crossover measured offline in
  // `docs/umap-on-anndata.md` §3 — a page has a much tighter patience budget than a script.
  const useDescent = choice === "descent" || (choice === "auto" && data.n > 40000);
  return useDescent
    ? { fn: (d, n, dim, k) => knnDescentCpu(d, n, dim, { k, seed: 42 }), label: "host NN-descent" }
    : { fn: (d, n, dim, k) => knnGpu(d, { n, dim, k }), label: "GPU exact" };
}

async function rebuildGraph(carryEmbedding: boolean): Promise<void> {
  const token = ++rebuildToken;
  const cols = activeColumns();
  if (cols.length === 0) {
    showBanner("no feature blocks selected");
    return;
  }
  const nNeighbors = Number($<HTMLInputElement>("nNeighbors").value);
  if (data.n <= nNeighbors) {
    showBanner(`need more cells than n_neighbors (${data.n} <= ${nNeighbors})`);
    return;
  }
  const knnChoice = knnFn();
  busy = true;
  setStatus(`${knnChoice.label} k-NN over ${data.n.toLocaleString()} cells x ${cols.length} features…`);
  const t0 = performance.now();

  try {
    const features = subsetColumns(data.values, data.n, data.dim, cols);
    const built = await umapGraphFor(features, data.n, cols.length, {
      nNeighbors,
      // Features are already reduced by the source — latent gene blocks, or principal
      // components for a real table — so a second PCA here would be reducing a reduction
      // and would break the correspondence between a toggle and a column.
      pca: false,
      knn: knnChoice.fn,
    });
    // A slower rebuild must not clobber a newer one — the sliders can outpace the GPU.
    if (token !== rebuildToken) return;

    graph = built.graph;
    knn = built.knn;
    $<HTMLSpanElement>("sBuild").textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    $<HTMLSpanElement>("sKnn").textContent = knnChoice.label;
    $<HTMLSpanElement>("sEdges").textContent = graph.nEdges.toLocaleString();

    // Carry the coordinates across the swap — this is the whole animation model.
    const carried = carryEmbedding && driver && driver.coords().length === data.n * 2 ? Float32Array.from(driver.coords()) : undefined;
    driver?.dispose();
    driver = gpuLayoutSelected() ? await gpuDriver(graph, carried) : hostDriver(graph, carried);
    if (token !== rebuildToken) return;
    if (!carried) {
      previous = undefined;
      speed = undefined;
      stress = undefined;
      canFade = false;
    }
    setStatus("");
  } catch (err) {
    setStatus(`graph build failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    busy = false;
  }
}

// --- per-frame derived channels -------------------------------------------------------

function updateSpeed(): void {
  if (!driver) return;
  const emb = driver.coords();
  if (!speed || speed.length !== data.n) speed = new Float32Array(data.n);
  if (!previous || previous.length !== emb.length) {
    previous = Float32Array.from(emb);
    speed.fill(0);
    return;
  }
  let total = 0;
  for (let i = 0; i < data.n; i++) {
    const dx = emb[i * 2]! - previous[i * 2]!;
    const dy = emb[i * 2 + 1]! - previous[i * 2 + 1]!;
    const s = Math.hypot(dx, dy);
    // Smooth so the colour reads as "unsettled", not as single-frame jitter.
    speed[i] = speed[i]! * 0.85 + s * 0.15;
    total += s;
  }
  previous.set(emb);
  $<HTMLSpanElement>("sSpeed").textContent = (total / data.n).toFixed(4);
}

/** Mean embedded length of each point's k-NN edges — how far its high-dimensional
 *  neighbours ended up. Recomputed lazily; it changes slowly compared to speed. */
function updateStress(): void {
  if (!driver || !knn) return;
  if (!stress || stress.length !== data.n) stress = new Float32Array(data.n);
  const emb = driver.coords();
  const k = knn.k;
  for (let i = 0; i < data.n; i++) {
    let acc = 0;
    for (let t = 0; t < k; t++) {
      const j = knn.indices[i * k + t]!;
      acc += Math.hypot(emb[i * 2]! - emb[j * 2]!, emb[i * 2 + 1]! - emb[j * 2 + 1]!);
    }
    stress[i] = acc / Math.max(k, 1);
  }
}

// --- rendering ------------------------------------------------------------------------

function fitTransform(emb: Float32Array): { s: number; ox: number; oy: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < data.n; i++) {
    const x = emb[i * 2]!;
    const y = emb[i * 2 + 1]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = 40;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const s = Math.min(w / Math.max(maxX - minX, 1e-6), h / Math.max(maxY - minY, 1e-6));
  return { s, ox: pad + (w - (maxX - minX) * s) / 2 - minX * s, oy: pad + (h - (maxY - minY) * s) / 2 - minY * s };
}

/** Percentile of a scalar channel. */
function percentile(values: Float32Array, p: number): number {
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] || 1;
}

/** Map a stress value to the ramp as a RATIO to the median, not a fraction of the
 *  maximum. Absolute stress is meaningless — it scales with the whole embedding — and
 *  normalising by the max puts every point mid-ramp, which is what the first version did
 *  and why it read as uniform noise. Against the median, 1.0 is "as well embedded as a
 *  typical point" and the ramp only lights up where a neighbourhood is genuinely
 *  stretched. */
function stressRamp(value: number, median: number): number {
  return (value / Math.max(median, 1e-9) - 0.6) / 1.4;
}

/** Above this many points, `fillRect` per point stops being free: the per-call overhead
 *  dominates and the frame budget goes on the renderer rather than the optimiser, which
 *  would make the throughput readout a measurement of canvas, not of UMAP. Past it, points
 *  are written straight into an ImageData buffer. */
const RECT_DRAW_LIMIT = 20000;

let pixels: ImageData | undefined;

/** Per-point colour as bytes, so both draw paths share one decision. */
function colourOf(i: number, mode: string, speedScale: number, stressMedian: number, pal: Rgb[]): Rgb {
  if (mode === "motion") return heat((speed?.[i] ?? 0) / speedScale);
  if (mode === "stress") return heat(stressRamp(stress?.[i] ?? 0, stressMedian));
  if (mode === "truth" && data.truth) return data.truthCyclic ? cyclic(data.truth[i]!) : heat(data.truth[i]!);
  return pal[data.label[i]! % pal.length]!;
}

function draw(): void {
  const w = canvas.width;
  const h = canvas.height;
  const bulk = data && data.n > RECT_DRAW_LIMIT;
  const showTrails = $<HTMLInputElement>("showTrails").checked && !bulk;

  if (showTrails && canFade) {
    // Fade what is already on the canvas instead of clearing it, so moving points leave
    // a wake and settled ones do not. The canvas IS the trail buffer.
    ctx2d.fillStyle = "rgba(18,20,26,0.18)";
    ctx2d.fillRect(0, 0, w, h);
  } else if (!bulk) {
    ctx2d.fillStyle = "#12141a";
    ctx2d.fillRect(0, 0, w, h);
  }
  canFade = true;
  if (!driver) return;

  const emb = driver.coords();
  if (emb.length < data.n * 2) return;
  const { s, ox, oy } = fitTransform(emb);
  const mode = $<HTMLSelectElement>("colourBy").value;

  if ($<HTMLInputElement>("showEdges").checked && graph && !bulk) {
    ctx2d.strokeStyle = "rgba(122,162,247,0.10)";
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    // Only the i<j direction, and at most a few thousand, or this dominates the frame.
    const stride = Math.max(1, Math.floor(graph.nEdges / 6000));
    for (let e = 0; e < graph.nEdges; e += 2 * stride) {
      const a = graph.head[e]!;
      const b = graph.tail[e]!;
      ctx2d.moveTo(emb[a * 2]! * s + ox, emb[a * 2 + 1]! * s + oy);
      ctx2d.lineTo(emb[b * 2]! * s + ox, emb[b * 2 + 1]! * s + oy);
    }
    ctx2d.stroke();
  }

  const speedScale = mode === "motion" && speed ? percentile(speed, 0.95) : 1;
  const stressMedian = mode === "stress" && stress ? percentile(stress, 0.5) : 1;
  const pal = palette(data.labelNames.length);

  if (bulk) {
    if (!pixels || pixels.width !== w || pixels.height !== h) pixels = ctx2d.createImageData(w, h);
    const buf = pixels.data;
    // Background, then one pixel per point. Cheaper than 100k canvas calls by well over an
    // order of magnitude, at the cost of trails and edges — both of which are illegible at
    // this density anyway.
    for (let p = 0; p < buf.length; p += 4) {
      buf[p] = 0x12;
      buf[p + 1] = 0x14;
      buf[p + 2] = 0x1a;
      buf[p + 3] = 255;
    }
    for (let i = 0; i < data.n; i++) {
      const x = Math.round(emb[i * 2]! * s + ox);
      const y = Math.round(emb[i * 2 + 1]! * s + oy);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      const [r, g, b] = colourOf(i, mode, speedScale, stressMedian, pal);
      const p = (y * w + x) * 4;
      buf[p] = r;
      buf[p + 1] = g;
      buf[p + 2] = b;
    }
    ctx2d.putImageData(pixels, 0, 0);
    return;
  }

  for (let i = 0; i < data.n; i++) {
    const x = emb[i * 2]! * s + ox;
    const y = emb[i * 2 + 1]! * s + oy;
    const [r, g, b] = colourOf(i, mode, speedScale, stressMedian, pal);
    ctx2d.fillStyle = `rgb(${r},${g},${b})`;
    ctx2d.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
}

// --- main loop ------------------------------------------------------------------------

async function frame(): Promise<void> {
  if (running && driver && graph && !busy) {
    const steps = Number($<HTMLInputElement>("steps").value);
    const initialAlpha = Number($<HTMLInputElement>("alpha").value);
    const minDist = Number($<HTMLInputElement>("minDist").value);
    // `LAYOUT_EPOCHS` is the decay horizon, not a stopping point: the loop keeps running
    // and the rate keeps falling, so a settled layout stays put until something changes.
    const t0 = performance.now();
    driver.step(steps, { initialAlpha, minDist });
    // One readback per FRAME, not per epoch — the whole point of the resident buffer.
    await driver.sync();
    stepMs = stepMs * 0.9 + ((performance.now() - t0) / Math.max(1, steps)) * 0.1;
    updateSpeed();
    if (driver.epoch % 8 === 0) updateStress();
    $<HTMLSpanElement>("sEpoch").textContent = String(driver.epoch);
    $<HTMLSpanElement>("sAlpha").textContent = driver.alphaNow(initialAlpha).toFixed(3);
    $<HTMLSpanElement>("sStep").textContent = `${stepMs.toFixed(2)} ms`;
  }
  draw();
  requestAnimationFrame(() => void frame());
}

/** Rebuild the driver around new coordinates (the GPU one owns a device buffer, so a
 *  host-side perturbation has to be re-uploaded rather than written in place). */
async function restartWith(coords: Float32Array, epoch: number): Promise<void> {
  if (!graph) return;
  driver?.dispose();
  driver = gpuLayoutSelected() ? await gpuDriver(graph, coords) : hostDriver(graph, coords);
  driver.reheat(epoch);
}

// --- wiring ---------------------------------------------------------------------------

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  canFade = false;
  pixels = undefined;
}

function bindSlider(id: string, fmt: (v: number) => string, onChange?: () => void): void {
  const el = $<HTMLInputElement>(id);
  const out = $<HTMLSpanElement>(`${id}V`);
  const sync = () => {
    out.textContent = fmt(Number(el.value));
  };
  el.addEventListener("input", sync);
  if (onChange) el.addEventListener("change", onChange);
  sync();
}

/** Cell counts as a geometric ladder rather than a linear slider: the interesting range
 *  spans two and a half orders of magnitude, and a linear control spends nine tenths of
 *  its travel in sizes that all behave identically. */
const CELL_STEPS = [500, 1000, 2000, 4000, 8000, 16000, 32000];

function renderBlockButtons(): void {
  const host = $<HTMLDivElement>("blocks");
  host.textContent = "";
  data.blocks.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.textContent = b.name;
    btn.className = active[i] ? "on" : "";
    btn.title = `${b.columns.length} feature${b.columns.length === 1 ? "" : "s"} (columns ${b.columns[0]}–${b.columns[b.columns.length - 1]})`;
    btn.addEventListener("click", async () => {
      active[i] = !active[i];
      btn.className = active[i] ? "on" : "";
      showBanner(`${b.name} ${active[i] ? "on" : "off"} — relaxing`);
      await rebuildGraph(true);
    });
    host.appendChild(btn);
  });
}

/** Colour options depend on the dataset: a continuous ground truth only exists for the
 *  generators that have one, and offering a dead entry is worse than omitting it. */
function renderColourOptions(): void {
  const sel = $<HTMLSelectElement>("colourBy");
  const wanted = sel.value;
  sel.textContent = "";
  const add = (value: string, text: string) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    sel.appendChild(o);
  };
  add("cluster", data.kind === "synthetic" ? "ground-truth label" : "obs label");
  if (data.truth) add("truth", `${data.truthName ?? "ground truth"} (continuous)`);
  add("motion", "speed (how unsettled)");
  add("stress", "local stress (neighbours pulled far)");
  sel.value = [...sel.options].some((o) => o.value === wanted) ? wanted : "cluster";
}

function describeDataset(): void {
  $<HTMLDivElement>("datasetNote").textContent = data.note;
  $<HTMLDivElement>("expect").textContent = data.expect ?? "";
  $<HTMLDivElement>("expect").style.display = data.expect ? "block" : "none";
  $<HTMLSpanElement>("sCells").textContent = data.n.toLocaleString();
}

async function adopt(next: UmapDataset, carry: boolean): Promise<void> {
  data = next;
  active = data.blocks.map(() => true);
  if (!carry) {
    driver?.dispose();
    driver = undefined;
    previous = undefined;
    speed = undefined;
    stress = undefined;
  }
  paletteCache = undefined;
  renderBlockButtons();
  renderColourOptions();
  describeDataset();
  await rebuildGraph(false);
}

async function regenerate(): Promise<void> {
  const key = $<HTMLSelectElement>("generator").value;
  const n = CELL_STEPS[Number($<HTMLInputElement>("nCells").value)]!;
  const noise = Number($<HTMLInputElement>("noise").value);
  // 0 means "whatever the generator asked for" — each shape needs a different amount, and
  // a single default across all of them would blur the thin ones and glue the fat ones.
  await adopt(syntheticDataset(key, { n, seed: 11, noise: noise > 0 ? noise : undefined }), false);
}

// --- real data ------------------------------------------------------------------------

function fillSelect(sel: HTMLSelectElement, items: { value: string; text: string }[], keep = true): void {
  const wanted = sel.value;
  sel.textContent = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = it.value;
    o.textContent = it.text;
    sel.appendChild(o);
  }
  if (keep && items.some((i) => i.value === wanted)) sel.value = wanted;
}

function refreshStoreControls(): void {
  const table = $<HTMLSelectElement>("tableSel").value;
  const info = catalog?.tables.find((t) => t.name === table);
  const vars = catalog?.vars[table];
  fillSelect(
    $<HTMLSelectElement>("matrixSel"),
    (vars?.matrices ?? ["X"]).map((m) => ({ value: m, text: m })),
  );
  fillSelect($<HTMLSelectElement>("labelSel"), [
    { value: "", text: "(none)" },
    ...labelColumnsOf(info).map((c) => ({ value: c.name, text: `${c.name} (${c.nCategories})` })),
  ]);
  setStatus(
    vars?.error
      ? `${table}: ${vars.error}`
      : `${table}: ${vars?.nCells.toLocaleString() ?? "?"} cells x ${vars?.nVars ?? "?"} genes (${vars?.encoding})`,
  );
}

async function inspect(): Promise<void> {
  const url = $<HTMLInputElement>("store").value.trim();
  if (!url) return;
  busy = true;
  setStatus("inspecting store…");
  try {
    catalog = await inspectStore(url);
    const usable = catalog.tables.filter((t) => !catalog?.vars[t.name]?.error);
    if (usable.length === 0) throw new Error("no table in this store has a readable X");
    fillSelect(
      $<HTMLSelectElement>("tableSel"),
      usable.map((t) => ({ value: t.name, text: `${t.name} (${t.nRows.toLocaleString()} rows)` })),
      false,
    );
    refreshStoreControls();
  } catch (err) {
    setStatus(`inspect failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    busy = false;
  }
}

async function loadStore(): Promise<void> {
  const url = $<HTMLInputElement>("store").value.trim();
  const table = $<HTMLSelectElement>("tableSel").value;
  if (!url || !table) {
    showBanner("inspect a store first");
    return;
  }
  busy = true;
  try {
    const next = await storeDataset(url, {
      table,
      matrix: $<HTMLSelectElement>("matrixSel").value,
      maxGenes: Number($<HTMLInputElement>("maxGenes").value),
      nHvg: Number($<HTMLInputElement>("maxGenes").value),
      maxCells: Number($<HTMLInputElement>("maxCells").value),
      nComponents: Number($<HTMLInputElement>("nComps").value),
      log1p: $<HTMLInputElement>("log1p").checked,
      labelColumn: $<HTMLSelectElement>("labelSel").value || undefined,
      onProgress: setStatus,
    });
    busy = false;
    await adopt(next, false);
  } catch (err) {
    setStatus(`load failed: ${err instanceof Error ? err.message : String(err)}`);
    busy = false;
  }
}

function setSourceMode(mode: string): void {
  $<HTMLDivElement>("syntheticPanel").style.display = mode === "synthetic" ? "block" : "none";
  $<HTMLDivElement>("storePanel").style.display = mode === "store" ? "block" : "none";
}

async function main(): Promise<void> {
  window.addEventListener("resize", resize);
  resize();

  fillSelect(
    $<HTMLSelectElement>("generator"),
    MANIFOLDS.map((m) => ({ value: m.key, text: m.label })),
    false,
  );
  $<HTMLSelectElement>("generator").value = "branching";
  const describeGenerator = () => {
    const spec = MANIFOLDS.find((m) => m.key === $<HTMLSelectElement>("generator").value);
    $<HTMLDivElement>("generatorNote").textContent = spec?.describe ?? "";
  };
  describeGenerator();

  bindSlider("nCells", (v) => CELL_STEPS[v]!.toLocaleString());
  bindSlider("noise", (v) => (v > 0 ? v.toFixed(2) : "auto"));
  bindSlider(
    "nNeighbors",
    (v) => String(v),
    () => void rebuildGraph(true),
  );
  bindSlider("minDist", (v) => v.toFixed(2));
  bindSlider("alpha", (v) => v.toFixed(2));
  bindSlider("steps", (v) => String(v));

  $<HTMLSelectElement>("sourceMode").addEventListener("change", (e) => setSourceMode((e.target as HTMLSelectElement).value));
  setSourceMode($<HTMLSelectElement>("sourceMode").value);
  $<HTMLSelectElement>("generator").addEventListener("change", () => {
    describeGenerator();
    void regenerate();
  });
  $<HTMLButtonElement>("inspect").addEventListener("click", () => void inspect());
  $<HTMLSelectElement>("tableSel").addEventListener("change", refreshStoreControls);
  $<HTMLButtonElement>("loadStore").addEventListener("click", () => void loadStore());

  $<HTMLSelectElement>("layoutBackend").addEventListener("change", () => {
    showBanner(`layout on ${gpuLayoutSelected() ? "GPU" : "host"}`);
    void rebuildGraph(true);
  });
  $<HTMLSelectElement>("knnBackend").addEventListener("change", () => void rebuildGraph(true));
  $<HTMLButtonElement>("regen").addEventListener("click", () => void regenerate());
  $<HTMLButtonElement>("reseed").addEventListener("click", () => void rebuildGraph(false));
  $<HTMLButtonElement>("pause").addEventListener("click", (e) => {
    running = !running;
    (e.target as HTMLButtonElement).textContent = running ? "Pause" : "Resume";
  });
  $<HTMLButtonElement>("kick").addEventListener("click", () => {
    // Jitter the layout to show it re-converging — the cheapest way to see that what is
    // on screen is a live optimum rather than a stored picture.
    if (!driver || !graph) return;
    const jittered = Float32Array.from(driver.coords());
    const rnd = mulberry32(driver.epoch + 1);
    for (let t = 0; t < jittered.length; t++) jittered[t] = jittered[t]! + (rnd() - 0.5) * 6;
    // Rewind the anneal, or nothing happens: a settled layout has no learning rate left,
    // AND its per-edge sampling schedule has run past the horizon. Both drivers expose
    // `reheat` for exactly this — setting the epoch alone leaves every edge un-due and
    // the layout frozen.
    const at = Math.min(driver.epoch, 200);
    void restartWith(jittered, at);
    showBanner("kicked — watch it re-converge");
  });

  try {
    await navigator.gpu?.requestAdapter();
    $<HTMLSpanElement>("sBackend").textContent = navigator.gpu ? "WebGPU" : "unavailable";
  } catch {
    $<HTMLSpanElement>("sBackend").textContent = "unavailable";
  }

  await regenerate();
  void frame();
}

void main();
