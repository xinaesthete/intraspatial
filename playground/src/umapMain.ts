// UMAP as a live optimisation you can watch — the interactive half of
// `docs/umap-on-anndata.md` §4.
//
// The point of this page is NOT to show a finished embedding. It is to show the
// embedding *forming*, and to make visible how the layout relates to the graph driving
// it. Three things follow from that:
//
//   • **One embedding, continuously stepped.** Changing the gene subset rebuilds the
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
// Rendering is canvas 2-D on purpose. The GPU carries the k-NN (`knnGpu`), which is the
// part that is actually expensive; drawing a few thousand points is not, and a 2-D
// context keeps the page readable as a demonstration of the algorithm rather than of a
// renderer. The heavy path is shared with the offline CLI — same `umapGraphFor`.

import { knnGpu } from "../../src/gpu/spatial/knn";
import { GpuUmapLayout } from "../../src/gpu/spatial/umapLayoutGpu";
import { subsetColumns, umapGraphFor } from "../../src/spatial/umap";
import type { FuzzyGraph, KnnResult } from "../../src/spatial/umapGraph";
import { fitAB, initLayout, type LayoutState, mulberry32, optimizeLayoutStep, reheatLayout } from "../../src/spatial/umapLayout";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// --- synthetic dataset ------------------------------------------------------------
//
// Gene "programmes" rather than raw columns: each programme is a block of genes that is
// up-regulated in some subset of the clusters. That makes the subset controls mean
// something — switching a programme off genuinely removes a distinction the manifold
// could have used, which is the thing worth watching.

interface Programme {
  readonly name: string;
  readonly columns: number[];
  /** Which clusters this programme separates. */
  readonly marks: number[];
}

interface Dataset {
  values: Float32Array;
  n: number;
  dim: number;
  label: Uint8Array;
  programmes: Programme[];
}

const GENES_PER_PROGRAMME = 6;

function makeDataset(n: number, nClusters: number, seed = 11): Dataset {
  const rnd = mulberry32(seed);
  const gauss = () => {
    const u = Math.max(rnd(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  };

  // One programme per cluster, plus two that merge pairs — so switching programmes off
  // can genuinely collapse two clusters into one, not merely blur them.
  const programmes: Programme[] = [];
  for (let c = 0; c < nClusters; c++) {
    programmes.push({ name: `P${c + 1}`, columns: [], marks: [c] });
  }
  if (nClusters >= 4) {
    programmes.push({ name: "shared A", columns: [], marks: [0, 1] });
    programmes.push({ name: "shared B", columns: [], marks: [2, 3] });
  }
  let col = 0;
  for (const p of programmes) {
    for (let g = 0; g < GENES_PER_PROGRAMME; g++) p.columns.push(col++);
  }
  const dim = col;

  const label = new Uint8Array(n);
  const values = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const c = i % nClusters;
    label[i] = c;
    for (let t = 0; t < dim; t++) values[i * dim + t] = gauss() * 0.6;
    for (const p of programmes) {
      if (!p.marks.includes(c)) continue;
      for (const g of p.columns) values[i * dim + g] = values[i * dim + g]! + 6;
    }
  }
  return { values, n, dim, label, programmes };
}

// --- state ------------------------------------------------------------------------

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

let data: Dataset;
let active: boolean[] = [];
let graph: FuzzyGraph | undefined;
let knn: KnnResult | undefined;
let driver: LayoutDriver | undefined;
let running = true;
let rebuildToken = 0;

/** Previous frame's coordinates, for the per-point speed channel. */
let previous: Float32Array | undefined;
let speed: Float32Array | undefined;
let stress: Float32Array | undefined;
/** True once the canvas holds a frame worth fading into. Cleared on resize and on a
 *  cold restart so trails never smear across a discontinuity. */
let canFade = false;

let backendLabel = "…";

function showBanner(text: string, ms = 1200): void {
  banner.textContent = text;
  banner.classList.add("show");
  window.setTimeout(() => banner.classList.remove("show"), ms);
}

// --- graph construction -----------------------------------------------------------

/** Columns of the currently-enabled programmes. Order is stable so a toggle changes the
 *  feature set without permuting what stays. */
function activeColumns(): number[] {
  const cols: number[] = [];
  data.programmes.forEach((p, i) => {
    if (active[i]) cols.push(...p.columns);
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

async function rebuildGraph(carryEmbedding: boolean): Promise<void> {
  const token = ++rebuildToken;
  const cols = activeColumns();
  if (cols.length === 0) {
    showBanner("no gene programmes selected");
    return;
  }
  const nNeighbors = Number($<HTMLInputElement>("nNeighbors").value);
  const t0 = performance.now();

  const features = subsetColumns(data.values, data.n, data.dim, cols);
  const built = await umapGraphFor(features, data.n, cols.length, {
    nNeighbors,
    // The features here are already a handful of programmes; PCA would be reducing
    // almost nothing, and skipping it keeps the toggles legible.
    pca: false,
    knn: (d, n, dim, k) => knnGpu(d, { n, dim, k }),
  });
  // A slower rebuild must not clobber a newer one — the sliders can outpace the GPU.
  if (token !== rebuildToken) return;

  graph = built.graph;
  knn = built.knn;
  $<HTMLSpanElement>("sBuild").textContent = `${(performance.now() - t0).toFixed(0)} ms`;
  $<HTMLSpanElement>("sEdges").textContent = graph.nEdges.toLocaleString();

  // Carry the coordinates across the swap — this is the whole animation model.
  const carried = carryEmbedding && driver ? Float32Array.from(driver.coords()) : undefined;
  driver?.dispose();
  driver = gpuLayoutSelected() ? await gpuDriver(graph, carried) : hostDriver(graph, carried);
  if (token !== rebuildToken) return;
  if (!carried) {
    previous = undefined;
    canFade = false;
  }
}

// --- per-frame derived channels ---------------------------------------------------

function updateSpeed(): void {
  if (!driver) return;
  const emb = driver.coords();
  speed ??= new Float32Array(data.n);
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
  stress ??= new Float32Array(data.n);
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

// --- rendering --------------------------------------------------------------------

const CLUSTER_COLOURS = ["#7aa2f7", "#9ece6a", "#e0af68", "#f7768e", "#bb9af7", "#7dcfff", "#ff9e64", "#73daca"];

/** Blue → amber → red, for the scalar channels. */
function heat(t: number): string {
  const u = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, u * 2));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.4 - Math.abs(u - 0.5) * 2.2)));
  const b = Math.round(255 * Math.max(0, 1 - u * 2));
  return `rgb(${r},${g},${b})`;
}

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

function draw(): void {
  const w = canvas.width;
  const h = canvas.height;
  const showTrails = $<HTMLInputElement>("showTrails").checked;

  if (showTrails && canFade) {
    // Fade what is already on the canvas instead of clearing it, so moving points leave
    // a wake and settled ones do not. The canvas IS the trail buffer.
    ctx2d.fillStyle = "rgba(18,20,26,0.18)";
    ctx2d.fillRect(0, 0, w, h);
  } else {
    ctx2d.fillStyle = "#12141a";
    ctx2d.fillRect(0, 0, w, h);
  }
  canFade = true;
  if (!driver) return;

  const emb = driver.coords();
  const { s, ox, oy } = fitTransform(emb);
  const mode = $<HTMLSelectElement>("colourBy").value;

  if ($<HTMLInputElement>("showEdges").checked && graph) {
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

  for (let i = 0; i < data.n; i++) {
    const x = emb[i * 2]! * s + ox;
    const y = emb[i * 2 + 1]! * s + oy;
    let colour: string;
    if (mode === "motion") colour = heat((speed?.[i] ?? 0) / speedScale);
    else if (mode === "stress") colour = heat(stressRamp(stress?.[i] ?? 0, stressMedian));
    else colour = CLUSTER_COLOURS[data.label[i]! % CLUSTER_COLOURS.length]!;
    ctx2d.fillStyle = colour;
    ctx2d.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
}

// --- main loop --------------------------------------------------------------------

async function frame(): Promise<void> {
  if (running && driver && graph) {
    const steps = Number($<HTMLInputElement>("steps").value);
    const initialAlpha = Number($<HTMLInputElement>("alpha").value);
    const minDist = Number($<HTMLInputElement>("minDist").value);
    // `LAYOUT_EPOCHS` is the decay horizon, not a stopping point: the loop keeps running
    // and the rate keeps falling, so a settled layout stays put until something changes.
    driver.step(steps, { initialAlpha, minDist });
    // One readback per FRAME, not per epoch — the whole point of the resident buffer.
    await driver.sync();
    updateSpeed();
    if (driver.epoch % 8 === 0) updateStress();
    $<HTMLSpanElement>("sEpoch").textContent = String(driver.epoch);
    $<HTMLSpanElement>("sAlpha").textContent = driver.alphaNow(initialAlpha).toFixed(3);
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

// --- wiring -----------------------------------------------------------------------

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  canFade = false;
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

function renderGeneButtons(): void {
  const host = $<HTMLDivElement>("genes");
  host.textContent = "";
  data.programmes.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = p.name;
    b.className = active[i] ? "on" : "";
    b.title = `genes ${p.columns[0]}-${p.columns[p.columns.length - 1]}, separates cluster(s) ${p.marks.map((m) => m + 1).join(", ")}`;
    b.addEventListener("click", async () => {
      active[i] = !active[i];
      b.className = active[i] ? "on" : "";
      showBanner(`${p.name} ${active[i] ? "on" : "off"} — relaxing`);
      await rebuildGraph(true);
    });
    host.appendChild(b);
  });
}

async function regenerate(): Promise<void> {
  const n = Number($<HTMLInputElement>("nCells").value);
  const clusters = Number($<HTMLInputElement>("nClusters").value);
  data = makeDataset(n, clusters);
  active = data.programmes.map(() => true);
  driver?.dispose();
  driver = undefined;
  previous = undefined;
  renderGeneButtons();
  await rebuildGraph(false);
}

async function main(): Promise<void> {
  window.addEventListener("resize", resize);
  resize();

  bindSlider("nCells", (v) => String(v));
  bindSlider("nClusters", (v) => String(v));
  bindSlider(
    "nNeighbors",
    (v) => String(v),
    () => void rebuildGraph(true),
  );
  bindSlider("minDist", (v) => v.toFixed(2));
  bindSlider("alpha", (v) => v.toFixed(2));
  bindSlider("steps", (v) => String(v));

  $<HTMLSelectElement>("layoutBackend").addEventListener("change", () => {
    showBanner(`layout on ${gpuLayoutSelected() ? "GPU" : "host"}`);
    void rebuildGraph(true);
  });
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
    backendLabel = navigator.gpu ? "GPU (WebGPU)" : "unavailable";
  } catch {
    backendLabel = "unavailable";
  }
  $<HTMLSpanElement>("sBackend").textContent = backendLabel;

  await regenerate();
  void frame();
}

void main();
