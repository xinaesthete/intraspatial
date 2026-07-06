// Plain-TS host for the Decision view (no React — the "thin illustrative host shell"
// of ADR-0008 §8). Wires the HTML controls to a framework-free DualView over a
// synthetic source, and renders the live fetch-budget HUD.
import Stats from "three/examples/jsm/libs/stats.module.js";
import { WebGPURenderer } from "three/webgpu";
import { type SyntheticSource, syntheticPlane, syntheticVolume } from "../../src/datasource";
import type { GpuBackend } from "../../src/gpu/graph/backend";
import { adoptDevice } from "../../src/gpu/interop";
import { type DecisionStats, DualView } from "./datasource/dualView";
import { levelColor } from "./datasource/overlays";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("stage");
const inset = $("inset");
const insetLabel = inset.querySelector<HTMLElement>(".inset-label");
let decisionCollapsed = false;
const qSlider = $<HTMLInputElement>("q");
const qVal = $("qval");
const pLevels = $<HTMLInputElement>("plevels");
const pLevelsVal = $("plevelsval");
const baseRes = $<HTMLSelectElement>("baseres");
const network = $<HTMLSelectElement>("network");

// Simulated network presets: [base ms, jitter ms]. Jitter staggers arrivals into a visible
// loading wave; without it every chunk of a selection would pop in at once.
const NET_PRESETS: Record<string, [number, number]> = {
  off: [0, 0],
  fast: [60, 80],
  "3g": [300, 350],
  slow: [1200, 900],
};
function applyNetwork(): void {
  const [base, jitter] = NET_PRESETS[network.value] ?? [0, 0];
  view?.setNetwork(base, jitter);
}
const texChk = $<HTMLInputElement>("tex");
const gridChk = $<HTMLInputElement>("grid");
const cmin = $<HTMLInputElement>("cmin");
const cmax = $<HTMLInputElement>("cmax");
const gamma = $<HTMLInputElement>("gamma");
const cminVal = $("cminval");
const cmaxVal = $("cmaxval");
const gammaVal = $("gammaval");
const solid = $<HTMLInputElement>("solid");
const solidVal = $("solidval");
const dread = $<HTMLInputElement>("dread");
const dwrite = $<HTMLInputElement>("dwrite");
const sourceSel = $<HTMLSelectElement>("source");

function applyTransfer(): void {
  cminVal.textContent = Number(cmin.value).toFixed(2);
  cmaxVal.textContent = Number(cmax.value).toFixed(2);
  gammaVal.textContent = Number(gamma.value).toFixed(2);
  view?.setTransfer(Number(cmin.value), Number(cmax.value), Number(gamma.value));
}
const bytesEl = $("bytes");
const chunksEl = $("chunks");
const levelsEl = $("levels");
const errEl = $("err");

const stats = new Stats();
stats.dom.style.left = "auto";
stats.dom.style.right = "12px";
document.body.appendChild(stats.dom);

const fmtBytes = (n: number): string => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

function renderStats(s: DecisionStats, levelCount: number): void {
  bytesEl.textContent = fmtBytes(s.totalBytes);
  chunksEl.textContent = String(s.chunkCount);
  levelsEl.replaceChildren();
  s.countByLevel.forEach((count, level) => {
    if (!count) return;
    const chip = document.createElement("span");
    chip.className = "lv";
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = `#${levelColor(level, levelCount - 1).getHexString()}`;
    const txt = document.createElement("span");
    txt.textContent = `L${level}: ${count}`;
    chip.append(sw, txt);
    levelsEl.append(chip);
  });
}

let view: DualView | null = null;
let renderer: WebGPURenderer | null = null;
let backend: GpuBackend | null = null;

// Adopt the renderer's own WebGPU device so our compute ops (the Mandelbulb brick generator)
// run on the SAME device as the atlas texture — sharing GPU memory without readback is a
// project priority (see AGENTS.md; src/gpu/interop/adoptDevice.ts). Cached: one device/renderer.
function getBackend(r: WebGPURenderer): GpuBackend {
  if (backend) return backend;
  const device = (r as unknown as { backend?: { device?: GPUDevice } }).backend?.device;
  if (!device) throw new Error("three WebGPURenderer has no backend device yet (call after init())");
  backend = adoptDevice(device, "three");
  return backend;
}

// One renderer/device for the whole page — reused across source switches (re-creating it
// per switch leaks GPU devices until WebGPU stops handing out new ones → black canvas).
async function getRenderer(): Promise<WebGPURenderer | null> {
  if (renderer) return renderer;
  if (!("gpu" in navigator)) {
    errEl.textContent = "WebGPU is not available in this view. Open http://localhost:5188/datasource.html in Chrome or Edge.";
    return null;
  }
  const r = new WebGPURenderer({ canvas, antialias: true, alpha: true });
  try {
    await r.init();
  } catch (e) {
    errEl.textContent = `WebGPU failed to start: ${e instanceof Error ? e.message : String(e)}. Try a hard reload, or open in Chrome/Edge.`;
    return null;
  }
  renderer = r;
  return r;
}

const VOL_CHUNK = 32;
const VOL_WORLD_EXTENT = 256; // hold the volume at a constant world size, so a finer base is the
// SAME object sampled more finely (voxels shrink) rather than a bigger one.

// The finest (level-0) resolution the pyramid samples the analytic Mandelbulb at. Pinning this
// at 256³ before meant more pyramid *levels* only stacked coarser tiers — you could never resolve
// finer than 256³. This knob raises the finest level so zooming in reveals genuinely finer
// structure. Whole-volume views of a finer base demand more chunks than the atlas holds at once
// (that's what the LOD fallback, step 3, makes graceful); zooming in stays cheap via frustum cull.
function makeVolumeSource(levelCount: number, baseRes: number): SyntheticSource {
  return syntheticVolume({ size: baseRes, chunk: VOL_CHUNK, levelCount, voxelSizeWorld: VOL_WORLD_EXTENT / baseRes });
}

async function mount(kind: string): Promise<void> {
  view?.dispose();
  errEl.textContent = "";
  const r = await getRenderer();
  if (!r) return;
  const levelCount = Number(pLevels.value);
  const combined = kind === "combined";
  const naive = kind === "vol-naive";
  const isVolume = kind === "volume" || combined || naive;
  const source: SyntheticSource = isVolume
    ? makeVolumeSource(levelCount, Number(baseRes.value))
    : syntheticPlane({ width: 2048, height: 2048, chunk: 64, levelCount });
  const backendForVolume = isVolume && !naive ? getBackend(r) : undefined;
  view = new DualView(canvas, inset, r, source, (s) => renderStats(s, source.ms.levelCount), combined, naive, stats, backendForVolume);
  view.setQ(Number(qSlider.value));
  view.setTextures(texChk.checked);
  view.setWireframe(gridChk.checked);
  view.setSolid(Number(solid.value));
  view.setDepthRead(dread.checked);
  view.setDepthWrite(dwrite.checked);
  applyNetwork();
  applyTransfer();
  view.setDecisionVisible(!decisionCollapsed); // preserve collapse state across re-mounts
  try {
    await view.start();
  } catch (e) {
    errEl.textContent = `Render init failed:\n${e instanceof Error ? e.message : String(e)}`;
    // eslint-disable-next-line no-console
    console.error(e);
  }
}

function fitCanvas(): void {
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  view?.resize(w, h);
}

qSlider.addEventListener("input", () => {
  qVal.textContent = Number(qSlider.value).toFixed(2);
  view?.setQ(Number(qSlider.value));
});
pLevels.addEventListener("input", () => {
  pLevelsVal.textContent = pLevels.value;
});
pLevels.addEventListener("change", () => void mount(sourceSel.value)); // re-mount: levelCount is baked into the Multiscale
baseRes.addEventListener("change", () => void mount(sourceSel.value)); // re-mount: base resolution is baked into the Multiscale
network.addEventListener("change", applyNetwork); // live: mutates the shared latency model, no re-mount

// Click the "decision" legend to collapse the inset into a little button (and back). Collapsing
// also skips the overlay build + inset render pass — the escape hatch when a large finest
// resolution makes rebuild() slow. stopPropagation so the click doesn't start an inset orbit-drag.
if (insetLabel) {
  const toggleDecision = (e: Event): void => {
    e.stopPropagation();
    decisionCollapsed = !decisionCollapsed;
    inset.classList.toggle("collapsed", decisionCollapsed);
    view?.setDecisionVisible(!decisionCollapsed);
  };
  insetLabel.addEventListener("click", toggleDecision);
  insetLabel.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't begin an orbit drag
}
for (const el of [cmin, cmax, gamma]) el.addEventListener("input", applyTransfer);
solid.addEventListener("input", () => {
  solidVal.textContent = Number(solid.value).toFixed(2);
  view?.setSolid(Number(solid.value));
});
dread.addEventListener("change", () => view?.setDepthRead(dread.checked));
dwrite.addEventListener("change", () => view?.setDepthWrite(dwrite.checked));
texChk.addEventListener("change", () => view?.setTextures(texChk.checked));
gridChk.addEventListener("change", () => view?.setWireframe(gridChk.checked));
sourceSel.addEventListener("change", () => void mount(sourceSel.value));
window.addEventListener("resize", fitCanvas);

void mount(sourceSel.value);
