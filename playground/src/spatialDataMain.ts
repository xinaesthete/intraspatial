// Plain-TS host for the SpatialData image demo (ADR-0010, slice 1a). Wires a real HTJ2K
// image element from a spatialdata.js store into the EXISTING DualView (Select + TileCache +
// three.js tile render + Decision inset) — the render pipeline is reused verbatim; only the
// Loader is new. No React (ADR-0008 §8).
import Stats from "three/examples/jsm/libs/stats.module.js";
import { WebGPURenderer } from "three/webgpu";
import { type DecisionStats, DualView } from "./datasource/dualView";
import { levelColor } from "./datasource/overlays";
import { openSpatialDataImage, type SpatialDataImage } from "./datasource/spatialDataLoader";

const DEFAULT_STORE = "http://localhost:8080/xenium_2.q0.001.htj2k.index-permutations.zarr";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("stage");
const inset = $("inset");
const qSlider = $<HTMLInputElement>("q");
const qVal = $("qval");
const elementSel = $<HTMLSelectElement>("element");
const storeInput = $<HTMLInputElement>("store");
const bytesEl = $("bytes");
const chunksEl = $("chunks");
const levelsEl = $("levels");
const errEl = $("err");
const labelEl = $("label");

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

async function getRenderer(): Promise<WebGPURenderer | null> {
  if (renderer) return renderer;
  if (!("gpu" in navigator)) {
    errEl.textContent = "WebGPU is not available in this view. Open in Chrome or Edge.";
    return null;
  }
  const r = new WebGPURenderer({ canvas, antialias: true, alpha: true });
  try {
    await r.init();
  } catch (e) {
    errEl.textContent = `WebGPU failed to start: ${e instanceof Error ? e.message : String(e)}.`;
    return null;
  }
  renderer = r;
  return r;
}

async function mount(): Promise<void> {
  view?.dispose();
  errEl.textContent = "";
  labelEl.textContent = "loading…";
  const r = await getRenderer();
  if (!r) return;
  let src: SpatialDataImage;
  try {
    src = await openSpatialDataImage(storeInput.value.trim() || DEFAULT_STORE, elementSel.value);
  } catch (e) {
    errEl.textContent = `Open failed:\n${e instanceof Error ? e.message : String(e)}`;
    labelEl.textContent = "—";
    // eslint-disable-next-line no-console
    console.error(e);
    return;
  }
  labelEl.textContent = src.label;
  view = new DualView(canvas, inset, r, src, (s) => renderStats(s, src.ms.levelCount), false, false, stats);
  (window as unknown as { __view: unknown }).__view = view; // debug handle (runtime-inspectable)
  view.setQ(Number(qSlider.value));
  try {
    await view.start();
  } catch (e) {
    errEl.textContent = `Render init failed:\n${e instanceof Error ? e.message : String(e)}`;
    // eslint-disable-next-line no-console
    console.error(e);
  }
}

qSlider.addEventListener("input", () => {
  qVal.textContent = Number(qSlider.value).toFixed(2);
  view?.setQ(Number(qSlider.value));
});
elementSel.addEventListener("change", () => void mount());
storeInput.addEventListener("change", () => void mount());
window.addEventListener("resize", () => view?.resize(canvas.clientWidth, canvas.clientHeight));

void mount();
