// Plain-TS host for the Decision view (no React — the "thin illustrative host shell"
// of ADR-0008 §8). Wires the HTML controls to a framework-free DualView over a
// synthetic source, and renders the live fetch-budget HUD.
import { WebGPURenderer } from "three/webgpu";
import { type SyntheticSource, syntheticPlane, syntheticVolume } from "../../src/datasource";
import { type DecisionStats, DualView } from "./datasource/dualView";
import { levelColor } from "./datasource/overlays";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("stage");
const inset = $("inset");
const qSlider = $<HTMLInputElement>("q");
const qVal = $("qval");
const pLevels = $<HTMLInputElement>("plevels");
const pLevelsVal = $("plevelsval");
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
    ? syntheticVolume({ size: 256, chunk: 32, levelCount })
    : syntheticPlane({ width: 2048, height: 2048, chunk: 64, levelCount });
  view = new DualView(canvas, inset, r, source, (s) => renderStats(s, source.ms.levelCount), combined, naive);
  view.setQ(Number(qSlider.value));
  view.setTextures(texChk.checked);
  view.setWireframe(gridChk.checked);
  view.setSolid(Number(solid.value));
  view.setDepthRead(dread.checked);
  view.setDepthWrite(dwrite.checked);
  applyTransfer();
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
