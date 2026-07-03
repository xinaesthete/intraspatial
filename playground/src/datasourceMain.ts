// Plain-TS host for the Decision view (no React — the "thin illustrative host shell"
// of ADR-0008 §8). Wires the HTML controls to a framework-free DualView over a
// synthetic source, and renders the live fetch-budget HUD.
import { syntheticPlane, syntheticVolume, type SyntheticSource } from "../../src/datasource";
import { DualView, type DecisionStats } from "./datasource/dualView";
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
const sourceSel = $<HTMLSelectElement>("source");
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

async function mount(kind: string): Promise<void> {
  view?.dispose();
  errEl.textContent = "";
  const levelCount = Number(pLevels.value);
  const source: SyntheticSource = kind === "volume"
    ? syntheticVolume({ size: 256, chunk: 32, levelCount })
    : syntheticPlane({ width: 2048, height: 2048, chunk: 64, levelCount });
  view = new DualView(canvas, inset, source, (s) => renderStats(s, source.ms.levelCount));
  view.setQ(Number(qSlider.value));
  view.setTextures(texChk.checked);
  view.setWireframe(gridChk.checked);
  try {
    await view.start();
  } catch (e) {
    errEl.textContent = `Render init failed:\n${e instanceof Error ? e.message : String(e)}`;
    // eslint-disable-next-line no-console
    console.error(e);
  }
}

function fitCanvas(): void {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  view?.resize(w, h);
}

qSlider.addEventListener("input", () => {
  qVal.textContent = Number(qSlider.value).toFixed(2);
  view?.setQ(Number(qSlider.value));
});
pLevels.addEventListener("input", () => { pLevelsVal.textContent = pLevels.value; });
pLevels.addEventListener("change", () => void mount(sourceSel.value)); // re-mount: levelCount is baked into the Multiscale
texChk.addEventListener("change", () => view?.setTextures(texChk.checked));
gridChk.addEventListener("change", () => view?.setWireframe(gridChk.checked));
sourceSel.addEventListener("change", () => void mount(sourceSel.value));
window.addEventListener("resize", fitCanvas);

void mount(sourceSel.value);
