// Plain-TS host for the Decision view (no React — the "thin illustrative host shell"
// of ADR-0008 §8). Wires the HTML controls to a framework-free DecisionView over a
// synthetic source, and renders the live fetch-budget HUD.
import { syntheticPlane, syntheticVolume, type SyntheticSource } from "../../src/datasource";
import { DecisionView, type DecisionStats } from "./datasource/decisionView";
import { levelColor } from "./datasource/overlays";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("stage");
const qSlider = $<HTMLInputElement>("q");
const qVal = $("qval");
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

let view: DecisionView | null = null;

async function mount(kind: string): Promise<void> {
  view?.dispose();
  errEl.textContent = "";
  const source: SyntheticSource = kind === "volume" ? syntheticVolume() : syntheticPlane();
  view = new DecisionView(canvas, source.ms, (s) => renderStats(s, source.ms.levelCount));
  view.setQ(Number(qSlider.value));
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
sourceSel.addEventListener("change", () => void mount(sourceSel.value));
window.addEventListener("resize", fitCanvas);

void mount(sourceSel.value);
