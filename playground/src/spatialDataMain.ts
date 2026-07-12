// Plain-TS host for the SpatialData image demo (ADR-0010, slice 1a). Wires a real HTJ2K
// image element from a spatialdata.js store into the EXISTING DualView (Select + TileCache +
// three.js tile render + Decision inset) — the render pipeline is reused verbatim; only the
// Loader is new. No React (ADR-0008 §8).
import Stats from "three/examples/jsm/libs/stats.module.js";
import { WebGPURenderer } from "three/webgpu";
import { formatBytes } from "../../src/datasource";
import { type DecisionStats, DualView } from "./datasource/dualView";
import { levelColor } from "./datasource/overlays";
import { openSpatialData, type SpatialDataHandle, type SpatialDataImage } from "./datasource/spatialDataLoader";
import { type BlendMode, ChannelComposite, type ChannelSettings } from "./datasource/tileChannelMaterial";

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
const blendSel = $<HTMLSelectElement>("blend");
const channelsEl = $("channelpanel");
const bytesEl = $("bytes");
const residentEl = $("resident");
const chunksEl = $("chunks");
const levelsEl = $("levels");
const errEl = $("err");
const labelEl = $("label");

const stats = new Stats();
stats.dom.style.left = "auto";
stats.dom.style.right = "12px";
document.body.appendChild(stats.dom);

function renderStats(s: DecisionStats, levelCount: number): void {
  bytesEl.textContent = formatBytes(s.totalBytes);
  residentEl.textContent = formatBytes(s.residentBytes);
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
let composite: ChannelComposite | null = null;
let channels: ChannelSettings[] = [];
let handle: SpatialDataHandle | null = null;
const applyChannels = (): void => composite?.update(channels, blendSel.value as BlendMode);

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

/** Populate the image dropdown from the open store, keeping the current pick if it survives. */
function populateImageOptions(names: string[]): void {
  const prev = elementSel.value;
  elementSel.replaceChildren();
  for (const n of names) {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n;
    elementSel.append(o);
  }
  if (names.includes(prev)) elementSel.value = prev;
}

/** Open (or re-open) the store at the URL, list its images into the dropdown, and mount one.
 *  Called on load and whenever the store URL changes. */
async function reopenStore(): Promise<void> {
  errEl.textContent = "";
  labelEl.textContent = "opening store…";
  handle = null;
  try {
    handle = await openSpatialData(storeInput.value.trim() || DEFAULT_STORE);
  } catch (e) {
    view?.dispose();
    view = null;
    elementSel.replaceChildren();
    errEl.textContent = `Open failed:\n${e instanceof Error ? e.message : String(e)}`;
    labelEl.textContent = "—";
    // eslint-disable-next-line no-console
    console.error(e);
    return;
  }
  if (handle.imageNames.length === 0) {
    view?.dispose();
    view = null;
    elementSel.replaceChildren();
    errEl.textContent = "No image elements found — is this a SpatialData store?";
    labelEl.textContent = "—";
    return;
  }
  populateImageOptions(handle.imageNames);
  await mountImage();
}

/** Build and mount the currently-selected image, reusing the open store handle. */
async function mountImage(): Promise<void> {
  if (!handle) return;
  view?.dispose();
  errEl.textContent = "";
  labelEl.textContent = "loading…";
  const r = await getRenderer();
  if (!r) return;
  let src: SpatialDataImage;
  try {
    src = await handle.image(elementSel.value);
  } catch (e) {
    errEl.textContent = `Load failed:\n${e instanceof Error ? e.message : String(e)}`;
    labelEl.textContent = "—";
    // eslint-disable-next-line no-console
    console.error(e);
    return;
  }
  labelEl.textContent = src.label;
  // The owned channel-composite material (GPU colour/window/composite; live via shared uniforms).
  channels = src.channels;
  composite = new ChannelComposite(channels, blendSel.value as BlendMode);
  buildChannelPanel(channels, applyChannels);
  view = new DualView(
    canvas,
    inset,
    r,
    { ...src, makeMaterial: composite.makeMaterial },
    (s) => renderStats(s, src.ms.levelCount),
    false,
    false,
    stats,
  );
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

const rgbHex = (c: readonly [number, number, number]): string =>
  `#${c
    .map((v) =>
      Math.round(Math.max(0, Math.min(1, v)) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
const hexRgb = (h: string): [number, number, number] => [
  Number.parseInt(h.slice(1, 3), 16) / 255,
  Number.parseInt(h.slice(3, 5), 16) / 255,
  Number.parseInt(h.slice(5, 7), 16) / 255,
];

/** A viv/MDV-style layers list: per channel a visibility toggle, colour swatch, and a
 *  contrast-window (min/max) — mutating the passed `channels` in place and calling `onChange`
 *  (which pushes them into the live GPU material). */
function buildChannelPanel(channels: ChannelSettings[], onChange: () => void): void {
  channelsEl.replaceChildren();
  channels.forEach((ch) => {
    const row = document.createElement("div");
    row.className = "chrow";
    const vis = document.createElement("input");
    vis.type = "checkbox";
    vis.checked = ch.visible;
    vis.title = "visible";
    vis.addEventListener("change", () => {
      ch.visible = vis.checked;
      onChange();
    });
    const col = document.createElement("input");
    col.type = "color";
    col.value = rgbHex(ch.color);
    col.title = "colour";
    col.addEventListener("input", () => {
      ch.color = hexRgb(col.value);
      onChange();
    });
    const name = document.createElement("span");
    name.className = "chname";
    name.textContent = ch.label;
    name.title = ch.label;
    const lo = document.createElement("input");
    lo.type = "range";
    lo.min = "0";
    lo.max = "1";
    lo.step = "0.005";
    lo.value = String(ch.contrastLimits[0]);
    lo.className = "chslider";
    const hi = document.createElement("input");
    hi.type = "range";
    hi.min = "0";
    hi.max = "1";
    hi.step = "0.005";
    hi.value = String(ch.contrastLimits[1]);
    hi.className = "chslider";
    const window = (): void => {
      const a = Number(lo.value);
      const b = Number(hi.value);
      ch.contrastLimits = [Math.min(a, b), Math.max(a, b)];
      onChange();
    };
    lo.addEventListener("input", window);
    hi.addEventListener("input", window);
    row.append(vis, col, name, lo, hi);
    channelsEl.append(row);
  });
}

qSlider.addEventListener("input", () => {
  qVal.textContent = Number(qSlider.value).toFixed(2);
  view?.setQ(Number(qSlider.value));
});
blendSel.addEventListener("change", applyChannels);
elementSel.addEventListener("change", () => void mountImage()); // reuse the open store, no re-read
storeInput.addEventListener("change", () => void reopenStore()); // new URL → re-list images
window.addEventListener("resize", () => view?.resize(canvas.clientWidth, canvas.clientHeight));

void reopenStore();
