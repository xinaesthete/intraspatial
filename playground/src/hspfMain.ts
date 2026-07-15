// Stand-alone host for the HsPf spatial simulation (ADR-0011): plain-DOM, no React — mirrors
// datasourceMain.ts. Loads the real HbS/weights GeoTIFFs, runs the GPU sim (state resident on the
// GPU), and drives the field renderer each frame. Parameters flow through the ParamSpec seam
// (defaultHspfParams → toHspfConfig), so a future filter-then-apply / Mutator UI can drive the
// same surface.
import { getDevice } from "../../src/gpu/device";
import { HspfSim, seedPfsa } from "../../src/gpu/sim/hspf/kernel";
import { makeNeighbourhood } from "../../src/gpu/sim/hspf/neighbourhood";
import { defaultHspfParams, type Params, toHspfConfig } from "../../src/gpu/sim/hspf/params";
import { scaffoldFromGeoTIFFs } from "../../src/gpu/sim/hspf/scaffold";
import { channelStyle, FieldRenderer } from "./hspf/fieldRenderer";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const SEED = 1;

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return res.arrayBuffer();
}

async function run(): Promise<void> {
  const canvas = $<HTMLCanvasElement>("stage");
  const errorEl = $("error");

  let device: GPUDevice;
  try {
    device = await getDevice();
  } catch (e) {
    errorEl.innerHTML =
      "<b>WebGPU is required.</b> This demo needs a recent desktop <code>Chrome</code> or <code>Edge</code> " +
      `with WebGPU enabled.<br><small>${(e as Error).message}</small>`;
    return;
  }

  const [hbsBuf, weightsBuf] = await Promise.all([fetchArrayBuffer("/hspf/hbsfilter.tif"), fetchArrayBuffer("/hspf/pf2000.tif")]);
  const scaffold = await scaffoldFromGeoTIFFs(hbsBuf, weightsBuf);

  // Size the canvas to the map's aspect ratio at ~620px wide.
  const displayW = 620;
  const displayH = Math.round((displayW * scaffold.height) / scaffold.width);
  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  const params: Params = defaultHspfParams();
  const cfg = () => toHspfConfig(params);
  const neighbourhoodFor = (p: ReturnType<typeof cfg>) => makeNeighbourhood({ ...p.neighbourhood, gridWidth: scaffold.width, seed: SEED });

  const initial = cfg();
  const sim = await HspfSim.create(device, scaffold, neighbourhoodFor(initial), seedPfsa(scaffold, initial.init), initial.hspf);
  const renderer = await FieldRenderer.create(device, canvas);

  // --- controls ---
  let playing = true; // auto-play so the field is visibly evolving on load
  let channel = 3;
  let contours = true;
  let stepsPerFrame = 2;

  const playpause = $<HTMLButtonElement>("playpause");
  const genEl = $("gen");
  const concentration = $<HTMLInputElement>("concentration");
  const twobite = $<HTMLInputElement>("twobite");
  const speed = $<HTMLInputElement>("speed");
  const concval = $("concval");
  const tbrval = $("tbrval");
  const speedval = $("speedval");

  concentration.value = String(params["spread.concentration"]);
  twobite.value = String(params["dynamics.twoBiteRate"]);
  const syncLabels = () => {
    concval.textContent = Number(concentration.value).toFixed(1);
    tbrval.textContent = `${Math.round(Number(twobite.value) * 100)}%`;
    speedval.textContent = String(stepsPerFrame);
  };
  syncLabels();
  playpause.textContent = playing ? "Pause" : "Play";
  sim.step(80); // warm-up so the map shows a developed pattern immediately, not a flat seed
  genEl.textContent = String(sim.iteration);

  playpause.addEventListener("click", () => {
    playing = !playing;
    playpause.textContent = playing ? "Pause" : "Play";
  });
  $<HTMLButtonElement>("reset").addEventListener("click", () => {
    sim.reset(seedPfsa(scaffold, cfg().init));
    genEl.textContent = "0";
  });
  $<HTMLSelectElement>("channel").addEventListener("change", (e) => {
    channel = Number((e.target as HTMLSelectElement).value);
  });
  $<HTMLInputElement>("contours").addEventListener("change", (e) => {
    contours = (e.target as HTMLInputElement).checked;
  });
  concentration.addEventListener("input", () => {
    params["spread.concentration"] = Number(concentration.value);
    sim.setNeighbourhood(neighbourhoodFor(cfg()), cfg().hspf);
    syncLabels();
  });
  twobite.addEventListener("input", () => {
    params["dynamics.twoBiteRate"] = Number(twobite.value);
    sim.setParams(cfg().hspf);
    syncLabels();
  });
  speed.addEventListener("input", () => {
    stepsPerFrame = Number(speed.value);
    syncLabels();
  });

  // --- render loop ---
  const frame = () => {
    if (playing) {
      sim.step(stepsPerFrame);
      genEl.textContent = String(sim.iteration);
    }
    const style = channelStyle(channel);
    style.contours = contours;
    renderer.render(sim, style);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

run().catch((e) => {
  const el = document.getElementById("error");
  if (el) el.textContent = `Error: ${(e as Error).message}`;
  console.error(e);
});
