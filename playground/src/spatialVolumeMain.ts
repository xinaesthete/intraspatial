// Plain-TS host for the SpatialData VOLUME demo — the 3-D sibling of spatialDataMain.ts.
// It owns its own scene/camera/loop rather than reusing `DualView`, for one specific reason:
// DualView drives the UNBOUNDED `select()`, and `NaiveVolumeRenderer` has no residency cap, so a
// whole-volume view would try to resident every selected brick at once. Here we drive
// `selectWithinBudget` (ADR-0008 §5 degrade-to-fit) and report honestly when even the coarsest level
// cannot fit — which is exactly the diagnostic you want when a pyramid is too shallow for 3-D.
//
// ADDITIVE: shares nothing mutable with the 2-D image/alignment prototype.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import { WebGPURenderer } from "three/webgpu";
import { type Camera, formatBytes, selectWithinBudget, worldAabbOfArrayBox } from "../../src/datasource";
import { NaiveVolumeRenderer } from "./datasource/naiveVolumeRenderer";
import { openSpatialDataVolume, type SpatialDataVolume, type SpatialDataVolumeHandle } from "./datasource/spatialDataVolume";

const DEFAULT_STORE = "http://localhost:8080/8090_13_Punch1_fused_htj2k_lossy.zarr/";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("stage");
const storeInput = $<HTMLInputElement>("store");
const elementSel = $<HTMLSelectElement>("element");
const channelSel = $<HTMLSelectElement>("channel");
const qSlider = $<HTMLInputElement>("q");
const qVal = $("qval");
const budgetSlider = $<HTMLInputElement>("budget");
const budgetVal = $("budgetval");
const cminSlider = $<HTMLInputElement>("cmin");
const cmaxSlider = $<HTMLInputElement>("cmax");
const gammaSlider = $<HTMLInputElement>("gamma");
const cminVal = $("cminval");
const cmaxVal = $("cmaxval");
const gammaVal = $("gammaval");
const autoBtn = $<HTMLButtonElement>("auto");
const rangeEl = $("range");
const bytesEl = $("bytes");
const residentEl = $("resident");
const chunksEl = $("chunks");
const levelEl = $("level");
const labelEl = $("label");
const errEl = $("err");
const pyramidEl = $("pyramid");

const stats = new Stats();
stats.dom.style.left = "auto";
stats.dom.style.right = "12px";
document.body.appendChild(stats.dom);

const ceilingBytes = (): number => Number(budgetSlider.value) * 1024 * 1024;

/** The pure `select` camera, from the three.js camera. */
function cameraFromThree(cam: THREE.PerspectiveCamera, viewportHeightPx: number): Camera {
  const fwd = new THREE.Vector3();
  cam.getWorldDirection(fwd);
  return {
    eye: [cam.position.x, cam.position.y, cam.position.z],
    forward: [fwd.x, fwd.y, fwd.z],
    up: [cam.up.x, cam.up.y, cam.up.z],
    fovY: (cam.fov * Math.PI) / 180,
    aspect: cam.aspect,
    near: cam.near,
    far: cam.far,
    viewportHeightPx,
  };
}

let renderer: WebGPURenderer | null = null;
let handle: SpatialDataVolumeHandle | null = null;
let disposeCurrent: (() => void) | null = null;

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

/** Warn when the coarsest level is far too big to ever fit — the shallow-pyramid diagnosis. */
function reportPyramid(vol: SpatialDataVolume): void {
  const coarsest = vol.levelDims[vol.levelDims.length - 1];
  if (!coarsest) return;
  const voxels = coarsest[0] * coarsest[1] * coarsest[2];
  const mb = (voxels * 4) / (1024 * 1024); // engine estimate: f32 samples
  const levels = vol.levelDims.length;
  const txt = `pyramid: ${levels} level${levels === 1 ? "" : "s"}; coarsest ${coarsest[0]}×${coarsest[1]}×${coarsest[2]} ≈ ${formatBytes(voxels * 4)}`;
  pyramidEl.textContent =
    mb > 512
      ? `⚠ ${txt}. That is too large to hold a whole-volume view — zoom into a sub-region, or add more pyramid levels (halve down to ~256³) when converting.`
      : txt;
  pyramidEl.style.color = mb > 512 ? "#fcd34d" : "#94a3b8";
}

async function reopenStore(): Promise<void> {
  errEl.textContent = "";
  labelEl.textContent = "opening store…";
  handle = null;
  try {
    handle = await openSpatialDataVolume(storeInput.value.trim() || DEFAULT_STORE);
  } catch (e) {
    disposeCurrent?.();
    disposeCurrent = null;
    elementSel.replaceChildren();
    errEl.textContent = `Open failed:\n${e instanceof Error ? e.message : String(e)}`;
    labelEl.textContent = "—";
    return;
  }
  const prev = elementSel.value;
  elementSel.replaceChildren();
  for (const n of handle.imageNames) {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n;
    elementSel.append(o);
  }
  if (handle.imageNames.includes(prev)) elementSel.value = prev;
  if (handle.imageNames.length === 0) {
    errEl.textContent = "No image elements found — is this a SpatialData / OME-Zarr store?";
    return;
  }
  await mountVolume();
}

async function mountVolume(): Promise<void> {
  if (!handle) return;
  disposeCurrent?.();
  disposeCurrent = null;
  errEl.textContent = "";
  labelEl.textContent = "loading…";
  const r = await getRenderer();
  if (!r) return;

  let vol: SpatialDataVolume;
  try {
    vol = await handle.volume(elementSel.value, { channel: Number(channelSel.value) || 0 });
  } catch (e) {
    errEl.textContent = `Load failed:\n${e instanceof Error ? e.message : String(e)}`;
    labelEl.textContent = "—";
    return;
  }
  labelEl.textContent = vol.label;
  reportPyramid(vol);

  // Channel options (the volume path renders one scalar channel at a time).
  const prevCh = channelSel.value;
  channelSel.replaceChildren();
  for (let c = 0; c < vol.channelCount; c++) {
    const o = document.createElement("option");
    o.value = String(c);
    o.textContent = `c${c}`;
    channelSel.append(o);
  }
  if (Number(prevCh) < vol.channelCount) channelSel.value = prevCh;

  const ms = vol.ms;
  const scene = new THREE.Scene();
  const b = worldAabbOfArrayBox(ms.worldFromArray, [0, 0, 0], [ms.voxelDims0[0], ms.voxelDims0[1], ms.voxelDims0[2]]);
  const centre = new THREE.Vector3((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
  const size = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) || 1;

  const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / Math.max(1, canvas.clientHeight), size * 0.001, size * 40);
  // Frame the whole box: pull back far enough that the bounding sphere fits the vertical fov, with a
  // little margin. (Starting part-way in would put the camera inside the volume — you'd just see fog.)
  const radius = 0.5 * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
  const dist = (radius / Math.sin((camera.fov * Math.PI) / 180 / 2)) * 1.15;
  camera.position.copy(centre).addScaledVector(new THREE.Vector3(0.55, 0.38, 0.74).normalize(), dist);
  camera.near = dist * 0.002;
  camera.far = dist * 8;
  camera.updateProjectionMatrix();
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.copy(centre);

  const volume = new NaiveVolumeRenderer(ms, vol.loader);
  scene.add(volume.group);
  volume.setTransfer(Number(cminSlider.value), Number(cmaxSlider.value), Number(gammaSlider.value));

  const fmt = (v: number): string => (v >= 0.1 ? v.toFixed(3) : v.toPrecision(3));
  const applyTransfer = (): void => {
    volume.setTransfer(Number(cminSlider.value), Number(cmaxSlider.value), Number(gammaSlider.value));
    cminVal.textContent = fmt(Number(cminSlider.value));
    cmaxVal.textContent = fmt(Number(cmaxSlider.value));
    gammaVal.textContent = Number(gammaSlider.value).toFixed(2);
  };
  cminSlider.addEventListener("input", applyTransfer);
  cmaxSlider.addEventListener("input", applyTransfer);
  gammaSlider.addEventListener("input", applyTransfer);

  /** Pick a window from sampled coarse bricks, and rescale the sliders to the data — the default
   *  0..1 range (step 0.005) can't even express a window on data that lives in the bottom few
   *  percent of its dtype range, which is why the volume looked empty before. */
  const applyAutoRange = async (): Promise<void> => {
    rangeEl.textContent = "sampling…";
    try {
      const r = await vol.autoRange();
      const sliderMax = Math.min(1, Math.max(r.hi * 2, r.max * 1.1, 1e-3));
      for (const s of [cminSlider, cmaxSlider]) {
        s.max = String(sliderMax);
        s.step = String(sliderMax / 500);
      }
      cminSlider.value = String(r.lo);
      cmaxSlider.value = String(r.hi);
      applyTransfer();
      rangeEl.textContent = `auto window ${fmt(r.lo)}–${fmt(r.hi)} · data ${fmt(r.min)}–${fmt(r.max)} · ${r.samples.toLocaleString()} samples`;
    } catch (e) {
      rangeEl.textContent = `auto-range failed: ${e instanceof Error ? e.message : String(e)}`;
    }
  };
  const onAuto = (): void => void applyAutoRange();
  autoBtn.addEventListener("click", onAuto);
  void applyAutoRange();

  const resize = (): void => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.setSize(w, h, false);
  };
  window.addEventListener("resize", resize);
  resize();

  r.setAnimationLoop(() => {
    stats.begin();
    controls.update();
    const cam = cameraFromThree(camera, canvas.clientHeight || 600);
    const res = selectWithinBudget(ms, cam, ceilingBytes(), { q: Number(qSlider.value) });
    if (res.ok) {
      const sel = res.value;
      volume.update(sel);
      errEl.textContent = "";
      bytesEl.textContent = formatBytes(sel.totalApproxBytes);
      chunksEl.textContent = String(sel.chunks.length);
      const used = sel.countByLevel.findIndex((n) => n > 0);
      levelEl.textContent = used === -1 ? "–" : `L${used}+`;
    } else {
      // Honest floor: even the coarsest level cannot fit the ceiling.
      errEl.textContent = "Working set doesn't fit even at the coarsest level — zoom in, raise the budget, or add pyramid levels.";
      chunksEl.textContent = "0";
      levelEl.textContent = "–";
    }
    residentEl.textContent = formatBytes(volume.byteLength);
    r.render(scene, camera);
    stats.end();
  });

  disposeCurrent = (): void => {
    r.setAnimationLoop(null);
    window.removeEventListener("resize", resize);
    cminSlider.removeEventListener("input", applyTransfer);
    cmaxSlider.removeEventListener("input", applyTransfer);
    gammaSlider.removeEventListener("input", applyTransfer);
    autoBtn.removeEventListener("click", onAuto);
    controls.dispose();
    // NaiveVolumeRenderer has no public dispose() (it's the A/B baseline), so free its GPU objects
    // by walking the group we own.
    volume.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.geometry.dispose();
      const mat = m.material as THREE.Material & { map?: THREE.Texture };
      mat.map?.dispose();
      mat.dispose();
    });
    scene.remove(volume.group);
  };
  (window as unknown as { __volume: unknown }).__volume = { vol, volume, camera };
}

qSlider.addEventListener("input", () => {
  qVal.textContent = Number(qSlider.value).toFixed(2);
});
budgetSlider.addEventListener("input", () => {
  budgetVal.textContent = `${budgetSlider.value} MB`;
});
elementSel.addEventListener("change", () => void mountVolume());
channelSel.addEventListener("change", () => void mountVolume());
storeInput.addEventListener("change", () => void reopenStore());

void reopenStore();
