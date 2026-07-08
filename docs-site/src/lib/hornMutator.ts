// The horn Mutator demo — driven by the toolbox's OWN library code, rendered on WebGPU.
// Geometry is our GPU code: each cell's horn is evaluated by the codegen'd swept kernel
// (`createSweptGpu`, src/geometry) into GPU-resident position/normal buffers that the render
// pass binds directly — no CPU tessellation, no round-trip. Because the kernel reads gene values
// from a uniform buffer (structure baked, values as data), all nine cells share ONE compute
// pipeline and morph smoothly: each eased frame just uploads a new `paramVector`.
//
// Breeding is `src/evo` (Todd–Latham Mutator): a `TraitSpace` over the horn's `ParamSpec`s, with
// `breed`/`randomSpecimen` moving real `Specimen`s. The between-generation easing is the library's
// `OnePole` trait-damping. This module owns only presentation — the WebGPU device, render
// pipeline, camera, and DOM wiring. Requires WebGPU; it fails gracefully if unavailable.

import tgpu from "typegpu";
import {
  breed,
  mulberry32,
  randomSpecimen,
  type Specimen,
  specimenToParams,
  type TraitSpace,
  traitSpaceFromParams,
} from "../../../src/evo";
import { gridIndices, horn, linear, ramp, type Swept } from "../../../src/geometry";
import { createSweptGpu, type SweptGpuHandle } from "../../../src/geometry/sweptGpu";
import { OnePole } from "../../../src/gpu/graph/onePole";
import type { ParamSpec } from "../../../src/gpu/graph/op";

// ── The horn's evolvable traits (a ParamSpec schema → a TraitSpace, exactly like the dancer) ──
const HORN_SPECS: ParamSpec[] = [
  { name: "baseR", type: "number", default: 0.3, min: 0.04, max: 0.9 },
  { name: "tipR", type: "number", default: 0.5, min: 0.02, max: 1.0 },
  { name: "exp", type: "number", default: 1.0, min: 0.45, max: 2.0 },
  { name: "len", type: "number", default: 4.0, min: 1.6, max: 6.2 },
  { name: "twist", type: "number", default: 360, min: 0, max: 1152 }, // degrees along the sweep (÷360 = turns)
  { name: "bend", type: "number", default: 60, min: 0, max: 170 }, // degrees along the sweep
  { name: "taper", type: "number", default: 0.6, min: 0.08, max: 1.3 }, // tip cross-section scale
];
const SPACE: TraitSpace = traitSpaceFromParams(HORN_SPECS);

interface GeneMeta {
  label: string;
  fmt: (v: number) => string;
}
const GENE_META: Record<string, GeneMeta> = {
  baseR: { label: "base radius", fmt: (v) => v.toFixed(2) },
  tipR: { label: "tip radius", fmt: (v) => v.toFixed(2) },
  exp: { label: "superellipse e", fmt: (v) => v.toFixed(2) },
  len: { label: "sweep length", fmt: (v) => v.toFixed(1) },
  twist: { label: "twist", fmt: (v) => `${(v / 360).toFixed(2)} turn` },
  bend: { label: "bend", fmt: (v) => `${v.toFixed(0)}°` },
  taper: { label: "taper", fmt: (v) => `×${v.toFixed(2)}` },
};
const GENE_ROWS = SPACE.traits
  .filter((t) => t.kind === "number")
  .map((t) => ({
    name: t.paramName,
    slot: t.slot,
    meta: GENE_META[t.paramName] ?? { label: t.paramName, fmt: (v: number) => v.toFixed(2) },
  }));
const BEND_SLOT = GENE_ROWS.find((g) => g.name === "bend")?.slot ?? 0;

const CELLS = 9;
const CELL_SLICES = 46;
const CELL_STACKS = 74;
const TAU = 0.13; // trait-damping time constant (seconds)
const FOV = (44 * Math.PI) / 180;

const num = (params: Record<string, unknown>, k: string): number => Number(params[k]);

/** Build a real `Swept` horn from a specimen's decoded params (the library grammar). All cells
 *  share this structure, so they share the GPU pipeline; only the paramVector differs. */
function hornFromSpecimen(sp: Specimen): Swept {
  const p = specimenToParams(SPACE, sp);
  return horn({ radius: linear(num(p, "baseR"), num(p, "tipR")), exponent: num(p, "exp"), length: num(p, "len") })
    .twist(ramp(num(p, "twist")))
    .bend(ramp(num(p, "bend")))
    .scale(linear(1, num(p, "taper")));
}

// ── mat4 (column-major, WebGPU clip-space z ∈ [0,1]) ─────────────────────────────────
type Vec3 = [number, number, number];
type Mat4 = number[];
const M4 = {
  perspectiveZO(fovy: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0];
  },
  lookAt(e: Vec3, c: Vec3, up: Vec3): Mat4 {
    const s = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const nr = (v: Vec3): Vec3 => {
      const l = Math.hypot(v[0], v[1], v[2]) || 1;
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    const cr = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const z = nr(s(e, c));
    const x = nr(cr(up, z));
    const y = cr(z, x);
    return [
      x[0],
      y[0],
      z[0],
      0,
      x[1],
      y[1],
      z[1],
      0,
      x[2],
      y[2],
      z[2],
      0,
      -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]),
      -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]),
      -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]),
      1,
    ];
  },
  mul(a: Mat4, b: Mat4): Mat4 {
    const o = new Array<number>(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        o[c * 4 + r] =
          (a[r] ?? 0) * (b[c * 4] ?? 0) +
          (a[4 + r] ?? 0) * (b[c * 4 + 1] ?? 0) +
          (a[8 + r] ?? 0) * (b[c * 4 + 2] ?? 0) +
          (a[12 + r] ?? 0) * (b[c * 4 + 3] ?? 0);
    return o;
  },
};

const RENDER_WGSL = `
struct Cam { mvp: mat4x4<f32>, eye: vec3<f32>, wire: f32, accent: vec3<f32>, pad: f32 };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) n: vec3<f32>, @location(1) wp: vec3<f32> };
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) nor: vec3<f32>) -> VSOut {
  var o: VSOut; o.pos = cam.mvp * vec4<f32>(p, 1.0); o.n = nor; o.wp = p; return o;
}
@fragment fn fs(inp: VSOut) -> @location(0) vec4<f32> {
  var N = normalize(inp.n); let V = normalize(cam.eye - inp.wp); if (dot(N, V) < 0.0) { N = -N; }
  let key = normalize(vec3<f32>(0.5, 0.9, 0.6)); let diff = max(dot(N, key), 0.0); let hemi = 0.5 + 0.5 * N.y;
  let ground = vec3<f32>(0.05, 0.08, 0.09); let sky = vec3<f32>(0.34, 0.52, 0.55);
  var col = mix(ground, cam.accent, hemi) * (0.28 + 0.85 * diff) + mix(ground, sky, hemi) * 0.28;
  let H = normalize(key + V); col += vec3<f32>(1.0, 0.92, 0.82) * pow(max(dot(N, H), 0.0), 42.0) * 0.5;
  col += cam.accent * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.5;
  if (cam.wire > 0.5) { col = mix(col, vec3<f32>(0.85, 0.95, 0.93), 0.6); }
  return vec4<f32>(pow(col, vec3<f32>(0.4545)), 1.0);
}`;

interface Cell {
  handle: SweptGpuHandle;
  cam: GPUBuffer;
  bind: GPUBindGroup;
  target: Specimen;
  filter: OnePole;
  dirty: boolean;
}

function fail(msg: string): void {
  const overlay = document.getElementById("overlay");
  if (overlay)
    overlay.innerHTML = `<div style="grid-column:1/-1;align-self:center;justify-self:center;color:var(--fg-dim);font-size:14px;text-align:center;padding:2rem;max-width:44ch">${msg}</div>`;
}

/** Mount the demo (elements are already in the DOM). Async: it acquires a WebGPU device. */
export async function mountHornMutator(): Promise<void> {
  const canvas = document.getElementById("gl") as HTMLCanvasElement | null;
  const overlay = document.getElementById("overlay");
  if (!canvas || !overlay) return;

  if (!navigator.gpu) {
    fail("This demo renders with <b>WebGPU</b>, which this browser doesn’t expose. Try a recent Chrome, Edge, or Safari.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter?.requestDevice().catch(() => null);
  if (!device) {
    fail("Couldn’t acquire a WebGPU device. Rendering is unavailable here.");
    return;
  }
  const ctx = canvas.getContext("webgpu");
  if (!ctx) {
    fail("Couldn’t create a WebGPU canvas context.");
    return;
  }
  const root = tgpu.initFromDevice({ device });
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "premultiplied" });

  // ── Render pipeline (shared explicit layout so tri/line pipelines share cell bind groups) ──
  const module = device.createShaderModule({ code: RENDER_WGSL });
  const camBGL = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [camBGL] });
  const vertexBuffers: GPUVertexBufferLayout[] = [
    { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
    { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
  ];
  const mkPipe = (topology: GPUPrimitiveTopology): GPURenderPipeline =>
    device.createRenderPipeline({
      layout: pipeLayout,
      vertex: { module, entryPoint: "vs", buffers: vertexBuffers },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology, cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  const triPipe = mkPipe("triangle-list");
  const linePipe = mkPipe("line-list");

  // Shared index buffers.
  const triIndices = gridIndices(CELL_SLICES, CELL_STACKS);
  const wireIndices = wireFromTris(triIndices);
  const triIdx = device.createBuffer({ size: triIndices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(triIdx, 0, triIndices);
  const wireIdx = device.createBuffer({ size: wireIndices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(wireIdx, 0, wireIndices);

  const rng = mulberry32(0x5eed);
  const nextSeed = (): number => (rng() * 0x100000000) >>> 0;

  const cells: Cell[] = Array.from({ length: CELLS }, () => {
    const target = randomSpecimen(SPACE, nextSeed());
    const handle = createSweptGpu(device, root, hornFromSpecimen(target), CELL_SLICES, CELL_STACKS);
    const cam = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({ layout: camBGL, entries: [{ binding: 0, resource: { buffer: cam } }] });
    return { handle, cam, bind, target, filter: new OnePole(SPACE.numCount, { tau: TAU, initial: target.pos }), dirty: false };
  });

  // ── State ────────────────────────────────────────────────────────────────────────────
  let selected = new Set<number>([0]);
  let primary = 0;
  let generation = 0;
  let spin = true;
  let wire = false;
  let rate = 0.18;
  let yaw = 0.7;
  let pitch = 0.28;
  let zoom = 1;
  let autoT = 0;

  // ── Depth target (recreated on resize) ────────────────────────────────────────────────
  let depthTex: GPUTexture | null = null;
  let depthView: GPUTextureView | null = null;
  function syncSize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h || !depthTex) {
      canvas.width = w;
      canvas.height = h;
      depthTex?.destroy();
      depthTex = device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
      depthView = depthTex.createView();
    }
  }

  const camScratch = new Float32Array(24);
  function writeCam(cell: Cell, vw: number, vh: number): void {
    const p = specimenToParams(SPACE, smoothed(cell));
    const len = num(p, "len");
    const maxR = Math.max(num(p, "baseR"), num(p, "tipR"));
    const bendFrac = cell.filter.value()[BEND_SLOT] ?? 0;
    const extent = 0.5 * len + 1.15 * maxR + 0.28 * len * bendFrac + 0.2;
    const aspect = vw / Math.max(1, vh);
    const fit = Math.min(aspect, 1);
    const eyeR = extent / Math.tan(FOV / 2) / (0.92 * fit) / zoom;
    const y = yaw + autoT;
    const eye: Vec3 = [eyeR * Math.cos(pitch) * Math.sin(y), eyeR * Math.sin(pitch), eyeR * Math.cos(pitch) * Math.cos(y)];
    const proj = M4.perspectiveZO(FOV, aspect, 0.02, 200);
    const view = M4.lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const model: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -len / 2, 1];
    const mvp = M4.mul(proj, M4.mul(view, model));
    for (let i = 0; i < 16; i++) camScratch[i] = mvp[i] ?? 0;
    camScratch[16] = eye[0];
    camScratch[17] = eye[1];
    camScratch[18] = eye[2] + len / 2; // object-space eye (undo model z-translate)
    camScratch[19] = wire ? 1 : 0;
    camScratch[20] = 0.9;
    camScratch[21] = 0.53;
    camScratch[22] = 0.29;
    camScratch[23] = 0;
    device.queue.writeBuffer(cell.cam, 0, camScratch);
  }

  const smoothed = (c: Cell): Specimen => ({
    pos: Float64Array.from(c.filter.value()),
    vel: c.target.vel,
    enable: c.target.enable,
    seed: c.target.seed,
  });

  // ── Trait-damping: re-evaluate the GPU geometry only while a cell is moving ────────────
  function easeCells(dt: number): void {
    for (const c of cells) {
      const cur = c.filter.value();
      let maxd = 0;
      for (let i = 0; i < SPACE.numCount; i++) maxd = Math.max(maxd, Math.abs((c.target.pos[i] ?? 0) - (cur[i] ?? 0)));
      if (maxd < 1e-4 && !c.dirty) continue;
      if (maxd < 1e-4) c.filter.reset(c.target.pos);
      else c.filter.push(c.target.pos, dt);
      c.handle.update(hornFromSpecimen(smoothed(c)).paramVector());
      c.dirty = false;
    }
  }

  function render(dt: number): void {
    if (spin) autoT += dt * 0.3;
    easeCells(dt);
    syncSize();
    if (!depthView) return;
    const W = canvas.width;
    const H = canvas.height;
    const cw = Math.floor(W / 3);
    const ch = Math.floor(H / 3);
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    pass.setPipeline(wire ? linePipe : triPipe);
    const idxBuf = wire ? wireIdx : triIdx;
    const idxCount = wire ? wireIndices.length : triIndices.length;
    for (let i = 0; i < CELLS; i++) {
      const col = i % 3;
      const row = (i / 3) | 0;
      const vw = col < 2 ? cw : W - 2 * cw;
      const vh = row < 2 ? ch : H - 2 * ch;
      const vx = col * cw;
      const vy = row * ch; // WebGPU viewport origin is top-left
      const cell = cells[i] as Cell;
      writeCam(cell, vw, vh);
      pass.setViewport(vx, vy, vw, vh, 0, 1);
      pass.setScissorRect(vx, vy, vw, vh);
      pass.setBindGroup(0, cell.bind);
      pass.setVertexBuffer(0, cell.handle.positions);
      pass.setVertexBuffer(1, cell.handle.normals);
      pass.setIndexBuffer(idxBuf, "uint32");
      pass.drawIndexed(idxCount);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  let lastT = 0;
  function loop(t: number): void {
    const dt = Math.min((t - lastT) / 1000, 0.05);
    lastT = t;
    render(dt);
    requestAnimationFrame(loop);
  }

  // ── Grid overlay: cells, selection, orbit ────────────────────────────────────────────
  const cellEls: HTMLElement[] = [];
  for (let i = 0; i < CELLS; i++) {
    const el = document.createElement("div");
    el.className = "cell";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `#${i + 1}`;
    el.appendChild(tag);
    overlay.appendChild(el);
    cellEls.push(el);
  }
  const selNEl = document.getElementById("selN");
  function refreshCellStyles(): void {
    cellEls.forEach((el, i) => {
      el.classList.toggle("sel", selected.has(i));
      el.classList.toggle("primary", i === primary);
      el.classList.toggle("elite", i === 0 && generation > 0);
    });
    if (selNEl) selNEl.textContent = String(selected.size);
  }
  const cellIndexAt = (clientX: number, clientY: number): number => {
    const r = overlay.getBoundingClientRect();
    const col = Math.min(2, Math.max(0, Math.floor((clientX - r.left) / (r.width / 3))));
    const row = Math.min(2, Math.max(0, Math.floor((clientY - r.top) / (r.height / 3))));
    return row * 3 + col;
  };
  let down = false;
  let moved = false;
  let sx = 0;
  let sy = 0;
  let pxp = 0;
  let pyp = 0;
  overlay.addEventListener("pointerdown", (e) => {
    down = true;
    moved = false;
    sx = pxp = e.clientX;
    sy = pyp = e.clientY;
    overlay.setPointerCapture(e.pointerId);
  });
  overlay.addEventListener("pointermove", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 5) moved = true;
    if (moved) {
      yaw -= (e.clientX - pxp) * 0.008;
      pitch = Math.max(-1.35, Math.min(1.35, pitch + (e.clientY - pyp) * 0.008));
    }
    pxp = e.clientX;
    pyp = e.clientY;
  });
  overlay.addEventListener("pointerup", (e) => {
    down = false;
    if (moved) return;
    const i = cellIndexAt(e.clientX, e.clientY);
    if (e.shiftKey || e.metaKey || e.ctrlKey) {
      if (selected.has(i) && selected.size > 1) selected.delete(i);
      else selected.add(i);
      primary = i;
    } else {
      selected = new Set([i]);
      primary = i;
    }
    refreshCellStyles();
    buildGenePanel();
  });
  overlay.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      zoom = Math.max(0.45, Math.min(2.6, zoom * (1 - Math.sign(e.deltaY) * 0.08)));
    },
    { passive: false },
  );

  // ── Parameters panel (folds away) ─────────────────────────────────────────────────────
  const geneList = document.getElementById("geneList");
  const panel = document.getElementById("params");
  const foldTab = document.getElementById("foldTab");
  function buildGenePanel(): void {
    if (!geneList) return;
    const c = cells[primary] as Cell;
    const specNameEl = document.getElementById("specName");
    if (specNameEl) specNameEl.textContent = `#${primary + 1}`;
    geneList.innerHTML = "";
    const params = specimenToParams(SPACE, c.target);
    for (const g of GENE_ROWS) {
      const row = document.createElement("div");
      row.className = "gene";
      const label = document.createElement("label");
      label.textContent = g.meta.label;
      const out = document.createElement("output");
      out.textContent = g.meta.fmt(num(params, g.name));
      const inp = document.createElement("input");
      inp.type = "range";
      inp.min = "0";
      inp.max = "1";
      inp.step = "0.001";
      inp.value = String(c.target.pos[g.slot] ?? 0);
      inp.style.gridColumn = "1 / -1";
      inp.addEventListener("input", () => {
        c.target.pos[g.slot] = Number(inp.value);
        c.dirty = true;
        out.textContent = g.meta.fmt(num(specimenToParams(SPACE, c.target), g.name));
        updateCode();
      });
      row.appendChild(label);
      row.appendChild(out);
      row.appendChild(inp);
      geneList.appendChild(row);
    }
    const code = document.createElement("div");
    code.className = "code";
    code.id = "code";
    geneList.appendChild(code);
    updateCode();
  }
  function updateCode(): void {
    const codeEl = document.getElementById("code");
    if (!codeEl) return;
    const p = specimenToParams(SPACE, (cells[primary] as Cell).target);
    const n = (x: string): string => `<span class="n">${x}</span>`;
    codeEl.innerHTML =
      `<span class="fn">horn</span>({ radius: <span class="k">linear</span>(${n(num(p, "baseR").toFixed(2))}, ${n(num(p, "tipR").toFixed(2))}),\n` +
      `      exponent: ${n(num(p, "exp").toFixed(2))}, length: ${n(num(p, "len").toFixed(1))} })\n` +
      `  .<span class="fn">twist</span>(<span class="k">ramp</span>(${n(num(p, "twist").toFixed(0))}))\n` +
      `  .<span class="fn">bend</span>(<span class="k">ramp</span>(${n(num(p, "bend").toFixed(0))}))\n` +
      `  .<span class="fn">scale</span>(<span class="k">linear</span>(1, ${n(num(p, "taper").toFixed(2))}))`;
  }
  const setFolded = (f: boolean): void => {
    panel?.classList.toggle("folded", f);
    foldTab?.classList.toggle("show", f);
  };
  document.getElementById("foldBtn")?.addEventListener("click", () => setFolded(true));
  foldTab?.addEventListener("click", () => setFolded(false));

  // ── Command bar ────────────────────────────────────────────────────────────────────────
  const genNEl = document.getElementById("genN");
  const bumpGen = (): void => {
    generation++;
    if (genNEl) genNEl.textContent = String(generation);
    selected = new Set([0]);
    primary = 0;
    refreshCellStyles();
    buildGenePanel();
  };
  document.getElementById("breed")?.addEventListener("click", () => {
    const others = [...selected].filter((i) => i !== primary);
    const parents = [(cells[primary] as Cell).target, ...others.map((i) => (cells[i] as Cell).target)];
    const kids = breed(SPACE, parents, CELLS, { rate, rng, keepElite: true });
    cells.forEach((c, i) => {
      c.target = kids[i] ?? c.target; // smooth stays → eases into the new genotype
    });
    bumpGen();
  });
  document.getElementById("random")?.addEventListener("click", () => {
    for (const c of cells) c.target = randomSpecimen(SPACE, nextSeed());
    bumpGen();
  });
  const rateInp = document.getElementById("rate") as HTMLInputElement | null;
  rateInp?.addEventListener("input", () => {
    rate = Number(rateInp.value);
    const rv = document.getElementById("rate_v");
    if (rv) rv.textContent = rate.toFixed(2);
  });
  (document.getElementById("spin") as HTMLInputElement | null)?.addEventListener("change", (e) => {
    spin = (e.target as HTMLInputElement).checked;
  });
  (document.getElementById("wire") as HTMLInputElement | null)?.addEventListener("change", (e) => {
    wire = (e.target as HTMLInputElement).checked;
  });

  if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
    spin = false;
    const s = document.getElementById("spin") as HTMLInputElement | null;
    if (s) s.checked = false;
  }

  refreshCellStyles();
  buildGenePanel();
  syncSize();
  requestAnimationFrame(loop);
}

/** Unique undirected edges of a triangle-index list, as a flat line-list index buffer. */
function wireFromTris(tris: Uint32Array): Uint32Array {
  const set = new Set<number>();
  const push = (a: number, b: number) => set.add(a < b ? a * 4294967296 + b : b * 4294967296 + a);
  for (let i = 0; i < tris.length; i += 3) {
    push(tris[i] ?? 0, tris[i + 1] ?? 0);
    push(tris[i + 1] ?? 0, tris[i + 2] ?? 0);
    push(tris[i + 2] ?? 0, tris[i] ?? 0);
  }
  const out = new Uint32Array(set.size * 2);
  let j = 0;
  for (const v of set) {
    out[j++] = Math.floor(v / 4294967296);
    out[j++] = v % 4294967296;
  }
  return out;
}
