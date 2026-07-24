// The co-location mode field as a lit terrain, with a metric over the mode axis for "select
// everything that looks like here".
//
// ## Why a terrain rather than a third colour channel
//
// `gramModes.ts` paints three modes through OKLab, which is the most a flat image can carry before
// the axes start competing for the same perceptual dimension. Height is a fourth channel that costs
// none of the first three: the eye reads a shaded surface as geometry, not as colour, so a displaced
// map shows *where a chosen quantity is high* while the colour still shows *what the co-location
// profile is*. The two are independent readings of the same resident rasters.
//
// Nothing is re-splatted and nothing is read back. The grid mesh samples `res.resident.buffer` in
// the vertex shader — the same buffer the matrix was reduced from — so a camera move is a redraw of
// data already on the device, and changing what drives the height is a uniform write.
//
// ## The metric over the mode axis — what the "magic wand" actually computes
//
// Pick a pixel; its standardised channel vector is `z_ref`. The natural question "where else looks
// like this?" is a distance in channel space, and the honest distance is **Mahalanobis**, because
// the channels are correlated — two channels that always co-occur should not count as two
// independent pieces of evidence. Writing `corr = V Λ Vᵀ`,
//
//     d²(x) = Δzᵀ corr⁻¹ Δz = Σ_k (Δy_k)² / λ_k        with Δy = Vᵀ Δz
//
// which is exactly: project onto the co-location modes, then divide each by its own variance. So
// the eigen-decomposition this module already has *is* the metric — no separate machinery. The
// shader takes a **whitening matrix** `A = Λ^{-1/2} Vᵀ`, truncated to the leading `m` modes, and
// computes `d = |A Δz|`. Two cases fall out of the one uniform:
//
//   * `m = K` — full Mahalanobis. Exact, and noisy: the trailing modes have tiny `λ`, so dividing
//     by them amplifies whatever numerical and sampling noise they hold.
//   * `m = 3` — distance in precisely the space the colour is drawn from, so "looks similar" and
//     "is selected" agree by construction. This is the useful default, and the reason for it is the
//     same reason the map is OKLab in the first place.
//
// This is ADR-0015's metric-over-an-open-axis, at its "full metric" end: identity gives Euclidean
// distance between raw channel vectors, a diagonal gives per-channel weights, and a full matrix
// gives a channel covariance. The Gram form is what makes the full case computable.

import type { Oklab } from "../../color/oklab";
import { getDevice } from "../device";
import type { GramMatrixGpuResult } from "./gramMatrix";
import { MARKER_WGSL } from "./markerWgsl";

/** Kept in step with `gramModes.ts` by being the same numbers — a mode map and its terrain must not
 *  disagree about what a colour means. */
const MAX_CHROMA = 0.11;
const BASE_L = 0.62;
const SPAN_L = 0.3;
/** Hue of the similarity ramp, in OKLab (a, b). Chosen away from the mode axes' typical directions
 *  so a similarity view is never mistaken for a mode view at a glance. */
const SIM_A = 0.6;
const SIM_B = -0.8;

const UNI_FLOATS = 40; // 160 bytes: mat4 + 6 vec4s

const TERRAIN_SHADER = /* wgsl */ `
struct Uni {
  mvp: mat4x4f,
  gridW: f32, gridH: f32, step: f32, rowFloats: f32,
  rasterH: f32, K: f32, aspectX: f32, aspectY: f32,
  heightScale: f32, heightSource: f32, colourBy: f32, m: f32,
  scaleL: f32, scaleA: f32, scaleB: f32, distScale: f32,
  markerX: f32, markerY: f32, markerOn: f32, lineW: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<f32>;
/** mean, 1/sd, and the three mode loadings per channel — the same 5-float interleave gramModes uses. */
@group(0) @binding(2) var<storage, read> chan: array<f32>;
/** K floats of reference z, then m*K floats of the whitening matrix A (row-major, m rows of K). */
@group(0) @binding(3) var<storage, read> wand: array<f32>;

fn oklabToSrgb(lab: vec3f) -> vec3f {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;
  let lin = vec3f(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
  let c = clamp(lin, vec3f(0.0), vec3f(1.0));
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

${MARKER_WGSL}

fn fetch(a: u32, col: u32, row: u32) -> f32 {
  return rasters[a * u32(U.rasterH) * u32(U.rowFloats) + row * u32(U.rowFloats) + col];
}

struct Sample { y: vec3f, d: f32 };

/** Standardise the K channels at one pixel, project onto the leading three modes, and measure the
 *  whitened distance to the wand reference. One pass over the channels serves both. */
fn sampleAt(col: u32, row: u32) -> Sample {
  let K = u32(U.K);
  let m = u32(U.m);
  var y = vec3f(0.0);
  var u: array<f32, 32>;               // MAX_CHANNELS; m <= K <= 32
  for (var k = 0u; k < m; k = k + 1u) { u[k] = 0.0; }
  for (var a = 0u; a < K; a = a + 1u) {
    let z = (fetch(a, col, row) - chan[a * 5u]) * chan[a * 5u + 1u];
    y = y + z * vec3f(chan[a * 5u + 2u], chan[a * 5u + 3u], chan[a * 5u + 4u]);
    let dz = z - wand[a];
    // u = A·Δz, accumulated channel-major so Δz is computed once for both uses.
    for (var k = 0u; k < m; k = k + 1u) { u[k] = u[k] + wand[K + k * K + a] * dz; }
  }
  var d2 = 0.0;
  for (var k = 0u; k < m; k = k + 1u) { d2 = d2 + u[k] * u[k]; }
  var out: Sample;
  out.y = y;
  out.d = sqrt(d2);
  return out;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) model: vec3f,
  @location(1) y: vec3f,
  @location(2) d: f32,
};

/** Height in model units. heightSource 0-2 pick a mode; 3 is similarity, raised at the wand point
 *  rather than sunk, because a selection should read as a peak and not as a hole. */
fn heightOf(s: Sample) -> f32 {
  let src = u32(U.heightSource);
  var h = 0.0;
  if (src == 0u) { h = s.y.x * U.scaleL; }
  else if (src == 1u) { h = s.y.y * U.scaleA; }
  else if (src == 2u) { h = s.y.z * U.scaleB; }
  else { h = 1.0 - clamp(s.d * U.distScale, 0.0, 1.0) * 2.0; }
  return clamp(h, -1.0, 1.0) * U.heightScale;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // Non-indexed grid: 6 vertices per quad, position derived from the index. An index buffer would
  // save vertex shader invocations but cost a buffer upload per resize; the mesh is regenerated on
  // every raster change, so deriving it is the cheaper side of that trade.
  let gw = u32(U.gridW) - 1u;
  let quad = vi / 6u;
  let c = vi % 6u;
  let qx = quad % gw;
  let qy = quad / gw;
  let dx = select(0u, 1u, c == 1u || c == 4u || c == 5u);
  let dy = select(0u, 1u, c == 2u || c == 3u || c == 5u);
  let gx = qx + dx;
  let gy = qy + dy;

  let s = sampleAt(gx * u32(U.step), gy * u32(U.step));

  // Model space: the window mapped to [-aspect/2, aspect/2] with its longer side 1, so the camera
  // is independent of both world units and raster resolution.
  let u = f32(gx) / max(U.gridW - 1.0, 1.0);
  let v = f32(gy) / max(U.gridH - 1.0, 1.0);
  let model = vec3f((u - 0.5) * U.aspectX, (0.5 - v) * U.aspectY, heightOf(s));

  var o: VOut;
  o.pos = U.mvp * vec4f(model, 1.0);
  o.model = model;
  o.y = s.y;
  o.d = s.d;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  // The surface normal from screen-space derivatives of the model position: exact for a triangle,
  // and it costs nothing next to sampling the neighbours in the vertex shader.
  let n = normalize(cross(dpdx(in.model), dpdy(in.model)));
  let lit = 0.45 + 0.55 * abs(dot(n, normalize(vec3f(0.35, 0.45, 0.82))));

  var lab: vec3f;
  if (U.colourBy < 0.5) {
    let t = clamp(in.y * vec3f(U.scaleL, U.scaleA, U.scaleB), vec3f(-1.0), vec3f(1.0));
    lab = vec3f(${BASE_L} + ${SPAN_L} * t.x, ${MAX_CHROMA} * t.y, ${MAX_CHROMA} * t.z);
  } else {
    // Similarity: 1 at the reference, falling to 0. Chroma AND lightness both carry it, so the
    // selection survives a greyscale reading — the same argument as mode 1 -> L on the flat map.
    let t = 1.0 - clamp(in.d * U.distScale, 0.0, 1.0);
    lab = vec3f(0.3 + 0.5 * t, ${MAX_CHROMA} * t * ${SIM_A}, ${MAX_CHROMA} * t * ${SIM_B});
  }
  // Shade in OKLab's lightness rather than by scaling sRGB: multiplying linear RGB would drag the
  // hue of saturated colours, which is exactly the signal the map carries.
  let rgb = oklabToSrgb(vec3f(lab.x * lit, lab.y, lab.z));

  // The two rule lines are loci in the surface's own XY, so each drapes over the relief as the
  // terrain's profile along one axis through the sample, and both stay put as the camera moves.
  // Deliberately NOT shaded: a mark that dims on a shadowed slope is exactly where you lose it.
  //
  // Their WIDTH is screen-space, not model-space, via the per-axis screen derivative of mp. On a
  // near-vertical face model XY barely changes per pixel, so a fixed model-space band smears across
  // the whole wall; scaling by fwidth holds each line at a couple of pixels whatever the surface is
  // doing underneath. The floor keeps it from vanishing on a face turned almost edge-on.
  let mp = in.model.xy - vec2f(U.markerX, U.markerY);
  let w = max(vec2f(U.lineW), fwidth(mp) * 1.6);
  return vec4f(markerOver(rgb, mp, w, U.markerOn), 1.0);
}
`;

/** Gathers the K raster values at one pixel, so the host can turn a click into a wand reference
 *  without downloading a raster. K floats out, one dispatch. */
const PROBE_SHADER = /* wgsl */ `
struct Uni { K: f32, col: f32, row: f32, rowFloats: f32, rasterH: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3u) {
  let a = gid.x;
  if (a >= u32(U.K)) { return; }
  out[a] = rasters[a * u32(U.rasterH) * u32(U.rowFloats) + u32(U.row) * u32(U.rowFloats) + u32(U.col)];
}
`;

export type HeightSource = "mode1" | "mode2" | "mode3" | "similarity";
export type ColourBy = "modes" | "similarity";

export interface OrbitCamera {
  /** Radians. 0 looks down the −Y axis of model space; elevation π/2 is straight down. */
  readonly azimuth: number;
  readonly elevation: number;
  readonly distance: number;
  /** Pan offset in model units. */
  readonly target: readonly [number, number, number];
}

export interface TerrainOptions {
  readonly vectors: Float64Array;
  readonly camera: OrbitCamera;
  readonly heightSource: HeightSource;
  readonly colourBy: ColourBy;
  /** Model-space amplitude of the displacement. 0 flattens to the 2-D map. */
  readonly heightScale: number;
  /** How many σ of each mode's own spread saturate the ramp — shared with `paintGramModes`. */
  readonly saturate?: number;
  /** Grid decimation: sample every `step`-th pixel. 1 is one vertex per pixel. */
  readonly step?: number;
  /** The wand: the standardised channel vector to measure distance FROM, plus how many modes the
   *  metric keeps. Absent means no reference, and every distance reads 0. */
  readonly reference?: Float64Array;
  readonly modesUsed?: number;
  /** Distance at which similarity reaches 0. */
  readonly distanceSpan?: number;
  /** Where the wand sample was taken, in raster pixels — the same coordinates `paintGramModes`
   *  takes, so the caller does not have to know this module's model space. */
  readonly marker?: { readonly col: number; readonly row: number };
}

export interface TerrainInfo {
  readonly sigmas: [number, number, number];
  readonly gridW: number;
  readonly gridH: number;
  readonly triangles: number;
}

interface Ctx {
  device: GPUDevice;
  pipeline: GPURenderPipeline;
  probe: GPUComputePipeline;
  format: GPUTextureFormat;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const module = device.createShaderModule({ code: TERRAIN_SHADER });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    const probeModule = device.createShaderModule({ code: PROBE_SHADER });
    const probe = device.createComputePipeline({ layout: "auto", compute: { module: probeModule, entryPoint: "probe" } });
    return { device, pipeline, probe, format };
  })();
  return ctxCache;
}

const surfaces = new WeakMap<HTMLCanvasElement, GPUCanvasContext>();
let depth: { tex: GPUTexture; view: GPUTextureView; w: number; h: number } | undefined;
let uniBuf: GPUBuffer | undefined;
let chanBuf: GPUBuffer | undefined;
let wandBuf: GPUBuffer | undefined;
let chanCap = 0;
let wandCap = 0;

// Pooled and grow-only; `.destroy()` segfaults Dawn-on-Node's teardown, the constraint every module
// in this directory works under.
function ensure(device: GPUDevice, buf: GPUBuffer | undefined, cap: number, want: number, usage: number) {
  if (buf && cap >= want) return { buf, cap };
  const grown = Math.max(want, cap * 2, 16);
  return { buf: device.createBuffer({ size: grown * 4, usage }), cap: grown };
}

/**
 * The 4×4 model-view-projection for an orbit camera, column-major for WGSL.
 *
 * Written out rather than pulled from `wgpu-matrix` because it is twenty lines and this is the only
 * camera in the module — a dependency here would have to agree about handedness and depth range
 * with a shader that is right next to it.
 */
function mvpOf(cam: OrbitCamera, aspect: number): Float32Array {
  const ce = Math.cos(cam.elevation);
  const eye: [number, number, number] = [
    cam.target[0] + cam.distance * ce * Math.sin(cam.azimuth),
    cam.target[1] - cam.distance * ce * Math.cos(cam.azimuth),
    cam.target[2] + cam.distance * Math.sin(cam.elevation),
  ];
  // Z is up in model space (the displacement axis), so the world up is +Z.
  const f = norm([cam.target[0] - eye[0], cam.target[1] - eye[1], cam.target[2] - eye[2]]);
  const s = norm(cross(f, [0, 0, 1]));
  const u = cross(s, f);
  const fovY = Math.PI / 4;
  const t = 1 / Math.tan(fovY / 2);
  const near = 0.02;
  const far = 100;
  // Reverse-less standard perspective with WebGPU's [0,1] depth range.
  const p = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
  const v = [s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1];
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let acc = 0;
      for (let k = 0; k < 4; k++) acc += (p[k * 4 + r] ?? 0) * (v[c * 4 + k] ?? 0);
      out[c * 4 + r] = acc;
    }
  }
  return out;
}

type V3 = [number, number, number];
const cross = (a: V3 | number[], b: V3 | number[]): V3 => [
  (a[1] ?? 0) * (b[2] ?? 0) - (a[2] ?? 0) * (b[1] ?? 0),
  (a[2] ?? 0) * (b[0] ?? 0) - (a[0] ?? 0) * (b[2] ?? 0),
  (a[0] ?? 0) * (b[1] ?? 0) - (a[1] ?? 0) * (b[0] ?? 0),
];
const dot = (a: number[], b: number[]) => (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);
const norm = (a: number[]): V3 => {
  const l = Math.hypot(a[0] ?? 0, a[1] ?? 0, a[2] ?? 0) || 1;
  return [(a[0] ?? 0) / l, (a[1] ?? 0) / l, (a[2] ?? 0) / l];
};

/**
 * The whitening matrix `A = Λ^{-1/2} Vᵀ` truncated to `m` modes, row-major (m rows of K).
 *
 * Eigenvalues are floored before the inverse square root: `corr` is PSD by construction but its
 * trailing eigenvalues can be zero to machine precision (K channels spanning fewer than K
 * dimensions — perfectly collinear genes will do it), and 1/√0 would turn one pixel of numerical
 * noise into an infinite distance.
 */
export function whiteningMatrix(vectors: Float64Array, values: Float64Array, K: number, m: number): Float64Array {
  const floor = Math.max(values[0] ?? 1, 1e-12) * 1e-6;
  const out = new Float64Array(m * K);
  for (let k = 0; k < m; k++) {
    const inv = 1 / Math.sqrt(Math.max(values[k] ?? 0, floor));
    for (let a = 0; a < K; a++) out[k * K + a] = inv * (vectors[k * K + a] ?? 0);
  }
  return out;
}

/** Standardise raw raster values at a pixel into the `z` a wand reference is expressed in. */
export function standardise(raw: ArrayLike<number>, mean: Float64Array, sd: Float64Array): Float64Array {
  const out = new Float64Array(mean.length);
  for (let a = 0; a < mean.length; a++) out[a] = sd[a]! > 0 ? ((raw[a] ?? 0) - mean[a]!) / sd[a]! : 0;
  return out;
}

let probeUni: GPUBuffer | undefined;
let probeOut: { raw: GPUBuffer; staging: GPUBuffer; cap: number } | undefined;

/**
 * Read the K channel densities at one window pixel — a click turned into a wand reference.
 *
 * `col`/`row` are WINDOW coordinates; the apron offset is applied here so callers work in the
 * coordinates the canvas shows.
 */
export async function probeChannels(res: GramMatrixGpuResult, col: number, row: number): Promise<Float64Array> {
  const { device, probe } = await getCtx();
  const K = res.labels.length;
  const { buffer, rowFloats } = res.resident;
  probeUni ??= device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  if (!probeOut || probeOut.cap < K) {
    const cap = Math.max(K, 64);
    probeOut = {
      raw: device.createBuffer({ size: cap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      staging: device.createBuffer({ size: cap * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      cap,
    };
  }
  const c = Math.max(0, Math.min(res.width - 1, Math.round(col)));
  const r = Math.max(0, Math.min(res.height - 1, Math.round(row)));
  device.queue.writeBuffer(probeUni, 0, new Float32Array([K, c, r, rowFloats, res.height, 0, 0, 0]));
  const bind = device.createBindGroup({
    layout: probe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: probeUni, size: 32 } },
      { binding: 1, resource: { buffer } },
      { binding: 2, resource: { buffer: probeOut.raw } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(probe);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(K / 64));
  pass.end();
  enc.copyBufferToBuffer(probeOut.raw, 0, probeOut.staging, 0, K * 4);
  device.queue.submit([enc.finish()]);
  await probeOut.staging.mapAsync(GPUMapMode.READ, 0, K * 4);
  const got = Float64Array.from(new Float32Array(probeOut.staging.getMappedRange(0, K * 4)));
  probeOut.staging.unmap();
  return got;
}

/**
 * Draw the mode field as a lit, displaced surface.
 *
 * Reads `res.resident` in place — call before the next `gramMatrixGpu`, exactly as with
 * `paintGramModes`.
 */
export async function paintGramTerrain(canvas: HTMLCanvasElement, res: GramMatrixGpuResult, opts: TerrainOptions): Promise<TerrainInfo> {
  const { device, pipeline, format } = await getCtx();
  const K = res.labels.length;
  const { mean, sd, buffer, rowFloats } = res.resident;
  const saturate = opts.saturate ?? 2.5;
  const step = Math.max(1, Math.floor(opts.step ?? 1));
  const m = Math.max(1, Math.min(opts.modesUsed ?? 3, K, 32));

  const gridW = Math.max(2, Math.floor((res.width - 1) / step) + 1);
  const gridH = Math.max(2, Math.floor((res.height - 1) / step) + 1);
  const quads = (gridW - 1) * (gridH - 1);

  const chan = new Float32Array(K * 5);
  for (let a = 0; a < K; a++) {
    chan[a * 5] = mean[a]!;
    chan[a * 5 + 1] = sd[a]! > 0 ? 1 / sd[a]! : 0;
    for (let k = 0; k < 3; k++) chan[a * 5 + 2 + k] = k < K ? (opts.vectors[k * K + a] ?? 0) : 0;
  }

  // σ_k = √λ_k via vᵀ·corr·v — the same identity `paintGramModes` uses, so the two views saturate
  // identically and a mode that looks flat on one is flat on the other.
  const sigmas: [number, number, number] = [0, 0, 0];
  const lambda = new Float64Array(Math.max(K, 3));
  for (let k = 0; k < K; k++) {
    let q = 0;
    for (let a = 0; a < K; a++) {
      for (let b = 0; b < K; b++) q += (opts.vectors[k * K + a] ?? 0) * res.corr[a * K + b]! * (opts.vectors[k * K + b] ?? 0);
    }
    lambda[k] = Math.max(q, 0);
    if (k < 3) sigmas[k] = Math.sqrt(lambda[k]!);
  }
  const scales = sigmas.map((s) => (s > 0 ? 1 / (saturate * s) : 0)) as [number, number, number];

  // zRef then A, one buffer: the shader walks both per channel and a single binding keeps that loop
  // to one storage access pattern.
  const wandData = new Float32Array(K + m * K);
  if (opts.reference) {
    const A = whiteningMatrix(opts.vectors, lambda, K, m);
    for (let a = 0; a < K; a++) wandData[a] = opts.reference[a] ?? 0;
    for (let i = 0; i < m * K; i++) wandData[K + i] = A[i] ?? 0;
  }
  // With no reference every distance is 0, which would read as "everything matches". Scale to 0 so
  // the similarity views are uniformly *dissimilar* until a point is actually picked.
  const distScale = opts.reference ? 1 / Math.max(opts.distanceSpan ?? 1.2, 1e-6) : 1e9;

  let ctx = surfaces.get(canvas);
  if (!ctx) {
    ctx = canvas.getContext("webgpu") as GPUCanvasContext;
    surfaces.set(canvas, ctx);
  }
  ctx.configure({ device, format, alphaMode: "opaque" });
  const cw = canvas.width;
  const ch = canvas.height;
  if (!depth || depth.w !== cw || depth.h !== ch) {
    const tex = device.createTexture({
      size: { width: cw, height: ch },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depth = { tex, view: tex.createView(), w: cw, h: ch };
  }

  uniBuf ??= device.createBuffer({ size: UNI_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const gotChan = ensure(device, chanBuf, chanCap, chan.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  chanBuf = gotChan.buf;
  chanCap = gotChan.cap;
  const gotWand = ensure(device, wandBuf, wandCap, wandData.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  wandBuf = gotWand.buf;
  wandCap = gotWand.cap;

  const long = Math.max(res.width, res.height);
  const aspectX = res.width / long;
  const aspectY = res.height / long;
  const mvp = mvpOf(opts.camera, cw / Math.max(ch, 1));
  const srcCode = { mode1: 0, mode2: 1, mode3: 2, similarity: 3 }[opts.heightSource];

  // Raster pixel -> model XY, using the SAME expression the vertex shader does, so the rule lines
  // land on the vertex they name rather than half a grid cell off when `step` > 1.
  const markerModel: [number, number] = [0, 0];
  if (opts.marker) {
    const u = opts.marker.col / step / Math.max(gridW - 1, 1);
    const v = opts.marker.row / step / Math.max(gridH - 1, 1);
    markerModel[0] = (u - 0.5) * aspectX;
    markerModel[1] = (0.5 - v) * aspectY;
  }

  const uni = new Float32Array(UNI_FLOATS);
  uni.set(mvp, 0);
  uni.set(
    [
      gridW,
      gridH,
      step,
      rowFloats,
      res.height,
      K,
      aspectX,
      aspectY,
      opts.heightScale,
      srcCode,
      opts.colourBy === "similarity" ? 1 : 0,
      m,
      scales[0],
      scales[1],
      scales[2],
      distScale,
      markerModel[0],
      markerModel[1],
      opts.marker ? 1 : 0,
      // Minimum half-width as a fraction of the model's long side, which is 1 by construction. It
      // only bites where the screen-space derivative would make the line thinner than this.
      0.0012,
    ],
    16,
  );
  device.queue.writeBuffer(uniBuf, 0, uni);
  device.queue.writeBuffer(chanBuf, 0, chan);
  device.queue.writeBuffer(wandBuf, 0, wandData);

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf, size: UNI_FLOATS * 4 } },
      { binding: 1, resource: { buffer } },
      { binding: 2, resource: { buffer: chanBuf } },
      { binding: 3, resource: { buffer: wandBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: ctx.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0.043, g: 0.063, b: 0.125, a: 1 },
      },
    ],
    depthStencilAttachment: { view: depth.view, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.draw(quads * 6);
  pass.end();
  device.queue.submit([enc.finish()]);

  return { sigmas, gridW, gridH, triangles: quads * 2 };
}

/** The colour a similarity value maps to, in host code — for a legend that cannot drift from the
 *  shader, because it is the same expression. */
export function similaritySwatch(t: number): Oklab {
  const c = Math.max(0, Math.min(1, t));
  return [0.3 + 0.5 * c, MAX_CHROMA * c * SIM_A, MAX_CHROMA * c * SIM_B];
}
