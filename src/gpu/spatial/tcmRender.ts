// TCM as two render passes — the interactive path.
//
// The whole statistic is two additive splats with a pointwise nonlinearity between them:
//
//   pass 1  ρ̂_B = K_r ⊛ β        B splatted through a radial kernel  → r32float texture
//   pass 2  Γ   = G_σ ⊛ (M·α)    A splatted as Gaussians, where each point's WEIGHT is
//                                 M = 𝔐( ρ̂_B(x_a)/ρ_B , α ), read from pass 1 in the VERTEX stage
//
// There is no neighbour search anywhere: pass 1 replaces the disk count with a rasterised kernel
// density, and the per-A-cell evaluation that eq 9 demands becomes one texture fetch per instance.
// Both passes are the fixed-function additive blend (src=ONE, dst=ONE) into an r32float target —
// the same no-atomics accumulation `splatDensity` uses, since core WGSL has no atomic<f32>.
//
// Why the fetch is in the VERTEX stage and not the fragment stage: 𝔐 is nonlinear and eq 14 applies
// it PER CELL, before smoothing. Evaluating it per fragment would compute (G_σ ⊛ α)·𝔐(m) instead of
// G_σ ⊛ (𝔐(m)·α) — a different field, differing by a Jensen gap that is largest exactly where a
// neighbourhood mixes clustered and excluded A cells. One fetch per point keeps the paper's
// statistic; one fetch per pixel would quietly replace it.
//
// The approximation this trades for the speed is raster sampling of ρ̂_B: the mark is bilinearly
// interpolated from a `markWidth × markHeight` grid rather than evaluated at x_a. That error falls
// with mark resolution and — the tangible part — falls much faster for smooth kernels, because the
// top-hat's discontinuity at |u| = r is exactly what a raster cannot represent.
// `src/spatial/tcmKernel.ts` is the continuous f64 oracle this is measured against.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { GAUSS_TRUNC, type KernelSpec, kernelCode, TOPHAT } from "../../spatial/kernels";
import type { CellCloud } from "../../spatial/tcm";
import type { TcmKernelParams } from "../../spatial/tcmKernel";
import { getDevice } from "../device";

// Uniform block shared by both passes: one flat Float32Array, no std140 padding puzzles.
const UNI_FLOATS = 12;

const SHADER = /* wgsl */ `
struct Uni {
  minX: f32, minY: f32, invSpanX: f32, invSpanY: f32,
  radius: f32, kernelCode: f32,     // pass 1: kernel support + selector (-1 = gaussian)
  sigma: f32, radiusSigma: f32,     // pass 2: Gamma bandwidth + square support in sigmas
  rhoB: f32, alpha: f32,            // pass 2: CSR density and the extreme threshold
  markW: f32, markH: f32,           // pass 2: mark texture size
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;   // packed [x0,y0,x1,y1,...]

const PI = 3.14159265358979;
const E_HALF = 0.011108996538242306;   // exp(-GAUSS_TRUNC^2 / 2), the truncated Gaussian's lost mass
const T2 = ${(GAUSS_TRUNC * GAUSS_TRUNC).toFixed(1)};

// Unit-mass radial kernels, matching src/spatial/kernels.ts exactly.
//   code >= 0 : polynomial order n, K = (n+1)/(pi r^2) (1 - d^2/r^2)^n
//   code <  0 : gaussian truncated at r = GAUSS_TRUNC * sigma
fn kernelAt(d2: f32, r: f32, code: f32) -> f32 {
  let r2 = r * r;
  let t = d2 / r2;
  if (t >= 1.0) { return 0.0; }
  if (code < 0.0) {
    let s2 = r2 / T2;
    return exp(-d2 / (2.0 * s2)) / (2.0 * PI * s2 * (1.0 - E_HALF));
  }
  return ((code + 1.0) / (PI * r2)) * pow(1.0 - t, code);
}

// eqs 10-13. Kept branch-for-branch identical to markToM() in src/spatial/tcm.ts, including the
// reciprocal form below CSR (which is NOT a linear mirror).
fn markToM(m: f32, alpha: f32) -> f32 {
  if (m >= alpha) { return 1.0; }
  if (m > 1.0) { return (m - 1.0) / (alpha - 1.0); }
  if (m <= 1.0 / alpha) { return -1.0; }
  return (1.0 - 1.0 / m) / (alpha - 1.0);
}

// Corners of a triangle-strip quad from the vertex index — no vertex buffers.
fn corner(vi: u32) -> vec2f {
  return vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
}
fn toNdc(world: vec2f) -> vec4f {
  return vec4f((world.x - U.minX) * U.invSpanX * 2.0 - 1.0,
               (world.y - U.minY) * U.invSpanY * 2.0 - 1.0, 0.0, 1.0);
}

// ---- pass 1: kernel density of B ----

struct KOut { @builtin(position) pos: vec4f, @location(0) off: vec2f };

@vertex
fn vsKernel(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> KOut {
  let c = corner(vi);
  let centre = vec2f(pts[2u * ii], pts[2u * ii + 1u]);
  var o: KOut;
  o.pos = toNdc(centre + c * U.radius);
  o.off = c * U.radius;
  return o;
}

@fragment
fn fsKernel(in: KOut) -> @location(0) vec4f {
  return vec4f(kernelAt(dot(in.off, in.off), U.radius, U.kernelCode), 0.0, 0.0, 0.0);
}

// ---- pass 2: Gamma = Gaussian splat of A, weighted by M ----

@group(0) @binding(2) var markTex: texture_2d<f32>;

// Bilinear fetch from the pass-1 target. textureLoad rather than a sampler: r32float is not
// filterable without the float32-filterable feature, and doing the lerp by hand costs four loads
// and works on every adapter.
fn sampleMark(p: vec2f) -> f32 {
  let W = i32(U.markW);
  let H = i32(U.markH);
  // Row 0 of the target is at maxY (NDC +1), hence the flip in y.
  let fx = (p.x - U.minX) * U.invSpanX * U.markW - 0.5;
  let fy = (1.0 - (p.y - U.minY) * U.invSpanY) * U.markH - 0.5;
  let x0 = floor(fx);
  let y0 = floor(fy);
  let tx = fx - x0;
  let ty = fy - y0;
  let i0 = clamp(i32(x0), 0, W - 1);
  let j0 = clamp(i32(y0), 0, H - 1);
  let i1 = clamp(i32(x0) + 1, 0, W - 1);
  let j1 = clamp(i32(y0) + 1, 0, H - 1);
  let v00 = textureLoad(markTex, vec2i(i0, j0), 0).r;
  let v10 = textureLoad(markTex, vec2i(i1, j0), 0).r;
  let v01 = textureLoad(markTex, vec2i(i0, j1), 0).r;
  let v11 = textureLoad(markTex, vec2i(i1, j1), 0).r;
  return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

struct TOut { @builtin(position) pos: vec4f, @location(0) off: vec2f, @location(1) weight: f32 };

@vertex
fn vsTcm(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TOut {
  let c = corner(vi);
  let centre = vec2f(pts[2u * ii], pts[2u * ii + 1u]);
  let R = U.sigma * U.radiusSigma;
  // eq 9 -> eqs 10-13, once per CELL. This is the load-bearing line.
  var m = 0.0;
  if (U.rhoB > 0.0) { m = sampleMark(centre) / U.rhoB; }
  var o: TOut;
  o.pos = toNdc(centre + c * R);
  o.off = c * R;
  o.weight = markToM(m, U.alpha);
  return o;
}

@fragment
fn fsTcm(in: TOut) -> @location(0) vec4f {
  // eq 14's normalisation verbatim: 1/(sigma*sqrt(2pi)). (It is the 1-D constant; that is what the
  // paper specifies and what the CPU oracle uses, so it stays.)
  let norm = 1.0 / (U.sigma * sqrt(2.0 * PI));
  let g = in.weight * norm * exp(-0.5 * dot(in.off, in.off) / (U.sigma * U.sigma));
  return vec4f(g, 0.0, 0.0, 0.0);
}
`;

type Root = ReturnType<typeof tgpu.initFromDevice>;

interface Ctx {
  device: GPUDevice;
  root: Root;
  kernelPipe: GPURenderPipeline;
  tcmPipe: GPURenderPipeline;
}
let ctxCache: Promise<Ctx> | undefined;

const ADDITIVE = {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
} as const satisfies GPUBlendState;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const module = device.createShaderModule({ code: SHADER });
    const mk = (vs: string, fs: string) =>
      device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: vs },
        primitive: { topology: "triangle-strip" },
        fragment: { module, entryPoint: fs, targets: [{ format: "r32float", blend: ADDITIVE }] },
      });
    // Pipelines are built ON DEMAND, not both up front: a caller that only wants a kernel density
    // should not pay for (or keep alive) the TCM pipeline.
    let kernelPipe: GPURenderPipeline | undefined;
    let tcmPipe: GPURenderPipeline | undefined;
    return {
      device,
      root,
      get kernelPipe() {
        kernelPipe ??= mk("vsKernel", "fsKernel");
        return kernelPipe;
      },
      get tcmPipe() {
        tcmPipe ??= mk("vsTcm", "fsTcm");
        return tcmPipe;
      },
    };
  })();
  return ctxCache;
}

// --- pooled resources (grown, never destroyed: destroying mid-process segfaults Dawn's teardown) ---

const align256 = (n: number) => Math.ceil(n / 256) * 256;

let ptsBuf: GPUBuffer | undefined;
let ptsCap = 0;
function ensurePts(device: GPUDevice, floats: number) {
  if (ptsBuf && ptsCap >= floats) return ptsBuf;
  ptsCap = Math.max(floats, ptsCap * 2, 4);
  ptsBuf = device.createBuffer({ size: ptsCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  return ptsBuf;
}

let uniBuf: GPUBuffer | undefined;
function ensureUni(device: GPUDevice) {
  uniBuf ??= device.createBuffer({ size: UNI_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  return uniBuf;
}

// Textures are pooled WITH their views. Creating a fresh `createView()` per call is the obvious
// way to write this and it works, but every view is another live Dawn object at process exit, and
// Dawn-on-Node's atexit walk is exactly what segfaults once there are enough of them — the failure
// lands after the results are computed, so it reads as a mysterious lost test run rather than a
// leak. One view per texture, reused.
interface Tex {
  tex: GPUTexture;
  view: GPUTextureView;
  w: number;
  h: number;
}
const texPool = new Map<string, Tex>();
function ensureTex(device: GPUDevice, key: string, w: number, h: number, usage: number): Tex {
  const got = texPool.get(key);
  if (got && got.w === w && got.h === h) return got;
  const tex = device.createTexture({ size: { width: w, height: h }, format: "r32float", usage });
  const made: Tex = { tex, view: tex.createView(), w, h };
  texPool.set(key, made);
  return made;
}

let rbRaw: GPUBuffer | undefined;
let rbWrap: ReturnType<Root["createBuffer"]> | undefined;
let rbCap = 0;
function ensureReadback(device: GPUDevice, root: Root, floats: number) {
  if (rbWrap && rbCap >= floats) return;
  rbCap = Math.max(floats, rbCap * 2, 1);
  rbRaw = device.createBuffer({
    size: rbCap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  rbWrap = root.createBuffer(d.arrayOf(d.f32, rbCap), rbRaw);
}

// Bind groups memoised on the identity of what they point at. Pooled buffers only change identity
// when they grow, so in steady state this is one bind group per pass for the whole process —
// again to keep Dawn's exit-time object graph small.
let kernelBg: { bg: GPUBindGroup; ub: GPUBuffer; pb: GPUBuffer } | undefined;
function kernelBindGroup(device: GPUDevice, pipeline: GPURenderPipeline, ub: GPUBuffer, pb: GPUBuffer): GPUBindGroup {
  if (kernelBg && kernelBg.ub === ub && kernelBg.pb === pb) return kernelBg.bg;
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: pb } },
    ],
  });
  kernelBg = { bg, ub, pb };
  return bg;
}

let tcmBg: { bg: GPUBindGroup; ub: GPUBuffer; pb: GPUBuffer; view: GPUTextureView } | undefined;
function tcmBindGroup(device: GPUDevice, pipeline: GPURenderPipeline, ub: GPUBuffer, pb: GPUBuffer, mark: Tex): GPUBindGroup {
  if (tcmBg && tcmBg.ub === ub && tcmBg.pb === pb && tcmBg.view === mark.view) return tcmBg.bg;
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: pb } },
      { binding: 2, resource: mark.view },
    ],
  });
  tcmBg = { bg, ub, pb, view: mark.view };
  return bg;
}

function packXY(xs: ArrayLike<number>, ys: ArrayLike<number>): Float32Array {
  const n = xs.length;
  const flat = new Float32Array(2 * Math.max(n, 1));
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  return flat;
}

/** Strip the 256-byte row padding `copyTextureToBuffer` forces, into a tight w*h grid. Row 0 of the
 *  texture is at maxY, so this also flips y to the project's row-0-at-minY convention. */
function unpad(padded: ArrayLike<number>, w: number, h: number, rowFloats: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let row = 0; row < h; row++) {
    const src = row * rowFloats;
    const dst = (h - 1 - row) * w;
    for (let col = 0; col < w; col++) out[dst + col] = padded[src + col]!;
  }
  return out;
}

export interface KernelDensityOptions {
  width: number;
  height: number;
  /** Kernel support radius in world units. */
  radius: number;
  /** Default `TOPHAT` — the paper's disk. */
  kernel?: KernelSpec;
  bbox: readonly [number, number, number, number];
}

/** Render a point cloud through a unit-mass radial kernel: ρ̂(x) = (K_r ⊛ points)(x).
 *
 *  This is the reusable half of the TCM — it is also what the Gram-matrix form of the N-way
 *  cross-PCF needs (one of these per cell type). Row 0 of the result is at `bbox` minY. */
export async function kernelDensityGpu(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: KernelDensityOptions): Promise<Float32Array> {
  const { device, root, kernelPipe } = await getCtx();
  const { width: w, height: h } = opts;
  const tex = ensureTex(device, "solo", w, h, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
  const bytesPerRow = align256(w * 4);
  ensureReadback(device, root, (bytesPerRow / 4) * h);

  const enc = device.createCommandEncoder();
  drawKernel(device, enc, kernelPipe, tex, xs, ys, opts.radius, opts.kernel ?? TOPHAT, opts.bbox);
  enc.copyTextureToBuffer({ texture: tex.tex }, { buffer: rbRaw!, bytesPerRow }, { width: w, height: h });
  device.queue.submit([enc.finish()]);

  return unpad((await rbWrap!.read()) as ArrayLike<number>, w, h, bytesPerRow / 4);
}

/** Encode one kernel-density pass into `enc`. Shared by the standalone op and the TCM's pass 1. */
function drawKernel(
  device: GPUDevice,
  enc: GPUCommandEncoder,
  pipeline: GPURenderPipeline,
  target: Tex,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  radius: number,
  kernel: KernelSpec,
  bbox: readonly [number, number, number, number],
) {
  const [minX, minY, maxX, maxY] = bbox;
  const uni = new Float32Array(UNI_FLOATS);
  uni.set([minX, minY, 1 / (maxX - minX || 1), 1 / (maxY - minY || 1), radius, kernelCode(kernel)]);
  const ub = ensureUni(device);
  device.queue.writeBuffer(ub, 0, uni);
  const pb = ensurePts(device, 2 * Math.max(xs.length, 1));
  device.queue.writeBuffer(pb, 0, packXY(xs, ys));

  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: target.view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, kernelBindGroup(device, pipeline, ub, pb));
  if (xs.length > 0) pass.draw(4, xs.length);
  pass.end();
}

export interface TcmRenderParams extends TcmKernelParams {
  /** Resolution of the intermediate B-density field. Defaults to the output grid; raising it is
   *  the knob that closes the gap to the continuous oracle (the mark is sampled from it). */
  markWidth?: number;
  markHeight?: number;
}

/** Γ_ab(x) by the two-pass render formulation — the interactive path, and a drop-in for
 *  `computeTcm` / `computeTcmKernel`. Row 0 is at minY. */
export async function computeTcmRender(a: CellCloud, b: CellCloud, p: TcmRenderParams): Promise<Float32Array> {
  const { device, root, kernelPipe, tcmPipe } = await getCtx();
  const { width: w, height: h, sigma } = p;
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const rhoB = b.xs.length / roiArea;
  const mw = p.markWidth ?? w;
  const mh = p.markHeight ?? h;

  const markTex = ensureTex(device, "mark", mw, mh, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING);
  const outTex = ensureTex(device, "gamma", w, h, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);
  const bytesPerRow = align256(w * 4);
  ensureReadback(device, root, (bytesPerRow / 4) * h);

  // Pass 1 — B through the mark kernel. Encoded and SUBMITTED on its own: pass 2 samples this
  // texture, and the point/uniform buffers are pooled, so the two passes cannot share an encoder
  // without the second overwriting the first's inputs before it runs.
  const enc1 = device.createCommandEncoder();
  drawKernel(device, enc1, kernelPipe, markTex, b.xs, b.ys, p.radius, p.kernel ?? TOPHAT, p.bbox);
  device.queue.submit([enc1.finish()]);

  // Pass 2 — A as weighted Gaussians, weight fetched from pass 1 per point.
  const uni = new Float32Array(UNI_FLOATS);
  uni.set([
    minX,
    minY,
    1 / (maxX - minX || 1),
    1 / (maxY - minY || 1),
    p.radius,
    kernelCode(p.kernel ?? TOPHAT),
    sigma,
    p.radiusSigma ?? 4,
    rhoB,
    p.alpha,
    mw,
    mh,
  ]);
  const ub = ensureUni(device);
  device.queue.writeBuffer(ub, 0, uni);
  const pb = ensurePts(device, 2 * Math.max(a.xs.length, 1));
  device.queue.writeBuffer(pb, 0, packXY(a.xs, a.ys));

  const enc2 = device.createCommandEncoder();
  const pass = enc2.beginRenderPass({
    colorAttachments: [{ view: outTex.view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
  });
  pass.setPipeline(tcmPipe);
  pass.setBindGroup(0, tcmBindGroup(device, tcmPipe, ub, pb, markTex));
  if (a.xs.length > 0) pass.draw(4, a.xs.length);
  pass.end();
  enc2.copyTextureToBuffer({ texture: outTex.tex }, { buffer: rbRaw!, bytesPerRow }, { width: w, height: h });
  device.queue.submit([enc2.finish()]);

  return unpad((await rbWrap!.read()) as ArrayLike<number>, w, h, bytesPerRow / 4);
}
