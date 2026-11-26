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
// the same no-atomics accumulation `splatDensity` uses, since core WGSL has no atomic<f32>. That
// blend needs the device feature `float32-blendable`; there is no fallback here.
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
//
// WEIGHTS. A and B are given as per-point weights, so both can be weightings of ONE point set (a
// table's x/y with a 0/1 column per cell type) and a pair costs two weight arrays, not two copies of
// the coordinates. A weight of 0 leaves a point out. With 0/1 weights this is exactly the TCM of the
// two subsets; other non-negative weights act as masses: ρ̂_B and ρ_B are weight per unit area, and
// an A cell's term in Γ is scaled by its weight.
//
// DEVICES AND OWNERSHIP. A `TcmRenderer` belongs to one device and owns every resource it renders
// with: the uniforms, the uploaded points, the mark and Γ textures, and its bind groups. Two renderers
// never share any of them, so any number of TCMs can be alive at once (a deck layer per pair, a grid
// of them), on any number of devices. Pipelines are shared per device, since they hold no data.
//
// A render is SYNCHRONOUS: it writes, encodes and submits in one call, with no await anywhere. So
// once it returns, nothing on the host still has to submit work that uses the renderer's buffers, and
// destroying them (`destroy()`) is always safe: WebGPU lets already-submitted work finish. That is
// the rule `DensityLayer` had to retrofit (deck finalizes layers mid-splat); here it holds by
// construction.
//
// The module-level functions (`renderTcm`, `computeTcmRender`, `renderKernelDensity`,
// `kernelDensityGpu`) are the earlier API, kept: they use one renderer per device and key, so their
// output is still overwritten by the next call with the same key.
import { type KernelSpec, kernelCode, TOPHAT } from "#src/spatial/kernels";
import type { CellCloud } from "#src/spatial/tcm";
import type { TcmKernelParams } from "#src/spatial/tcmKernel";
import { checkBindingSize, getDevice, perDevice, sized, writeView } from "../device";
import { KERNEL_WGSL } from "./kernelWgsl";

// Uniform block shared by both passes: one flat Float32Array, no std140 padding puzzles.
const UNI_FLOATS = 20;

const SHADER = /* wgsl */ `
struct Uni {
  minX: f32, minY: f32, invSpanX: f32, invSpanY: f32,         // the Γ grid (pass 2's target)
  markMinX: f32, markMinY: f32, markInvX: f32, markInvY: f32, // the mark grid (pass 1's target)
  radius: f32, kernelCode: f32,     // pass 1: kernel support + selector (-1 = gaussian)
  sigma: f32, radiusSigma: f32,     // pass 2: Gamma bandwidth + square support in sigmas
  rhoB: f32, alpha: f32,            // pass 2: CSR density and the extreme threshold
  markW: f32, markH: f32,           // pass 2: mark texture size
  pad0: f32, pad1: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;   // packed [x0,y0,x1,y1,...]
@group(0) @binding(3) var<storage, read> wts: array<f32>;   // one weight per point; 0 = left out

${KERNEL_WGSL}

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
fn toMarkNdc(world: vec2f) -> vec4f {
  return vec4f((world.x - U.markMinX) * U.markInvX * 2.0 - 1.0,
               (world.y - U.markMinY) * U.markInvY * 2.0 - 1.0, 0.0, 1.0);
}
// A point of weight 0 is drawn as a quad of zero area: all four corners at one place off screen.
const OFF = vec4f(2.0, 2.0, 0.0, 1.0);

// ---- pass 1: kernel density of B ----

struct KOut { @builtin(position) pos: vec4f, @location(0) off: vec2f, @location(1) weight: f32 };

@vertex
fn vsKernel(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> KOut {
  let c = corner(vi);
  let centre = vec2f(pts[2u * ii], pts[2u * ii + 1u]);
  let w = wts[ii];
  var o: KOut;
  o.pos = select(OFF, toMarkNdc(centre + c * U.radius), w > 0.0);
  o.off = c * U.radius;
  o.weight = w;
  return o;
}

@fragment
fn fsKernel(in: KOut) -> @location(0) vec4f {
  return vec4f(in.weight * kernelAt(dot(in.off, in.off), U.radius, U.kernelCode), 0.0, 0.0, 0.0);
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
  let fx = (p.x - U.markMinX) * U.markInvX * U.markW - 0.5;
  let fy = (1.0 - (p.y - U.markMinY) * U.markInvY) * U.markH - 0.5;
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
  let w = wts[ii];
  let R = U.sigma * U.radiusSigma;
  // eq 9 -> eqs 10-13, once per CELL. This is the load-bearing line.
  var m = 0.0;
  if (U.rhoB > 0.0) { m = sampleMark(centre) / U.rhoB; }
  var o: TOut;
  o.pos = select(OFF, toNdc(centre + c * R), w > 0.0);
  o.off = c * R;
  o.weight = w * markToM(m, U.alpha);
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

// max |Γ| over a field copied out of the Γ texture, written as the u32 bit pattern of an f32 (see
// `fieldMax.ts`, which this follows: abs() clears the sign bit, so bit order is numeric order and
// `atomicMax` on u32 is an exact float max). Rows are `rowFloats` apart, as `copyTextureToBuffer`
// pads them.
const ABS_MAX_WG = 256;
const ABS_MAX_GROUPS = 1024;
const ABS_MAX_SHADER = /* wgsl */ `
struct Uni { w: u32, h: u32, rowFloats: u32, stride: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;

var<workgroup> part: array<f32, ${ABS_MAX_WG}>;

@compute @workgroup_size(${ABS_MAX_WG})
fn absMax(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = U.w * U.h;
  var m = 0.0;
  for (var i = gid.x; i < n; i = i + U.stride) {
    let row = i / U.w;
    m = max(m, abs(src[row * U.rowFloats + (i - row * U.w)]));
  }
  part[li] = m;
  workgroupBarrier();
  for (var s = ${ABS_MAX_WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) {
      part[li] = max(part[li], part[li + s]);
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicMax(&dst[0], bitcast<u32>(part[0]));
  }
}
`;

const ADDITIVE = {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
} as const satisfies GPUBlendState;

/** `createShaderModule` never throws, and a module that failed to compile only shows up as a cascade
 *  of "invalid due to a previous error". Report the WGSL diagnostics without awaiting them, so that
 *  building a pipeline stays synchronous. */
function reportCompile(module: GPUShaderModule, label: string): void {
  void module
    .getCompilationInfo?.()
    .then((info) => {
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0)
        console.error(`WGSL compile failed in '${label}':\n${errors.map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
    })
    .catch(() => undefined);
}

/** Pipelines, per device: they hold no data, so every renderer on a device shares them. Each is built
 *  on first use, so a caller that only wants a kernel density never builds the TCM pipeline. */
const pipelines = perDevice((device: GPUDevice) => {
  let module: GPUShaderModule | undefined;
  const render = () => {
    if (!module) {
      module = device.createShaderModule({ label: "tcmRender", code: SHADER });
      reportCompile(module, "tcmRender");
    }
    return module;
  };
  const mk = (vs: string, fs: string) =>
    device.createRenderPipeline({
      label: `tcmRender:${fs}`,
      layout: "auto",
      vertex: { module: render(), entryPoint: vs },
      primitive: { topology: "triangle-strip" },
      fragment: { module: render(), entryPoint: fs, targets: [{ format: "r32float", blend: ADDITIVE }] },
    });
  let kernel: GPURenderPipeline | undefined;
  let tcm: GPURenderPipeline | undefined;
  let absMax: GPUComputePipeline | undefined;
  return {
    get kernel() {
      kernel ??= mk("vsKernel", "fsKernel");
      return kernel;
    },
    get tcm() {
      tcm ??= mk("vsTcm", "fsTcm");
      return tcm;
    },
    get absMax() {
      if (!absMax) {
        const m = device.createShaderModule({ label: "tcmRender:absMax", code: ABS_MAX_SHADER });
        reportCompile(m, "tcmRender:absMax");
        absMax = device.createComputePipeline({ label: "tcmRender:absMax", layout: "auto", compute: { module: m, entryPoint: "absMax" } });
      }
      return absMax;
    },
  };
});

const align256 = (n: number) => Math.ceil(n / 256) * 256;

/** Floats per row when a `width`-texel r32float field is copied out of its texture: rows are padded
 *  to 256 bytes. A buffer receiving `TcmRenderOptions.out` holds `tcmRowFloats(width) × height`
 *  floats, row 0 at the grid's maxY. */
export function tcmRowFloats(width: number): number {
  return align256(width * 4) / 4;
}

/** An r32float render target with its one view. Pooled WITH the view: every view is another live
 *  Dawn object at process exit, so one is made per texture and reused. */
export interface FieldTexture {
  tex: GPUTexture;
  view: GPUTextureView;
  w: number;
  h: number;
}

/** A grid in world units: `width × height` cells tiling `bbox` = [minX, minY, maxX, maxY]. */
export interface TcmGrid {
  bbox: readonly [number, number, number, number];
  width: number;
  height: number;
}

/** Points on the device: `points` is [x, y] per point (f32), `weights` one f32 per point. Both need
 *  STORAGE usage and must hold at least `count` points (and at least one, even when `count` is 0). */
export interface ResidentPoints {
  points: GPUBuffer;
  weights: GPUBuffer;
  count: number;
}

export interface TcmResidentInput {
  /** The cells Γ is about. May share `points` (and even `weights`) with `b`. */
  a: ResidentPoints;
  /** The cells whose density marks each A cell. */
  b: ResidentPoints;
  /** ρ_B: B's total weight per unit area of the region the statistic is taken over (Mode 1's ROI).
   *  An A cell's mark is ρ̂_B(x_a) / ρ_B. */
  rhoB: number;
}

export interface TcmRenderOptions {
  /** m_ab neighbourhood radius in world units (paper: 100 µm). */
  radius: number;
  /** TCM Gaussian bandwidth in world units (paper: 50 µm). */
  sigma: number;
  /** Extreme clustering/exclusion threshold α (paper: 5). */
  alpha: number;
  /** Mark kernel. Default `TOPHAT`, the paper's disk. */
  kernel?: KernelSpec;
  /** Γ Gaussian support half-extent in units of σ. Default 4. */
  radiusSigma?: number;
  /** Where Γ is rendered. */
  grid: TcmGrid;
  /** Where ρ̂_B is rendered, to be sampled at each A cell. It must cover every A cell whose Gaussian
   *  reaches `grid` (`grid` padded by `sigma · radiusSigma`); a cell outside it reads the edge. Its
   *  resolution is the accuracy knob. Default: `grid`. */
  markGrid?: TcmGrid;
  /** Also copy Γ into caller-owned buffers, in the same submit:
   *  - `buffer` (STORAGE | COPY_DST): `tcmRowFloats(grid.width) × grid.height` f32, row 0 at maxY;
   *  - `max` (STORAGE | COPY_DST, ≥ 4 bytes): max |Γ| over the grid, as the u32 bits of an f32. */
  out?: { buffer: GPUBuffer; max?: GPUBuffer };
}

export interface KernelDensityOptions {
  width: number;
  height: number;
  /** Kernel support radius in world units. */
  radius: number;
  /** Default `TOPHAT` — the paper's disk. */
  kernel?: KernelSpec;
  bbox: readonly [number, number, number, number];
  /** Which pooled render target the module-level functions use. Callers that keep two densities on
   *  screen at once (an A/B pair, say) must pass distinct keys, or the second render overwrites the
   *  first's texture. (A `TcmRenderer` of one's own avoids the question.) */
  key?: string;
  /** The device to render on. Default: the library's own (`getDevice()`). */
  device?: GPUDevice;
}

export interface TcmRenderParams extends TcmKernelParams {
  /** Resolution of the intermediate B-density field. Defaults to the output grid; raising it is
   *  the knob that closes the gap to the continuous oracle (the mark is sampled from it). */
  markWidth?: number;
  markHeight?: number;
  /** The device to render on (module-level functions only). Default: the library's own. */
  device?: GPUDevice;
}

/** A unit of 1 for each of `n` points. */
function ones(n: number): Float32Array {
  return new Float32Array(Math.max(n, 1)).fill(1);
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
function unpad(padded: Float32Array, w: number, h: number, rowFloats: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let row = 0; row < h; row++) {
    const src = row * rowFloats;
    const dst = (h - 1 - row) * w;
    out.set(padded.subarray(src, src + w), dst);
  }
  return out;
}

function gridUniforms(uni: Float32Array, at: number, g: TcmGrid): void {
  const [minX, minY, maxX, maxY] = g.bbox;
  uni.set([minX, minY, 1 / (maxX - minX || 1), 1 / (maxY - minY || 1)], at);
}

interface Staging {
  buf: GPUBuffer;
  busy: boolean;
}

/** A grow-only host-upload buffer. Replaced (and the old one destroyed) when too small: everything
 *  that used it has been submitted already, since a render never awaits. */
interface Upload {
  buf?: GPUBuffer;
}

/**
 * The two-pass TCM render on one device, owning everything it renders with — see "DEVICES AND
 * OWNERSHIP" at the top of this file. Render calls are synchronous; outputs live until the next
 * render on the same renderer (or `destroy()`).
 */
export class TcmRenderer {
  readonly device: GPUDevice;
  readonly label: string;
  private readonly uni: GPUBuffer;
  private absMaxUni?: GPUBuffer;
  private mark?: FieldTexture;
  private gamma?: FieldTexture;
  private readonly uploads = { aPts: {} as Upload, aW: {} as Upload, bPts: {} as Upload, bW: {} as Upload };
  private staging: Staging[] = [];
  private kernelBg?: { bg: GPUBindGroup; key: unknown[] };
  private tcmBg?: { bg: GPUBindGroup; key: unknown[] };
  private destroyed = false;

  /** `label` prefixes every GPU resource's label, so a device error names its owner (a deck layer
   *  passes its id). Throws if the device lacks `float32-blendable`, which both passes need. */
  constructor(device: GPUDevice, label = "tcmRender") {
    if (!device.features.has("float32-blendable")) {
      throw new Error(
        `${label}: this device lacks 'float32-blendable', which the TCM render needs (it blends into r32float). ` +
          "Request it when creating the device; not every adapter offers it.",
      );
    }
    this.device = device;
    this.label = label;
    this.uni = device.createBuffer({
      label: `${label}:uniforms`,
      size: UNI_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Γ by the two-pass render, from points already on this renderer's device. Returns the Γ texture
   *  (row 0 at the grid's maxY), which this renderer owns and the next render overwrites. */
  renderResident(input: TcmResidentInput, o: TcmRenderOptions): FieldTexture {
    this.live();
    const device = this.device;
    const pipes = pipelines(device);
    const markGrid = o.markGrid ?? o.grid;
    const mark = (this.mark = this.target(this.mark, "mark", markGrid.width, markGrid.height));
    const gamma = (this.gamma = this.target(this.gamma, "gamma", o.grid.width, o.grid.height));
    // The max |Γ| reduction binds the whole padded field; check the device allows it before anything
    // is encoded, so an oversized grid fails here rather than as a silent validation error.
    if (o.out?.max) checkBindingSize(device, this.label, tcmRowFloats(o.grid.width) * o.grid.height * 4);

    const uni = new Float32Array(UNI_FLOATS);
    gridUniforms(uni, 0, o.grid);
    gridUniforms(uni, 4, markGrid);
    uni.set(
      [o.radius, kernelCode(o.kernel ?? TOPHAT), o.sigma, o.radiusSigma ?? 4, input.rhoB, o.alpha, markGrid.width, markGrid.height],
      8,
    );
    writeView(device.queue, this.uni, uni);

    // Both passes in one encoder: the render-pass boundary orders pass 2's reads of the mark after
    // pass 1's writes, and each pass binds its own points, so neither overwrites the other's inputs.
    const enc = device.createCommandEncoder({ label: `${this.label}:tcm` });
    this.drawKernel(enc, pipes.kernel, mark, input.b);
    const pass = enc.beginRenderPass({
      label: `${this.label}:gamma`,
      colorAttachments: [{ view: gamma.view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipes.tcm);
    pass.setBindGroup(0, this.tcmBindGroup(pipes.tcm, input.a, mark));
    if (input.a.count > 0) pass.draw(4, input.a.count);
    pass.end();
    if (o.out) this.encodeOut(enc, gamma, o.out);
    device.queue.submit([enc.finish()]);
    return gamma;
  }

  /** Γ from host arrays: A and B as separate clouds, weight 1 each, and ρ_B = |B| / area(bbox) — the
   *  `computeTcm` / `computeTcmKernel` contract. Uploads into buffers this renderer owns. */
  render(a: CellCloud, b: CellCloud, p: TcmRenderParams, out?: TcmRenderOptions["out"]): FieldTexture {
    this.live();
    const [minX, minY, maxX, maxY] = p.bbox;
    const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
    const grid: TcmGrid = { bbox: p.bbox, width: p.width, height: p.height };
    return this.renderResident(
      { a: this.uploadCloud(a, "a"), b: this.uploadCloud(b, "b"), rhoB: b.xs.length / roiArea },
      {
        radius: p.radius,
        sigma: p.sigma,
        alpha: p.alpha,
        kernel: p.kernel,
        radiusSigma: p.radiusSigma,
        grid,
        markGrid: { bbox: p.bbox, width: p.markWidth ?? p.width, height: p.markHeight ?? p.height },
        out,
      },
    );
  }

  /** Pass 1 alone: ρ̂(x) = (K_r ⊛ points)(x), unit-mass kernels, into this renderer's mark texture
   *  (row 0 at `bbox` maxY). Also what the Gram-matrix form of the N-way cross-PCF needs. */
  renderDensity(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: Omit<KernelDensityOptions, "key" | "device">): FieldTexture {
    this.live();
    const mark = (this.mark = this.target(this.mark, "mark", opts.width, opts.height));
    const grid: TcmGrid = { bbox: opts.bbox, width: opts.width, height: opts.height };
    const uni = new Float32Array(UNI_FLOATS);
    gridUniforms(uni, 0, grid);
    gridUniforms(uni, 4, grid);
    uni.set([opts.radius, kernelCode(opts.kernel ?? TOPHAT)], 8);
    writeView(this.device.queue, this.uni, uni);
    const enc = this.device.createCommandEncoder({ label: `${this.label}:density` });
    this.drawKernel(enc, pipelines(this.device).kernel, mark, this.uploadCloud({ xs, ys }, "b"));
    this.device.queue.submit([enc.finish()]);
    return mark;
  }

  /** Download a texture this renderer returned, as a tight `w × h` grid with row 0 at minY. The copy
   *  is encoded NOW, so the result is the texture as it is at this call, whatever renders follow. */
  read(t: FieldTexture): Promise<Float32Array> {
    this.live();
    const rowFloats = tcmRowFloats(t.w);
    const bytes = rowFloats * t.h * 4;
    let s = this.staging.find((x) => !x.busy && x.buf.size >= bytes);
    if (!s) {
      // Idle buffers too small for this read are replaced, not kept: a series of growing reads would
      // otherwise hold one buffer per size. Busy ones stay until their reads finish.
      for (const x of this.staging) if (!x.busy) x.buf.destroy();
      this.staging = this.staging.filter((x) => x.busy);
      s = {
        buf: this.device.createBuffer({
          label: `${this.label}:readback`,
          size: bytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        busy: false,
      };
      this.staging.push(s);
    }
    const staging = s;
    staging.busy = true;
    const enc = this.device.createCommandEncoder({ label: `${this.label}:read` });
    enc.copyTextureToBuffer({ texture: t.tex }, { buffer: staging.buf, bytesPerRow: rowFloats * 4 }, { width: t.w, height: t.h });
    this.device.queue.submit([enc.finish()]);
    return staging.buf
      .mapAsync(GPUMapMode.READ, 0, bytes)
      .then(() => {
        // Unpadded straight from the mapped range, which `unpad` copies out of before the unmap.
        try {
          return unpad(new Float32Array(staging.buf.getMappedRange(0, bytes)), t.w, t.h, rowFloats);
        } finally {
          staging.buf.unmap();
        }
      })
      .finally(() => {
        staging.busy = false;
      });
  }

  /** Free every GPU resource this renderer owns. Safe at any time: nothing it encodes is waiting to
   *  be submitted (renders never await), and submitted work completes regardless. A read in flight
   *  rejects. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uni.destroy();
    this.absMaxUni?.destroy();
    this.mark?.tex.destroy();
    this.gamma?.tex.destroy();
    for (const u of Object.values(this.uploads)) u.buf?.destroy();
    for (const s of this.staging) s.buf.destroy();
    this.staging = [];
    this.kernelBg = undefined;
    this.tcmBg = undefined;
  }

  private live(): void {
    if (this.destroyed) throw new Error(`${this.label}: used after destroy()`);
  }

  private target(cur: FieldTexture | undefined, name: string, w: number, h: number): FieldTexture {
    if (cur && cur.w === w && cur.h === h) return cur;
    const max = this.device.limits.maxTextureDimension2D;
    if (!(w >= 1 && h >= 1 && w <= max && h <= max) || !Number.isInteger(w) || !Number.isInteger(h)) {
      throw new Error(`${this.label}: a ${w}×${h} ${name} grid is not a valid texture size on this device (1–${max} per side)`);
    }
    cur?.tex.destroy();
    const tex = this.device.createTexture({
      label: `${this.label}:${name}`,
      size: { width: w, height: h },
      format: "r32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    return { tex, view: tex.createView({ label: `${this.label}:${name}` }), w, h };
  }

  private upload(u: Upload, name: string, data: Float32Array): GPUBuffer {
    if (!u.buf || u.buf.size < data.byteLength) {
      u.buf?.destroy();
      // Grown by doubling; bindings are sized to the call (`sized`), never to the buffer.
      const size = Math.max(data.byteLength, (u.buf?.size ?? 0) * 2, 16);
      u.buf = this.device.createBuffer({ label: `${this.label}:${name}`, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    writeView(this.device.queue, u.buf, data);
    return u.buf;
  }

  private uploadCloud(c: { xs: ArrayLike<number>; ys: ArrayLike<number> }, which: "a" | "b"): ResidentPoints {
    const n = c.xs.length;
    const up = this.uploads;
    return {
      points: this.upload(which === "a" ? up.aPts : up.bPts, `${which}:points`, packXY(c.xs, c.ys)),
      weights: this.upload(which === "a" ? up.aW : up.bW, `${which}:weights`, ones(n)),
      count: n,
    };
  }

  private drawKernel(enc: GPUCommandEncoder, pipeline: GPURenderPipeline, target: FieldTexture, b: ResidentPoints): void {
    const pass = enc.beginRenderPass({
      label: `${this.label}:mark`,
      colorAttachments: [{ view: target.view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.kernelBindGroup(pipeline, b));
    if (b.count > 0) pass.draw(4, b.count);
    pass.end();
  }

  // Bind groups memoised on the identity of what they point at, plus the point count: the bindings
  // are sized to THIS call (`sized` in `../device.ts`), because a grow-only buffer bound whole can
  // cross maxStorageBufferBindingSize long before the data does, and the pass then silently draws
  // nothing. In steady state this is one bind group per pass.
  private kernelBindGroup(pipeline: GPURenderPipeline, b: ResidentPoints): GPUBindGroup {
    const n = Math.max(b.count, 1);
    const key = [pipeline, b.points, b.weights, n];
    if (this.kernelBg && sameKey(this.kernelBg.key, key)) return this.kernelBg.bg;
    const bg = this.device.createBindGroup({
      label: `${this.label}:mark`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: sized(b.points, n * 8) },
        { binding: 3, resource: sized(b.weights, n * 4) },
      ],
    });
    this.kernelBg = { bg, key };
    return bg;
  }

  private tcmBindGroup(pipeline: GPURenderPipeline, a: ResidentPoints, mark: FieldTexture): GPUBindGroup {
    const n = Math.max(a.count, 1);
    const key = [pipeline, a.points, a.weights, n, mark.view];
    if (this.tcmBg && sameKey(this.tcmBg.key, key)) return this.tcmBg.bg;
    const bg = this.device.createBindGroup({
      label: `${this.label}:gamma`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: sized(a.points, n * 8) },
        { binding: 2, resource: mark.view },
        { binding: 3, resource: sized(a.weights, n * 4) },
      ],
    });
    this.tcmBg = { bg, key };
    return bg;
  }

  private encodeOut(enc: GPUCommandEncoder, gamma: FieldTexture, out: { buffer: GPUBuffer; max?: GPUBuffer }): void {
    const rowFloats = tcmRowFloats(gamma.w);
    const bytes = rowFloats * gamma.h * 4;
    if (out.buffer.size < bytes) {
      throw new Error(
        `${this.label}: the output buffer holds ${out.buffer.size} bytes; a ${gamma.w}×${gamma.h} field needs ${bytes} (tcmRowFloats(width) × height × 4)`,
      );
    }
    enc.copyTextureToBuffer(
      { texture: gamma.tex },
      { buffer: out.buffer, bytesPerRow: rowFloats * 4 },
      { width: gamma.w, height: gamma.h },
    );
    if (!out.max) return;
    const n = gamma.w * gamma.h;
    const groups = Math.max(1, Math.min(ABS_MAX_GROUPS, Math.ceil(n / ABS_MAX_WG)));
    this.absMaxUni ??= this.device.createBuffer({
      label: `${this.label}:absMax`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    writeView(this.device.queue, this.absMaxUni, new Uint32Array([gamma.w, gamma.h, rowFloats, groups * ABS_MAX_WG]));
    enc.clearBuffer(out.max, 0, 4);
    const pipeline = pipelines(this.device).absMax;
    const pass = enc.beginComputePass({ label: `${this.label}:absMax` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        label: `${this.label}:absMax`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.absMaxUni } },
          { binding: 1, resource: sized(out.buffer, bytes) },
          { binding: 2, resource: sized(out.max, 4) },
        ],
      }),
    );
    pass.dispatchWorkgroups(groups);
    pass.end();
  }
}

function sameKey(a: unknown[], b: unknown[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

// --- the module-level API: one renderer per device and key ---

const shared = perDevice(() => new Map<string, TcmRenderer>());

async function sharedRenderer(device: GPUDevice | undefined, key: string): Promise<TcmRenderer> {
  const dev = device ?? (await getDevice());
  const map = shared(dev);
  let r = map.get(key);
  if (!r) {
    r = new TcmRenderer(dev, `tcmRender:${key}`);
    map.set(key, r);
  }
  return r;
}

/** Render a point cloud through a unit-mass radial kernel: ρ̂(x) = (K_r ⊛ points)(x).
 *
 *  This is the reusable half of the TCM — it is also what the Gram-matrix form of the N-way
 *  cross-PCF needs (one of these per cell type). Row 0 of the texture is at `bbox` maxY; it is
 *  pooled under `key` and overwritten by the next call with that key. */
export async function renderKernelDensity(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: KernelDensityOptions): Promise<FieldTexture> {
  const r = await sharedRenderer(opts.device, `density:${opts.key ?? "solo"}`);
  return r.renderDensity(xs, ys, opts);
}

/** As `renderKernelDensity`, but downloaded (row 0 at minY). For callers that consume the numbers;
 *  for display, render and paint (`paintFieldTexture`) so the field never leaves the device. */
export async function kernelDensityGpu(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: KernelDensityOptions): Promise<Float32Array> {
  const r = await sharedRenderer(opts.device, `density:${opts.key ?? "solo"}`);
  // No await between the render and the read's copy: a concurrent call cannot overwrite in between.
  return r.read(r.renderDensity(xs, ys, opts));
}

/** Γ_ab(x) by the two-pass render formulation — the interactive path, and a drop-in for
 *  `computeTcm` / `computeTcmKernel`. Row 0 is at minY.
 *
 *  This form DOWNLOADS the result. For display, prefer `renderTcm` + `paintFieldTexture`, or a
 *  `TcmRenderer` writing into buffers a renderer draws from: the field then stays on the device
 *  (ADR-0017 invariant 4). The download is for callers that consume the numbers — the oracle
 *  comparison, tests. */
export async function computeTcmRender(a: CellCloud, b: CellCloud, p: TcmRenderParams): Promise<Float32Array> {
  const r = await sharedRenderer(p.device, "tcm");
  return r.read(r.render(a, b, p));
}

/** The two render passes alone, leaving Γ in its r32float target.
 *
 *  The texture is POOLED (one per device) and is overwritten by the next call — paint or copy it
 *  before then, and do not hold on to it; for several at once, make a `TcmRenderer` each. Row 0 is
 *  at maxY (the render's own orientation), which is also the canvas's, so a paint needs no flip. */
export async function renderTcm(a: CellCloud, b: CellCloud, p: TcmRenderParams): Promise<FieldTexture> {
  const r = await sharedRenderer(p.device, "tcm");
  return r.render(a, b, p);
}
