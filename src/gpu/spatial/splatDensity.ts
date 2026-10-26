// GPU kernel-density splat — rasterise a labelled/weighted point cloud into a
// continuous density field (a 2D KDE). This is the points -> grid *bridge*: once
// the cloud is a grid, the image-front primitives (blur, Getis-Ord, threshold,
// wavelet) apply unchanged.
//
// Method: the no-atomics **additive render** path. Each point is drawn as a small
// instanced quad covering its kernel footprint; the fragment shader evaluates the
// Gaussian and writes it; **additive blending** (src=ONE, dst=ONE, op=ADD) into an
// `r32float` render target accumulates the overlaps in fixed-function hardware. No
// `atomic<f32>` (which core WGSL lacks), no compute scatter contention.
//
// Deliberately dependency-light: this is *raw WebGPU* (only `getDevice()` from the
// project) — no TypeGPU resolve, no deck.gl / luma.gl / MDV. The WGSL below and the
// "render each layer to a float target, then composite" shape are exactly what a
// deck.gl custom layer or an MDV / SpatialData.js overlay would need, so it
// translates rather than locks us in.
//
// Stays on the Dawn-stable path: pipeline built once and cached; point buffer and
// render target pooled and grown, never `.destroy()`d per call (destroying buffers
// mid-process segfaults Dawn-on-Node's teardown). Large grids/point counts are for
// the browser harness; Node validates correctness at small sizes.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice, MAX_WORKGROUPS_PER_DIM, sized } from "../device";
import { rawBindGroup } from "../graph/residentBind";
import { halfNarrower } from "../halfTexture";
import { scratchTexture } from "../scratchTexture";
import { typegpuComputePipeline } from "../typegpuPipeline";
import { splatDensityAtomic } from "./splatDensityAtomic";

const SHADER = /* wgsl */ `
struct Uni {
  minX: f32, minY: f32,
  invSpanX: f32, invSpanY: f32,
  sigma: f32, radiusSigma: f32,
  // Floats per point in \`pts\`: 3 for the host path (x, y, weight), 2 for the Tier-2 path,
  // whose input is the graph's packed [x0,y0,x1,y1,...] points value bound directly. Reusing
  // the two former pad slots keeps the uniform layout byte-identical.
  stride: f32,
  defaultWeight: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) off: vec2f,   // world offset (fragment - point centre)
  @location(1) weight: f32,
};

// 4 vertices per instance as a triangle-strip quad; corners generated from the
// vertex index, so no vertex buffers are needed.
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let base = u32(U.stride) * ii;
  let cx = pts[base];
  let cy = pts[base + 1u];
  // Stride 2 carries no per-point weight, so fall back to the uniform one. Branch rather than
  // select(), which evaluates both arms — that would read pts[base + 2u] past the end of a
  // tightly-sized stride-2 buffer on the last point. (Robust buffer access makes that defined
  // rather than dangerous, but not reading it at all is clearer.)
  var w = U.defaultWeight;
  if (U.stride > 2.5) {
    w = pts[base + 2u];
  }
  let R = U.sigma * U.radiusSigma;                       // square support half-extent (world)
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let world = vec2f(cx, cy) + corner * R;
  let ndcX = (world.x - U.minX) * U.invSpanX * 2.0 - 1.0;
  let ndcY = (world.y - U.minY) * U.invSpanY * 2.0 - 1.0;
  var o: VSOut;
  o.pos = vec4f(ndcX, ndcY, 0.0, 1.0);
  o.off = corner * R;
  o.weight = w;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let d2 = dot(in.off, in.off);                          // squared world distance to centre
  let g = in.weight * exp(-0.5 * d2 / (U.sigma * U.sigma));
  return vec4f(g, 0.0, 0.0, 0.0);
}
`;

// --- de-pad (Tier-2 only) ---
//
// `copyTextureToBuffer` requires `bytesPerRow` to be a multiple of 256, so the texture always
// lands in a row-padded staging buffer. The host path strips that padding in a JS loop; the
// resident path cannot, because the whole point is that the value never reaches the host. This
// compute pass does the same strip on-device, writing tightly-packed w*h into the leased output.
//
// (When `w * 4` happens to be 256-aligned the copy could target the output directly and skip
// this pass. Not special-cased: one path is easier to trust, and the pass is trivially cheap.)
const DEPAD_WG = 64;

const DepadParams = d.struct({ w: d.u32, h: d.u32, rowFloats: d.u32 });

const depadLayout = tgpu
  .bindGroupLayout({
    params: { uniform: DepadParams },
    src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  })
  .$name("splatDensity:depad");

const depadFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [DEPAD_WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const w = depadLayout.$.params.w;
    if (i < w * depadLayout.$.params.h) {
      // Integer division, spelled explicitly. `i / w` on u32 operands transpiles to FLOAT
      // division here — the resolver even warns about the implicit u32→f32 conversions — so a
      // fractional `row` silently scrambles the source index (row 8.875 reads 8.875*rowFloats,
      // landing most of a row away). That produced a plausible-looking but wrong grid: mass in
      // the right ballpark, in the wrong cells. Keep the truncation explicit.
      const row = d.u32(std.floor(d.f32(i) / d.f32(w)));
      const col = i - row * w;
      depadLayout.$.dst[i] = depadLayout.$.src[row * depadLayout.$.params.rowFloats + col]!;
    }
  })
  .$name("splatDepad");

/** Formats a density field can be STORED in. Accumulation is always f32 — see
 *  `docs/field-precision.md` for the measurements that forced that, and `../halfTexture.ts`. */
export type DensityFormat = "r32float" | "r16float";

/**
 * Everything this module makes on a device: the pipelines, and the pooled (grown, never destroyed)
 * point buffer, uniform block, render target and row-padded staging buffer.
 *
 * Kept PER DEVICE rather than in module singletons, so a caller holding its own device — a deck
 * layer drawing with whatever device its canvas was given — passes it in and gets state made on
 * that device, however many devices a page has (several canvases, or a canvas remounted with a new
 * device while the old one is never destroyed). Held weakly: a device nobody references takes its
 * state with it.
 */
interface Gpu {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  /** The additive-blend render pipeline; absent on a device without `float32-blendable`, where
   *  creating it would be a validation error (the resident path falls back to compute there). */
  pipeline?: GPURenderPipeline;
  depad?: Promise<{ pipeline: GPUComputePipeline; params: ReturnType<Gpu["root"]["createBuffer"]> }>;
  ptsBuf?: GPUBuffer;
  ptsCap: number;
  uniBuf?: GPUBuffer;
  target?: GPUTexture;
  texW: number;
  texH: number;
  bytesPerRow: number;
  // Readback goes through TypeGPU's `.read()` (the project's Dawn-on-Node-stable path) rather than a
  // raw `mapAsync` on a pooled MAP_READ buffer, which crashed the vitest worker on teardown. We own
  // the underlying GPUBuffer (so the raw `copyTextureToBuffer` can target it) and wrap it for `.read()`.
  rbRaw?: GPUBuffer;
  rbWrap?: ReturnType<Gpu["root"]["createBuffer"]>;
  rbCap: number;
}

const perDevice = new WeakMap<GPUDevice, Promise<Gpu>>();

/** This module's state on `device`, or on the library's own device when none is given. */
async function getGpu(device?: GPUDevice): Promise<Gpu> {
  const dev = device ?? (await getDevice());
  let gpu = perDevice.get(dev);
  if (!gpu) {
    gpu = Promise.resolve(buildGpu(dev));
    perDevice.set(dev, gpu);
  }
  return gpu;
}

function buildGpu(device: GPUDevice): Gpu {
  const root = tgpu.initFromDevice({ device });
  const base = { device, root, ptsCap: 0, texW: 0, texH: 0, bytesPerRow: 0, rbCap: 0 };
  if (!device.features.has("float32-blendable")) return base;
  const module = device.createShaderModule({ label: "splatDensity:splat", code: SHADER });
  const pipeline = device.createRenderPipeline({
    label: "splatDensity:splat",
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    primitive: { topology: "triangle-strip" },
    fragment: {
      module,
      entryPoint: "fs",
      targets: [
        {
          format: "r32float",
          // additive accumulation — the whole point of the render path
          blend: {
            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
          },
        },
      ],
    },
  });
  return { ...base, pipeline };
}

const align256 = (n: number) => Math.ceil(n / 256) * 256;

/** The de-pad pipeline and its uniform block, built once per device like the render pipeline. */
function getDepad(g: Gpu) {
  g.depad ??= (async () => {
    const pipeline = typegpuComputePipeline(
      g.device,
      g.root,
      tgpu.resolveWithContext([depadFn], { names: "strict" }),
      "splatDepad",
      "splatDensity:depad",
    );
    return { pipeline, params: g.root.createBuffer(DepadParams).$usage("uniform").$name("splatDensity:depadParams") };
  })();
  return g.depad;
}

function ensurePts(g: Gpu, floats: number): GPUBuffer {
  if (g.ptsBuf && g.ptsCap >= floats) return g.ptsBuf;
  g.ptsCap = Math.max(floats, g.ptsCap * 2, 3);
  g.ptsBuf = g.device.createBuffer({
    label: "splatDensity:points",
    size: g.ptsCap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  return g.ptsBuf;
}

function ensureReadback(g: Gpu, floats: number): void {
  if (g.rbWrap && g.rbCap >= floats) return;
  g.rbCap = Math.max(floats, g.rbCap * 2, 1);
  g.rbRaw = g.device.createBuffer({
    label: "splatDensity:readback",
    size: g.rbCap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  g.rbWrap = g.root.createBuffer(d.arrayOf(d.f32, g.rbCap), g.rbRaw);
}

function ensureTarget(g: Gpu, w: number, h: number): GPUTexture {
  if (g.target && g.texW === w && g.texH === h) return g.target;
  g.texW = w;
  g.texH = h;
  g.bytesPerRow = align256(w * 4); // r32float = 4 bytes/texel; rows must be 256-aligned
  g.target = g.device.createTexture({
    label: "splatDensity:target",
    size: { width: w, height: h },
    format: "r32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  return g.target;
}

export interface SplatOptions {
  /** Output grid resolution. */
  width: number;
  height: number;
  /** Gaussian bandwidth in WORLD units. */
  sigma: number;
  /** Kernel square-support half-extent, in units of sigma. Default 4 (captures
   *  ~all the mass; contributions past it are dropped — see distance-decay). */
  radiusSigma?: number;
  /** World bounds [minX, minY, maxX, maxY]. Default: the points' bounds padded by
   *  the kernel support so edge kernels are not clipped. */
  bbox?: [number, number, number, number];
  /** Per-point weights (default 1 each). */
  weights?: ArrayLike<number>;
}

export interface DensityField {
  /** Row-major width*height density. Row 0 is the TOP of the bbox (worldY = maxY). */
  data: Float32Array;
  width: number;
  height: number;
  bbox: [number, number, number, number];
}

/** Splat a point cloud into a Gaussian KDE density grid on the GPU. */
export async function splatDensityGpu(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: SplatOptions): Promise<DensityField> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("splatDensity: xs and ys length mismatch");
  const { width: w, height: h, sigma } = opts;
  if (w <= 0 || h <= 0) throw new Error("splatDensity: width/height must be > 0");
  if (sigma <= 0) throw new Error("splatDensity: sigma must be > 0");
  const radiusSigma = opts.radiusSigma ?? 4;
  const weights = opts.weights;

  const g = await getGpu();
  const { device } = g;

  let bbox = opts.bbox;
  if (!bbox) {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xs[i]!,
        y = ys[i]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const pad = sigma * radiusSigma;
    bbox = [minX - pad, minY - pad, maxX + pad, maxY + pad];
  }
  const pbuf = ensurePts(g, 3 * n);
  const flat = new Float32Array(3 * Math.max(n, 1));
  for (let i = 0; i < n; i++) {
    flat[3 * i] = xs[i]!;
    flat[3 * i + 1] = ys[i]!;
    flat[3 * i + 2] = weights ? weights[i]! : 1;
  }
  device.queue.writeBuffer(pbuf, 0, flat);

  const rowFloats = renderToStaging(g, { buffer: pbuf, n, stride: 3, defaultWeight: 1 }, w, h, sigma, radiusSigma, bbox);

  const padded = (await g.rbWrap!.read()) as ArrayLike<number>; // 256-byte-padded rows
  const data = new Float32Array(w * h);
  for (let row = 0; row < h; row++) {
    const base = row * rowFloats;
    for (let col = 0; col < w; col++) data[row * w + col] = padded[base + col]!;
  }

  return { data, width: w, height: h, bbox };
}

/** Where the points live for one splat. `stride` is floats per point in `buffer`. */
interface PointSource {
  buffer: GPUBuffer;
  n: number;
  /** 3 = (x, y, weight); 2 = packed (x, y) with `defaultWeight` for all points. */
  stride: 2 | 3;
  defaultWeight: number;
}

/** Additively render the points into the r32float target and copy it into the row-padded
 *  staging buffer. Shared by both paths — they differ only in where the points come from and
 *  what happens to the staging buffer afterwards. Returns floats per padded row. */
function renderToStaging(
  g: Gpu,
  pts: PointSource,
  w: number,
  h: number,
  sigma: number,
  radiusSigma: number,
  bbox: [number, number, number, number],
): number {
  const target = ensureTarget(g, w, h);
  const rowFloats = g.bytesPerRow / 4;
  ensureReadback(g, rowFloats * h);

  const enc = g.device.createCommandEncoder({ label: "splatDensity:toStaging" });
  renderPoints(g, enc, target.createView(), pts, sigma, radiusSigma, bbox);
  enc.copyTextureToBuffer({ texture: target }, { buffer: g.rbRaw!, bytesPerRow: g.bytesPerRow }, { width: w, height: h });
  g.device.queue.submit([enc.finish()]);

  return rowFloats;
}

/** The additive splat itself: encode one render pass drawing `pts` into `view`. Factored out so
 *  the texture path can target a caller-owned texture and stop, while the host path follows it
 *  with a copy into staging. */
function renderPoints(
  g: Gpu,
  enc: GPUCommandEncoder,
  view: GPUTextureView,
  pts: PointSource,
  sigma: number,
  radiusSigma: number,
  bbox: [number, number, number, number],
  /** The field's extent. The attachment may be a REUSED scratch target that is larger (see
   *  `../scratchTexture.ts`), and without a viewport the splat would spread across all of it —
   *  the same points at the wrong scale, cropped back to the right size on the way out, which
   *  looks like a plausible field rather than an error. */
  extent?: { width: number; height: number },
): void {
  const [minX, minY, maxX, maxY] = bbox;
  const uni = new Float32Array([
    minX,
    minY,
    1 / (maxX - minX || 1),
    1 / (maxY - minY || 1),
    sigma,
    radiusSigma,
    pts.stride,
    pts.defaultWeight,
  ]);
  const { device, pipeline } = g;
  if (!pipeline) {
    throw new Error(
      "splatDensity: this device lacks 'float32-blendable', which the render path needs " +
        "(splatDensityResident falls back to a compute splat; the host and texture paths do not)",
    );
  }
  g.uniBuf ??= device.createBuffer({
    label: "splatDensity:uniforms",
    size: uni.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(g.uniBuf, 0, uni);
  const pass = enc.beginRenderPass({
    label: "splatDensity:splat",
    colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
  });
  if (extent) pass.setViewport(0, 0, extent.width, extent.height, 0, 1);
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      label: "splatDensity:splat",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: g.uniBuf } },
        // n * stride floats, not the whole pooled buffer. `ensurePts` doubles on growth, so
        // binding the buffer whole crosses maxStorageBufferBindingSize at ~5.6M points once a
        // larger call has been through — and past it the draw silently splats nothing. This is
        // the kernel MDV's note wants to point at 35M rows (420 MB), so it is on the path.
        { binding: 1, resource: sized(pts.buffer, pts.n * pts.stride * 4) },
      ],
    }),
  );
  if (pts.n > 0) pass.draw(4, pts.n); // 4-vertex strip quad, one instance per point
  pass.end();
}

export interface ResidentSplatOptions {
  width: number;
  height: number;
  sigma: number;
  radiusSigma?: number;
  /** REQUIRED here, unlike the host path: deriving it from the points would mean reading them
   *  back, which is the transfer this whole path exists to avoid. Callers that want the
   *  points' own bounds compute them host-side (or, eventually, with a GPU reduction). */
  bbox: [number, number, number, number];
  /** Floats per point in `points`: 2 = the graph's packed `[x, y, ...]`, every point weight 1
   *  (the default); 3 = `[x, y, weight, ...]`. Honoured by `splatDensityResident`. */
  stride?: 2 | 3;
  /** The device `points` and the output live on. Default: the library's own (`getDevice()`). A
   *  renderer with its own device — a deck layer — passes it, and needs no `adoptDevice`. */
  device?: GPUDevice;
  /** How `splatDensityResident` accumulates. `"blend"`: additive blending into r32float (fast; needs
   *  `float32-blendable`). `"atomic"`: an exact fixed-point compute scatter (`splatDensityAtomic.ts`).
   *  Default: blend where the device can, atomic where it cannot. */
  accumulate?: "blend" | "atomic";
  /** The largest point weight (default 1). Only the atomic path uses it, to set its fixed-point
   *  scale; a weight above it is clamped there. */
  maxWeight?: number;
}

/** Tier-2 form, TEXTURE output (ADR-0017): splat a GPU-resident point cloud straight into a
 *  caller-owned r32float texture, and stop there.
 *
 *  This is the render path's natural product — the additive blend already targets a texture — so
 *  nothing further happens: no `copyTextureToBuffer`, no de-padding pass. A consumer that renders
 *  (a paint to canvas, another pass) uses it as-is; one that needs a storage buffer gets the copy
 *  from the executor's bridge, which runs only when such a consumer actually exists. */
export async function splatDensityToTexture(points: GPUBuffer, n: number, target: GPUTexture, opts: ResidentSplatOptions): Promise<void> {
  const { width: w, height: h, sigma } = opts;
  if (w <= 0 || h <= 0) throw new Error("splatDensity: width/height must be > 0");
  if (sigma <= 0) throw new Error("splatDensity: sigma must be > 0");
  const g = await getGpu(opts.device);
  const { device } = g;

  // ACCUMULATE IN f32, ALWAYS — even when the field is stored as f16.
  //
  // Blending additively straight into an r16float target looks like the obvious way to halve the
  // memory, and it is wrong in a way that does not announce itself. f16 carries 10 mantissa bits,
  // so once a texel's running total exceeds ~1024× one splat's contribution, `total + c == total`
  // and the contribution disappears. The loss is systematic, not noise: measured on a 128² field
  // (`pnpm bench:density-precision`), a 6400-point cloud lost 9% of its total mass and 25000
  // points lost 75%, while the same fields stored f16 AFTER an f32 accumulation lose ~0.02%.
  // Dense data is exactly where a density field is worth computing, so that is the case that must
  // not quietly degrade.
  const half = target.format === "r16float";
  // The last await. The accumulator is a scratch texture shared by every call on this device, and
  // the uniforms `renderPoints` writes are shared too, so from here to the submit runs without
  // yielding. A concurrent call that slipped into a later await destroyed this call's accumulator
  // (a different size does not fit the slot) and overwrote its uniforms (`scratchTexture.ts`).
  const narrow = half ? await halfNarrower(device) : undefined;
  const accum = half ? ensureAccum(device, w, h) : target;

  const enc = device.createCommandEncoder({ label: "splatDensity:toTexture" });
  renderPoints(g, enc, accum.createView(), { buffer: points, n, stride: 2, defaultWeight: 1 }, sigma, opts.radiusSigma ?? 4, opts.bbox, {
    width: w,
    height: h,
  });
  narrow?.(enc, accum, target, w, h);
  device.queue.submit([enc.finish()]);
}

// The f32 accumulator behind every f16 field. ONE texture, reused — the saving f16 buys is in the
// fields a graph RETAINS (a layer stack, a memo, a cache), not in this single transient, so one
// full-precision scratch costs one field's worth of memory however many half-precision ones are
// alive. It is reused for cost, and REPLACED PROPERLY when the size changes: `scratchTexture`
// destroys the one it supersedes rather than leaving a multi-megabyte allocation for a collector
// that cannot see it. `releaseScratchTextures("splatDensity")` gives it back.
function ensureAccum(device: GPUDevice, w: number, h: number): GPUTexture {
  return scratchTexture(device, "splatDensity:accum", {
    width: w,
    height: h,
    format: "r32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
}

/** Tier-2 form (ADR-0017): splat a GPU-resident point cloud into a GPU-resident density grid,
 *  with no host transfer in either direction.
 *
 *  `points` is bound directly — no host repacking: the graph's packed `[x0,y0,x1,y1,...]` points
 *  value (every point weight 1), or `[x, y, weight, ...]` with `stride: 3`. `dst` is a
 *  caller-owned (leased) buffer of at least `width*height` f32. */
export async function splatDensityResident(points: GPUBuffer, n: number, dst: GPUBuffer, opts: ResidentSplatOptions): Promise<void> {
  const { width: w, height: h, sigma } = opts;
  if (w <= 0 || h <= 0) throw new Error("splatDensity: width/height must be > 0");
  if (sigma <= 0) throw new Error("splatDensity: sigma must be > 0");
  const dev = opts.device ?? (await getDevice());
  if (opts.accumulate === "atomic" || !dev.features.has("float32-blendable")) {
    // No float blending on this device (Firefox's WebGPU): the compute path, same output layout.
    await splatDensityAtomic(dev, points, n, dst, {
      width: w,
      height: h,
      sigma,
      radiusSigma: opts.radiusSigma ?? 4,
      bbox: opts.bbox,
      stride: opts.stride ?? 2,
      maxWeight: opts.maxWeight ?? 1,
    });
    return;
  }
  // The de-pad dispatch is 1-D, and past the per-dimension workgroup cap WebGPU drops the dispatch
  // without an error — the caller would get an all-zero field.
  if (Math.ceil((w * h) / DEPAD_WG) > MAX_WORKGROUPS_PER_DIM) {
    throw new Error(`splatDensity: ${w}×${h} exceeds the resident path's ${MAX_WORKGROUPS_PER_DIM * DEPAD_WG} cells`);
  }
  const radiusSigma = opts.radiusSigma ?? 4;

  // Every await happens BEFORE any encoding. The render target and the staging buffer are shared by
  // every call on a device, so the splat, its copy into staging and the de-pad out of staging must reach the
  // queue in one submit: with an await between them, a second caller (two layers updating in the
  // same frame) would splat into the shared staging buffer before the first had de-padded it, and
  // the first caller would get the second one's field. Nothing would look wrong but the picture.
  const g = await getGpu(dev);
  const depad = await getDepad(g);
  const { device, root } = g;

  const target = ensureTarget(g, w, h);
  const rowFloats = g.bytesPerRow / 4;
  ensureReadback(g, rowFloats * h);
  // queue.writeBuffer lands before the submit below, and after any earlier caller's submit.
  depad.params.write({ w, h, rowFloats });

  const enc = device.createCommandEncoder({ label: "splatDensity:resident" });
  renderPoints(
    g,
    enc,
    target.createView(),
    { buffer: points, n, stride: opts.stride ?? 2, defaultWeight: 1 },
    sigma,
    radiusSigma,
    opts.bbox,
  );
  enc.copyTextureToBuffer({ texture: target }, { buffer: g.rbRaw!, bytesPerRow: g.bytesPerRow }, { width: w, height: h });
  // Strip the 256-byte row padding on-device, straight into the caller's buffer.
  const pass = enc.beginComputePass({ label: "splatDensity:depad" });
  pass.setPipeline(depad.pipeline);
  pass.setBindGroup(0, rawBindGroup(device, root, depadLayout, [depad.params, g.rbRaw!, dst], "splatDensity:depad"));
  pass.dispatchWorkgroups(Math.ceil((w * h) / DEPAD_WG));
  pass.end();
  device.queue.submit([enc.finish()]);
}
