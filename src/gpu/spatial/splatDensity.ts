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
import { getDevice } from "../device";
import { rawBindGroup } from "../graph/residentBind";

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

const depadLayout = tgpu.bindGroupLayout({
  params: { uniform: DepadParams },
  src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

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

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPURenderPipeline;
}
let pipeCache: Promise<Pipe> | undefined;

function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const module = device.createShaderModule({ code: SHADER });
    const pipeline = device.createRenderPipeline({
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
    return { device, root, pipeline };
  })();
  return pipeCache;
}

// Pooled, grown (never destroyed): the point buffer, and the render target +
// readback buffer (keyed by W,H).
let ptsBuf: GPUBuffer | undefined;
let ptsCap = 0;
let uniBuf: GPUBuffer | undefined;
let target: GPUTexture | undefined;
// Readback goes through TypeGPU's `.read()` (the project's Dawn-on-Node-stable
// path) rather than a raw `mapAsync` on a pooled MAP_READ buffer, which crashed
// the vitest worker on teardown. We own the underlying GPUBuffer (so the raw
// `copyTextureToBuffer` can target it) and wrap it for `.read()`.
let rbRaw: GPUBuffer | undefined;
let rbWrap: ReturnType<ReturnType<typeof tgpu.initFromDevice>["createBuffer"]> | undefined;
let rbCap = 0;
let texW = 0;
let texH = 0;
let bytesPerRow = 0;

const align256 = (n: number) => Math.ceil(n / 256) * 256;

// De-pad pipeline + its uniform block, built once per device (like the render pipeline above).
let depadCache: Promise<{ pipeline: GPUComputePipeline; params: ReturnType<Pipe["root"]["createBuffer"]> }> | undefined;

function getDepad(device: GPUDevice, root: Pipe["root"]) {
  depadCache ??= (async () => {
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([depadFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const layout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: "splatDepad" } });
    return { pipeline, params: root.createBuffer(DepadParams).$usage("uniform") };
  })();
  return depadCache;
}

function ensurePts(device: GPUDevice, floats: number) {
  if (ptsBuf && ptsCap >= floats) return ptsBuf;
  ptsCap = Math.max(floats, ptsCap * 2, 3);
  ptsBuf = device.createBuffer({ size: ptsCap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  return ptsBuf;
}

function ensureReadback(device: GPUDevice, root: Pipe["root"], floats: number) {
  if (rbWrap && rbCap >= floats) return;
  rbCap = Math.max(floats, rbCap * 2, 1);
  rbRaw = device.createBuffer({
    size: rbCap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  rbWrap = root.createBuffer(d.arrayOf(d.f32, rbCap), rbRaw);
}

function ensureTarget(device: GPUDevice, w: number, h: number) {
  if (target && texW === w && texH === h) return;
  texW = w;
  texH = h;
  bytesPerRow = align256(w * 4); // r32float = 4 bytes/texel; rows must be 256-aligned
  target = device.createTexture({
    size: { width: w, height: h },
    format: "r32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
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

  const { device, root, pipeline } = await getPipe();

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
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const pbuf = ensurePts(device, 3 * n);
  const flat = new Float32Array(3 * Math.max(n, 1));
  for (let i = 0; i < n; i++) {
    flat[3 * i] = xs[i]!;
    flat[3 * i + 1] = ys[i]!;
    flat[3 * i + 2] = weights ? weights[i]! : 1;
  }
  device.queue.writeBuffer(pbuf, 0, flat);

  const rowFloats = renderToStaging(
    device,
    root,
    pipeline,
    { buffer: pbuf, n, stride: 3, defaultWeight: 1 },
    w,
    h,
    sigma,
    radiusSigma,
    bbox,
  );

  const padded = (await rbWrap!.read()) as ArrayLike<number>; // 256-byte-padded rows
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
  device: GPUDevice,
  root: Pipe["root"],
  pipeline: GPURenderPipeline,
  pts: PointSource,
  w: number,
  h: number,
  sigma: number,
  radiusSigma: number,
  bbox: [number, number, number, number],
): number {
  const [minX, minY, maxX, maxY] = bbox;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  ensureTarget(device, w, h);
  const rowFloats = bytesPerRow / 4;
  ensureReadback(device, root, rowFloats * h);

  const enc = device.createCommandEncoder();
  renderPoints(device, enc, pipeline, target!.createView(), pts, sigma, radiusSigma, bbox);
  enc.copyTextureToBuffer({ texture: target! }, { buffer: rbRaw!, bytesPerRow }, { width: w, height: h });
  device.queue.submit([enc.finish()]);

  return rowFloats;
}

/** The additive splat itself: encode one render pass drawing `pts` into `view`. Factored out so
 *  the texture path can target a caller-owned texture and stop, while the host path follows it
 *  with a copy into staging. */
function renderPoints(
  device: GPUDevice,
  enc: GPUCommandEncoder,
  pipeline: GPURenderPipeline,
  view: GPUTextureView,
  pts: PointSource,
  sigma: number,
  radiusSigma: number,
  bbox: [number, number, number, number],
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
  uniBuf ??= device.createBuffer({ size: uni.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniBuf, 0, uni);
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: { buffer: pts.buffer } },
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
  const { device, pipeline } = await getPipe();
  const enc = device.createCommandEncoder();
  renderPoints(
    device,
    enc,
    pipeline,
    target.createView(),
    { buffer: points, n, stride: 2, defaultWeight: 1 },
    sigma,
    opts.radiusSigma ?? 4,
    opts.bbox,
  );
  device.queue.submit([enc.finish()]);
}

/** Tier-2 form (ADR-0017): splat a GPU-resident point cloud into a GPU-resident density grid,
 *  with no host transfer in either direction.
 *
 *  `points` is the graph's packed `[x0,y0,x1,y1,...]` points value bound directly — no host
 *  repacking — and every point carries weight 1. `dst` is a caller-owned (leased) buffer of at
 *  least `width*height` f32. */
export async function splatDensityResident(points: GPUBuffer, n: number, dst: GPUBuffer, opts: ResidentSplatOptions): Promise<void> {
  const { width: w, height: h, sigma } = opts;
  if (w <= 0 || h <= 0) throw new Error("splatDensity: width/height must be > 0");
  if (sigma <= 0) throw new Error("splatDensity: sigma must be > 0");
  const radiusSigma = opts.radiusSigma ?? 4;

  const { device, root, pipeline } = await getPipe();
  const rowFloats = renderToStaging(
    device,
    root,
    pipeline,
    { buffer: points, n, stride: 2, defaultWeight: 1 },
    w,
    h,
    sigma,
    radiusSigma,
    opts.bbox,
  );

  // Strip the 256-byte row padding on-device, straight into the caller's buffer.
  const depad = await getDepad(device, root);
  depad.params.write({ w, h, rowFloats });
  const bind = rawBindGroup(device, root, depadLayout, [depad.params, rbRaw!, dst]);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(depad.pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil((w * h) / DEPAD_WG));
  pass.end();
  device.queue.submit([enc.finish()]);
}
