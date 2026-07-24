// The N-way association matrix as a render plus a matmul — the GPU twin of `src/spatial/gram.ts`.
//
// Two passes, and the shape of them is the whole point:
//
//   1. **Splat.** One additive render per channel through a unit-mass radial kernel, into an
//      `r32float` target that is then copied into a slice of one big storage buffer. Cost is
//      `O(Σ n_c · kernel footprint)` — a scatter with blending, no neighbour search, no atomics.
//   2. **Reduce.** One compute dispatch forming every inner product `⟨M_a, M_b⟩` at once: one
//      workgroup per channel pair, each striding the whole raster and finishing with a
//      shared-memory tree reduction.
//
// Neither pass's cost depends on the interaction radius, which is what makes this the right shape
// for the statistic. The bucket-grid path in `crossPcf.ts` walks every neighbour inside `r`, so it
// scales as `O(n · ρ · πr²)` — quadratic in the radius. Here `r` only changes how far a quad
// spreads its mass, and the reduction does not know `r` exists at all.
//
// **Precision.** The counting in `crossPcf.ts` is integer, so that path is *exactly* the CPU
// statistic. This one is not, and does not pretend to be: the rasterisation is a quadrature and
// the accumulation is f32. `gramMatrix.gpu.test.ts` measures the residual against the f64 CPU
// oracle rather than asserting bit-parity. The one number worth knowing up front is that the
// **positive semi-definiteness survives f32 exactly** — `MMᵀ` is PSD because of what it is, not
// because of how precisely it was computed, which is the property `gram.ts` builds on.
//
// Written in raw WGSL rather than TGSL for two reasons: the reduction needs `var<workgroup>`
// shared memory and a `workgroupBarrier`, and it indexes rasters with `q / width` — integer
// division on `u32`, which in TGSL "use gpu" kernels silently lowers to *float* division.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { ChannelCloud, GramParams, GramResult } from "../../spatial/gram";
import { EPANECHNIKOV, kernelCode, roughness } from "../../spatial/kernels";
import { getDevice } from "../device";
import { KERNEL_WGSL } from "./kernelWgsl";

const REDUCE_WG = 256;
const SPLAT_UNI_FLOATS = 8; // 32 bytes
const REDUCE_UNI_FLOATS = 8;

const SPLAT_SHADER = /* wgsl */ `
struct Uni {
  minX: f32, minY: f32, invSpanX: f32, invSpanY: f32,
  radius: f32, kernelCode: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;   // stride 3: x, y, weight

${KERNEL_WGSL}

fn corner(vi: u32) -> vec2f {
  return vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
}

struct KOut { @builtin(position) pos: vec4f, @location(0) off: vec2f, @location(1) @interpolate(flat) w: f32 };

@vertex
fn vsSplat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> KOut {
  let c = corner(vi);
  let centre = vec2f(pts[3u * ii], pts[3u * ii + 1u]);
  let world = centre + c * U.radius;
  var o: KOut;
  o.pos = vec4f((world.x - U.minX) * U.invSpanX * 2.0 - 1.0,
                (world.y - U.minY) * U.invSpanY * 2.0 - 1.0, 0.0, 1.0);
  o.off = c * U.radius;
  // Flat: the weight is a per-CELL mark, not something to interpolate across the quad.
  o.w = pts[3u * ii + 2u];
  return o;
}

@fragment
fn fsSplat(in: KOut) -> @location(0) vec4f {
  return vec4f(in.w * kernelAt(dot(in.off, in.off), U.radius, U.kernelCode), 0.0, 0.0, 0.0);
}
`;

// Both reductions share this preamble. `rowFloats` is the texture-copy row stride in floats
// (copyTextureToBuffer pads rows to 256 bytes), so the padding is skipped by indexing rather than
// removed by a separate de-pad pass.
const REDUCE_SHADER = /* wgsl */ `
struct Uni { K: f32, width: f32, height: f32, rowFloats: f32, pad0: f32, pad1: f32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<f32>;
@group(0) @binding(2) var<storage, read> means: array<f32>;
@group(0) @binding(3) var<storage, read_write> sums: array<f32>;      // K
@group(0) @binding(4) var<storage, read_write> raw: array<f32>;       // K*K
@group(0) @binding(5) var<storage, read_write> centred: array<f32>;   // K*K

var<workgroup> scratch: array<f32, ${REDUCE_WG}>;

fn at(channel: u32, q: u32, width: u32, rowFloats: u32) -> f32 {
  let row = q / width;
  let col = q - row * width;
  return rasters[channel * u32(U.height) * rowFloats + row * rowFloats + col];
}

fn treeReduce(lid: u32, v: f32) -> f32 {
  scratch[lid] = v;
  workgroupBarrier();
  var stride = ${REDUCE_WG}u >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) { scratch[lid] = scratch[lid] + scratch[lid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  return scratch[0];
}

// Pass 1 — Σ_p M_a(p), one workgroup per channel. Its only job is to give pass 2 an exact mean, so
// that pass 2 accumulates CENTRED products. Deriving the centred sum on the host as
// (raw - P·μ_a·μ_b) is algebraically identical and numerically much worse: for weakly correlated
// channels those two terms very nearly cancel, and raw was accumulated in f32.
@compute @workgroup_size(${REDUCE_WG})
fn sumChannels(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let a = wid.x;
  let width = u32(U.width);
  let rowFloats = u32(U.rowFloats);
  let P = width * u32(U.height);
  var acc = 0.0;
  var q = lid.x;
  loop {
    if (q >= P) { break; }
    acc = acc + at(a, q, width, rowFloats);
    q = q + ${REDUCE_WG}u;
  }
  let total = treeReduce(lid.x, acc);
  if (lid.x == 0u) { sums[a] = total; }
}

// Pass 2 — the matmul. One workgroup per ORDERED pair, but only the upper triangle does work and
// it writes both halves: the matrix is symmetric by construction, and computing it twice would
// also let the two halves disagree in the last f32 bit.
@compute @workgroup_size(${REDUCE_WG})
fn gramPairs(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let K = u32(U.K);
  let a = wid.x / K;
  let b = wid.x - a * K;
  if (b < a) { return; }
  let width = u32(U.width);
  let rowFloats = u32(U.rowFloats);
  let P = width * u32(U.height);
  let ma = means[a];
  let mb = means[b];

  var accRaw = 0.0;
  var accCen = 0.0;
  var q = lid.x;
  loop {
    if (q >= P) { break; }
    let va = at(a, q, width, rowFloats);
    let vb = at(b, q, width, rowFloats);
    accRaw = accRaw + va * vb;
    accCen = accCen + (va - ma) * (vb - mb);
    q = q + ${REDUCE_WG}u;
  }
  let r = treeReduce(lid.x, accRaw);
  workgroupBarrier();
  let c = treeReduce(lid.x, accCen);
  if (lid.x == 0u) {
    raw[a * K + b] = r;
    raw[b * K + a] = r;
    centred[a * K + b] = c;
    centred[b * K + a] = c;
  }
}
`;

type Root = ReturnType<typeof tgpu.initFromDevice>;

interface Ctx {
  device: GPUDevice;
  root: Root;
  splat: GPURenderPipeline;
  sumChannels: GPUComputePipeline;
  gramPairs: GPUComputePipeline;
  /** Explicit, and shared by both compute pipelines. `layout: "auto"` would derive a *different*
   *  layout for each — one containing only the bindings that entry point happens to reference —
   *  so a single bind group could not satisfy both, and `sumChannels` (which never touches
   *  `means`, `raw` or `centred`) would reject them as unexpected entries. */
  reduceLayout: GPUBindGroupLayout;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const splatModule = device.createShaderModule({ code: SPLAT_SHADER });
    const reduceModule = device.createShaderModule({ code: REDUCE_SHADER });
    const splat = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: splatModule, entryPoint: "vsSplat" },
      fragment: {
        module: splatModule,
        entryPoint: "fsSplat",
        targets: [
          {
            format: "r32float",
            // Additive blending IS the sum in `M(x) = Σ_i w_i J(x − x_i)`. No atomics, no
            // read-modify-write; the raster op does it.
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
    });
    const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    const reduceLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        storage(1, "read-only-storage"), // rasters
        storage(2, "read-only-storage"), // means
        storage(3, "storage"), // sums
        storage(4, "storage"), // raw
        storage(5, "storage"), // centred
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [reduceLayout] });
    const mk = (entryPoint: string) =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module: reduceModule, entryPoint } });
    return { device, root, splat, sumChannels: mk("sumChannels"), gramPairs: mk("gramPairs"), reduceLayout };
  })();
  return ctxCache;
}

// Pooled, grow-only, never destroyed — `.destroy()` segfaults Dawn-on-Node's teardown, the same
// constraint crossPcf.ts and splatDensity.ts work under.
const bufPool = new Map<string, { buf: GPUBuffer; cap: number }>();

function ensureBuf(device: GPUDevice, key: string, floats: number, usage: number): GPUBuffer {
  const got = bufPool.get(key);
  if (got && got.cap >= floats) return got.buf;
  const cap = Math.max(floats, (got?.cap ?? 0) * 2, 4);
  const buf = device.createBuffer({ size: cap * 4, usage });
  bufPool.set(key, { buf, cap });
  return buf;
}

let scratchTex: { tex: GPUTexture; view: GPUTextureView; w: number; h: number } | undefined;

function ensureTex(device: GPUDevice, w: number, h: number) {
  if (scratchTex && scratchTex.w === w && scratchTex.h === h) return scratchTex;
  const tex = device.createTexture({
    size: { width: w, height: h },
    format: "r32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  scratchTex = { tex, view: tex.createView(), w, h };
  return scratchTex;
}

// Readback goes through a TypeGPU-wrapped buffer: `.read()` is the only Dawn-on-Node-stable
// readback in this project (see crossPcf.ts).
const readbacks = new Map<string, { raw: GPUBuffer; wrap: ReturnType<Root["createBuffer"]>; cap: number }>();

function ensureReadback(device: GPUDevice, root: Root, key: string, floats: number) {
  const got = readbacks.get(key);
  if (got && got.cap >= floats) return got;
  const cap = Math.max(floats, (got?.cap ?? 0) * 2, 4);
  const raw = device.createBuffer({
    size: cap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const made = { raw, wrap: root.createBuffer(d.arrayOf(d.f32, cap), raw), cap };
  readbacks.set(key, made);
  return made;
}

const align256 = (n: number) => Math.ceil(n / 256) * 256;

/** Interleave every channel's points into one `[x, y, w, …]` buffer, recording where each starts
 *  so the draws can be issued with `firstInstance` instead of one upload per channel. */
// The return type is inferred rather than annotated: a bare `Float32Array` annotation widens to
// `Float32Array<ArrayBufferLike>`, which `queue.writeBuffer` rejects (it will not accept a possibly
// SharedArrayBuffer-backed view). Inference keeps the precise `Float32Array<ArrayBuffer>`.
function packChannels(channels: readonly ChannelCloud[]) {
  const counts = channels.map((c) => c.xs.length);
  const total = counts.reduce((a, b) => a + b, 0);
  const data = new Float32Array(3 * Math.max(total, 1));
  const offsets: number[] = [];
  const mass = new Float64Array(channels.length);
  let at = 0;
  channels.forEach((c, k) => {
    offsets.push(at);
    let w = 0;
    for (let i = 0; i < c.xs.length; i++) {
      const wi = c.weights ? (c.weights[i] ?? 0) : 1;
      data[3 * at] = c.xs[i] ?? 0;
      data[3 * at + 1] = c.ys[i] ?? 0;
      data[3 * at + 2] = wi;
      w += wi;
      at++;
    }
    mass[k] = w;
  });
  return { data, offsets, mass };
}

/** The splatted channel rasters, left **on the device**.
 *
 *  Downloading K full rasters is exactly the readback this formulation exists to avoid, so the
 *  mode projection consumes them in place (`paintGramModes`). The `mean`/`sd` are carried alongside
 *  because the projection standardises with the same numbers `corr` was built from — recomputing
 *  them would risk the map and the matrix disagreeing.
 *
 *  **Lifetime: valid until the next `gramMatrixGpu` call.** The buffer is pooled and grow-only (a
 *  `.destroy()` segfaults Dawn-on-Node's teardown), so a second call overwrites it in place. */
export interface ResidentRasters {
  readonly buffer: GPUBuffer;
  /** Row stride in floats — `copyTextureToBuffer` pads rows to 256 bytes, and the padding is
   *  skipped by indexing rather than removed by a de-pad pass. */
  readonly rowFloats: number;
  readonly mean: Float64Array;
  readonly sd: Float64Array;
}

/** What the GPU path returns: the same statistics as `GramResult`, with the rasters kept as a
 *  device handle rather than a host array. */
export type GramMatrixGpuResult = Pick<
  GramResult,
  "labels" | "mass" | "c" | "g" | "corr" | "selfTerm" | "width" | "height" | "bbox" | "pixelArea"
> & { readonly resident: ResidentRasters };

/**
 * The N-way Gram matrix on the GPU. Same statistic, same normalisation and same result shape as
 * `gramMatrix` in `src/spatial/gram.ts`, which is its f64 oracle.
 */
export async function gramMatrixGpu(channels: readonly ChannelCloud[], p: GramParams): Promise<GramMatrixGpuResult> {
  const { device, root, splat, sumChannels, gramPairs, reduceLayout } = await getCtx();
  const K = channels.length;
  const { width: w, height: h } = p;
  const [minX, minY, maxX, maxY] = p.bbox;
  const kernel = p.kernel ?? EPANECHNIKOV;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const P = w * h;
  const pixelArea = roiArea / P;

  const { data, offsets, mass } = packChannels(channels);
  const rowFloats = align256(w * 4) / 4;

  const ptsBuf = ensureBuf(device, "pts", data.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const uniSplat = ensureBuf(device, "uniSplat", SPLAT_UNI_FLOATS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const uniReduce = ensureBuf(device, "uniReduce", REDUCE_UNI_FLOATS, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const rasters = ensureBuf(
    device,
    "rasters",
    K * h * rowFloats,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const meansBuf = ensureBuf(device, "means", K, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sums = ensureReadback(device, root, "sums", K);
  const rawOut = ensureReadback(device, root, "raw", K * K);
  const cenOut = ensureReadback(device, root, "centred", K * K);

  device.queue.writeBuffer(ptsBuf, 0, data);
  device.queue.writeBuffer(
    uniSplat,
    0,
    new Float32Array([minX, minY, 1 / (maxX - minX), 1 / (maxY - minY), p.radius, kernelCode(kernel), 0, 0]),
  );
  device.queue.writeBuffer(uniReduce, 0, new Float32Array([K, w, h, rowFloats, 0, 0, 0, 0]));

  const tex = ensureTex(device, w, h);
  const splatBind = device.createBindGroup({
    layout: splat.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniSplat, size: SPLAT_UNI_FLOATS * 4 } },
      { binding: 1, resource: { buffer: ptsBuf } },
    ],
  });

  // One encoder for the whole splat phase: render channel k, copy it into its slice, clear and
  // reuse the same target for k+1. Sequential within an encoder is well-defined, and it keeps the
  // texture footprint at one raster regardless of K.
  const enc = device.createCommandEncoder();
  for (let k = 0; k < K; k++) {
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: tex.view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    pass.setPipeline(splat);
    pass.setBindGroup(0, splatBind);
    const n = channels[k]!.xs.length;
    if (n > 0) pass.draw(4, n, 0, offsets[k]!);
    pass.end();
    enc.copyTextureToBuffer(
      { texture: tex.tex },
      { buffer: rasters, offset: k * h * rowFloats * 4, bytesPerRow: rowFloats * 4, rowsPerImage: h },
      { width: w, height: h },
    );
  }
  device.queue.submit([enc.finish()]);

  const reduceBind = () =>
    device.createBindGroup({
      layout: reduceLayout,
      entries: [
        { binding: 0, resource: { buffer: uniReduce, size: REDUCE_UNI_FLOATS * 4 } },
        { binding: 1, resource: { buffer: rasters } },
        { binding: 2, resource: { buffer: meansBuf } },
        { binding: 3, resource: { buffer: sums.raw } },
        { binding: 4, resource: { buffer: rawOut.raw } },
        { binding: 5, resource: { buffer: cenOut.raw } },
      ],
    });

  const enc2 = device.createCommandEncoder();
  const pass1 = enc2.beginComputePass();
  pass1.setPipeline(sumChannels);
  pass1.setBindGroup(0, reduceBind());
  pass1.dispatchWorkgroups(K);
  pass1.end();
  device.queue.submit([enc2.finish()]);

  // The means round-trip to the host. It is K floats — the whole point of pass 1 is that this is
  // the ONLY readback between the two reductions, not a per-pixel one.
  const sumVals = (await sums.wrap.read()) as ArrayLike<number>;
  const mean = new Float64Array(K);
  for (let k = 0; k < K; k++) mean[k] = (sumVals[k] ?? 0) / P;
  device.queue.writeBuffer(meansBuf, 0, Float32Array.from(mean));

  const enc3 = device.createCommandEncoder();
  const pass2 = enc3.beginComputePass();
  pass2.setPipeline(gramPairs);
  pass2.setBindGroup(0, reduceBind());
  pass2.dispatchWorkgroups(K * K);
  pass2.end();
  device.queue.submit([enc3.finish()]);

  const rawVals = (await rawOut.wrap.read()) as ArrayLike<number>;
  const cenVals = (await cenOut.wrap.read()) as ArrayLike<number>;

  // Normalisation in f64 on the host — it is O(K²), i.e. free, and keeping it here means the CPU
  // and GPU paths share one definition of the statistic rather than two (the crossPcf.ts rule).
  const selfAtZero = roughness(kernel, p.radius);
  const c = new Float64Array(K * K);
  const g = new Float64Array(K * K);
  const corr = new Float64Array(K * K);
  const selfTerm = new Float64Array(K * K);
  const sd = new Float64Array(K);
  for (let a = 0; a < K; a++) sd[a] = Math.sqrt(Math.max((cenVals[a * K + a] ?? 0) / P, 0));

  for (let a = 0; a < K; a++) {
    for (let b = 0; b < K; b++) {
      const ca = channels[a]!;
      const cb = channels[b]!;
      let shared = 0;
      if (a === b || (ca.xs === cb.xs && ca.ys === cb.ys)) {
        const n = Math.min(ca.xs.length, cb.xs.length);
        for (let i = 0; i < n; i++) {
          shared += (ca.weights ? (ca.weights[i] ?? 0) : 1) * (cb.weights ? (cb.weights[i] ?? 0) : 1);
        }
      }
      selfTerm[a * K + b] = selfAtZero * shared;

      const cab = (rawVals[a * K + b] ?? 0) * pixelArea;
      c[a * K + b] = cab;
      const expected = (mass[a]! * mass[b]!) / roiArea;
      g[a * K + b] = expected > 0 ? (cab - selfTerm[a * K + b]!) / expected : 0;
      const denom = sd[a]! * sd[b]!;
      corr[a * K + b] = a === b ? (sd[a]! > 0 ? 1 : 0) : denom > 0 ? (cenVals[a * K + b] ?? 0) / P / denom : 0;
    }
  }

  return {
    labels: channels.map((ch) => ch.label),
    mass,
    c,
    g,
    corr,
    selfTerm,
    width: w,
    height: h,
    bbox: p.bbox,
    pixelArea,
    resident: { buffer: rasters, rowFloats, mean, sd },
  };
}
