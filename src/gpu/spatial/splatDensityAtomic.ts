// The KDE splat without float blending: a compute scatter into an exact 64-bit fixed-point
// accumulator, for devices that lack `float32-blendable` (Firefox's WebGPU, at the time of writing).
//
// The render path (`splatDensity.ts`) accumulates by additive blending into r32float, which needs
// that feature. The obvious substitute — blend into r16float, which core WebGPU can — is the one this
// repo measured losing 75% of the mass at 25k points (`docs/field-precision.md`): f16 stops adding
// once a texel's total is ~1024× one contribution. So instead each point is a thread that walks its
// kernel footprint and atomically adds `w · g` as fixed point.
//
// Fixed point needs a scale and a range. Core WGSL has 32-bit integer atomics only, so a cell's sum
// is held as two u32 words, lo and hi, and a carry is detected from the value `atomicAdd` returns:
// `old + q` wrapped iff `old > 0xFFFFFFFF − q`, and then hi gets 1. That is an exact 64-bit add, in
// any order — so the result is deterministic, unlike a float scatter. The scale is 2³¹ per largest
// weight, so one contribution always fits a u32; evaluating it in f32 bounds its error to ~1e−7 of
// that weight. The 64-bit sum holds 2³³ largest-weight points before it could overflow.
//
// It evaluates the Gaussian at the same cell centres, over the same square support (±radiusSigma·σ),
// as the rasteriser does, so the two paths agree to rounding. Weights must be ≥ 0 (a density's are);
// a negative or non-finite weight contributes nothing.
//
// It is slower than hardware blending — every footprint cell is an atomic, contended where points
// are dense — and it is meant as the fallback, not the default.
import { compileShader, dispatchGrid } from "../device";

const WG = 64;

const SPLAT = /* wgsl */ `
struct Uni {
  minX: f32, maxY: f32, cellX: f32, cellY: f32,
  inv2s2: f32, R: f32, scale: f32, defaultWeight: f32,
  n: u32, stride: u32, w: u32, h: u32,
  rowThreads: u32, cells: u32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
/** [lo for every cell | hi for every cell] */
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;

@compute @workgroup_size(${WG})
fn splat(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * U.rowThreads;
  if (i >= U.n) { return; }
  let b = i * U.stride;
  let px = pts[b];
  let py = pts[b + 1u];
  var wt = U.defaultWeight;
  if (U.stride == 3u) { wt = pts[b + 2u]; }
  if (!(wt > 0.0)) { return; }

  // Cells whose centre lies within [p − R, p + R]: column c's centre is minX + (c + ½)·cellX, and
  // row r's is maxY − (r + ½)·cellY (row 0 at the top, as the render path lays the field out).
  let c0 = max(i32(ceil((px - U.R - U.minX) / U.cellX - 0.5)), 0);
  let c1 = min(i32(floor((px + U.R - U.minX) / U.cellX - 0.5)), i32(U.w) - 1);
  let r0 = max(i32(ceil((U.maxY - (py + U.R)) / U.cellY - 0.5)), 0);
  let r1 = min(i32(floor((U.maxY - (py - U.R)) / U.cellY - 0.5)), i32(U.h) - 1);
  for (var r = r0; r <= r1; r++) {
    let dy = U.maxY - (f32(r) + 0.5) * U.cellY - py;
    for (var c = c0; c <= c1; c++) {
      let dx = U.minX + (f32(c) + 0.5) * U.cellX - px;
      // Clamped: a weight above maxWeight breaks the contract, but must not reach an undefined u32().
      let q = u32(min(wt * exp(-(dx * dx + dy * dy) * U.inv2s2) * U.scale + 0.5, 4294967295.0));
      if (q == 0u) { continue; }
      let idx = u32(r) * U.w + u32(c);
      let old = atomicAdd(&acc[idx], q);
      if (old > 0xFFFFFFFFu - q) {
        atomicAdd(&acc[U.cells + idx], 1u);
      }
    }
  }
}
`;

const RESOLVE = /* wgsl */ `
struct Uni { cells: u32, rowThreads: u32, invScale: f32, pad: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> acc: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG})
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * U.rowThreads;
  if (i >= U.cells) { return; }
  dst[i] = (f32(acc[U.cells + i]) * 4294967296.0 + f32(acc[i])) * U.invScale;
}
`;

interface Ctx {
  splat: GPUComputePipeline;
  resolve: GPUComputePipeline;
  splatUni: GPUBuffer;
  resolveUni: GPUBuffer;
  acc?: GPUBuffer;
}

/** Per device, held weakly — see `splatDensity.ts`'s `Gpu`. */
const perDevice = new WeakMap<GPUDevice, Promise<Ctx>>();

function getCtx(device: GPUDevice): Promise<Ctx> {
  let ctx = perDevice.get(device);
  if (!ctx) {
    ctx = (async () => {
      const [splatModule, resolveModule] = await Promise.all([
        compileShader(device, SPLAT, "splatDensityAtomic:splat"),
        compileShader(device, RESOLVE, "splatDensityAtomic:resolve"),
      ]);
      return {
        splat: device.createComputePipeline({ layout: "auto", compute: { module: splatModule, entryPoint: "splat" } }),
        resolve: device.createComputePipeline({ layout: "auto", compute: { module: resolveModule, entryPoint: "resolve" } }),
        splatUni: device.createBuffer({
          label: "splatDensityAtomic:splatUniforms",
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        resolveUni: device.createBuffer({
          label: "splatDensityAtomic:resolveUniforms",
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      };
    })();
    perDevice.set(device, ctx);
  }
  return ctx;
}

export interface AtomicSplatOptions {
  width: number;
  height: number;
  sigma: number;
  radiusSigma: number;
  bbox: [number, number, number, number];
  /** 2 = [x, y] (weight 1), 3 = [x, y, w]. */
  stride: 2 | 3;
  /** The largest weight among the points (≥ any `w`); sets the fixed-point scale. */
  maxWeight: number;
}

/** Splat into `dst` (width·height f32, row 0 = bbox maxY) — the same layout `splatDensityResident`
 *  produces — without float blending. All awaits come before any encoding (one submit). */
export async function splatDensityAtomic(
  device: GPUDevice,
  points: GPUBuffer,
  n: number,
  dst: GPUBuffer,
  opts: AtomicSplatOptions,
): Promise<void> {
  const ctx = await getCtx(device);
  const { width: w, height: h, sigma, radiusSigma, bbox, stride } = opts;
  const cells = w * h;
  const [minX, minY, maxX, maxY] = bbox;
  const scale = 2 ** 31 / Math.max(opts.maxWeight, Number.MIN_VALUE);

  const bytes = cells * 8;
  if (!ctx.acc || ctx.acc.size < bytes) {
    ctx.acc?.destroy();
    ctx.acc = device.createBuffer({
      label: "splatDensityAtomic:accumulator",
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }
  const splatGrid = dispatchGrid(Math.ceil(Math.max(n, 1) / WG));
  const resolveGrid = dispatchGrid(Math.ceil(cells / WG));

  const su = new ArrayBuffer(64);
  new Float32Array(su, 0, 8).set([
    minX,
    maxY,
    (maxX - minX) / w,
    (maxY - minY) / h,
    1 / (2 * sigma * sigma),
    sigma * radiusSigma,
    scale,
    1,
  ]);
  new Uint32Array(su, 32, 8).set([n, stride, w, h, splatGrid.x * WG, cells, 0, 0]);
  device.queue.writeBuffer(ctx.splatUni, 0, su);
  const ru = new ArrayBuffer(16);
  new Uint32Array(ru, 0, 2).set([cells, resolveGrid.x * WG]);
  new Float32Array(ru, 8, 1).set([1 / scale]);
  device.queue.writeBuffer(ctx.resolveUni, 0, ru);

  const enc = device.createCommandEncoder();
  enc.clearBuffer(ctx.acc, 0, bytes);
  const splat = enc.beginComputePass();
  splat.setPipeline(ctx.splat);
  splat.setBindGroup(
    0,
    device.createBindGroup({
      layout: ctx.splat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ctx.splatUni } },
        { binding: 1, resource: { buffer: points, size: Math.max(4, n * stride * 4) } },
        { binding: 2, resource: { buffer: ctx.acc, size: bytes } },
      ],
    }),
  );
  if (n > 0) splat.dispatchWorkgroups(splatGrid.x, splatGrid.y);
  splat.end();
  const resolve = enc.beginComputePass();
  resolve.setPipeline(ctx.resolve);
  resolve.setBindGroup(
    0,
    device.createBindGroup({
      layout: ctx.resolve.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ctx.resolveUni } },
        { binding: 1, resource: { buffer: ctx.acc, size: bytes } },
        { binding: 2, resource: { buffer: dst, size: cells * 4 } },
      ],
    }),
  );
  resolve.dispatchWorkgroups(resolveGrid.x, resolveGrid.y);
  resolve.end();
  device.queue.submit([enc.finish()]);
}
