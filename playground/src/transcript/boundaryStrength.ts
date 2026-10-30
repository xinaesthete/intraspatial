// PROTOTYPE (docs/wand-metric-and-regions.md, experiment 1): where the channel mixture changes
// fastest, measured in the wand's metric. Throwaway until a region rule is chosen from what it shows.
//
// The metric is the note's PPCA form, as a K×K matrix A so the shader is one `u = A·z`: the leading
// q modes at 1/√λ_k, the rest at √w/σ with σ² the mean of their λ. q = K is today's "every mode",
// q = 3 with w = 0 is today's "3 modes". Boundary strength is the Di Zenzo structure tensor of u,
// G = Σ_k ∇u_k ∇u_kᵀ, and its larger eigenvalue's root: the most the whitened distance changes per
// step, in the steepest direction. It is scaled to "per kernel radius" so a θ survives a zoom.
//
// Experiment 1b: the modes after q are spatially rough, so a gradient amplifies them well beyond
// their share of the variance. Their gradient is therefore taken from a Gaussian-blurred copy of
// the channels (blurring z then projecting = blurring those modes, the projection being linear),
// at a σ set in kernel radii. The main modes keep the raw rasters' gradient.

import { compileShader, getDevice } from "../../../src/gpu/device";
import type { GramMatrixGpuResult } from "../../../src/gpu/spatial/gramMatrix";

/** Raster unpacking shared by both shaders: the layout `ResidentRasters` documents. */
const FETCH = /* wgsl */ `
fn fetch(a: u32, col: u32, row: u32) -> f32 {
  let base = a * u32(U.height) * u32(U.rowWords) + row * u32(U.rowWords);
  if (U.half != 0.0) {
    let pair = unpack2x16float(rasters[base + (col >> 1u)]);
    return select(pair.y, pair.x, (col & 1u) == 0u);
  }
  return bitcast<f32>(rasters[base + col]);
}
`;

/** One axis of a separable Gaussian, edge-replicated. The horizontal pass reads the rasters, the
 *  vertical one the horizontal pass's output; both write plain f32, channel-major, stride width. */
const blurShader = (horizontal: boolean) => /* wgsl */ `
struct Uni { K: f32, width: f32, height: f32, rowWords: f32, half: f32, halfWidth: f32, sigma: f32, pad: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> ${horizontal ? "rasters: array<u32>" : "src: array<f32>"};
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
${horizontal ? FETCH : ""}
fn sampleAt(a: u32, col: u32, row: u32) -> f32 {
  ${horizontal ? "return fetch(a, col, row);" : "return src[(a * u32(U.height) + row) * u32(U.width) + col];"}
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = i32(U.width);
  let h = i32(U.height);
  if (i32(gid.x) >= w || i32(gid.y) >= h) { return; }
  let r = i32(U.halfWidth);
  let inv = 1.0 / max(2.0 * U.sigma * U.sigma, 1e-6);
  for (var a = 0u; a < u32(U.K); a = a + 1u) {
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
      let wt = exp(-f32(i * i) * inv);
      ${
        horizontal
          ? "let c = u32(clamp(i32(gid.x) + i, 0, w - 1)); acc = acc + wt * sampleAt(a, c, gid.y);"
          : "let c = u32(clamp(i32(gid.y) + i, 0, h - 1)); acc = acc + wt * sampleAt(a, gid.x, c);"
      }
      wsum = wsum + wt;
    }
    dst[(a * u32(h) + gid.y) * u32(w) + gid.x] = acc / wsum;
  }
}
`;

/** u = A·z at every pixel, mode-major: the wand's coordinates, so d to any seed is |u − u_seed|. */
const PROJECT = /* wgsl */ `
struct Uni { K: f32, width: f32, height: f32, rowWords: f32, half: f32, perR: f32, q: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<u32>;
@group(0) @binding(2) var<storage, read> chan: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> uOut: array<f32>;
${FETCH}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = u32(U.width);
  let h = u32(U.height);
  if (gid.x >= w || gid.y >= h) { return; }
  let K = u32(U.K);
  var z: array<f32, 32>;
  for (var a = 0u; a < K; a = a + 1u) { z[a] = (fetch(a, gid.x, gid.y) - chan[a * 2u]) * chan[a * 2u + 1u]; }
  for (var k = 0u; k < K; k = k + 1u) {
    var acc = 0.0;
    for (var a = 0u; a < K; a = a + 1u) { acc = acc + A[k * K + a] * z[a]; }
    uOut[(k * h + gid.y) * w + gid.x] = acc;
  }
}
`;

const SHADER = /* wgsl */ `
struct Uni { K: f32, width: f32, height: f32, rowWords: f32, half: f32, perR: f32, q: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<u32>;
/** Per channel: mean, 1/sd. */
@group(0) @binding(2) var<storage, read> chan: array<f32>;
/** K×K, row-major: row k is mode k scaled by its PPCA weight. */
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
/** The channels blurred for the modes after q: f32, channel-major, stride width. */
@group(0) @binding(5) var<storage, read> blurred: array<f32>;
${FETCH}
fn fetchS(a: u32, col: u32, row: u32) -> f32 { return blurred[(a * u32(U.height) + row) * u32(U.width) + col]; }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = u32(U.width);
  let h = u32(U.height);
  if (gid.x >= w || gid.y >= h) { return; }
  let K = u32(U.K);
  // Central differences, one-sided at the raster's edge.
  let xl = select(gid.x - 1u, 0u, gid.x == 0u);
  let xr = min(gid.x + 1u, w - 1u);
  let yu = select(gid.y - 1u, 0u, gid.y == 0u);
  let yd = min(gid.y + 1u, h - 1u);
  let sx = 1.0 / f32(max(xr - xl, 1u));
  let sy = 1.0 / f32(max(yd - yu, 1u));
  // ∇z per channel, then ∇u = A·∇z: the standardisation's mean cancels in a difference.
  var gx: array<f32, 32>;
  var gy: array<f32, 32>;
  var bx: array<f32, 32>;
  var by: array<f32, 32>;
  for (var a = 0u; a < K; a = a + 1u) {
    let s = chan[a * 2u + 1u];
    gx[a] = (fetch(a, xr, gid.y) - fetch(a, xl, gid.y)) * s * sx;
    gy[a] = (fetch(a, gid.x, yd) - fetch(a, gid.x, yu)) * s * sy;
    bx[a] = (fetchS(a, xr, gid.y) - fetchS(a, xl, gid.y)) * s * sx;
    by[a] = (fetchS(a, gid.x, yd) - fetchS(a, gid.x, yu)) * s * sy;
  }
  let q = u32(U.q);
  var gxx = 0.0; var gyy = 0.0; var gxy = 0.0;
  for (var k = 0u; k < K; k = k + 1u) {
    var ux = 0.0; var uy = 0.0;
    let isMain = k < q;
    for (var a = 0u; a < K; a = a + 1u) {
      let m = A[k * K + a];
      ux = ux + m * select(bx[a], gx[a], isMain);
      uy = uy + m * select(by[a], gy[a], isMain);
    }
    gxx = gxx + ux * ux; gyy = gyy + uy * uy; gxy = gxy + ux * uy;
  }
  let half = 0.5 * (gxx - gyy);
  let lmax = 0.5 * (gxx + gyy) + sqrt(half * half + gxy * gxy);
  // Strength, then the tensor itself (per radius²): its components interpolate without the sign
  // ambiguity an eigenvector has, so the overlay and the ridge test take the direction from them.
  let r2 = U.perR * U.perR;
  let o = (gid.y * w + gid.x) * 4u;
  out[o] = sqrt(max(lmax, 0.0)) * U.perR;
  out[o + 1u] = gxx * r2;
  out[o + 2u] = gyy * r2;
  out[o + 3u] = gxy * r2;
}
`;

/** The PPCA metric as a K×K matrix (row k = mode k · weight), and the noise floor it used. */
export function ppcaMetric(
  vectors: Float64Array,
  values: Float64Array,
  K: number,
  q: number,
  w: number,
): { A: Float32Array<ArrayBuffer>; sigma2: number } {
  const qq = Math.max(1, Math.min(q, K));
  // Same floor as whiteningMatrix, so q = K reproduces today's "every mode".
  const floor = Math.max(values[0] ?? 1, 1e-12) * 1e-6;
  let tail = 0;
  for (let k = qq; k < K; k++) tail += Math.max(values[k] ?? 0, 0);
  const sigma2 = qq < K ? Math.max(tail / (K - qq), floor) : 0;
  const A = new Float32Array(K * K);
  for (let k = 0; k < K; k++) {
    const weight = k < qq ? 1 / Math.sqrt(Math.max(values[k] ?? 0, floor)) : Math.sqrt(Math.max(0, w) / sigma2);
    for (let a = 0; a < K; a++) A[k * K + a] = weight * (vectors[k * K + a] ?? 0);
  }
  return { A, sigma2 };
}

export interface BoundaryField {
  /** Per raster pixel, row-major, on the device: strength, then the tensor's xx, yy, xy. */
  readonly buffer: GPUBuffer;
  /** The same, on the host, for the ridge-bounded fill. */
  readonly field: Float32Array;
  /** The wand's coordinates u = A·z per pixel, mode-major (K planes of width×height), on the host. */
  readonly u: Float32Array;
  readonly K: number;
  /** Raster pixels per kernel radius. */
  readonly perR: number;
  /** Quantiles of the strength over the window, for the readout and a θ that means something. */
  readonly quantiles: { readonly p50: number; readonly p90: number; readonly p99: number; readonly max: number };
}

let ctx:
  | Promise<{
      device: GPUDevice;
      pipeline: GPUComputePipeline;
      blurH: GPUComputePipeline;
      blurV: GPUComputePipeline;
      project: GPUComputePipeline;
    }>
  | undefined;
type Bufs = {
  uni: GPUBuffer;
  blurUni: GPUBuffer;
  chan: GPUBuffer;
  A: GPUBuffer;
  out: GPUBuffer;
  staging: GPUBuffer;
  tmp: GPUBuffer;
  smooth: GPUBuffer;
  u: GPUBuffer;
  uStaging: GPUBuffer;
};
/** One buffer set per caller-named slot: each slot's `out` is drawn by a layer, so two scales
 *  computed in turn must not share it. */
const slots = new Map<string, Bufs>();

function fit(device: GPUDevice, old: GPUBuffer | undefined, size: number, usage: number, label: string): GPUBuffer {
  if (old && old.size >= size) return old;
  old?.destroy();
  return device.createBuffer({ label, size: Math.max(size, 16), usage });
}

/**
 * Boundary strength for `res` under metric `A`, in whitened distance per kernel radius.
 * `mean`/`sd` are the standardisation the map is painted with (a locked basis's, or res's own).
 * `q` is how many rows of `A` are main modes; the rest take their gradient from the channels blurred
 * at σ = `smoothR` kernel radii (0 = no blur). One call at a time; the returned buffer is reused.
 */
export async function boundaryStrength(
  res: GramMatrixGpuResult,
  rasters: GPUBuffer,
  mean: ArrayLike<number>,
  sd: ArrayLike<number>,
  A: Float32Array<ArrayBuffer>,
  radius: number,
  q: number,
  smoothR: number,
  slot = "main",
): Promise<BoundaryField> {
  ctx ??= (async () => {
    const device = await getDevice();
    const make = async (code: string, label: string) =>
      device.createComputePipeline({
        label,
        layout: "auto",
        compute: { module: await compileShader(device, code, label), entryPoint: "main" },
      });
    return {
      device,
      pipeline: await make(SHADER, "boundaryStrength"),
      blurH: await make(blurShader(true), "boundaryStrength:blurH"),
      blurV: await make(blurShader(false), "boundaryStrength:blurV"),
      project: await make(PROJECT, "boundaryStrength:project"),
    };
  })();
  const { device, pipeline, blurH, blurV, project } = await ctx;
  const K = res.labels.length;
  const { width, height } = res;
  const n = width * height;
  const chan = new Float32Array(K * 2);
  for (let a = 0; a < K; a++) {
    chan[a * 2] = mean[a] ?? 0;
    chan[a * 2 + 1] = (sd[a] ?? 0) > 0 ? 1 / sd[a]! : 0;
  }
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  let bufs = slots.get(slot);
  bufs = {
    uni: bufs?.uni ?? device.createBuffer({ label: "boundary:uni", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    blurUni:
      bufs?.blurUni ??
      device.createBuffer({ label: "boundary:blurUni", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    tmp: fit(device, bufs?.tmp, K * n * 4, GPUBufferUsage.STORAGE, "boundary:tmp"),
    smooth: fit(device, bufs?.smooth, K * n * 4, GPUBufferUsage.STORAGE, "boundary:smooth"),
    u: fit(device, bufs?.u, K * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "boundary:u"),
    uStaging: fit(device, bufs?.uStaging, K * n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "boundary:uStaging"),
    chan: fit(device, bufs?.chan, chan.byteLength, STORAGE, "boundary:chan"),
    A: fit(device, bufs?.A, A.byteLength, STORAGE, "boundary:A"),
    out: fit(device, bufs?.out, n * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "boundary:out"),
    staging: fit(device, bufs?.staging, n * 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "boundary:staging"),
  };
  slots.set(slot, bufs);
  // Raster pixels per kernel radius: the per-pixel strength times this is per radius.
  const pixel = (res.bbox[2] - res.bbox[0]) / width;
  const perR = pixel > 0 ? radius / pixel : 1;
  device.queue.writeBuffer(
    bufs.uni,
    0,
    new Float32Array([K, width, height, res.resident.rowWords, res.resident.precision === "f16" ? 1 : 0, perR, q, 0]),
  );
  const sigma = Math.max(0, smoothR) * perR;
  const halfWidth = Math.min(48, Math.ceil(3 * sigma));
  device.queue.writeBuffer(
    bufs.blurUni,
    0,
    new Float32Array([K, width, height, res.resident.rowWords, res.resident.precision === "f16" ? 1 : 0, halfWidth, sigma, 0]),
  );
  device.queue.writeBuffer(bufs.chan, 0, chan);
  device.queue.writeBuffer(bufs.A, 0, A);
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufs.uni } },
      { binding: 1, resource: { buffer: rasters } },
      { binding: 2, resource: { buffer: bufs.chan } },
      { binding: 3, resource: { buffer: bufs.A } },
      { binding: 4, resource: { buffer: bufs.out } },
      { binding: 5, resource: { buffer: bufs.smooth } },
    ],
  });
  const blur = (p: GPUComputePipeline, src: GPUBuffer, dst: GPUBuffer) =>
    device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufs!.blurUni } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: dst } },
      ],
    });
  const enc = device.createCommandEncoder({ label: "boundaryStrength" });
  for (const [p, bg] of [
    [blurH, blur(blurH, rasters, bufs.tmp)],
    [blurV, blur(blurV, bufs.tmp, bufs.smooth)],
  ] as const) {
    const bp = enc.beginComputePass();
    bp.setPipeline(p);
    bp.setBindGroup(0, bg);
    bp.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    bp.end();
  }
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  const pp = enc.beginComputePass();
  pp.setPipeline(project);
  pp.setBindGroup(
    0,
    device.createBindGroup({
      layout: project.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufs.uni } },
        { binding: 1, resource: { buffer: rasters } },
        { binding: 2, resource: { buffer: bufs.chan } },
        { binding: 3, resource: { buffer: bufs.A } },
        { binding: 4, resource: { buffer: bufs.u } },
      ],
    }),
  );
  pp.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pp.end();
  enc.copyBufferToBuffer(bufs.out, 0, bufs.staging, 0, n * 16);
  enc.copyBufferToBuffer(bufs.u, 0, bufs.uStaging, 0, K * n * 4);
  device.queue.submit([enc.finish()]);
  await Promise.all([bufs.staging.mapAsync(GPUMapMode.READ, 0, n * 16), bufs.uStaging.mapAsync(GPUMapMode.READ, 0, K * n * 4)]);
  const field = new Float32Array(bufs.staging.getMappedRange(0, n * 16).slice(0));
  bufs.staging.unmap();
  const u = new Float32Array(bufs.uStaging.getMappedRange(0, K * n * 4).slice(0));
  bufs.uStaging.unmap();
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++) s[i] = field[i * 4]!;
  s.sort();
  const at = (p: number) => s[Math.min(n - 1, Math.floor(p * n))] ?? 0;
  return { buffer: bufs.out, field, u, K, perR, quantiles: { p50: at(0.5), p90: at(0.9), p99: at(0.99), max: s[n - 1] ?? 0 } };
}

/** Bilinear strength at a raster position (pixel centres at whole numbers), edge-clamped. */
function strengthAt(f: Float32Array, width: number, height: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(width - 1, x));
  const cy = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const v = (c: number, r: number) => f[(r * width + c) * 4]!;
  return (v(x0, y0) * (1 - fx) + v(x1, y0) * fx) * (1 - fy) + (v(x0, y1) * (1 - fx) + v(x1, y1) * fx) * fy;
}

/**
 * Ridge pixels: the discrete zero-crossing of the strength's derivative across the edge (Cumani's
 * rule as Canny's non-maximum suppression). A pixel is on a ridge when its strength is at least θ and
 * no less than the strength one pixel either side along the tensor's leading direction. The
 * result is 8-connected lines, which a 4-connected fill cannot cross.
 */
export function ridges(f: Float32Array, width: number, height: number, theta: number): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const i = (r * width + c) * 4;
      const s = f[i]!;
      if (s < theta) continue;
      const ang = 0.5 * Math.atan2(2 * f[i + 3]!, f[i + 1]! - f[i + 2]!);
      const nx = Math.cos(ang);
      const ny = Math.sin(ang);
      if (s >= strengthAt(f, width, height, c + nx, r + ny) && s >= strengthAt(f, width, height, c - nx, r - ny)) out[r * width + c] = 1;
    }
  }
  return out;
}

/** The 4-connected cell holding `seed` whose pixels are not ridge pixels: 1 inside, 0 outside. */
export function ridgeCell(ridge: Uint8Array, width: number, height: number, seed: { col: number; row: number }): Uint8Array {
  const cell = new Uint8Array(width * height);
  const c = Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  const r = Math.max(0, Math.min(height - 1, Math.round(seed.row)));
  const start = r * width + c;
  if (ridge[start]) return cell;
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  cell[start] = 1;
  queue[tail++] = start;
  const visit = (i: number) => {
    if (cell[i] === 0 && ridge[i] === 0) {
      cell[i] = 1;
      queue[tail++] = i;
    }
  };
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    if (x > 0) visit(i - 1);
    if (x < width - 1) visit(i + 1);
    if (i >= width) visit(i - width);
    if (i < width * (height - 1)) visit(i + width);
  }
  return cell;
}

/** A separable Gaussian on a host raster, edge-replicated; σ in pixels (≤ 0 returns a copy). */
function gaussian(src: Float32Array, width: number, height: number, sigma: number): Float32Array {
  if (sigma <= 0) return src.slice();
  const r = Math.min(48, Math.ceil(3 * sigma));
  const w = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) sum += w[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < w.length; i++) w[i]! /= sum;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += w[i + r]! * src[y * width + Math.max(0, Math.min(width - 1, x + i))]!;
      tmp[y * width + x] = acc;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) acc += w[i + r]! * tmp[Math.max(0, Math.min(height - 1, y + i)) * width + x]!;
      out[y * width + x] = acc;
    }
  }
  return out;
}

/**
 * The Marr–Hildreth cell for one seed: the wand distance d to the seed's mixture, smoothed at
 * σ = `smoothR` radii, and the 4-connected region around the seed where its Laplacian is ≥ 0.
 * Inside, d is a bowl (curving up); the boundary is where the rise away from the seed's mixture
 * stops accelerating: the shoulder of the step, which is a closed curve because it is the zero set
 * of one scalar field. `tol` > 0 also stops the fill where d itself exceeds it (on the raw d,
 * so the gate means what the wand's tolerance means); 0 leaves it off.
 */
export function laplacianCell(
  f: BoundaryField,
  width: number,
  height: number,
  seed: { col: number; row: number },
  smoothR: number,
  tol = 0,
  /** false: a plain flood fill on `d ≤ tol` (the Laplacian test is skipped). */
  useLaplacian = true,
): { mask: Uint8Array; d: Float32Array; dEdge: number; gated: number } {
  const n = width * height;
  const { u, K } = f;
  const c = Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  const r = Math.max(0, Math.min(height - 1, Math.round(seed.row)));
  const start = r * width + c;
  // The seed's mixture, interpolated bilinearly at the seed itself: read at the nearest pixel, it
  // would step every time the pointer crossed one (a third of r on the usual raster).
  const fx = Math.max(0, Math.min(width - 1, seed.col));
  const fy = Math.max(0, Math.min(height - 1, seed.row));
  const x0 = Math.min(width - 2, Math.floor(fx));
  const y0 = Math.min(height - 2, Math.floor(fy));
  const tx = width > 1 ? fx - x0 : 0;
  const ty = height > 1 ? fy - y0 : 0;
  const corners = [
    [Math.max(0, y0) * width + Math.max(0, x0), (1 - tx) * (1 - ty)],
    [Math.max(0, y0) * width + Math.min(width - 1, x0 + 1), tx * (1 - ty)],
    [Math.min(height - 1, y0 + 1) * width + Math.max(0, x0), (1 - tx) * ty],
    [Math.min(height - 1, y0 + 1) * width + Math.min(width - 1, x0 + 1), tx * ty],
  ] as const;
  const raw = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    let ref = 0;
    for (const [i, wt] of corners) ref += wt * u[k * n + i]!;
    for (let i = 0; i < n; i++) {
      const dv = u[k * n + i]! - ref;
      raw[i]! += dv * dv;
    }
  }
  for (let i = 0; i < n; i++) raw[i] = Math.sqrt(raw[i]!);
  const d = gaussian(raw, width, height, smoothR * f.perR);
  const lap = (i: number) => {
    const x = i % width;
    const y = (i - x) / width;
    const at = (xx: number, yy: number) => d[Math.max(0, Math.min(height - 1, yy)) * width + Math.max(0, Math.min(width - 1, xx))]!;
    return at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * d[i]!;
  };
  const mask = new Uint8Array(n);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;
  mask[start] = 1;
  queue[tail++] = start;
  // The seed is in regardless: the smoothed bowl's floor can be flat to rounding.
  let edgeSum = 0;
  let edgeCount = 0;
  /** Edge pixels stopped by the gate rather than by the Laplacian. */
  let gated = 0;
  const visit = (i: number) => {
    if (mask[i] !== 0) return;
    const curving = !useLaplacian || lap(i) >= 0;
    const near = tol <= 0 || raw[i]! <= tol;
    if (curving && near) {
      mask[i] = 1;
      queue[tail++] = i;
    } else {
      mask[i] = 2; // a boundary pixel: seen, outside
      if (curving) gated++;
      edgeSum += raw[i]!;
      edgeCount++;
    }
  };
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    if (x > 0) visit(i - 1);
    if (x < width - 1) visit(i + 1);
    if (i >= width) visit(i - width);
    if (i < n - width) visit(i + width);
  }
  for (let i = 0; i < n; i++) if (mask[i] === 2) mask[i] = 0;
  return { mask, d: raw, dEdge: edgeCount > 0 ? edgeSum / edgeCount : 0, gated: edgeCount > 0 ? gated / edgeCount : 0 };
}

/** 4-connected component of `set` holding `start`. */
function component(set: Uint8Array, width: number, height: number, start: number): Uint8Array {
  const out = new Uint8Array(width * height);
  if (!set[start]) return out;
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  out[start] = 1;
  queue[tail++] = start;
  while (head < tail) {
    const i = queue[head++]!;
    const x = i % width;
    for (const j of [x > 0 ? i - 1 : -1, x < width - 1 ? i + 1 : -1, i - width, i + width]) {
      if (j >= 0 && j < width * height && set[j] && !out[j]) {
        out[j] = 1;
        queue[tail++] = j;
      }
    }
  }
  return out;
}

/**
 * Morphological opening of a cell by a disk of radius `rho` pixels (erode, then dilate), then the
 * piece the seed belongs to. Opening removes every part of the cell narrower than the disk (the
 * branches and the necks between them) and leaves the rest exactly as it was. When the seed itself
 * sat on a removed branch the result is empty and `moved` says so: a piece elsewhere would be a
 * patch the pointer is not on, and it jumps about as the pointer moves.
 */
export function openCell(
  cell: Uint8Array,
  width: number,
  height: number,
  seed: { col: number; row: number },
  rho: number,
): { mask: Uint8Array; moved: boolean } {
  const n = width * height;
  const r = Math.max(0, rho);
  const start = Math.max(0, Math.min(height - 1, Math.round(seed.row))) * width + Math.max(0, Math.min(width - 1, Math.round(seed.col)));
  if (r < 0.5) return { mask: cell, moved: false };
  const R = Math.ceil(r);
  const disk: [number, number][] = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= r * r) disk.push([dx, dy]);
  // Outside the raster counts as outside the cell, so a cell is eroded from the window's edge too.
  const eroded = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!cell[y * width + x]) continue;
      let keep = 1;
      for (const [dx, dy] of disk) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height || !cell[yy * width + xx]) {
          keep = 0;
          break;
        }
      }
      eroded[y * width + x] = keep;
    }
  }
  const opened = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!eroded[y * width + x]) continue;
      for (const [dx, dy] of disk) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < width && yy < height) opened[yy * width + xx] = 1;
      }
    }
  }
  if (opened[start]) return { mask: component(opened, width, height, start), moved: false };
  return { mask: new Uint8Array(n), moved: true };
}

/**
 * Two scales: the broad cell picks the patch, the narrow cell places its edge. The result is the
 * narrow cell inside the broad one, where the broad one is first grown by `slack` narrow-raster pixels
 * (its edge is blurred over about r₂, so the narrow edge may sit a little outside it); then the
 * click's piece. The broad cell may be on a coarser raster over the same bbox; it is sampled
 * nearest at each narrow pixel's centre.
 */
export function focusCell(
  narrow: Uint8Array,
  nw: number,
  nh: number,
  broad: Uint8Array,
  bw: number,
  bh: number,
  seed: { col: number; row: number },
  slack: number,
): Uint8Array {
  const n = nw * nh;
  const inBroad = new Uint8Array(n);
  for (let r = 0; r < nh; r++) {
    const br = Math.max(0, Math.min(bh - 1, Math.round(((r + 0.5) * bh) / nh - 0.5)));
    for (let c = 0; c < nw; c++) {
      const bc = Math.max(0, Math.min(bw - 1, Math.round(((c + 0.5) * bw) / nw - 0.5)));
      inBroad[r * nw + c] = broad[br * bw + bc]!;
    }
  }
  let grown = inBroad;
  const R = Math.max(0, slack);
  if (R >= 0.5) {
    grown = new Uint8Array(n);
    const Ri = Math.ceil(R);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        if (!inBroad[y * nw + x]) continue;
        for (let dy = -Ri; dy <= Ri; dy++) {
          for (let dx = -Ri; dx <= Ri; dx++) {
            if (dx * dx + dy * dy > R * R) continue;
            const xx = x + dx;
            const yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < nw && yy < nh) grown[yy * nw + xx] = 1;
          }
        }
      }
    }
  }
  const both = new Uint8Array(n);
  for (let i = 0; i < n; i++) both[i] = narrow[i]! & grown[i]!;
  const start = Math.max(0, Math.min(nh - 1, Math.round(seed.row))) * nw + Math.max(0, Math.min(nw - 1, Math.round(seed.col)));
  return component(both, nw, nh, start);
}
