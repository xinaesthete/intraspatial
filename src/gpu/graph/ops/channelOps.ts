// Ops over a grid's channel axis (planar: plane k is `data[k·n … (k+1)·n)`, ADR-0004/0015): read the
// Gram's raster stack into one, mix channels per pixel, and measure where the mixture changes fastest.
// Together with `convolveSeparable` (which blurs each plane) they compose the wand's boundary field
// (docs/wand-metric-and-regions.md):
//
//   x = gramChannels(snapshot) → u = channelMix(x; A·diag(1/sd), −A·diag(1/sd)·mean)
//   strength = channelStructureTensor(u)            (the Di Zenzo tensor, per kernel radius)
//
// Resident (Tier-2) with raw-WGSL kernels and CPU references. Channel counts are capped at 32: the
// kernels hold one pixel's channels in a private array.
import {
  checkBindingSize,
  compileShader,
  deviceScoped,
  dispatchGrid,
  MAX_WORKGROUPS_PER_DIM,
  sized,
  submitDispatch,
} from "#src/gpu/device";
import { ensureBuf } from "#src/gpu/scan/prefixSum";
import { channelAxis, type FieldValue, type Shape, type TensorAxis } from "../handle";
import type { ExecCtx, OpType, Params } from "../op";

const WG = 64;
/** The most channels a kernel holds per pixel. */
export const MAX_CHANNELS = 32;

function gridOf(s: Shape, who: string): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error(`${who}: input must be a grid`);
  return { width: s.width, height: s.height };
}

function channelsOf(v: { axes?: readonly TensorAxis[] }, who: string): number {
  const K = channelAxis(v)?.length ?? 1;
  if (K > MAX_CHANNELS) throw new Error(`${who}: ${K} channels, at most ${MAX_CHANNELS}`);
  return K;
}

const channelAxes = (K: number): TensorAxis[] => [{ name: "c", type: "channel", length: K }];

/** One raw-WGSL pipeline per device, over uniform + read-only storages + one output. */
function pipelineFor(label: string, code: string, reads: number) {
  return deviceScoped(async (device: GPUDevice) => {
    const entries: GPUBindGroupLayoutEntry[] = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }];
    for (let i = 1; i <= reads; i++)
      entries.push({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } });
    entries.push({ binding: reads + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
    const layout = device.createBindGroupLayout({ label, entries });
    const module = await compileShader(device, code, label);
    const pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    return { device, layout, pipeline };
  });
}

/** Run a per-pixel kernel: `uni` (u32/f32 words, the first two `n` and `gridX`), then the bindings. */
async function runPerPixel(
  ctx: ExecCtx,
  get: ReturnType<typeof pipelineFor>,
  label: string,
  n: number,
  uni: (gridX: number) => ArrayBuffer,
  bindings: GPUBufferBinding[],
): Promise<void> {
  const { device, layout, pipeline } = await get(await ctx.backend.getDevice());
  const { x: gridX, y: gridY } = dispatchGrid(Math.ceil(n / WG), MAX_WORKGROUPS_PER_DIM);
  const u = ensureBuf(device, `${label}:uni`, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(u, 0, uni(gridX));
  const bind = device.createBindGroup({
    label,
    layout,
    entries: [{ binding: 0, resource: sized(u, 32) }, ...bindings.map((resource, i) => ({ binding: i + 1, resource }))],
  });
  submitDispatch(device, label, pipeline, bind, gridX, gridY);
}

/** A small host array as a pooled storage buffer. */
function uploadSmall(device: GPUDevice, key: string, data: Float32Array): GPUBufferBinding {
  const buf = ensureBuf(device, key, Math.max(16, data.byteLength), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return sized(buf, Math.max(4, data.byteLength));
}

const PIXEL = /* wgsl */ `
fn pixel(gid: vec3u) -> u32 { return gid.x + gid.y * U.gridX * ${WG}u; }
`;

// --- gramChannels ---------------------------------------------------------------------------------

/** The Gram's raster stack as `gramMatrixGpu` leaves it: per channel, `height` rows of `rowWords`
 *  words, each word one f32 texel or (half) two f16 texels. Row 0 is the top row (bbox maxY). */
export interface GramRasters {
  readonly kind: "gramRasters";
  /** On the device, for `execute`. */
  readonly buffer?: GPUBuffer;
  /** On the host, for `cpuGolden`. */
  readonly words?: Uint32Array;
  readonly rowWords: number;
  readonly half: boolean;
}

function rastersOf(v: FieldValue): GramRasters {
  const p = v.payload as GramRasters | undefined;
  if (p?.kind !== "gramRasters") throw new Error("gramChannels: input carries no gramRasters payload");
  return p;
}

const GRAM_CHANNELS = /* wgsl */ `
struct Uni { n: u32, gridX: u32, K: u32, width: u32, rowWords: u32, half: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> rasters: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
${PIXEL}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = pixel(gid);
  if (i >= U.n) { return; }
  let col = i % U.width;
  let row = i / U.width;
  let height = U.n / U.width;
  for (var a = 0u; a < U.K; a = a + 1u) {
    let base = (a * height + row) * U.rowWords;
    var v: f32;
    if (U.half != 0u) {
      let pair = unpack2x16float(rasters[base + (col >> 1u)]);
      v = select(pair.y, pair.x, (col & 1u) == 0u);
    } else {
      v = bitcast<f32>(rasters[base + col]);
    }
    dst[a * U.n + i] = v;
  }
}
`;
const gramChannelsPipe = pipelineFor("gramChannels", GRAM_CHANNELS, 1);

/** Unpack one f16 from the low (even column) or high (odd) half of a word. */
function f16(word: number, high: boolean): number {
  const h = high ? word >>> 16 : word & 0xffff;
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export const gramChannelsOp: OpType = {
  name: "gramChannels",
  label: "Gram channels",
  category: "Channels",
  describe: "The Gram's packed raster stack (f32 or f16 rows) as a planar f32 grid with a channel axis.",
  inputs: [{ name: "rasters", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  resident: true,
  async execute(ctx, inputs) {
    const v = inputs[0]!;
    const r = rastersOf(v);
    if (!r.buffer) throw new Error("gramChannels: the rasters are not on the device");
    const { width, height } = gridOf(v.shape, "gramChannels");
    const K = channelsOf(v, "gramChannels");
    const n = width * height;
    const device = await ctx.backend.getDevice();
    checkBindingSize(device, `gramChannels: ${K} channels at ${width}x${height}`, K * n * 4);
    const out = await ctx.backend.lease(K * n * 4);
    await runPerPixel(
      ctx,
      gramChannelsPipe,
      "gramChannels",
      n,
      (gridX) => new Uint32Array([n, gridX, K, width, r.rowWords, r.half ? 1 : 0, 0, 0]).buffer,
      [sized(r.buffer, K * height * r.rowWords * 4), sized(out.buffer, K * n * 4)],
    );
    return [{ shape: v.shape, dtype: "f32", axes: channelAxes(K), buffer: out }];
  },
  cpuGolden(inputs) {
    const v = inputs[0]!;
    const r = rastersOf(v);
    if (!r.words) throw new Error("gramChannels: the rasters are not on the host");
    const { width, height } = gridOf(v.shape, "gramChannels");
    const K = channelsOf(v, "gramChannels");
    const n = width * height;
    const out = new Float32Array(K * n);
    const asF32 = new Float32Array(r.words.buffer, r.words.byteOffset, r.words.length);
    for (let a = 0; a < K; a++)
      for (let row = 0; row < height; row++)
        for (let col = 0; col < width; col++) {
          const base = (a * height + row) * r.rowWords;
          out[a * n + row * width + col] = r.half ? f16(r.words[base + (col >> 1)]!, (col & 1) === 1) : asF32[base + col]!;
        }
    return [{ shape: v.shape, dtype: "f32", axes: channelAxes(K), data: out }];
  },
};

// --- channelMix -----------------------------------------------------------------------------------

/** The mix's matrix (rows × K, row-major) and offset (rows), checked against the input's K. */
function mixOf(params: Params, K: number, who: string): { M: Float32Array; b: Float32Array; rows: number } {
  const M = Float32Array.from((params.matrix as ArrayLike<number> | undefined) ?? []);
  const rows = M.length / Math.max(1, K);
  if (!Number.isInteger(rows) || rows < 1) throw new Error(`${who}: matrix has ${M.length} entries, not a multiple of ${K} channels`);
  if (rows > MAX_CHANNELS) throw new Error(`${who}: ${rows} output channels, at most ${MAX_CHANNELS}`);
  const b = new Float32Array(rows);
  b.set(Array.from((params.offset as ArrayLike<number> | undefined) ?? []).slice(0, rows));
  return { M, b, rows };
}

const CHANNEL_MIX = /* wgsl */ `
struct Uni { n: u32, gridX: u32, K: u32, rows: u32, pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> M: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
${PIXEL}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = pixel(gid);
  if (i >= U.n) { return; }
  var x: array<f32, ${MAX_CHANNELS}>;
  for (var a = 0u; a < U.K; a = a + 1u) { x[a] = src[a * U.n + i]; }
  for (var k = 0u; k < U.rows; k = k + 1u) {
    var acc = b[k];
    for (var a = 0u; a < U.K; a = a + 1u) { acc = acc + M[k * U.K + a] * x[a]; }
    dst[k * U.n + i] = acc;
  }
}
`;
const channelMixPipe = pipelineFor("channelMix", CHANNEL_MIX, 3);

export const channelMixOp: OpType = {
  name: "channelMix",
  label: "Mix channels",
  category: "Channels",
  describe: "Per pixel, out = M·x + b over the channel axis: a projection, a standardisation, or both.",
  help: {
    detail: "M is rows × K (row-major) and b has one entry per row. Standardise-then-project is one mix: M = A·diag(1/sd), b = −M·mean.",
    math: String.raw`y_k(p) = b_k + \sum_a M_{ka}\, x_a(p)`,
  },
  inputs: [{ name: "in", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => [inputs[0]!],
  inferAxes(inputs, params) {
    const K = inputs[0]?.find((a) => a.type === "channel")?.length ?? 1;
    return [channelAxes(mixOf(params, K, "channelMix").rows)];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const v = inputs[0]!;
    if (!v.buffer) throw new Error("channelMix: resident op received a non-resident input");
    const { width, height } = gridOf(v.shape, "channelMix");
    const K = channelsOf(v, "channelMix");
    const { M, b, rows } = mixOf(params, K, "channelMix");
    const n = width * height;
    const device = await ctx.backend.getDevice();
    const out = await ctx.backend.lease(rows * n * 4);
    await runPerPixel(ctx, channelMixPipe, "channelMix", n, (gridX) => new Uint32Array([n, gridX, K, rows, 0, 0, 0, 0]).buffer, [
      sized(v.buffer.buffer, K * n * 4),
      uploadSmall(device, "channelMix:M", M),
      uploadSmall(device, "channelMix:b", b),
      sized(out.buffer, rows * n * 4),
    ]);
    return [{ shape: v.shape, dtype: "f32", axes: channelAxes(rows), buffer: out }];
  },
  cpuGolden(inputs, params) {
    const v = inputs[0]!;
    const { width, height } = gridOf(v.shape, "channelMix");
    const K = channelsOf(v, "channelMix");
    const { M, b, rows } = mixOf(params, K, "channelMix");
    const n = width * height;
    const x = v.data!;
    const out = new Float32Array(rows * n);
    for (let k = 0; k < rows; k++)
      for (let i = 0; i < n; i++) {
        let acc = b[k]!;
        for (let a = 0; a < K; a++) acc += M[k * K + a]! * x[a * n + i]!;
        out[k * n + i] = acc;
      }
    return [{ shape: v.shape, dtype: "f32", axes: channelAxes(rows), data: out }];
  },
};

// --- channelStructureTensor -----------------------------------------------------------------------

const VEC4 = { kind: "vec", n: 4 } as const;

const STRUCTURE_TENSOR = /* wgsl */ `
struct Uni { n: u32, gridX: u32, K: u32, width: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
${PIXEL}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = pixel(gid);
  if (i >= U.n) { return; }
  let w = U.width;
  let h = U.n / w;
  let x = i % w;
  let y = i / w;
  // Central differences, one-sided at the raster's edge.
  let xl = select(x - 1u, 0u, x == 0u);
  let xr = min(x + 1u, w - 1u);
  let yu = select(y - 1u, 0u, y == 0u);
  let yd = min(y + 1u, h - 1u);
  let sx = 1.0 / f32(max(xr - xl, 1u));
  let sy = 1.0 / f32(max(yd - yu, 1u));
  var gxx = 0.0; var gyy = 0.0; var gxy = 0.0;
  for (var k = 0u; k < U.K; k = k + 1u) {
    let o = k * U.n;
    let ux = (src[o + y * w + xr] - src[o + y * w + xl]) * sx;
    let uy = (src[o + yd * w + x] - src[o + yu * w + x]) * sy;
    gxx = gxx + ux * ux; gyy = gyy + uy * uy; gxy = gxy + ux * uy;
  }
  let half = 0.5 * (gxx - gyy);
  let lmax = 0.5 * (gxx + gyy) + sqrt(half * half + gxy * gxy);
  let s2 = U.scale * U.scale;
  dst[i * 4u] = sqrt(max(lmax, 0.0)) * U.scale;
  dst[i * 4u + 1u] = gxx * s2;
  dst[i * 4u + 2u] = gyy * s2;
  dst[i * 4u + 3u] = gxy * s2;
}
`;
const structureTensorPipe = pipelineFor("channelStructureTensor", STRUCTURE_TENSOR, 1);

export const channelStructureTensorOp: OpType = {
  name: "channelStructureTensor",
  label: "Structure tensor (channels)",
  category: "Channels",
  describe: "Where the channel mixture changes fastest: the Di Zenzo tensor and the root of its larger eigenvalue.",
  help: {
    detail:
      "Per pixel a vec4: the strength √λ_max, then the tensor's xx, yy, xy, all scaled by `scale` (e.g. pixels per kernel radius, so the strength is per radius) and its square. Central differences, one-sided at the raster's edge. In a whitened metric the strength is how fast the distance between mixtures grows per step, in the steepest direction.",
    math: String.raw`G = \sum_k \nabla u_k \nabla u_k^\top,\qquad s = \text{scale}\,\sqrt{\lambda_{\max}(G)}`,
  },
  inputs: [{ name: "in", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [{ name: "scale", type: "number", default: 1, min: 0, max: 100, step: 0.1, describe: "length unit, in pixels" }],
  inferShapes: (inputs) => [inputs[0]!],
  inferElements: () => [VEC4],
  inferAxes: () => [undefined],
  resident: true,
  async execute(ctx, inputs, params) {
    const v = inputs[0]!;
    if (!v.buffer) throw new Error("channelStructureTensor: resident op received a non-resident input");
    const { width, height } = gridOf(v.shape, "channelStructureTensor");
    const K = channelsOf(v, "channelStructureTensor");
    const n = width * height;
    const out = await ctx.backend.lease(n * 16);
    const scale = (params.scale as number | undefined) ?? 1;
    await runPerPixel(
      ctx,
      structureTensorPipe,
      "channelStructureTensor",
      n,
      (gridX) => {
        const buf = new ArrayBuffer(32);
        new Uint32Array(buf).set([n, gridX, K, width]);
        new Float32Array(buf, 16, 1)[0] = scale;
        return buf;
      },
      [sized(v.buffer.buffer, K * n * 4), sized(out.buffer, n * 16)],
    );
    return [{ shape: v.shape, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const v = inputs[0]!;
    const { width: w, height: h } = gridOf(v.shape, "channelStructureTensor");
    const K = channelsOf(v, "channelStructureTensor");
    const n = w * h;
    const u = v.data!;
    const scale = (params.scale as number | undefined) ?? 1;
    const out = new Float32Array(n * 4);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const xl = Math.max(0, x - 1);
        const xr = Math.min(w - 1, x + 1);
        const yu = Math.max(0, y - 1);
        const yd = Math.min(h - 1, y + 1);
        const sx = 1 / Math.max(xr - xl, 1);
        const sy = 1 / Math.max(yd - yu, 1);
        let gxx = 0;
        let gyy = 0;
        let gxy = 0;
        for (let k = 0; k < K; k++) {
          const o = k * n;
          const ux = (u[o + y * w + xr]! - u[o + y * w + xl]!) * sx;
          const uy = (u[o + yd * w + x]! - u[o + yu * w + x]!) * sy;
          gxx += ux * ux;
          gyy += uy * uy;
          gxy += ux * uy;
        }
        const half = 0.5 * (gxx - gyy);
        const lmax = 0.5 * (gxx + gyy) + Math.sqrt(half * half + gxy * gxy);
        const i = (y * w + x) * 4;
        out[i] = Math.sqrt(Math.max(lmax, 0)) * scale;
        out[i + 1] = gxx * scale * scale;
        out[i + 2] = gyy * scale * scale;
        out[i + 3] = gxy * scale * scale;
      }
    return [{ shape: v.shape, dtype: "f32", element: VEC4, data: out }];
  },
};

/** The channel pack, for the composer's registrar and the wand. */
export const CHANNEL_OPS: OpType[] = [gramChannelsOp, channelMixOp, channelStructureTensorOp];
