// Separable grey-scale morphology on a grid — erosion (local min) and dilation (local max)
// over a (2r+1)² square structuring element, plus opening (erode→dilate) and closing
// (dilate→erode) composed from them.
//
// A square element is separable EXACTLY: min over a square is the min of per-row minima, so
// this is the same two-pass horizontal-then-vertical gather as `convolveSeparable` with
// `min`/`max` in place of the weighted sum. Unlike convolution there is no arithmetic, so the
// GPU path is bit-exact against a direct 2-D window (see the test). Binary morphology is the
// 0/1 special case; the oracle is psychogeo `codec_eval/foliage.py::binary_reduce`/`opening`
// (clamp-to-edge = np.pad mode="edge"), where an r=1 opening is what stops a building's
// one-pixel perimeter reading as canopy and an r≈25 opening of LZ is a bare-earth estimate.
//
// Authored as a `"use gpu"` kernel (gather form, no workgroup memory — fits TGSL). Pipeline
// built once and layout-bound; buffers pooled/grown; readback via TypeGPU `.read()`.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice } from "../device";
import { type BindEntry, rawBindGroup } from "../graph/residentBind";

const WG = 64;

export type MorphMode = "erode" | "dilate";
export type MorphOp = MorphMode | "open" | "close";

const Params = d.struct({
  w: d.u32,
  h: d.u32,
  r: d.u32,
  stepX: d.i32,
  stepY: d.i32,
  /** 0 = erode (min), 1 = dilate (max). */
  mode: d.u32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const morphFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const total = layout.$.params.w * layout.$.params.h;
    if (i < total) {
      const col = i % layout.$.params.w;
      const row = i / layout.$.params.w;
      const taps = 2 * layout.$.params.r + 1;
      // Seed from the centre tap so the accumulator is always a real sample (no ±inf literals).
      let acc = layout.$.src[i]!;
      for (let t = d.u32(0); t < taps; t++) {
        const off = d.i32(t) - d.i32(layout.$.params.r);
        const c = std.clamp(d.i32(col) + off * layout.$.params.stepX, 0, d.i32(layout.$.params.w) - 1);
        const rr = std.clamp(d.i32(row) + off * layout.$.params.stepY, 0, d.i32(layout.$.params.h) - 1);
        const v = layout.$.src[d.u32(rr) * layout.$.params.w + d.u32(c)]!;
        if (layout.$.params.mode === 0) {
          acc = std.min(acc, v);
        } else {
          acc = std.max(acc, v);
        }
      }
      layout.$.dst[i] = acc;
    }
  })
  .$name("morphSep");

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;
function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([morphFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({
      bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)),
    });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "morphSep" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function buf(root: Root, n: number) {
  return root.createBuffer(d.arrayOf(d.f32, Math.max(1, n))).$usage("storage");
}
function makeParams(root: Root) {
  return root.createBuffer(Params).$usage("uniform");
}
function makePool(root: Root, cap: number, params: ReturnType<typeof makeParams>) {
  // Two grid-sized scratch buffers: one holds the horizontal pass, the other the intermediate
  // of a composed opening/closing (which needs a full erode result before the dilate starts).
  return { cap, a: buf(root, cap), b: buf(root, cap), params };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, cells: number) {
  if (pool && pool.cap >= cells) return pool;
  const cap = Math.max(cells, pool?.cap ?? 0, 1);
  pool = makePool(root, cap, pool?.params ?? makeParams(root));
  return pool;
}

function modeFlag(mode: MorphMode): number {
  return mode === "erode" ? 0 : 1;
}

/** The two passes of one erosion or dilation, from `src` into `dst`, using `scratch` for the
 *  horizontal intermediate. All three must be distinct buffers. Per-dependent-stage submits
 *  (ADR-0017 invariant 2): the vertical pass reads what the horizontal wrote. */
function runPasses(
  pipe: Pipe,
  params: ReturnType<typeof makeParams>,
  src: BindEntry,
  scratch: BindEntry,
  dst: BindEntry,
  width: number,
  height: number,
  r: number,
  mode: MorphMode,
): void {
  const { device, root, pipeline } = pipe;
  const groups = Math.ceil((width * height) / WG);
  const run = (from: BindEntry, to: BindEntry, stepX: number, stepY: number) => {
    params.write({ w: width, h: height, r, stepX, stepY, mode: modeFlag(mode) });
    const bind = rawBindGroup(device, root, layout, [params, from, to]);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  run(src, scratch, 1, 0); // horizontal: src -> scratch
  run(scratch, dst, 0, 1); // vertical:   scratch -> dst
}

/** Tier-2 form (ADR-0017): morphology of a GPU-resident grid into a GPU-resident destination,
 *  no host transfer. `src` and `dst` are caller-owned (the executor's lease) and must be distinct
 *  (the two passes read and write across the whole grid). `open`/`close` run two full
 *  erode/dilate rounds through the module's scratch pool. */
export async function morphologyResident(
  src: GPUBuffer,
  dst: GPUBuffer,
  width: number,
  height: number,
  radius: number,
  op: MorphOp,
): Promise<void> {
  if (src === dst) throw new Error("morphologyResident: src and dst must be distinct buffers");
  checkRadius(radius);
  const pipe = await getPipe();
  const p = ensurePool(pipe.root, width * height);
  if (op === "erode" || op === "dilate") {
    runPasses(pipe, p.params, src, p.a, dst, width, height, radius, op);
  } else {
    const [first, second]: [MorphMode, MorphMode] = op === "open" ? ["erode", "dilate"] : ["dilate", "erode"];
    runPasses(pipe, p.params, src, p.a, p.b, width, height, radius, first);
    runPasses(pipe, p.params, p.b, p.a, dst, width, height, radius, second);
  }
}

function checkRadius(radius: number): void {
  if (!Number.isInteger(radius) || radius < 1) throw new Error("morphology: radius must be an integer >= 1");
}

/** Host-array form: erode or dilate a row-major width*height grid over a (2r+1)² square
 *  with clamp-to-edge boundaries. Returns a new grid. */
export async function morphologyGpu(
  grid: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
  op: MorphOp,
): Promise<Float32Array> {
  const cells = width * height;
  if (grid.length !== cells) throw new Error("morphology: grid length != width*height");
  checkRadius(radius);
  const pipe = await getPipe();
  const { device, root } = pipe;
  // Host form needs its own in/out pair distinct from the scratch pool the resident path uses.
  const io = ensureIo(root, cells);
  device.queue.writeBuffer(root.unwrap(io.src), 0, Float32Array.from(grid) as BufferSource);
  await morphologyResident(root.unwrap(io.src), root.unwrap(io.dst), width, height, radius, op);
  const got = (await io.dst.read()) as ArrayLike<number>;
  return Float32Array.from({ length: cells }, (_, i) => got[i]!);
}

let io: { cap: number; src: ReturnType<typeof buf>; dst: ReturnType<typeof buf> } | undefined;
function ensureIo(root: Root, cells: number) {
  if (io && io.cap >= cells) return io;
  const cap = Math.max(cells, io?.cap ?? 0, 1);
  io = { cap, src: buf(root, cap), dst: buf(root, cap) };
  return io;
}

/** Opening = erode then dilate: removes anything thinner than the (2r+1) element, keeps blobs. */
export function openingGpu(grid: ArrayLike<number>, width: number, height: number, radius: number): Promise<Float32Array> {
  return morphologyGpu(grid, width, height, radius, "open");
}

/** Closing = dilate then erode: fills gaps narrower than the (2r+1) element. */
export function closingGpu(grid: ArrayLike<number>, width: number, height: number, radius: number): Promise<Float32Array> {
  return morphologyGpu(grid, width, height, radius, "close");
}

/** CPU reference, separable (matches the GPU passes exactly) — the `cpuGolden` for the graph op. */
export function morphologyCpu(grid: ArrayLike<number>, width: number, height: number, radius: number, op: MorphOp): Float32Array {
  checkRadius(radius);
  const pass1 = (src: ArrayLike<number>, mode: MorphMode): Float32Array => {
    const pick = mode === "erode" ? Math.min : Math.max;
    const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
    const tmp = new Float32Array(width * height);
    const out = new Float32Array(width * height);
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        let acc = src[row * width + col]!;
        for (let t = -radius; t <= radius; t++) acc = pick(acc, src[row * width + clamp(col + t, width - 1)]!);
        tmp[row * width + col] = acc;
      }
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        let acc = tmp[row * width + col]!;
        for (let t = -radius; t <= radius; t++) acc = pick(acc, tmp[clamp(row + t, height - 1) * width + col]!);
        out[row * width + col] = acc;
      }
    return out;
  };
  if (op === "erode" || op === "dilate") return pass1(grid, op);
  return op === "open" ? pass1(pass1(grid, "erode"), "dilate") : pass1(pass1(grid, "dilate"), "erode");
}
