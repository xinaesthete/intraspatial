// Decimating block reduction on a grid — every `factor`×`factor` block of input cells collapses
// to ONE output cell (mean, min or max). The first engine op that changes grid size.
//
// Output is ceil(w/factor) × ceil(h/factor). A block that hangs off the right/bottom edge (w or h
// not a multiple of `factor`) reduces over the cells that exist: a partial block's mean is the
// mean of its present cells, not of a zero-padded full block. Reference semantics: psychogeo
// `block_mean` (block mean over full blocks); the edge rule is ours and documented here.
// `factor` 1 is the identity (a copy) and is accepted — the graph op clamps to 2..64 because a
// factor-1 node is a no-op the composer should not offer.
//
// Authored as a `"use gpu"` kernel, one thread per OUTPUT cell looping over its block: no
// atomics, no workgroup memory, deterministic summation order (so the CPU golden matches the
// mean to f32 rounding and min/max bit-exactly). Pipeline built once; buffers pooled/grown.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { dispatchGrid, getDevice, writeView } from "../device";
import { rawBindGroup } from "../graph/residentBind";

const WG = 64;

export type DecimateMode = "mean" | "min" | "max";

const Params = d.struct({
  w: d.u32,
  h: d.u32,
  ow: d.u32,
  oh: d.u32,
  factor: d.u32,
  /** 0 = mean, 1 = min, 2 = max. */
  mode: d.u32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const decimateFn = tgpu
  .computeFn({
    in: { gid: d.builtin.globalInvocationId, nwg: d.builtin.numWorkgroups },
    workgroupSize: [WG],
  })((input) => {
    "use gpu";
    // Folded 2-D dispatch (dispatchGrid): linear thread id across x then y.
    const i = input.gid.x + input.gid.y * input.nwg.x * WG;
    const total = layout.$.params.ow * layout.$.params.oh;
    if (i < total) {
      // u32 `/` is float division in TGSL — spell integer division out.
      const orow = d.u32(std.floor(d.f32(i) / d.f32(layout.$.params.ow)));
      const ocol = i - orow * layout.$.params.ow;
      const x0 = ocol * layout.$.params.factor;
      const y0 = orow * layout.$.params.factor;
      const x1 = std.min(x0 + layout.$.params.factor, layout.$.params.w);
      const y1 = std.min(y0 + layout.$.params.factor, layout.$.params.h);
      // Seed from the block's first cell (always present) so min/max need no ±inf literals.
      let acc = layout.$.src[y0 * layout.$.params.w + x0]!;
      let count = d.f32(1);
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          if (x !== x0 || y !== y0) {
            const v = layout.$.src[y * layout.$.params.w + x]!;
            if (layout.$.params.mode === 0) {
              acc = acc + v;
              count = count + d.f32(1);
            } else if (layout.$.params.mode === 1) {
              acc = std.min(acc, v);
            } else {
              acc = std.max(acc, v);
            }
          }
        }
      }
      if (layout.$.params.mode === 0) {
        acc = acc / count;
      }
      layout.$.dst[i] = acc;
    }
  })
  .$name("decimateBlock");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([decimateFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({
      bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)),
    });
    const pipeline = device.createComputePipeline({
      layout: pipeLayout,
      compute: { module, entryPoint: "decimateBlock" },
    });
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
let params: ReturnType<typeof makeParams> | undefined;
function ensureParams(root: Root) {
  params ??= makeParams(root);
  return params;
}

function modeFlag(mode: DecimateMode): number {
  return mode === "mean" ? 0 : mode === "min" ? 1 : 2;
}

function checkFactor(factor: number): void {
  if (!Number.isInteger(factor) || factor < 1) throw new Error("decimate: factor must be an integer >= 1");
}

/** Output dimensions of a decimation: ceil(w/f) × ceil(h/f). */
export function decimatedSize(width: number, height: number, factor: number): { width: number; height: number } {
  checkFactor(factor);
  return { width: Math.ceil(width / factor), height: Math.ceil(height / factor) };
}

/** Tier-2 form (ADR-0017): decimate a GPU-resident width×height grid into a GPU-resident
 *  destination of at least `ceil(w/f)·ceil(h/f)` f32 cells, no host transfer. `src` and `dst`
 *  are caller-owned and must be distinct (every output cell reads a whole input block). */
export async function decimateResident(
  src: GPUBuffer,
  dst: GPUBuffer,
  width: number,
  height: number,
  factor: number,
  mode: DecimateMode,
): Promise<void> {
  if (src === dst) throw new Error("decimateResident: src and dst must be distinct buffers");
  const { width: ow, height: oh } = decimatedSize(width, height, factor);
  const pipe = await getPipe();
  const { device, root, pipeline } = pipe;
  const p = ensureParams(root);
  p.write({ w: width, h: height, ow, oh, factor, mode: modeFlag(mode) });
  const bind = rawBindGroup(device, root, layout, [p, src, dst]);
  const { x, y } = dispatchGrid(Math.ceil((ow * oh) / WG));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(x, y);
  pass.end();
  device.queue.submit([enc.finish()]);
}

/** Host-array form: block-reduce a row-major width×height grid by `factor`. Returns the
 *  ceil(w/f)×ceil(h/f) grid (row-major). */
export async function decimateGpu(
  grid: ArrayLike<number>,
  width: number,
  height: number,
  factor: number,
  mode: DecimateMode,
): Promise<Float32Array> {
  const cells = width * height;
  if (grid.length !== cells) throw new Error("decimate: grid length != width*height");
  const { width: ow, height: oh } = decimatedSize(width, height, factor);
  const pipe = await getPipe();
  const { device, root } = pipe;
  const io = ensureIo(root, cells, ow * oh);
  writeView(device.queue, root.unwrap(io.src), Float32Array.from(grid));
  await decimateResident(root.unwrap(io.src), root.unwrap(io.dst), width, height, factor, mode);
  const got = (await io.dst.read()) as ArrayLike<number>;
  return Float32Array.from({ length: ow * oh }, (_, i) => got[i]!);
}

let io: { capIn: number; capOut: number; src: ReturnType<typeof buf>; dst: ReturnType<typeof buf> } | undefined;
function ensureIo(root: Root, cellsIn: number, cellsOut: number) {
  if (io && io.capIn >= cellsIn && io.capOut >= cellsOut) return io;
  const capIn = Math.max(cellsIn, io?.capIn ?? 0, 1);
  const capOut = Math.max(cellsOut, io?.capOut ?? 0, 1);
  io = { capIn, capOut, src: buf(root, capIn), dst: buf(root, capOut) };
  return io;
}

/** CPU reference — same block loop and summation order as the kernel (the `cpuGolden` for the
 *  graph op). Accumulates in f32 so the mean matches the GPU to the last rounding step. */
export function decimateCpu(grid: ArrayLike<number>, width: number, height: number, factor: number, mode: DecimateMode): Float32Array {
  if (grid.length !== width * height) throw new Error("decimate: grid length != width*height");
  const { width: ow, height: oh } = decimatedSize(width, height, factor);
  const out = new Float32Array(ow * oh);
  const acc32 = new Float32Array(1);
  for (let orow = 0; orow < oh; orow++)
    for (let ocol = 0; ocol < ow; ocol++) {
      const x0 = ocol * factor;
      const y0 = orow * factor;
      const x1 = Math.min(x0 + factor, width);
      const y1 = Math.min(y0 + factor, height);
      acc32[0] = grid[y0 * width + x0]!;
      let count = 1;
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x++) {
          if (x === x0 && y === y0) continue;
          const v = grid[y * width + x]!;
          if (mode === "mean") {
            acc32[0] = acc32[0]! + v;
            count++;
          } else if (mode === "min") acc32[0] = Math.min(acc32[0]!, v);
          else acc32[0] = Math.max(acc32[0]!, v);
        }
      out[orow * ow + ocol] = mode === "mean" ? acc32[0]! / count : acc32[0]!;
    }
  return out;
}
