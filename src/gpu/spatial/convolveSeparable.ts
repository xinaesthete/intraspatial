// Separable 2D convolution on a grid — the windowing primitive. A 1D kernel is
// applied horizontally then vertically (two passes), so an NxN window costs O(N)
// per texel, not O(N^2). With a box kernel it is a local sum/mean; with a Gaussian
// it is smoothing. This is the grid consumer the KDE splat feeds into, and the
// shared "window" both fronts use (boxcar -> taper; see the toolbox doc).
//
// Authored as a `"use gpu"` kernel (gather form: each output texel reads its taps
// directly — no workgroup shared memory, so it fits TGSL). Edges use clamp-to-edge
// extension. Pipeline built once and layout-bound; buffers pooled/grown; readback
// via TypeGPU `.read()` (the Dawn-stable path).
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { deviceScoped, writeView } from "../device";
import { type BindEntry, rawBindGroup } from "../graph/residentBind";
import { typegpuComputePipeline } from "../typegpuPipeline";

const WG = 64;

const Params = d.struct({
  w: d.u32,
  h: d.u32,
  r: d.u32,
  stepX: d.i32,
  stepY: d.i32,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: Params },
    src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    wts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  })
  .$name("convolveSeparable");

const convFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const total = layout.$.params.w * layout.$.params.h;
    if (i < total) {
      const col = i % layout.$.params.w;
      const row = i / layout.$.params.w;
      const taps = 2 * layout.$.params.r + 1;
      let acc = d.f32(0);
      for (let t = d.u32(0); t < taps; t++) {
        const off = d.i32(t) - d.i32(layout.$.params.r);
        const c = std.clamp(d.i32(col) + off * layout.$.params.stepX, 0, d.i32(layout.$.params.w) - 1);
        const rr = std.clamp(d.i32(row) + off * layout.$.params.stepY, 0, d.i32(layout.$.params.h) - 1);
        acc = acc + layout.$.src[d.u32(rr) * layout.$.params.w + d.u32(c)]! * layout.$.wts[t]!;
      }
      layout.$.dst[i] = acc;
    }
  })
  .$name("convSep");

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
}
/** Per device (default: the library's own) — `deviceScoped` in `../device` says why. */
const getPipe = deviceScoped(async (device: GPUDevice): Promise<Pipe> => {
  const root = tgpu.initFromDevice({ device });
  const pipeline = typegpuComputePipeline(
    device,
    root,
    tgpu.resolveWithContext([convFn], { names: "strict" }),
    "convSep",
    "convolveSeparable",
  );
  return { device, root, pipeline };
});

type Root = Pipe["root"];
function buf(root: Root, n: number, label: string) {
  return root
    .createBuffer(d.arrayOf(d.f32, Math.max(1, n)))
    .$usage("storage")
    .$name(label);
}
function makeParams(root: Root) {
  return root.createBuffer(Params).$usage("uniform").$name("convolveSeparable:params");
}
function makePool(root: Root, cap: number, wcap: number, params: ReturnType<typeof makeParams>) {
  return {
    cap,
    wcap,
    a: buf(root, cap, "convolveSeparable:a"),
    b: buf(root, cap, "convolveSeparable:b"),
    w: buf(root, wcap, "convolveSeparable:w"),
    params,
  };
}
/** Per root, so per device: a pool's buffers belong to the device its root wraps. */
const pools = new WeakMap<Root, ReturnType<typeof makePool>>();
function ensurePool(root: Root, cells: number, weights: number) {
  let pool = pools.get(root);
  if (pool && pool.cap >= cells && pool.wcap >= weights) return pool;
  const cap = Math.max(cells, pool?.cap ?? 0, 1);
  const wcap = Math.max(weights, pool?.wcap ?? 0, 1);
  pool = makePool(root, cap, wcap, pool?.params ?? makeParams(root));
  pools.set(root, pool);
  return pool;
}

/** A box kernel of the given radius (all ones → a local *sum* over a
 *  (2r+1) window per axis; divide by (2r+1)^2 for a mean). */
export function boxKernel(radius: number): Float32Array {
  return new Float32Array(2 * radius + 1).fill(1);
}

/** A normalised 1D Gaussian kernel (sums to 1). */
export function gaussianKernel(sigma: number, radius?: number): Float32Array {
  const r = radius ?? Math.max(1, Math.ceil(3 * sigma));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp((-0.5 * (i * i)) / (sigma * sigma));
    k[i + r] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i]! /= sum;
  return k;
}

/** Separable convolution of a row-major width*height grid with a 1D `kernel`
 *  (odd length, centred), applied on both axes. Returns a new grid. `device` defaults to the
 *  library's own. */
export async function convolveSeparableGpu(
  grid: ArrayLike<number>,
  width: number,
  height: number,
  kernel: ArrayLike<number>,
  device?: GPUDevice,
): Promise<Float32Array> {
  const cells = width * height;
  if (grid.length !== cells) throw new Error("convolveSeparable: grid length != width*height");
  if (kernel.length % 2 === 0) throw new Error("convolveSeparable: kernel length must be odd");
  const r = (kernel.length - 1) / 2;

  const { device: dev, root, pipeline } = await getPipe(device);
  const p = ensurePool(root, cells, kernel.length);

  writeView(dev.queue, root.unwrap(p.a), Float32Array.from(grid));
  writeView(dev.queue, root.unwrap(p.w), Float32Array.from(kernel));

  const groups = Math.ceil(cells / WG);
  const run = (src: typeof p.a, dst: typeof p.a, stepX: number, stepY: number) => {
    p.params.write({ w: width, h: height, r, stepX, stepY });
    const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, src, wts: p.w, dst }));
    const enc = dev.createCommandEncoder({ label: "convolveSeparable" });
    const pass = enc.beginComputePass({ label: "convolveSeparable" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    dev.queue.submit([enc.finish()]);
  };

  run(p.a, p.b, 1, 0); // horizontal: a -> b
  run(p.b, p.a, 0, 1); // vertical:   b -> a

  const got = (await p.a.read()) as ArrayLike<number>;
  return Float32Array.from({ length: cells }, (_, i) => got[i]!);
}

/** Tier-2 form (ADR-0017): convolve a GPU-resident grid into a GPU-resident destination, with
 *  no host transfer in either direction. Same two-pass kernel as `convolveSeparableGpu` — only
 *  the ends differ, which is the whole of what residency changes.
 *
 *  `src` and `dst` are pooled buffers owned by the caller (the executor's lease). They must be
 *  distinct: the two passes read and write across the whole grid, so writing in place would
 *  violate invariant 1's single-writer rule. The module pool supplies the intermediate only.
 *  `device` is the one they live on; default: the library's own. */
export async function convolveSeparableResident(
  src: GPUBuffer,
  dst: GPUBuffer,
  width: number,
  height: number,
  kernel: ArrayLike<number>,
  device?: GPUDevice,
): Promise<void> {
  if (src === dst) throw new Error("convolveSeparableResident: src and dst must be distinct buffers");
  if (kernel.length % 2 === 0) throw new Error("convolveSeparable: kernel length must be odd");
  const cells = width * height;
  const r = (kernel.length - 1) / 2;

  const { device: dev, root, pipeline } = await getPipe(device);
  const p = ensurePool(root, cells, kernel.length);
  writeView(dev.queue, root.unwrap(p.w), Float32Array.from(kernel));

  const groups = Math.ceil(cells / WG);
  // Raw bind groups over the pooled buffers — see residentBind.ts for why these are not wrapped
  // as TypeGPU buffers. Order matches the `layout` declaration: params, src, wts, dst.
  const run = (from: BindEntry, to: BindEntry, stepX: number, stepY: number) => {
    p.params.write({ w: width, h: height, r, stepX, stepY });
    const bind = rawBindGroup(dev, root, layout, [p.params, from, p.w, to], "convolveSeparable");
    const enc = dev.createCommandEncoder({ label: "convolveSeparable" });
    const pass = enc.beginComputePass({ label: "convolveSeparable" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    dev.queue.submit([enc.finish()]);
  };

  // Per-dependent-stage submits (invariant 2): the vertical pass reads what the horizontal
  // wrote, so the boundary between them is explicit rather than assumed.
  run(src, p.b, 1, 0); // horizontal: src -> scratch
  run(p.b, dst, 0, 1); // vertical:   scratch -> dst
}
