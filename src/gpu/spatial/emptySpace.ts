// Empty-space function (a.k.a. the spherical contact / F-function) — for each of
// M random sample locations, the distance to the nearest data point. Where the
// nearest-neighbour distribution looks at gaps *between points*, this looks at the
// size of *empty space*: large values mean big voids. It is the natural complement
// to ANNI for describing how a pattern fills (or fails to fill) a region.
//
// Same brute-force min-distance kernel as nnDistance, but over a separate query set
// (and without excluding self). Authored in `"use gpu"`; layout-bound pipeline,
// pooled buffers, TypeGPU `.read()` readback.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice, writeView } from "../device";

const WG = 64;
const FAR = 3.4e38;

const Params = d.struct({ n: d.u32, m: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  data: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per data point
  query: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per sample
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // min dist per sample
});

const esFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const q = input.gid.x;
    if (q < layout.$.params.m) {
      const qx = layout.$.query[2 * q]!;
      const qy = layout.$.query[2 * q + 1]!;
      let best = d.f32(FAR);
      for (let j = d.u32(0); j < layout.$.params.n; j++) {
        const dx = layout.$.data[2 * j]! - qx;
        const dy = layout.$.data[2 * j + 1]! - qy;
        best = std.min(best, std.sqrt(dx * dx + dy * dy));
      }
      layout.$.out[q] = best;
    }
  })
  .$name("emptySpace");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([esFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "emptySpace" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function sb(root: Root, n: number) {
  return root.createBuffer(d.arrayOf(d.f32, Math.max(1, n))).$usage("storage");
}
function makeParams(root: Root) {
  return root.createBuffer(Params).$usage("uniform");
}
function makePool(root: Root, nCap: number, mCap: number, params: ReturnType<typeof makeParams>) {
  return { nCap, mCap, data: sb(root, 2 * nCap), query: sb(root, 2 * mCap), out: sb(root, mCap), params };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, n: number, m: number) {
  if (pool && pool.nCap >= n && pool.mCap >= m) return pool;
  pool = makePool(root, Math.max(n, pool?.nCap ?? 0, 1), Math.max(m, pool?.mCap ?? 0, 1), pool?.params ?? makeParams(root));
  return pool;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface EmptySpaceOptions {
  /** Number of random sample locations. Default 1024. */
  numSamples?: number;
  /** World bounds to sample within. Default: the data's bounding box. */
  bbox?: [number, number, number, number];
  /** RNG seed (sampling is reproducible). Default 1. */
  seed?: number;
}

export interface EmptySpaceResult {
  /** Nearest-data-point distance for each random sample. */
  distances: Float32Array;
  /** Mean empty-space distance — a compact "typical void radius". */
  mean: number;
  bbox: [number, number, number, number];
}

/** Empty-space (F) function: distances from random locations to the nearest point. */
export async function emptySpaceGpu(xs: ArrayLike<number>, ys: ArrayLike<number>, opts: EmptySpaceOptions = {}): Promise<EmptySpaceResult> {
  const n = xs.length;
  if (n < 1) throw new Error("emptySpace: need at least 1 data point");
  const m = opts.numSamples ?? 1024;

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
    bbox = [minX, minY, maxX, maxY];
  }
  const [minX, minY, maxX, maxY] = bbox;

  const rnd = mulberry32(opts.seed ?? 1);
  const query = new Float32Array(2 * m);
  for (let i = 0; i < m; i++) {
    query[2 * i] = minX + rnd() * (maxX - minX);
    query[2 * i + 1] = minY + rnd() * (maxY - minY);
  }
  const data = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    data[2 * i] = xs[i]!;
    data[2 * i + 1] = ys[i]!;
  }

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n, m);
  writeView(device.queue, root.unwrap(p.data), data);
  writeView(device.queue, root.unwrap(p.query), query);
  p.params.write({ n, m });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, data: p.data, query: p.query, out: p.out }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(m / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.out.read()) as ArrayLike<number>;
  const distances = Float32Array.from({ length: m }, (_, i) => got[i]!);
  let sum = 0;
  for (let i = 0; i < m; i++) sum += distances[i]!;
  return { distances, mean: sum / m, bbox };
}
