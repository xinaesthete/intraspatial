// k-th nearest-neighbour distance — for each point, the distance to its k-th
// closest other point. This is a local density estimate: ρ_i = d(i, x_k) is small
// where points are dense, large where sparse, and (for points on an m-manifold)
// scales like q(x)^(-1/m). It is the local bandwidth used by the CkNN / self-tuning
// constructions (Berry & Sauer 2016; the "self-tuning" kernel of Zelnik-Manor &
// Perona) — see `cknn.ts` and docs/fuzzy-tda-and-windowing.md.
//
// Brute force O(N·k): each thread keeps the k smallest distances seen in a local
// fixed-size array. Local mutable arrays are not expressible in TGSL in this
// TypeGPU version, so this is a **WGSL template** kernel (resolveWithContext),
// per ADR-0003 — the first non-`"use gpu"` spatial primitive. The array is sized
// to a compile-time MAX_K; the actual k (≤ MAX_K) is a uniform.
//
// With `cell` (see `IndexedQueryOptions`) the uniform-grid index is built in the same command
// buffer and the candidate loop is the 3×3 stencil — O(N·k) — calling `gridIndexQuery.ts`'s
// TGSL helpers as template externals. Brute force stays the default and the golden.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice, sized } from "../device";
import { type GridIndexCtx, getGridIndexCtx } from "./gridIndex";
import { cellCoord, cellRange, encodeQueryIndex, type IndexedQueryOptions, LATTICE_BYTES, Lattice, StartArray } from "./gridIndexQuery";

const WG = 64;
const MAX_K = 32;

const Params = d.struct({ n: d.u32, k: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per point
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // ρ_i per point
});

const layoutIdx = tgpu.bindGroupLayout({
  params: { uniform: Params },
  lat: { uniform: Lattice },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  start: { storage: StartArray, access: "readonly" },
  items: { storage: StartArray, access: "readonly" },
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

// The per-candidate body, shared by both entry points: keep the k smallest.
const CONSIDER = /* wgsl */ `
    let dx = pts[2u * j] - xi;
    let dy = pts[2u * j + 1u] - yi;
    let dist = sqrt(dx * dx + dy * dy);
    // replace the current largest of the k smallest if this is smaller
    var maxIdx: u32 = 0u;
    var maxVal: f32 = best[0];
    for (var t: u32 = 1u; t < k; t = t + 1u) {
      if (best[t] > maxVal) { maxVal = best[t]; maxIdx = t; }
    }
    if (dist < maxVal) { best[maxIdx] = dist; }
`;

// Slots never filled keep the FAR sentinel (WGSL refuses a constant +Inf), which the host
// reports as +Inf: a point with fewer than k candidates — only reachable on the indexed
// path, since brute force requires k < n.
const FAR = 3.4e38;
const PROLOGUE = /* wgsl */ `
  let i = gid.x;
  let n = params.n;
  if (i >= n) { return; }
  let k = min(params.k, ${MAX_K}u);
  let xi = pts[2u * i];
  let yi = pts[2u * i + 1u];
  var best: array<f32, ${MAX_K}u>;
  for (var t: u32 = 0u; t < k; t = t + 1u) { best[t] = ${FAR}; }
`;

// the k-th nearest distance is the largest among the k smallest
const EPILOGUE = /* wgsl */ `
  var ans: f32 = best[0];
  for (var t: u32 = 1u; t < k; t = t + 1u) {
    if (best[t] > ans) { ans = best[t]; }
  }
  out[i] = ans;
`;

const TEMPLATE = /* wgsl */ `
@compute @workgroup_size(${WG})
fn kthnn(@builtin(global_invocation_id) gid: vec3u) {
  ${PROLOGUE}
  for (var j: u32 = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
    ${CONSIDER}
  }
  ${EPILOGUE}
}
`;

const TEMPLATE_IDX = /* wgsl */ `
@compute @workgroup_size(${WG})
fn kthnnIdx(@builtin(global_invocation_id) gid: vec3u) {
  ${PROLOGUE}
  let c = cellCoord(vec2f(xi, yi), lat);
  for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
    for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
      let r = cellRange(vec2i(c.x + dx, c.y + dy), lat, &start);
      for (var s = r.x; s < r.y; s = s + 1u) {
        let j = items[s];
        if (j == i) { continue; }
        ${CONSIDER}
      }
    }
  }
  ${EPILOGUE}
}
`;

interface Pipe {
  device: GPUDevice;
  root: ReturnType<typeof tgpu.initFromDevice>;
  pipeline: GPUComputePipeline;
  pipelineIdx: GPUComputePipeline;
  indexCtx: GridIndexCtx;
}
let pipeCache: Promise<Pipe> | undefined;
function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const build = (template: string, externals: Record<string, object>, entryPoint: string) => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({ template, externals, names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      return device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint } });
    };
    const indexCtx = await getGridIndexCtx();
    return {
      device,
      root,
      pipeline: build(TEMPLATE, { ...layout.bound }, "kthnn"),
      pipelineIdx: build(TEMPLATE_IDX, { ...layoutIdx.bound, cellCoord, cellRange }, "kthnnIdx"),
      indexCtx,
    };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makePool(root: Root, n: number) {
  return {
    n,
    pts: root.createBuffer(d.arrayOf(d.f32, 2 * n)).$usage("storage"),
    out: root.createBuffer(d.arrayOf(d.f32, n)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, n: number) {
  if (pool && pool.n >= n) return pool;
  pool = makePool(root, Math.max(n, pool?.n ?? 0, 1));
  return pool;
}

export const KTH_NEIGHBOR_MAX_K = MAX_K;

/** For each point, the distance to its k-th nearest other point (a local density
 *  estimate ρ_i). `k` must be in 1..32 and < N.
 *
 *  With `opts.cell` the query is indexed (see `IndexedQueryOptions`): exact wherever the true
 *  k-th neighbour is within `cell`; a point with fewer than k others in its 3×3 stencil gets
 *  `+Infinity`. */
export async function kthNeighborDistanceGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  k: number,
  opts: IndexedQueryOptions = {},
): Promise<Float32Array> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("kthNeighborDistance: xs and ys length mismatch");
  if (k < 1 || k > MAX_K) throw new Error(`kthNeighborDistance: k must be in 1..${MAX_K}`);
  if (k >= n) throw new Error("kthNeighborDistance: need k < N");

  const { device, root, pipeline, pipelineIdx, indexCtx } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  device.queue.writeBuffer(root.unwrap(p.pts), 0, flat as BufferSource);
  p.params.write({ n, k });

  const enc = device.createCommandEncoder();
  let bind: GPUBindGroup;
  if (opts.cell === undefined) {
    bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, out: p.out }));
  } else {
    const q = encodeQueryIndex(indexCtx, root.unwrap(p.pts), n, xs, ys, opts.cell, opts.bounds, enc, "kthNeighborDistance:index");
    bind = device.createBindGroup({
      layout: root.unwrap(layoutIdx),
      entries: [
        { binding: 0, resource: { buffer: root.unwrap(p.params) } },
        { binding: 1, resource: sized(q.lat, LATTICE_BYTES) },
        { binding: 2, resource: sized(root.unwrap(p.pts), 2 * n * 4) },
        { binding: 3, resource: sized(q.index.start, (q.index.M + 1) * 4) },
        { binding: 4, resource: sized(q.index.items, n * 4) },
        { binding: 5, resource: sized(root.unwrap(p.out), n * 4) },
      ],
    });
  }
  const pass = enc.beginComputePass();
  pass.setPipeline(opts.cell === undefined ? pipeline : pipelineIdx);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.out.read()) as ArrayLike<number>;
  // FAR rounds down in f32, so the sentinel test is a threshold, not equality.
  return Float32Array.from({ length: n }, (_, i) => (got[i]! >= 3e38 ? Number.POSITIVE_INFINITY : got[i]!));
}
