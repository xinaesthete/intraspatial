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
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../device";

const WG = 64;
const MAX_K = 32;

const Params = d.struct({ n: d.u32, k: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per point
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // ρ_i per point
});

const TEMPLATE = /* wgsl */ `
@compute @workgroup_size(${WG})
fn kthnn(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = params.n;
  if (i >= n) { return; }
  let k = min(params.k, ${MAX_K}u);
  let xi = pts[2u * i];
  let yi = pts[2u * i + 1u];

  var best: array<f32, ${MAX_K}u>;
  for (var t: u32 = 0u; t < k; t = t + 1u) { best[t] = 3.4e38; }

  for (var j: u32 = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
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
  }

  // the k-th nearest distance is the largest among the k smallest
  var ans: f32 = best[0];
  for (var t: u32 = 1u; t < k; t = t + 1u) {
    if (best[t] > ans) { ans = best[t]; }
  }
  out[i] = ans;
}
`;

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext({
      template: TEMPLATE,
      externals: { ...layout.bound },
      names: "strict",
    });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "kthnn" } });
    return { device, root, pipeline };
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
 *  estimate ρ_i). `k` must be in 1..32 and < N. */
export async function kthNeighborDistanceGpu(xs: ArrayLike<number>, ys: ArrayLike<number>, k: number): Promise<Float32Array> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("kthNeighborDistance: xs and ys length mismatch");
  if (k < 1 || k > MAX_K) throw new Error(`kthNeighborDistance: k must be in 1..${MAX_K}`);
  if (k >= n) throw new Error("kthNeighborDistance: need k < N");

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  device.queue.writeBuffer(root.unwrap(p.pts), 0, flat as BufferSource);
  p.params.write({ n, k });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, out: p.out }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.out.read()) as ArrayLike<number>;
  return Float32Array.from({ length: n }, (_, i) => got[i]!);
}
