// Adaptive (per-point bandwidth) fuzzy adjacency — the UMAP-style fuzzy simplicial
// set 1-skeleton. Where `fuzzyAdjacency` uses one global σ, this gives each point
// its own local bandwidth σ_i (typically its k-th nearest-neighbour distance ρ_i,
// from `kthNeighborDistance`), so the membership adapts to local density:
//
//   a_ij = exp(-d_ij² / 2σ_i²)        (i's view of j)
//   b_ij = exp(-d_ij² / 2σ_j²)        (j's view of i)
//   μ_ij = a_ij + b_ij − a_ij·b_ij    (probabilistic t-conorm — fuzzy union)
//
// The t-conorm is UMAP's symmetrisation: an edge is strong if *either* endpoint
// considers the other a close neighbour. μ is symmetric (μ_ij = μ_ji) and 0 on the
// diagonal. Dense regions (small ρ) get tight kernels; sparse regions (large ρ)
// reach further — the self-tuning move of Zelnik-Manor & Perona, the density side
// of CkNN (see `cknn.ts`). Feed `1 − μ` to the persistence reducer for a
// membership-sweep filtration.
//
// `"use gpu"` kernel, layout-bound pipeline, pooled buffers, `.read()` readback —
// the project's Dawn-stable pattern, matching `fuzzyAdjacency.ts`.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice } from "../device";
import { kthNeighborDistanceGpu } from "./kthNeighborDistance";

const WG = 64;

const Params = d.struct({
  n: d.u32,
  scale: d.f32, // σ_i = scale * ρ_i
  minSigma: d.f32, // floor so a zero ρ_i doesn't divide by zero
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per point
  rho: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // local bandwidth ρ_i
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // n*n membership
});

const adaptiveFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const n = layout.$.params.n;
    if (i < n) {
      const xi = layout.$.pts[2 * i]!;
      const yi = layout.$.pts[2 * i + 1]!;
      const si = std.max(layout.$.params.scale * layout.$.rho[i]!, layout.$.params.minSigma);
      const invI = 1 / (2 * si * si);
      for (let j = d.u32(0); j < n; j++) {
        let mu = d.f32(0);
        if (j !== i) {
          const dx = layout.$.pts[2 * j]! - xi;
          const dy = layout.$.pts[2 * j + 1]! - yi;
          const d2 = dx * dx + dy * dy;
          const sj = std.max(layout.$.params.scale * layout.$.rho[j]!, layout.$.params.minSigma);
          const invJ = 1 / (2 * sj * sj);
          const a = std.exp(-d2 * invI);
          const b = std.exp(-d2 * invJ);
          mu = a + b - a * b;
        }
        layout.$.out[i * n + j] = mu;
      }
    }
  })
  .$name("fuzzyAdjAdaptive");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([adaptiveFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "fuzzyAdjAdaptive" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makePool(root: Root, n: number) {
  return {
    n,
    pts: root.createBuffer(d.arrayOf(d.f32, 2 * n)).$usage("storage"),
    rho: root.createBuffer(d.arrayOf(d.f32, n)).$usage("storage"),
    out: root.createBuffer(d.arrayOf(d.f32, n * n)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, n: number) {
  if (pool && pool.n >= n) return pool;
  pool = makePool(root, Math.max(n, pool?.n ?? 0, 1));
  return pool;
}

export interface FuzzyAdjacencyAdaptiveOptions {
  /** Multiplier on each point's local bandwidth: σ_i = scale·ρ_i. Default 1. */
  scale?: number;
  /** Floor on σ_i so a degenerate ρ_i = 0 doesn't divide by zero. Default 1e-6. */
  minSigma?: number;
}

export interface FuzzyAdjacency {
  /** Row-major n*n symmetric membership matrix in [0,1]; diagonal 0. */
  membership: Float32Array;
  n: number;
}

/** Adaptive fuzzy adjacency given a precomputed per-point bandwidth ρ. */
export async function fuzzyAdjacencyAdaptiveFromRhoGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  rho: ArrayLike<number>,
  opts: FuzzyAdjacencyAdaptiveOptions = {},
): Promise<FuzzyAdjacency> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("fuzzyAdjacencyAdaptive: xs and ys length mismatch");
  if (rho.length !== n) throw new Error("fuzzyAdjacencyAdaptive: rho length != point count");
  if (n < 2) throw new Error("fuzzyAdjacencyAdaptive: need at least 2 points");

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  device.queue.writeBuffer(root.unwrap(p.pts), 0, flat as BufferSource);
  device.queue.writeBuffer(root.unwrap(p.rho), 0, Float32Array.from(rho) as BufferSource);
  p.params.write({ n, scale: opts.scale ?? 1, minSigma: opts.minSigma ?? 1e-6 });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, rho: p.rho, out: p.out }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.out.read()) as ArrayLike<number>;
  return { membership: Float32Array.from({ length: n * n }, (_, i) => got[i]!), n };
}

export interface FuzzyAdjacencyAdaptiveKOptions extends FuzzyAdjacencyAdaptiveOptions {
  /** Neighbour rank defining the local bandwidth ρ_i (1..32, < N). Default 4. */
  k?: number;
}

/** Adaptive fuzzy adjacency, computing the local bandwidth ρ_i internally as each
 *  point's k-th nearest-neighbour distance (composes `kthNeighborDistance`). */
export async function fuzzyAdjacencyAdaptiveGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: FuzzyAdjacencyAdaptiveKOptions = {},
): Promise<FuzzyAdjacency> {
  const k = opts.k ?? 4;
  const rho = await kthNeighborDistanceGpu(xs, ys, k);
  return fuzzyAdjacencyAdaptiveFromRhoGpu(xs, ys, rho, opts);
}
