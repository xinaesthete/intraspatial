// Continuous k-Nearest Neighbours (CkNN) — Berry & Sauer, "Consistent Manifold
// Representation for Topological Data Analysis" (2016).
//
// CkNN connects points i,j when  d(i,j) < δ·√(ρ_i ρ_j),  where ρ_i = d(i, k-th NN)
// is a local density estimate and δ is a single continuous scale. Equivalently, it
// is ordinary connectivity on a **density-rescaled distance**
//
//     d̃_ij = d_ij / √(ρ_i ρ_j).
//
// That one matrix is the whole story:
//   • topology  — feed d̃ to a Vietoris-Rips persistence engine (CPU: Ripser/GUDHI);
//                 because ρ adapts to local density, a *single* graph captures all
//                 scales at once ("consistent homology"), and it gets the right
//                 Betti numbers even on non-compact manifolds where a fixed radius
//                 cannot. Thresholding at δ gives the unweighted CkNN graph.
//   • geometry  — the self-tuning kernel is just  K_ij = exp(-d̃_ij²)  (since
//                 d̃² = d²/(ρ_iρ_j)); its graph Laplacian is a consistent estimator
//                 of the Laplace-de Rham operator.
//
// So this primitive (GPU builds d̃) plus two trivial CPU readouts covers both the
// unweighted-for-topology and weighted-for-geometry sides of the paper. See
// docs/fuzzy-tda-and-windowing.md.
//
// Composition: `kthNeighborDistanceGpu` for ρ, then a `"use gpu"` kernel for d̃.
// Dense N×N output (the form persistence wants); small-N regime only, as ever.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice, writeView } from "../device";
import { kthNeighborDistanceGpu } from "./kthNeighborDistance";

const WG = 64;
const Params = d.struct({ n: d.u32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  rho: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // n*n
});

const cknnFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const n = layout.$.params.n;
    if (i < n) {
      const xi = layout.$.pts[2 * i]!;
      const yi = layout.$.pts[2 * i + 1]!;
      const ri = layout.$.rho[i]!;
      for (let j = d.u32(0); j < n; j++) {
        let dt = d.f32(0);
        if (j !== i) {
          const dx = layout.$.pts[2 * j]! - xi;
          const dy = layout.$.pts[2 * j + 1]! - yi;
          const dij = std.sqrt(dx * dx + dy * dy);
          const den = std.sqrt(ri * layout.$.rho[j]!);
          dt = dij / std.max(den, d.f32(1e-20));
        }
        layout.$.out[i * n + j] = dt;
      }
    }
  })
  .$name("cknnRescaled");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([cknnFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "cknnRescaled" } });
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

export interface CknnResult {
  /** Row-major n*n rescaled distance d̃_ij = d_ij / √(ρ_i ρ_j); diagonal 0. Feed to
   *  Vietoris-Rips persistence, or threshold at δ for the CkNN graph. */
  rescaled: Float32Array;
  /** The local bandwidths ρ_i = distance to the k-th nearest neighbour. */
  rho: Float32Array;
  n: number;
}

/** CkNN rescaled-distance matrix for a point cloud. `k` is the neighbour rank for
 *  the local bandwidth (1..32, < N); larger k = smoother density estimate. */
export async function cknnRescaledDistanceGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: { k?: number } = {},
): Promise<CknnResult> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("cknn: xs and ys length mismatch");
  const k = opts.k ?? 5;

  // ρ_i = k-th NN distance (composed GPU primitive).
  const rho = await kthNeighborDistanceGpu(xs, ys, k);

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n);
  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  writeView(device.queue, root.unwrap(p.pts), flat);
  writeView(device.queue, root.unwrap(p.rho), rho);
  p.params.write({ n });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, rho: p.rho, out: p.out }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.out.read()) as ArrayLike<number>;
  return { rescaled: Float32Array.from({ length: n * n }, (_, i) => got[i]!), rho, n };
}

/** The unweighted CkNN graph at scale δ: edge i~j iff d̃_ij < δ. Returns a row-major
 *  n*n Uint8Array adjacency (0/1), diagonal 0. (CPU, trivial — for topology.) */
export function cknnGraph(rescaled: ArrayLike<number>, n: number, delta: number): Uint8Array {
  const a = new Uint8Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i !== j && rescaled[i * n + j]! < delta) a[i * n + j] = 1;
    }
  return a;
}

/** The self-tuning (weighted) kernel K_ij = exp(-d̃_ij²); its Laplacian is a
 *  consistent estimator of the Laplace-de Rham operator. (CPU — for geometry.) */
export function selfTuningWeights(rescaled: ArrayLike<number>, n: number): Float32Array {
  const w = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const dt = rescaled[i * n + j]!;
        w[i * n + j] = Math.exp(-dt * dt);
      }
    }
  return w;
}
