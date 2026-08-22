// Fuzzy adjacency — a kernel-weighted graph over a point cloud. Where a *hard*
// adjacency matrix records 1 if two points are within radius R and 0 otherwise
// (a boxcar in distance), this records a smooth **membership** in [0,1]:
//   μ_ij = exp(-d_ij^2 / 2σ^2),   μ_ii = 0,   μ = 0 beyond the support radius.
//
// This is the same boxcar -> taper move we make for quadrats, applied to the
// connectivity of a point set. The fuzzy/weighted graph is the substrate for
// "fuzzier" topological data analysis: fuzzy simplicial sets (the construction
// underpinning UMAP) and weighted Vietoris-Rips filtrations, where an edge fades
// in with distance rather than snapping on at a single threshold — making the
// resulting topology far less brittle to the choice of radius. See
// docs/fuzzy-tda-and-windowing.md.
//
// Output is a dense NxN membership matrix (row-major), which is the natural form
// for small N (the only regime Node+Dawn validates anyway); a sparse, spatial-
// index-backed version is future work for large N.
//
// `"use gpu"` kernel, layout-bound pipeline (built once), pooled/grown buffers,
// TypeGPU `.read()` readback — the project's Dawn-stable pattern.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice, writeView } from "../device";

const WG = 64;

const Params = d.struct({
  n: d.u32,
  inv2s2: d.f32, // 1 / (2 σ²)
  maxD2: d.f32, // squared support radius; contributions beyond it are 0
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" }, // x,y per point
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" }, // n*n membership
});

const fuzzyFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const n = layout.$.params.n;
    if (i < n) {
      const xi = layout.$.pts[2 * i]!;
      const yi = layout.$.pts[2 * i + 1]!;
      for (let j = d.u32(0); j < n; j++) {
        const dx = layout.$.pts[2 * j]! - xi;
        const dy = layout.$.pts[2 * j + 1]! - yi;
        const d2 = dx * dx + dy * dy;
        let mu = d.f32(0);
        if (j !== i) {
          if (d2 <= layout.$.params.maxD2) {
            mu = std.exp(-d2 * layout.$.params.inv2s2);
          }
        }
        layout.$.out[i * n + j] = mu;
      }
    }
  })
  .$name("fuzzyAdj");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([fuzzyFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "fuzzyAdj" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makePool(root: Root, n: number) {
  return {
    n,
    pts: root.createBuffer(d.arrayOf(d.f32, 2 * n)).$usage("storage"),
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

export interface FuzzyAdjacencyOptions {
  /** Kernel bandwidth σ (world units). Edge membership = exp(-d²/2σ²). */
  sigma: number;
  /** Support radius in units of σ; memberships beyond it are exactly 0
   *  (sparsity / principled truncation). Default 3. */
  radiusSigma?: number;
}

export interface FuzzyAdjacency {
  /** Row-major n*n membership matrix in [0,1]; m[i*n+j] = μ_ij, diagonal 0. */
  membership: Float32Array;
  n: number;
}

/** Fuzzy (kernel-weighted) adjacency matrix for a point cloud. */
export async function fuzzyAdjacencyGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: FuzzyAdjacencyOptions,
): Promise<FuzzyAdjacency> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("fuzzyAdjacency: xs and ys length mismatch");
  if (n < 2) throw new Error("fuzzyAdjacency: need at least 2 points");
  if (!(opts.sigma > 0)) throw new Error("fuzzyAdjacency: sigma must be > 0");
  const radiusSigma = opts.radiusSigma ?? 3;

  const { device, root, pipeline } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  writeView(device.queue, root.unwrap(p.pts), flat);
  const support = radiusSigma * opts.sigma;
  p.params.write({ n, inv2s2: 1 / (2 * opts.sigma * opts.sigma), maxD2: support * support });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, out: p.out }));
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
