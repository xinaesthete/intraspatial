// GPU nearest-neighbour distance — for each of N points, the Euclidean distance
// to its closest other point. Foundational for the discrete-cell ("MuSpAn-style")
// spatial front: the Average Nearest Neighbour Index, the nearest-neighbour
// distribution, and (with random sample points) the empty-space function all sit
// on top of this.
//
// The kernel is authored in TypeScript as a `"use gpu"` TGSL function (transpiled
// to WGSL by unplugin-typegpu — see vitest.gpu.config.ts), then *resolved to WGSL*
// and executed via a raw compute pipeline bound to a bind-group layout. This keeps
// authoring in type-safe TS while running on the project's Dawn-stable path:
//   - the pipeline references the LAYOUT (not specific buffers), so it is built
//     ONCE and survives buffer growth (only the bind group is recreated);
//   - buffers are pooled with createBuffer/$usage and grown without `.destroy()`
//     (destroying buffers mid-process segfaults Dawn-on-Node's exit teardown; the
//     guarded `"use gpu"` pipeline, which closes over buffers and thus rebuilds on
//     growth, hit the same crash — hence this layout-bound design).
//
// Brute force O(N^2): one thread per point loops over all points. No atomics /
// shared memory, so it fits TGSL directly; the index-accelerated O(N*k) version
// (uniform-grid spatial index) comes later.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice } from "../device";

const WG = 64;
const FAR = 3.4e38; // f32 max-ish sentinel for "no neighbour yet"

const Params = d.struct({ n: d.u32 });

// Points stored flat: x at 2*i, y at 2*i+1. (Flat f32 dodges vec2f write-shape
// quirks and writes cleanly via queue.writeBuffer.)
const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  outb: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const nnFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const count = layout.$.params.n;
    if (i < count) {
      const xi = layout.$.pts[2 * i]!;
      const yi = layout.$.pts[2 * i + 1]!;
      let best = d.f32(FAR);
      for (let j = d.u32(0); j < count; j++) {
        const dx = layout.$.pts[2 * j]! - xi;
        const dy = layout.$.pts[2 * j + 1]! - yi;
        const dist = std.sqrt(dx * dx + dy * dy);
        if (j !== i) {
          best = std.min(best, dist);
        }
      }
      layout.$.outb[i] = best;
    }
  })
  .$name("nnDist");

// Pipeline is expensive and immutable across calls → cache per device. It binds
// the *layout*, so it never needs rebuilding when buffers grow.
interface Pipe {
  root: ReturnType<typeof tgpu.initFromDevice>;
  device: GPUDevice;
  pipeline: GPUComputePipeline;
}
let pipeCache: Promise<Pipe> | undefined;

function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([nnFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({
      bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)),
    });
    const pipeline = device.createComputePipeline({
      layout: pipeLayout,
      compute: { module, entryPoint: "nnDist" },
    });
    return { root, device, pipeline };
  })();
  return pipeCache;
}

// Pooled buffers, grown (never destroyed) across calls.
type Root = Pipe["root"];
function makePool(root: Root, cap: number) {
  return {
    cap,
    pts: root.createBuffer(d.arrayOf(d.f32, 2 * cap)).$usage("storage"),
    outb: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, n: number) {
  if (pool && pool.cap >= n) return pool;
  pool = makePool(root, Math.max(n, pool?.cap ?? 0, 1));
  return pool;
}

/** For each point, the Euclidean distance to its nearest other point.
 *  `xs`/`ys` are parallel coordinate arrays of equal length N (>= 2).
 *  Returns a Float32Array of length N. */
export async function nearestNeighborDistancesGpu(xs: ArrayLike<number>, ys: ArrayLike<number>): Promise<Float32Array> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("nnDistance: xs and ys length mismatch");
  if (n < 2) throw new Error("nnDistance: need at least 2 points");

  const { root, device, pipeline } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  device.queue.writeBuffer(root.unwrap(p.pts), 0, flat as BufferSource);
  p.params.write({ n });

  const bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, outb: p.outb }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.outb.read()) as ArrayLike<number>;
  return Float32Array.from({ length: n }, (_, i) => got[i]!);
}
