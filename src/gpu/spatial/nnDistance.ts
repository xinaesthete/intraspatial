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
// shared memory, so it fits TGSL directly. With `cell` the uniform-grid index
// (`gridIndex.ts`) is built in the same command buffer and the thread walks the
// 3×3 stencil instead — O(N·k) — under the contract in `IndexedQueryOptions`.
// Brute force stays the default and the golden.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice, sized, writeView } from "../device";
import { type GridIndexCtx, getGridIndexCtx } from "./gridIndex";
import { cellCoord, cellRange, encodeQueryIndex, type IndexedQueryOptions, LATTICE_BYTES, Lattice, StartArray } from "./gridIndexQuery";

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

// The indexed kernel's bindings: the brute-force set plus the lattice and the index.
const layoutIdx = tgpu.bindGroupLayout({
  params: { uniform: Params },
  lat: { uniform: Lattice },
  pts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  start: { storage: StartArray, access: "readonly" },
  items: { storage: StartArray, access: "readonly" },
  outb: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const nnIdxFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const count = layoutIdx.$.params.n;
    if (i < count) {
      const xi = layoutIdx.$.pts[2 * i]!;
      const yi = layoutIdx.$.pts[2 * i + 1]!;
      const c = cellCoord(d.vec2f(xi, yi), layoutIdx.$.lat);
      let best = d.f32(FAR);
      for (let dy = d.i32(-1); dy <= 1; dy++) {
        for (let dx = d.i32(-1); dx <= 1; dx++) {
          const r = cellRange(d.vec2i(c.x + dx, c.y + dy), layoutIdx.$.lat, d.ref(layoutIdx.$.start));
          for (let s = r.x; s < r.y; s++) {
            const j = layoutIdx.$.items[s]!;
            const ddx = layoutIdx.$.pts[2 * j]! - xi;
            const ddy = layoutIdx.$.pts[2 * j + 1]! - yi;
            const dist = std.sqrt(ddx * ddx + ddy * ddy);
            if (j !== i) {
              best = std.min(best, dist);
            }
          }
        }
      }
      layoutIdx.$.outb[i] = best;
    }
  })
  .$name("nnDistIdx");

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
  /** Indexed variant; its layout is `layoutIdx`. */
  pipelineIdx: GPUComputePipeline;
  indexCtx: GridIndexCtx;
}
let pipeCache: Promise<Pipe> | undefined;

function getPipe(): Promise<Pipe> {
  pipeCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const build = (fn: typeof nnFn, entryPoint: string) => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([fn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({
        bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)),
      });
      return device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint } });
    };
    const indexCtx = await getGridIndexCtx();
    return { root, device, pipeline: build(nnFn, "nnDist"), pipelineIdx: build(nnIdxFn, "nnDistIdx"), indexCtx };
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
 *  Returns a Float32Array of length N.
 *
 *  With `opts.cell` the query is indexed (see `IndexedQueryOptions`): exact wherever the true
 *  nearest neighbour is within `cell`; a point with no other point in its 3×3 stencil gets
 *  `+Infinity`. */
export async function nearestNeighborDistancesGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: IndexedQueryOptions = {},
): Promise<Float32Array> {
  const n = xs.length;
  if (ys.length !== n) throw new Error("nnDistance: xs and ys length mismatch");
  if (n < 2) throw new Error("nnDistance: need at least 2 points");

  const { root, device, pipeline, pipelineIdx, indexCtx } = await getPipe();
  const p = ensurePool(root, n);

  const flat = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  writeView(device.queue, root.unwrap(p.pts), flat);
  p.params.write({ n });

  const enc = device.createCommandEncoder();
  let bind: GPUBindGroup;
  if (opts.cell === undefined) {
    bind = root.unwrap(root.createBindGroup(layout, { params: p.params, pts: p.pts, outb: p.outb }));
  } else {
    const q = encodeQueryIndex(indexCtx, root.unwrap(p.pts), n, xs, ys, opts.cell, opts.bounds, enc, "nnDistance:index");
    bind = device.createBindGroup({
      layout: root.unwrap(layoutIdx),
      entries: [
        { binding: 0, resource: { buffer: root.unwrap(p.params) } },
        { binding: 1, resource: sized(q.lat, LATTICE_BYTES) },
        { binding: 2, resource: sized(root.unwrap(p.pts), 2 * n * 4) },
        { binding: 3, resource: sized(q.index.start, (q.index.M + 1) * 4) },
        { binding: 4, resource: sized(q.index.items, n * 4) },
        { binding: 5, resource: sized(root.unwrap(p.outb), n * 4) },
      ],
    });
  }
  const pass = enc.beginComputePass();
  pass.setPipeline(opts.cell === undefined ? pipeline : pipelineIdx);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await p.outb.read()) as ArrayLike<number>;
  // The kernel's "no neighbour" sentinel is FAR; the documented value is +Infinity. (FAR
  // rounds down in f32, so the test is a threshold, not equality.)
  return Float32Array.from({ length: n }, (_, i) => (got[i]! >= 3e38 ? Number.POSITIVE_INFINITY : got[i]!));
}
