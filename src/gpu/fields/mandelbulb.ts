// Mandelbulb field generator — a reusable GPU compute primitive that fills a 3D brick
// with the power-8 Mandelbulb escape fraction (0 = escaped fast, 1 = inside), sampling
// an axis-aligned sub-box of the unit cube. It is the GPU counterpart of the CPU
// `mandelbulbField` in `src/datasource/syntheticLoader.ts`: the datasource's synthetic
// volume moves its per-voxel compute off the main thread and onto the device, so the
// brick-atlas renderer never janks generating chunks (see AGENTS.md — prefer our op
// library over host shaders; and docs/decisions/0009-rendering-as-ops.md).
//
// Authored in the same `"use gpu"` TGSL style as the graph ops (`ops/threshold.ts`):
// a layout-bound pipeline cached per backend root, Dawn-stable, portable across the
// Node/Dawn and browser (incl. an *adopted* host-renderer device) backends. It is NOT
// yet a registered `OpType` because the op-graph `Shape` vocabulary is 2-D-only
// (`grid` = width×height); promote it to a first-class 3-D op when the graph gains a
// volume shape. Until then it is a standalone kernel with a CPU golden.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { GpuBackend, Root } from "../graph/backend";

/** A brick request: sample a `dims`-sized grid whose voxel (i,j,k) centre maps to the
 *  normalised unit-cube coordinate `origin + (idx + 0.5)·step` per axis. For a chunk of
 *  the synthetic volume this is `origin = base/levelDims`, `step = extent/(dims·levelDims)`
 *  — collapsing to `(base + idx + 0.5)/levelDims`, exactly the CPU loader's sampling. */
export interface MandelbulbRegion {
  readonly dims: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly step: readonly [number, number, number];
  /** Fractal power (CPU default 8). */
  readonly power?: number;
  /** Escape-iteration budget = the value's quantum is `1/iters` (CPU default 8). */
  readonly iters?: number;
}

const DEFAULT_POWER = 8;
const DEFAULT_ITERS = 8;

const Params = d.struct({
  nx: d.u32,
  ny: d.u32,
  nz: d.u32,
  power: d.u32,
  ox: d.f32,
  oy: d.f32,
  oz: d.f32,
  iters: d.u32,
  sx: d.f32,
  sy: d.f32,
  sz: d.f32,
  _pad: d.f32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

// 3-D dispatch: the global invocation id IS the voxel coordinate, so there is no
// float-division index math (the `d.u32(i / w)` gotcha the 2-D kernels wrestle with).
const WG: [number, number, number] = [4, 4, 4];

const mandelbulbFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: WG })((input) => {
    "use gpu";
    const p = layout.$.params;
    const i = input.gid.x;
    const j = input.gid.y;
    const k = input.gid.z;
    if (i < p.nx && j < p.ny && k < p.nz) {
      const u = p.ox + (d.f32(i) + 0.5) * p.sx;
      const v = p.oy + (d.f32(j) + 0.5) * p.sy;
      const w = p.oz + (d.f32(k) + 0.5) * p.sz;
      const cx = (u * 2 - 1) * 1.2;
      const cy = (v * 2 - 1) * 1.2;
      const cz = (w * 2 - 1) * 1.2;
      const power = d.f32(p.power);
      // The orbit starts at the origin; its first update is deterministic — every trig
      // term is annihilated by zr = |origin|^power = 0 — so it lands exactly on (cx,cy,cz).
      // We do that step explicitly to skip the degenerate atan2(0,0): undefined in WGSL, it
      // returns NaN and 0·NaN poisons the whole orbit (JS Math.atan2(0,0)=0, so the CPU is
      // fine). The origin (iteration 0) can never escape (|origin|=0<2), so starting the
      // escape checks at n=1 on the point (cx,cy,cz) matches the CPU integrator's indexing.
      let x = cx;
      let y = cy;
      let z = cz;
      let iEsc = p.iters;
      // Fixed-trip loop, no `break`: latch the FIRST iteration whose radius escapes (r>2)
      // via a plain min-compare, advancing the orbit every iteration (single-level
      // reassignment — the pattern the other kernels use reliably). Interior points stay
      // bounded/finite; escaped points may diverge afterwards, but `iEsc` only latches its
      // first value, so the escape fraction matches the CPU integrator (which `break`s there).
      for (let n = d.u32(1); n < p.iters; n++) {
        const r = std.sqrt(x * x + y * y + z * z);
        if (r > 2) {
          if (n < iEsc) {
            iEsc = n;
          }
        }
        const rr = std.max(r, d.f32(1e-9));
        const theta = std.acos(z / rr) * power;
        const phi = std.atan2(y, x) * power;
        const zr = std.pow(r, power);
        const st = std.sin(theta);
        x = cx + zr * st * std.cos(phi);
        y = cy + zr * st * std.sin(phi);
        z = cz + zr * std.cos(theta);
      }
      const outIdx = (k * p.ny + j) * p.nx + i;
      layout.$.dst[outIdx] = d.f32(iEsc) / d.f32(p.iters);
    }
  })
  .$name("mandelbulb");

// Buffer types (incl. their `$usage` flags) are inferred from `makePool` so `createBindGroup`
// gets the precise storage/uniform types — annotating with the broad `createBuffer` return
// widens them and breaks the bind-group typing.
function makePool(root: Root, n: number) {
  const cap = Math.max(1, n);
  return {
    cap,
    dst: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    params: root.createBuffer(Params).$usage("uniform"),
  };
}
type Pool = ReturnType<typeof makePool>;
interface Pipe {
  pipeline: GPUComputePipeline;
  pool: Pool;
}

// One pipeline + pool per backend root (roots are device singletons — WeakMap keeps this
// correct across the Node device, the browser device, and an adopted three.js device).
const pipes = new WeakMap<object, Promise<Pipe>>();

function getPipe(device: GPUDevice, root: Root): Promise<Pipe> {
  let p = pipes.get(root as object);
  if (!p) {
    p = (async () => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([mandelbulbFn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "mandelbulb" } });
      return { pipeline, pool: makePool(root, 1) };
    })();
    pipes.set(root as object, p);
  }
  return p;
}

function ensurePool(root: Root, pipe: Pipe, n: number): Pool {
  if (pipe.pool.cap >= n) return pipe.pool;
  pipe.pool = makePool(root, Math.max(n, pipe.pool.cap * 2));
  return pipe.pool;
}

/** Fill a brick on the GPU and read it back as a host `Float32Array` (row-major, length
 *  `dims[0]·dims[1]·dims[2]`, values in [0,1]). The readback is async (`.read()` maps the
 *  buffer off the main thread) — the resident, no-readback path (compute straight into the
 *  atlas texture) is the ADR-0009 follow-up flagged in AGENTS.md. */
export async function mandelbulbBrickGpu(backend: GpuBackend, region: MandelbulbRegion): Promise<Float32Array> {
  const [nx, ny, nz] = region.dims;
  const n = nx * ny * nz;
  const device = await backend.getDevice();
  const root = await backend.getRoot();
  const pipe = await getPipe(device, root);
  const pool = ensurePool(root, pipe, n);

  pool.params.write({
    nx,
    ny,
    nz,
    power: region.power ?? DEFAULT_POWER,
    ox: region.origin[0],
    oy: region.origin[1],
    oz: region.origin[2],
    iters: region.iters ?? DEFAULT_ITERS,
    sx: region.step[0],
    sy: region.step[1],
    sz: region.step[2],
    _pad: 0,
  });

  const bind = root.unwrap(root.createBindGroup(layout, { params: pool.params, dst: pool.dst }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe.pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(nx / WG[0]), Math.ceil(ny / WG[1]), Math.ceil(nz / WG[2]));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await pool.dst.read()) as ArrayLike<number>;
  return Float32Array.from({ length: n }, (_, idx) => got[idx] ?? 0);
}

/** CPU reference for {@link mandelbulbBrickGpu} — the golden the kernel is validated
 *  against, and a no-GPU fallback. Mirrors `mandelbulbField` (parameterised power/iters). */
export function mandelbulbBrickCpu(region: MandelbulbRegion): Float32Array {
  const [nx, ny, nz] = region.dims;
  const power = region.power ?? DEFAULT_POWER;
  const iters = region.iters ?? DEFAULT_ITERS;
  const out = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) {
    const w = region.origin[2] + (k + 0.5) * region.step[2];
    for (let j = 0; j < ny; j++) {
      const v = region.origin[1] + (j + 0.5) * region.step[1];
      for (let i = 0; i < nx; i++) {
        const u = region.origin[0] + (i + 0.5) * region.step[0];
        out[(k * ny + j) * nx + i] = mandelbulbAt(u, v, w, power, iters);
      }
    }
  }
  return out;
}

function mandelbulbAt(u: number, v: number, w: number, power: number, maxIter: number): number {
  const cx = (u * 2 - 1) * 1.2;
  const cy = (v * 2 - 1) * 1.2;
  const cz = (w * 2 - 1) * 1.2;
  let x = 0;
  let y = 0;
  let z = 0;
  let i = 0;
  for (; i < maxIter; i++) {
    const r = Math.hypot(x, y, z);
    if (r > 2) break;
    const rr = r < 1e-9 ? 1e-9 : r;
    const theta = Math.acos(z / rr) * power;
    const phi = Math.atan2(y, x) * power;
    const zr = r ** power;
    x = cx + zr * Math.sin(theta) * Math.cos(phi);
    y = cy + zr * Math.sin(theta) * Math.sin(phi);
    z = cz + zr * Math.cos(theta);
  }
  return i / maxIter;
}
