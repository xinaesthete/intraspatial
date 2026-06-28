// Reaction–diffusion (Gray–Scott) — the first *iterative* primitive. Where every
// other op produces one snapshot from its inputs, a simulation advances state over
// time: each call is one explicit Euler step of the two-species Gray–Scott system,
//
//   U' = U + dt·( Du·∇²U − U·V² + F·(1−U) )
//   V' = V + dt·( Dv·∇²V + U·V² − (F+k)·V )
//
// with a 5-point toroidal Laplacian. A pattern grows by running many steps; the loop
// lives in the caller (or the graph, ping-ponging the state grids), not the kernel —
// matching the runtime's per-stage model (see docs/gpu-simulation-toolbox.md).
//
// `"use gpu"` kernel, layout-bound pipeline, pooled buffers that ping-pong between
// steps, `.read()` readback — the project's Dawn-stable pattern.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice } from "../device";

const WG = 64;

const Params = d.struct({
  w: d.u32,
  h: d.u32,
  du: d.f32,
  dv: d.f32,
  feed: d.f32,
  kill: d.f32,
  dt: d.f32,
});

const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  uIn: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  vIn: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  uOut: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  vOut: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const stepFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const w = layout.$.params.w;
    const h = layout.$.params.h;
    const total = w * h;
    if (i < total) {
      const col = i % w;
      // `i / w` transpiles to FLOAT division (f32(i)/f32(w)); wrap in u32 so `row`
      // is an integer and `row * w` indexing stays exact.
      const row = d.u32(i / w);
      // 5-point Laplacian, clamp-to-edge (Neumann) boundaries. Indices built with
      // std.clamp on i32 — the proven pattern from convolveSeparable; building them
      // with `%` inline mis-transpiled the precedence and scattered the neighbours.
      const cm = d.u32(std.clamp(d.i32(col) - 1, 0, d.i32(w) - 1));
      const cp = d.u32(std.clamp(d.i32(col) + 1, 0, d.i32(w) - 1));
      const rm = d.u32(std.clamp(d.i32(row) - 1, 0, d.i32(h) - 1));
      const rp = d.u32(std.clamp(d.i32(row) + 1, 0, d.i32(h) - 1));
      const left = row * w + cm;
      const right = row * w + cp;
      const up = rm * w + col;
      const down = rp * w + col;

      const uc = layout.$.uIn[i]!;
      const vc = layout.$.vIn[i]!;
      const lapU = layout.$.uIn[left]! + layout.$.uIn[right]! + layout.$.uIn[up]! + layout.$.uIn[down]! - 4 * uc;
      const lapV = layout.$.vIn[left]! + layout.$.vIn[right]! + layout.$.vIn[up]! + layout.$.vIn[down]! - 4 * vc;

      const uvv = uc * vc * vc;
      const dt = layout.$.params.dt;
      layout.$.uOut[i] = uc + dt * (layout.$.params.du * lapU - uvv + layout.$.params.feed * (1 - uc));
      layout.$.vOut[i] = vc + dt * (layout.$.params.dv * lapV + uvv - (layout.$.params.feed + layout.$.params.kill) * vc);
    }
  })
  .$name("grayScottStep");

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
    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([stepFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "grayScottStep" } });
    return { device, root, pipeline };
  })();
  return pipeCache;
}

type Root = Pipe["root"];
function makePool(root: Root, n: number) {
  const cap = Math.max(1, n);
  const sbuf = () => root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage");
  return { n, uA: sbuf(), vA: sbuf(), uB: sbuf(), vB: sbuf(), params: root.createBuffer(Params).$usage("uniform") };
}
let pool: ReturnType<typeof makePool> | undefined;
function ensurePool(root: Root, n: number) {
  if (pool && pool.n >= n) return pool;
  pool = makePool(root, Math.max(n, pool?.n ?? 0));
  return pool;
}

export interface GrayScottParams {
  /** Diffusion rate of U (default 0.16). */
  du?: number;
  /** Diffusion rate of V (default 0.08). */
  dv?: number;
  /** Feed rate F (default 0.06). */
  feed?: number;
  /** Kill rate k (default 0.062). */
  kill?: number;
  /** Explicit-Euler time step (default 1). */
  dt?: number;
}

export interface GrayScottState {
  /** Row-major width*height concentration of U. */
  u: Float32Array;
  /** Row-major width*height concentration of V. */
  v: Float32Array;
  width: number;
  height: number;
}

const DEFAULTS: Required<GrayScottParams> = { du: 0.16, dv: 0.08, feed: 0.06, kill: 0.062, dt: 1 };

/** Advance a Gray–Scott state by `steps` explicit-Euler steps on the GPU. State
 *  grids ping-pong in a pooled pair; only the final state is read back. */
export async function grayScottStepsGpu(
  state: GrayScottState,
  steps: number,
  params: GrayScottParams = {},
): Promise<GrayScottState> {
  const { width: w, height: h } = state;
  const n = w * h;
  if (state.u.length !== n || state.v.length !== n) throw new Error("grayScott: u/v length != width*height");
  if (steps < 1) return state;
  const p = { ...DEFAULTS, ...params };

  const { device, root, pipeline } = await getPipe();
  const pool = ensurePool(root, n);

  device.queue.writeBuffer(root.unwrap(pool.uA), 0, Float32Array.from(state.u) as BufferSource);
  device.queue.writeBuffer(root.unwrap(pool.vA), 0, Float32Array.from(state.v) as BufferSource);
  pool.params.write({ w, h, du: p.du, dv: p.dv, feed: p.feed, kill: p.kill, dt: p.dt });

  const groups = Math.ceil(n / WG);
  let uIn = pool.uA, vIn = pool.vA, uOut = pool.uB, vOut = pool.vB;
  for (let s = 0; s < steps; s++) {
    const bind = root.unwrap(root.createBindGroup(layout, { params: pool.params, uIn, vIn, uOut, vOut }));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]); // per-step submit (resource-sync baseline)
    [uIn, uOut] = [uOut, uIn];
    [vIn, vOut] = [vOut, vIn];
  }

  const u = (await uIn.read()) as ArrayLike<number>;
  const v = (await vIn.read()) as ArrayLike<number>;
  return {
    u: Float32Array.from({ length: n }, (_, i) => u[i]!),
    v: Float32Array.from({ length: n }, (_, i) => v[i]!),
    width: w,
    height: h,
  };
}

/** One Gray–Scott step on the CPU — the golden, and a reference for the graph op's
 *  validate/fallback. Toroidal 5-point Laplacian, explicit Euler. */
export function grayScottStepCpu(state: GrayScottState, params: GrayScottParams = {}): GrayScottState {
  const { width: w, height: h } = state;
  const p = { ...DEFAULTS, ...params };
  const n = w * h;
  const u = state.u, v = state.v;
  const uo = new Float32Array(n), vo = new Float32Array(n);
  const clamp = (x: number, hi: number) => (x < 0 ? 0 : x > hi ? hi : x);
  const idx = (c: number, r: number) => clamp(r, h - 1) * w + clamp(c, w - 1);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c;
      const uc = u[i]!, vc = v[i]!;
      const lapU = u[idx(c - 1, r)]! + u[idx(c + 1, r)]! + u[idx(c, r - 1)]! + u[idx(c, r + 1)]! - 4 * uc;
      const lapV = v[idx(c - 1, r)]! + v[idx(c + 1, r)]! + v[idx(c, r - 1)]! + v[idx(c, r + 1)]! - 4 * vc;
      const uvv = uc * vc * vc;
      uo[i] = uc + p.dt * (p.du * lapU - uvv + p.feed * (1 - uc));
      vo[i] = vc + p.dt * (p.dv * lapV + uvv - (p.feed + p.kill) * vc);
    }
  }
  return { u: uo, v: vo, width: w, height: h };
}

/** A standard Gray–Scott seed: U=1 everywhere, V=0, with a small square of V=0.5,
 *  U=0.5 perturbation at the centre (plus optional jitter to break symmetry). */
export function seedGrayScott(width: number, height: number, jitter = 0): GrayScottState {
  const n = width * height;
  const u = new Float32Array(n).fill(1);
  const v = new Float32Array(n);
  const cx = Math.floor(width / 2), cy = Math.floor(height / 2);
  const r = Math.max(2, Math.floor(Math.min(width, height) / 10));
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= width || y >= height) continue;
      const i = y * width + x;
      u[i] = 0.5;
      v[i] = 0.25;
    }
  }
  if (jitter > 0) {
    // deterministic hash jitter (no Math.random in the kernel path; keep CPU seed reproducible)
    for (let i = 0; i < n; i++) {
      const hsh = ((i * 2654435761) >>> 0) / 0xffffffff;
      v[i] = Math.min(1, Math.max(0, v[i]! + (hsh - 0.5) * jitter));
    }
  }
  return { u, v, width, height };
}
