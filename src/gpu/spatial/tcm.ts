// GPU Topographical Correlation Map (TCM) — the device path for `src/spatial/tcm.ts`.
//
// The CPU version is one serial loop doing two unrelated jobs; on the GPU they separate cleanly
// into two atomic-free `"use gpu"` (TGSL) kernels, which is why neither needs a WGSL template:
//
//   1. `tcmMarks`  — one thread per A cell. Counts B within `radius` through the 3×3 neighbourhood
//                    of a CSR `BucketGrid` over B, forms m_ab (eq 9) and the transformed mark
//                    M_ab (eqs 10–13) in-register. Pure map over A: no cross-thread traffic.
//   2. `tcmSplat`  — one thread per OUTPUT CELL, gathering the A points whose kernel window covers
//                    it (via a second bucket grid, over A, sized to the kernel support). Gather,
//                    not scatter: each cell owns its own accumulator, so there is no contention
//                    and no need for the f32 atomics core WGSL lacks.
//
// The gather deliberately reproduces the CPU reference's arithmetic exactly — the same ±kr SQUARE
// window in cells, the same cell-centre sample points, the same 1/(σ√2π) normalisation, and row 0
// at minY. It is not the `splatDensity` render path, which uses a σ-relative support and puts row 0
// at maxY: that is a fine KDE but a different function, and this one has to stay comparable with
// `computeTcmReference`, the parity oracle. The residual difference is summation ORDER (bucket
// order here, A-index order on the CPU), so the two agree to f32 rounding, not bit-exactly.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { latticeFor } from "../../spatial/bucketGrid";
import type { CellCloud, TcmParams } from "../../spatial/tcm";
import { getDevice, writeView } from "../device";
import { encodeGridIndex, type GridIndexCtx, getGridIndexCtx } from "./gridIndex";

const WG = 64;

// --- kernel 1: per-A-cell transformed mark M_ab ---

const MarkParams = d.struct({
  nA: d.u32,
  cols: d.u32,
  rows: d.u32,
  pad: d.u32,
  minX: d.f32,
  minY: d.f32,
  cell: d.f32,
  r2: d.f32,
  /** ρ_B · A_radius — the CSR expectation m_ab is measured against (eq 9). */
  expected: d.f32,
  alpha: d.f32,
});

const markLayout = tgpu.bindGroupLayout({
  params: { uniform: MarkParams },
  aPts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  bPts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  bStart: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  bItems: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  marks: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const marksFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = markLayout.$.params;
    if (i < P.nA) {
      const ax = markLayout.$.aPts[2 * i]!;
      const ay = markLayout.$.aPts[2 * i + 1]!;
      // Bucket coords are computed in f32 and clamped BEFORE the u32 cast: the raw floor can be
      // negative for a point outside the grid bounds, and u32(-1) is not a small number.
      const c0 = d.u32(std.clamp(std.floor((ax - P.minX) / P.cell), d.f32(0), d.f32(P.cols) - 1));
      const r0 = d.u32(std.clamp(std.floor((ay - P.minY) / P.cell), d.f32(0), d.f32(P.rows) - 1));
      // 3×3 window without signed arithmetic: max(x,1)-1 is a clamped decrement on u32.
      const cLo = std.max(c0, d.u32(1)) - 1;
      const cHi = std.min(c0 + 1, P.cols - 1);
      const rLo = std.max(r0, d.u32(1)) - 1;
      const rHi = std.min(r0 + 1, P.rows - 1);
      let count = d.f32(0);
      for (let rr = rLo; rr <= rHi; rr++) {
        for (let cc = cLo; cc <= cHi; cc++) {
          const b = rr * P.cols + cc;
          const hi = markLayout.$.bStart[b + 1]!;
          for (let k = markLayout.$.bStart[b]!; k < hi; k++) {
            const j = markLayout.$.bItems[k]!;
            const dx = markLayout.$.bPts[2 * j]! - ax;
            const dy = markLayout.$.bPts[2 * j + 1]! - ay;
            if (dx * dx + dy * dy < P.r2) {
              count = count + 1;
            }
          }
        }
      }
      let m = d.f32(0);
      if (P.expected > 0) {
        m = count / P.expected;
      }
      // markToM (eqs 10–13) as an overwrite cascade rather than else-if: the default is the
      // reciprocal branch (1/α < m < 1), guarded against m = 0, then each stricter case wins.
      let M = (1 - 1 / std.max(m, d.f32(1e-20))) / (P.alpha - 1);
      if (m <= 1 / P.alpha) {
        M = -1;
      }
      if (m > 1) {
        M = (m - 1) / (P.alpha - 1);
      }
      if (m >= P.alpha) {
        M = 1;
      }
      markLayout.$.marks[i] = M;
    }
  })
  .$name("tcmMarks");

// --- kernel 2: gather the marked Gaussians into the output grid ---

const SplatParams = d.struct({
  w: d.u32,
  h: d.u32,
  cols: d.u32,
  rows: d.u32,
  minX: d.f32,
  minY: d.f32,
  cw: d.f32,
  ch: d.f32,
  /** Kernel square-support half-extent in GRID CELLS (the reference's `kr`). */
  kr: d.f32,
  inv2s2: d.f32,
  norm: d.f32,
  /** Bucket side of the grid over A (≥ the kernel's world reach, so 3×3 suffices). */
  cell: d.f32,
});

const splatLayout = tgpu.bindGroupLayout({
  params: { uniform: SplatParams },
  aPts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  marks: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  aStart: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  aItems: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const splatFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const idx = input.gid.x;
    const P = splatLayout.$.params;
    if (idx < P.w * P.h) {
      // Integer division, spelled out: `idx / P.w` on u32 operands transpiles to FLOAT division
      // here, which silently scrambles the row/col split (see splatDensity's depad kernel).
      const row = d.u32(std.floor(d.f32(idx) / d.f32(P.w)));
      const col = idx - row * P.w;
      const wx = P.minX + (d.f32(col) + 0.5) * P.cw; // cell CENTRE, as the reference samples
      const wy = P.minY + (d.f32(row) + 0.5) * P.ch;
      const c0 = d.u32(std.clamp(std.floor((wx - P.minX) / P.cell), d.f32(0), d.f32(P.cols) - 1));
      const r0 = d.u32(std.clamp(std.floor((wy - P.minY) / P.cell), d.f32(0), d.f32(P.rows) - 1));
      const cLo = std.max(c0, d.u32(1)) - 1;
      const cHi = std.min(c0 + 1, P.cols - 1);
      const rLo = std.max(r0, d.u32(1)) - 1;
      const rHi = std.min(r0 + 1, P.rows - 1);
      let acc = d.f32(0);
      for (let rr = rLo; rr <= rHi; rr++) {
        for (let cc = cLo; cc <= cHi; cc++) {
          const b = rr * P.cols + cc;
          const hi = splatLayout.$.aStart[b + 1]!;
          for (let k = splatLayout.$.aStart[b]!; k < hi; k++) {
            const j = splatLayout.$.aItems[k]!;
            const ax = splatLayout.$.aPts[2 * j]!;
            const ay = splatLayout.$.aPts[2 * j + 1]!;
            // The reference's window, restated as a gather: cell (col,row) is inside point j's
            // [floor(c−kr), ceil(c+kr)] box. Compared in f32 so the lower bound may go negative
            // without wrapping.
            const ci = (ax - P.minX) / P.cw;
            const cj = (ay - P.minY) / P.ch;
            const inX = d.f32(col) >= std.floor(ci - P.kr) && d.f32(col) <= std.ceil(ci + P.kr);
            const inY = d.f32(row) >= std.floor(cj - P.kr) && d.f32(row) <= std.ceil(cj + P.kr);
            if (inX && inY) {
              const dx = wx - ax;
              const dy = wy - ay;
              acc = acc + splatLayout.$.marks[j]! * P.norm * std.exp(-(dx * dx + dy * dy) * P.inv2s2);
            }
          }
        }
      }
      splatLayout.$.out[idx] = acc;
    }
  })
  .$name("tcmSplat");

// --- device plumbing ---

type Root = ReturnType<typeof tgpu.initFromDevice>;

// Concrete factories, so the cached handles keep their STRUCT type. `ReturnType<Root["createBuffer"]>`
// erases to TgpuBuffer<AnyData>, which no longer satisfies the bind-group layout's uniform slot.
const mkMarkParams = (root: Root) => root.createBuffer(MarkParams).$usage("uniform");
const mkSplatParams = (root: Root) => root.createBuffer(SplatParams).$usage("uniform");

interface Ctx {
  device: GPUDevice;
  root: Root;
  index: GridIndexCtx;
  marks: GPUComputePipeline;
  splat: GPUComputePipeline;
  markParams: ReturnType<typeof mkMarkParams>;
  splatParams: ReturnType<typeof mkSplatParams>;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const build = (fn: typeof marksFn | typeof splatFn, entryPoint: string) => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([fn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const layout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      return device.createComputePipeline({ layout, compute: { module, entryPoint } });
    };
    return {
      device,
      root,
      index: await getGridIndexCtx(),
      marks: build(marksFn, "tcmMarks"),
      splat: build(splatFn, "tcmSplat"),
      markParams: mkMarkParams(root),
      splatParams: mkSplatParams(root),
    };
  })();
  return ctxCache;
}

// Grow-only buffer pools, keyed by role. Never `.destroy()`d: destroying buffers mid-process
// segfaults Dawn-on-Node's teardown.
function mkF32(root: Root, n: number) {
  return root.createBuffer(d.arrayOf(d.f32, n)).$usage("storage");
}
function mkU32(root: Root, n: number) {
  return root.createBuffer(d.arrayOf(d.u32, n)).$usage("storage");
}
const f32Pool = new Map<string, { buf: ReturnType<typeof mkF32>; cap: number }>();

function ensureF32(root: Root, key: string, n: number) {
  const got = f32Pool.get(key);
  if (got && got.cap >= n) return got.buf;
  const cap = Math.max(n, (got?.cap ?? 0) * 2, 1);
  const buf = mkF32(root, cap);
  f32Pool.set(key, { buf, cap });
  return buf;
}

/** The index's `start`/`items` come back from `encodeGridIndex` as raw pooled buffers; the
 *  TypeGPU bind group wants wrappers. One wrapper per underlying buffer, cached for as long as
 *  the pool keeps that buffer (it only ever swaps on growth). */
const u32Wraps = new WeakMap<GPUBuffer, ReturnType<typeof mkU32>>();
function wrapU32(root: Root, raw: GPUBuffer) {
  let w = u32Wraps.get(raw);
  if (!w) {
    w = root.createBuffer(d.arrayOf(d.u32, raw.size / 4), raw).$usage("storage");
    u32Wraps.set(raw, w);
  }
  return w;
}

function packXY(xs: ArrayLike<number>, ys: ArrayLike<number>): Float32Array {
  const n = xs.length;
  const flat = new Float32Array(2 * Math.max(n, 1));
  for (let i = 0; i < n; i++) {
    flat[2 * i] = xs[i]!;
    flat[2 * i + 1] = ys[i]!;
  }
  return flat;
}

/** Per-A-cell transformed mark M_ab ∈ [−1, 1] (eqs 9–13) on the GPU. Exposed on its own because it
 *  is the interesting per-cell quantity — a cell-level "is this A cell in a B-rich place?" score
 *  that can be coloured straight onto the scatter, independent of the Γ raster. */
export async function crossMarksGpu(a: CellCloud, b: CellCloud, p: TcmParams): Promise<Float32Array> {
  const { device, root, index, marks: pipeline, markParams } = await getCtx();
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const nA = a.xs.length;
  const nB = b.xs.length;
  // Over B's own extent as the GPU reads it (f32), cell = the mark radius.
  const grid = latticeFor(Float32Array.from(b.xs), Float32Array.from(b.ys), p.radius);

  const aPts = ensureF32(root, "aPts", 2 * Math.max(nA, 1));
  const bPts = ensureF32(root, "bPts", 2 * Math.max(nB, 1));
  const out = ensureF32(root, "marks", Math.max(nA, 1));

  writeView(device.queue, root.unwrap(aPts), packXY(a.xs, a.ys));
  writeView(device.queue, root.unwrap(bPts), packXY(b.xs, b.ys));
  // The index over B is built on the device, in the same submit, from the B points just uploaded.
  const enc = device.createCommandEncoder();
  const bIdx = encodeGridIndex(index, root.unwrap(bPts), nB, grid, enc, { keyPrefix: "tcm:b" });
  const bStart = wrapU32(root, bIdx.cellOffsets);
  const bItems = wrapU32(root, bIdx.pointIds);
  markParams.write({
    nA,
    cols: grid.cols,
    rows: grid.rows,
    pad: 0,
    minX: grid.minX,
    minY: grid.minY,
    cell: grid.cell,
    r2: p.radius * p.radius,
    expected: (b.xs.length / roiArea) * Math.PI * p.radius * p.radius,
    alpha: p.alpha,
  });

  const bind = root.unwrap(root.createBindGroup(markLayout, { params: markParams, aPts, bPts, bStart, bItems, marks: out }));
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  if (nA > 0) pass.dispatchWorkgroups(Math.ceil(nA / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await out.read()) as ArrayLike<number>;
  return Float32Array.from({ length: nA }, (_, i) => got[i]!);
}

/** GPU TCM Γ_ab(x) (eq 14) as a row-major `width×height` grid — the drop-in for `computeTcm`,
 *  validated against `computeTcmReference`. Both kernels run back to back in one submission; only
 *  the finished grid crosses back to the host. */
export async function computeTcmGpu(a: CellCloud, b: CellCloud, p: TcmParams): Promise<Float32Array> {
  const { device, root, index, marks: marksPipe, splat: splatPipe, markParams, splatParams } = await getCtx();
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const { width: w, height: h, sigma } = p;
  const nA = a.xs.length;
  const cw = (maxX - minX) / w;
  const ch = (maxY - minY) / h;
  const kr = Math.max(1, Math.ceil((3 * sigma) / Math.min(cw, ch)));

  // Grid over B at the mark radius; grid over A at the kernel's world reach. A point's window can
  // extend (kr + 1.5) cells past its own centre, so (kr + 2) cells is a safe bucket side — with it,
  // every A point that can reach an output cell is in that cell's 3×3 bucket neighbourhood.
  const nB = b.xs.length;
  const bGrid = latticeFor(Float32Array.from(b.xs), Float32Array.from(b.ys), p.radius);
  const aGrid = latticeFor(a.xs, a.ys, (kr + 2) * Math.max(cw, ch), p.bbox);

  const aPts = ensureF32(root, "aPts", 2 * Math.max(nA, 1));
  const bPts = ensureF32(root, "bPts", 2 * Math.max(nB, 1));
  const marks = ensureF32(root, "marks", Math.max(nA, 1));
  const out = ensureF32(root, "out", w * h);

  writeView(device.queue, root.unwrap(aPts), packXY(a.xs, a.ys));
  writeView(device.queue, root.unwrap(bPts), packXY(b.xs, b.ys));
  // Both indexes are built on the device in this one submit. Distinct `keyPrefix`es: the builds
  // share the scan pool, and under one prefix the second would overwrite the first's offsets.
  const enc = device.createCommandEncoder();
  const bIdx = encodeGridIndex(index, root.unwrap(bPts), nB, bGrid, enc, { keyPrefix: "tcm:b" });
  const aIdx = encodeGridIndex(index, root.unwrap(aPts), nA, aGrid, enc, { keyPrefix: "tcm:a" });
  const bStart = wrapU32(root, bIdx.cellOffsets);
  const bItems = wrapU32(root, bIdx.pointIds);
  const aStart = wrapU32(root, aIdx.cellOffsets);
  const aItems = wrapU32(root, aIdx.pointIds);

  markParams.write({
    nA,
    cols: bGrid.cols,
    rows: bGrid.rows,
    pad: 0,
    minX: bGrid.minX,
    minY: bGrid.minY,
    cell: bGrid.cell,
    r2: p.radius * p.radius,
    expected: (b.xs.length / roiArea) * Math.PI * p.radius * p.radius,
    alpha: p.alpha,
  });
  splatParams.write({
    w,
    h,
    cols: aGrid.cols,
    rows: aGrid.rows,
    minX,
    minY,
    cw,
    ch,
    kr,
    inv2s2: 1 / (2 * sigma * sigma),
    norm: 1 / (sigma * Math.sqrt(2 * Math.PI)),
    cell: aGrid.cell,
  });

  const markBind = root.unwrap(root.createBindGroup(markLayout, { params: markParams, aPts, bPts, bStart, bItems, marks }));
  const splatBind = root.unwrap(root.createBindGroup(splatLayout, { params: splatParams, aPts, marks, aStart, aItems, out }));
  const pass = enc.beginComputePass();
  pass.setPipeline(marksPipe);
  pass.setBindGroup(0, markBind);
  if (nA > 0) pass.dispatchWorkgroups(Math.ceil(nA / WG));
  pass.setPipeline(splatPipe);
  pass.setBindGroup(0, splatBind);
  pass.dispatchWorkgroups(Math.ceil((w * h) / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await out.read()) as ArrayLike<number>;
  return Float32Array.from({ length: w * h }, (_, i) => got[i]!);
}
