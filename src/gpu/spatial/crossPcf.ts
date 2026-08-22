// GPU cross-PCF — the pair-counting half of `src/spatial/pcf.ts` moved onto the device.
//
// Both kernels are the same shape: one thread per anchor cell, which walks the 3×3 bucket
// neighbourhood of a uniform-grid index (cell size = the query radius, so every in-range
// neighbour is in those 9 buckets) and `atomicAdd`s into a small histogram. The index is the
// `src/spatial/bucketGrid.ts` offset list, built ON THE DEVICE by `gridIndex.ts` from the points
// already uploaded for the query pass, in the same submit — the host never sees `start`/`items`. The anchors are embarrassingly
// parallel; the histogram is tiny and shared, which is exactly what integer atomics are for. They
// are authored as WGSL templates rather than TGSL because that is where ADR-0003 puts kernels
// carrying `array<atomic<u32>>` — the TCM kernels next door, which need no atomics, are TGSL.
//
// **The counts are integers, so this is not an approximation of the CPU path — it is the same
// arithmetic.** The one place the two can disagree is bin/radius CLASSIFICATION: the GPU tests
// `d² < r²` and floors `√d²/dr` in f32 where the CPU does it in f64, so a pair sitting within a
// float ulp of a bin edge can land on either side. Measure-zero for real coordinates; the parity
// tests state the tolerance rather than pretending it is bit-exact.
//
// Normalisation (ρ_B, annulus areas, g) stays on the host in f64: it is O(nBins) or O(N²) over
// type counts, i.e. free, and keeping it there means the GPU and CPU paths share one definition
// of the statistic rather than two.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { type GridLattice, latticeFor } from "../../spatial/bucketGrid";
import type { LabelledCells, PcfMatrixParams, PcfMatrixResult, PcfParams, PcfResult } from "../../spatial/pcf";
import type { CellCloud } from "../../spatial/tcm";
import { checkBindingSize, getDevice, sized } from "../device";
import { encodeGridIndex, type GridIndexCtx, getGridIndexCtx } from "./gridIndex";

const WG = 64;

// Uniforms are all-f32 (cast to u32 in the shader) so the block is one flat Float32Array with no
// std140 padding puzzles — the same trick `splatDensity`'s uniform uses.
const UNI_FLOATS = 12; // 48 bytes, a multiple of 16

const HIST_SHADER = /* wgsl */ `
struct Uni {
  nA: f32, cols: f32, rows: f32, nBins: f32,
  minX: f32, minY: f32, cell: f32, rMax2: f32,
  invDr: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> aPts: array<f32>;        // [x0,y0,x1,y1,...]
@group(0) @binding(2) var<storage, read> bPts: array<f32>;
@group(0) @binding(3) var<storage, read> start: array<u32>;       // grid-index offsets over B
@group(0) @binding(4) var<storage, read> items: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(${WG})
fn pcfHist(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(U.nA)) { return; }
  let ax = aPts[2u * i];
  let ay = aPts[2u * i + 1u];
  let cols = i32(U.cols);
  let rows = i32(U.rows);
  let nBins = u32(U.nBins);
  let c0 = clamp(i32(floor((ax - U.minX) / U.cell)), 0, cols - 1);
  let r0 = clamp(i32(floor((ay - U.minY) / U.cell)), 0, rows - 1);
  for (var dr = -1; dr <= 1; dr = dr + 1) {
    let rr = r0 + dr;
    if (rr < 0 || rr >= rows) { continue; }
    for (var dc = -1; dc <= 1; dc = dc + 1) {
      let cc = c0 + dc;
      if (cc < 0 || cc >= cols) { continue; }
      let b = u32(rr * cols + cc);
      let lo = start[b];
      let hi = start[b + 1u];
      for (var k = lo; k < hi; k = k + 1u) {
        let j = items[k];
        let dx = bPts[2u * j] - ax;
        let dy = bPts[2u * j + 1u] - ay;
        let d2 = dx * dx + dy * dy;
        if (d2 < U.rMax2) {
          let bin = min(nBins - 1u, u32(sqrt(d2) * U.invDr));
          atomicAdd(&counts[bin], 1u);
        }
      }
    }
  }
}
`;

const MATRIX_SHADER = /* wgsl */ `
struct Uni {
  n: f32, cols: f32, rows: f32, nTypes: f32,
  minX: f32, minY: f32, cell: f32, r2: f32,
  pad0: f32, pad1: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> typeIdx: array<u32>;     // dense type index per cell
@group(0) @binding(3) var<storage, read> start: array<u32>;
@group(0) @binding(4) var<storage, read> items: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;  // N*N, row-major

@compute @workgroup_size(${WG})
fn pcfMatrix(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(U.n)) { return; }
  let ax = pts[2u * i];
  let ay = pts[2u * i + 1u];
  let a = typeIdx[i];
  let N = u32(U.nTypes);
  let cols = i32(U.cols);
  let rows = i32(U.rows);
  let c0 = clamp(i32(floor((ax - U.minX) / U.cell)), 0, cols - 1);
  let r0 = clamp(i32(floor((ay - U.minY) / U.cell)), 0, rows - 1);
  for (var dr = -1; dr <= 1; dr = dr + 1) {
    let rr = r0 + dr;
    if (rr < 0 || rr >= rows) { continue; }
    for (var dc = -1; dc <= 1; dc = dc + 1) {
      let cc = c0 + dc;
      if (cc < 0 || cc >= cols) { continue; }
      let b = u32(rr * cols + cc);
      let lo = start[b];
      let hi = start[b + 1u];
      for (var k = lo; k < hi; k = k + 1u) {
        let j = items[k];
        if (j == i) { continue; }                     // exclude self-pairs
        let dx = pts[2u * j] - ax;
        let dy = pts[2u * j + 1u] - ay;
        if (dx * dx + dy * dy < U.r2) {
          atomicAdd(&counts[a * N + typeIdx[j]], 1u);
        }
      }
    }
  }
}
`;

type Root = ReturnType<typeof tgpu.initFromDevice>;

interface Ctx {
  device: GPUDevice;
  root: Root;
  index: GridIndexCtx;
  hist: GPUComputePipeline;
  matrix: GPUComputePipeline;
}
let ctxCache: Promise<Ctx> | undefined;

function getCtx(): Promise<Ctx> {
  ctxCache ??= (async () => {
    const device = await getDevice();
    const root = tgpu.initFromDevice({ device });
    const index = await getGridIndexCtx();
    const mk = (code: string, entryPoint: string) =>
      device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint } });
    return { device, root, index, hist: mk(HIST_SHADER, "pcfHist"), matrix: mk(MATRIX_SHADER, "pcfMatrix") };
  })();
  return ctxCache;
}

// Pooled, grow-only buffers (never `.destroy()`d — that segfaults Dawn-on-Node's teardown).
interface Slot {
  buf: GPUBuffer;
  cap: number;
}
const pool = new Map<string, Slot>();

function ensure(device: GPUDevice, key: string, floats: number, extraUsage = 0): GPUBuffer {
  const got = pool.get(key);
  if (got && got.cap >= floats) return got.buf;
  const cap = Math.max(floats, (got?.cap ?? 0) * 2, 4);
  const buf = device.createBuffer({
    size: cap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
  });
  pool.set(key, { buf, cap });
  return buf;
}

// The counts buffer is read back, so it needs a TypeGPU wrapper: `.read()` is the only
// Dawn-on-Node-stable readback in this project (a raw `mapAsync` on a pooled buffer crashed the
// vitest worker on teardown).
let countsRaw: GPUBuffer | undefined;
let countsWrap: ReturnType<Root["createBuffer"]> | undefined;
let countsCap = 0;

function ensureCounts(device: GPUDevice, root: Root, n: number) {
  if (countsWrap && countsCap >= n) return;
  countsCap = Math.max(n, countsCap * 2, 4);
  countsRaw = device.createBuffer({
    size: countsCap * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  countsWrap = root.createBuffer(d.arrayOf(d.u32, countsCap), countsRaw);
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

/** Which of the two uploaded point sets the index is built over: the histogram kernel indexes
 *  B and walks it from A; the matrix kernel indexes the one cloud it has. */
type IndexOver = "a" | "b";

async function runCounts(
  pipeline: GPUComputePipeline,
  uni: Float32Array,
  ptsA: Float32Array,
  second: Float32Array | Uint32Array,
  index: { over: IndexOver; n: number; lattice: GridLattice },
  threads: number,
  nCounts: number,
): Promise<Uint32Array> {
  const { device, root, index: indexCtx } = await getCtx();
  const uniBuf = ensure(device, "uni", UNI_FLOATS, GPUBufferUsage.UNIFORM);
  const aBuf = ensure(device, "a", ptsA.length);
  const bBuf = ensure(device, "b", second.length);
  ensureCounts(device, root, nCounts);
  // Every buffer here is one entry (or two floats) per point, so they cross together at ~16.8M
  // points. Throw rather than let an over-large binding turn the pass into a silent no-op that
  // returns the previous call's histogram. (The index build checks its own bindings.)
  const widest = Math.max(ptsA.length, second.length, nCounts);
  checkBindingSize(device, `crossPcf: ${ptsA.length / 2} points`, widest * 4);

  device.queue.writeBuffer(uniBuf, 0, uni);
  device.queue.writeBuffer(aBuf, 0, ptsA);
  device.queue.writeBuffer(bBuf, 0, second);
  device.queue.writeBuffer(countsRaw!, 0, new Uint32Array(countsCap)); // clear

  // Index build and query in one command buffer: the pass ordering inside a submit is what
  // makes `start`/`items` complete before the query pass reads them.
  const enc = device.createCommandEncoder();
  const grid = encodeGridIndex(indexCtx, index.over === "a" ? aBuf : bBuf, index.n, index.lattice, enc, { keyPrefix: "crossPcf:index" });

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf, size: UNI_FLOATS * 4 } },
      // Sized to THIS call, not to the pooled buffer: the pool doubles on growth, and a
      // binding past maxStorageBufferBindingSize is a silent no-op returning the previous
      // call's histogram. See `sized` in `../device.ts`.
      { binding: 1, resource: sized(aBuf, ptsA.length * 4) },
      { binding: 2, resource: sized(bBuf, second.length * 4) },
      { binding: 3, resource: sized(grid.start, (grid.M + 1) * 4) },
      { binding: 4, resource: sized(grid.items, Math.max(grid.n, 1) * 4) },
      { binding: 5, resource: sized(countsRaw!, nCounts * 4) },
    ],
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  if (threads > 0) pass.dispatchWorkgroups(Math.ceil(threads / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await countsWrap!.read()) as ArrayLike<number>;
  const out = new Uint32Array(nCounts);
  for (let i = 0; i < nCounts; i++) out[i] = got[i]!;
  return out;
}

/** GPU `crossPCF` — same statistic, same parameters, same result shape as the CPU
 *  `src/spatial/pcf.ts` version (eq 8, Mode 1: global ρ_B, full-annulus area). */
export async function crossPCFGpu(a: CellCloud, b: CellCloud, p: PcfParams): Promise<PcfResult> {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const rhoB = b.xs.length / roiArea;
  const dr = p.rMax / p.nBins;
  const nA = a.xs.length;

  const bPts = packXY(b.xs, b.ys);
  // Over B's own extent (as read by the GPU, i.e. f32), cell = rMax.
  const grid = latticeFor(Float32Array.from(b.xs), Float32Array.from(b.ys), p.rMax);
  const uni = new Float32Array(UNI_FLOATS);
  uni.set([nA, grid.cols, grid.rows, p.nBins, grid.minX, grid.minY, grid.cell, p.rMax * p.rMax, 1 / dr]);
  const index = { over: "b" as const, n: b.xs.length, lattice: grid };
  const counts = await runCounts((await getCtx()).hist, uni, packXY(a.xs, a.ys), bPts, index, nA, p.nBins);

  const r: number[] = [];
  const g: number[] = [];
  const out: number[] = [];
  for (let k = 0; k < p.nBins; k++) {
    const r0 = k * dr;
    const r1 = (k + 1) * dr;
    const annulus = Math.PI * (r1 * r1 - r0 * r0);
    const expected = nA * rhoB * annulus;
    r.push((r0 + r1) / 2);
    g.push(expected > 0 ? counts[k]! / expected : 0);
    out.push(counts[k]!);
  }
  return { r, g, counts: out };
}

/** GPU `crossPCFMatrix` — all N² ordered type pairs from one batched pass over every cell.
 *  This is the hover-linked matrix in the cell-stats demo: the whole N² statistic for the cost of
 *  one pass, not N² passes. */
export async function crossPCFMatrixGpu(cells: LabelledCells, p: PcfMatrixParams): Promise<PcfMatrixResult> {
  const [minX, minY, maxX, maxY] = p.bbox;
  const roiArea = Math.max((maxX - minX) * (maxY - minY), 1e-12);
  const n = cells.xs.length;
  const r = p.radius;

  const types = [...new Set(cells.typeId)].sort((x, y) => x - y);
  const idx = new Map<number, number>();
  types.forEach((t, i) => {
    idx.set(t, i);
  });
  const N = types.length;
  const nPer = new Array<number>(N).fill(0);
  const ti = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const k = idx.get(cells.typeId[i]!)!;
    ti[i] = k;
    nPer[k]!++;
  }

  const grid = latticeFor(cells.xs, cells.ys, r, p.bbox);
  const uni = new Float32Array(UNI_FLOATS);
  uni.set([n, grid.cols, grid.rows, N, grid.minX, grid.minY, grid.cell, r * r]);
  const index = { over: "a" as const, n, lattice: grid };
  const counts = await runCounts((await getCtx()).matrix, uni, packXY(cells.xs, cells.ys), ti, index, n, N * N);

  const diskArea = Math.PI * r * r;
  const g = new Float64Array(N * N);
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const expected = nPer[a]! * (nPer[b]! / roiArea) * diskArea;
      g[a * N + b] = expected > 0 ? counts[a * N + b]! / expected : 0;
    }
  }
  return { types, counts: nPer, g };
}
