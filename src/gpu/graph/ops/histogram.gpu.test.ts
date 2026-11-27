import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice, readBackBytes } from "#src/gpu/device";
import { gramMatrixGpu } from "#src/gpu/spatial/gramMatrix";
import { histogramGpu, residentRasterPlane } from "#src/gpu/spatial/histogram";
import { channelsFromLabels } from "#src/spatial/gram";
import { type HistogramResult, histogramCpu, histogramRow, histogramSummary } from "#src/spatial/histogram";
import { EPANECHNIKOV } from "#src/spatial/kernels";
import type { FieldValue } from "../handle";
import { Graph, pull, rasterPlaneValue } from "../index";

// GPU vs the f64 golden (`histogramCpu`): counts, CDF, totals and edges EXACTLY (the bin rule is
// shared bit for bit — see src/spatial/histogram.ts), moments to f32 rounding. Sizes stay small:
// CPU work in a Dawn fork is what kills it (gramMatrix.gpu.test.ts), and every comparison is one
// aggregated `expect`, never a per-element loop.

const W = 67; // not a multiple of anything, so the last workgroup is partial
const H = 45;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

function mismatch(a: ArrayLike<number>, b: ArrayLike<number>, tol = 0): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    if (x === y || (Number.isNaN(x) && Number.isNaN(y))) continue;
    if (!(Math.abs(x - y) <= tol * Math.max(1, Math.abs(y)))) return `index ${i}: got ${x}, want ${y}`;
  }
  return undefined;
}

/** Every difference between a GPU result and the golden, or undefined. */
function diff(gpu: Omit<HistogramResult, "bins" | "rows">, cpu: HistogramResult): string | undefined {
  return (
    mismatch([gpu.lo, gpu.hi], [cpu.lo, cpu.hi]) ??
    mismatch(gpu.counts, cpu.counts) ??
    mismatch(gpu.cdf, cpu.cdf) ??
    mismatch(gpu.totals, cpu.totals) ??
    mismatch(gpu.moments, cpu.moments, 1e-5)
  );
}

async function runOp(g: Graph, op: string, inputs: Record<string, ReturnType<Graph["grid"]>>, params = {}) {
  const outs = g.op(op, inputs, params);
  const get = async (i: number, mode?: "cpu") => (await pull(g, outs[i]!, mode ? { mode } : {})).data!;
  const read = async (mode?: "cpu") => {
    const [counts, cdf, edges, totals, moments] = await Promise.all([0, 1, 2, 3, 4].map((i) => get(i, mode)));
    return {
      lo: edges![0]!,
      hi: edges![1]!,
      counts: counts as Uint32Array,
      cdf: cdf as Uint32Array,
      totals: totals as Uint32Array,
      moments: Float64Array.from(moments!),
    };
  };
  return { gpu: await read(), cpu: await read("cpu"), shape: (await pull(g, outs[0]!)).shape };
}

/** Signed, skewed values with NaN / ±Inf sprinkled in. */
function signedField(seed: number): Float32Array {
  const rnd = lcg(seed);
  const v = new Float32Array(W * H);
  for (let i = 0; i < v.length; i++) v[i] = Math.log(rnd() + 1e-4) * 1.7 + 2.5 + rnd();
  v[3] = Number.NaN;
  v[100] = Infinity;
  v[101] = -Infinity;
  v[W * H - 1] = Number.NaN;
  return v;
}

const f16ToF32 = (h: number): number => {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 31;
  const m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? Number.NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
};

/** Plane `k` of a ResidentRasters-layout word buffer, unpadded, as host floats. */
function decodePlane(words: Uint32Array, k: number, w: number, h: number, rowWords: number, half: boolean): Float32Array {
  const out = new Float32Array(w * h);
  const f32 = new Float32Array(words.buffer, words.byteOffset, words.length);
  for (let y = 0; y < h; y++) {
    const base = (k * h + y) * rowWords;
    for (let x = 0; x < w; x++) {
      out[y * w + x] = half ? f16ToF32((words[base + (x >> 1)]! >>> (x & 1 ? 16 : 0)) & 0xffff) : f32[base + x]!;
    }
  }
  return out;
}

describe("histogram graph op", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("matches the golden on signed values with NaN/±Inf, and the summary agrees", async () => {
    const field = signedField(1);
    const g = new Graph();
    const { gpu, cpu, shape } = await runOp(g, "histogram", { field: g.grid(field, W, H) }, { bins: 64 });
    expect(shape).toEqual({ kind: "matrix", rows: 1, cols: 64 });
    expect(diff(gpu, { ...cpu, bins: 64, rows: 1 })).toBeUndefined();
    expect(gpu.lo).toBeLessThan(0);
    expect(Array.from(gpu.totals)).toEqual([W * H - 4, 0, 0, 4]);
    expect(gpu.cdf[63]).toBe(W * H - 4);
    const s = histogramSummary(histogramRow({ ...gpu, bins: 64 }));
    const c = histogramSummary(histogramRow({ ...cpu, bins: 64 }));
    expect(mismatch([s.p10, s.q1, s.median, s.q3, s.p90, s.iqr], [c.p10, c.q1, c.median, c.q3, c.p90, c.iqr], 1e-6)).toBeUndefined();
  });

  it("constant field: zero range becomes [v-0.5, v+0.5], everything in the middle bin", async () => {
    const g = new Graph();
    const { gpu, cpu } = await runOp(g, "histogram", { field: g.grid(new Float32Array(W * H).fill(-4.25), W, H) }, { bins: 8 });
    expect(diff(gpu, { ...cpu, bins: 8, rows: 1 })).toBeUndefined();
    expect([gpu.lo, gpu.hi]).toEqual([-4.75, -3.75]);
    expect(gpu.counts[4]).toBe(W * H);
    expect(histogramSummary(histogramRow({ ...gpu, bins: 8 })).median).toBe(-4.25);
  });

  it("fixed range: under/over counted per row, not binned", async () => {
    const g = new Graph();
    const { gpu, cpu } = await runOp(g, "histogram", { field: g.grid(signedField(2), W, H) }, { bins: 32, range: "fixed", lo: 0, hi: 3 });
    expect(diff(gpu, { ...cpu, bins: 32, rows: 1 })).toBeUndefined();
    expect([gpu.lo, gpu.hi]).toEqual([0, 3]);
    expect(gpu.totals[1]).toBeGreaterThan(0);
    expect(gpu.totals[2]).toBeGreaterThan(0);
  });

  it("masked: inside and outside rows from one dispatch, u32 and f32 masks", async () => {
    const field = signedField(3);
    const u = new Uint32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) u[y * W + x] = (x - 30) ** 2 + (y - 20) ** 2 < 300 ? 1 : 0;
    for (const dtype of ["u32", "f32"] as const) {
      const g = new Graph();
      const data = dtype === "u32" ? u : Float32Array.from(u);
      const mask: FieldValue = { shape: { kind: "grid", width: W, height: H }, dtype, data };
      const { gpu, cpu, shape } = await runOp(g, "histogramMasked", { field: g.grid(field, W, H), mask: g.source(mask) }, { bins: 48 });
      expect(shape).toEqual({ kind: "matrix", rows: 2, cols: 48 });
      expect(diff(gpu, { ...cpu, bins: 48, rows: 2 }), dtype).toBeUndefined();
      const inside = gpu.cdf[47]!;
      expect(inside).toBeGreaterThan(100);
      expect(inside + gpu.cdf[95]! + gpu.totals[3]! + gpu.totals[7]!).toBe(W * H);
    }
  });

  it("extreme values: a constant 2^24 field and ranges near f32 max still bin, and match", async () => {
    const big = new Float32Array(W * H);
    for (let i = 0; i < big.length; i++) big[i] = (i % 3) * 1e38 - (i % 5 === 0 ? 2e38 : 0);
    for (const [label, field] of [
      ["constant 2^24", new Float32Array(W * H).fill(2 ** 24)],
      ["near f32 max", big],
    ] as const) {
      const g = new Graph();
      const { gpu, cpu } = await runOp(g, "histogram", { field: g.grid(field, W, H) }, { bins: 16 });
      // Near f32 max the GPU's f32 SUM overflows to Inf (documented on histogramResident), so the
      // moments are compared for the constant case only; counts, CDF, totals and edges always.
      const moments = label === "near f32 max" ? cpu.moments : gpu.moments;
      expect(diff({ ...gpu, moments }, { ...cpu, bins: 16, rows: 1 }), label).toBeUndefined();
      expect(gpu.cdf[15], label).toBe(W * H);
    }
  });

  it("past the workgroup-private cap, bins merge globally and still match", async () => {
    const g = new Graph();
    const { gpu, cpu } = await runOp(g, "histogram", { field: g.grid(signedField(4), W, H) }, { bins: 5000 });
    expect(diff(gpu, { ...cpu, bins: 5000, rows: 1 })).toBeUndefined();
  });

  it("an f16 ResidentRasters plane with padded rows (synthetic, NaN/Inf/negative/subnormal texels)", async () => {
    const device = await getDevice();
    const K = 3;
    const w = 50; // f16 row = 100 B -> padded to 256 B = 64 words, 25 of them used
    const h = 21;
    const rowWords = 64;
    const words = new Uint32Array(K * h * rowWords).fill(0x7e007e00); // padding = NaN: must be skipped
    const rnd = lcg(5);
    for (let k = 0; k < K; k++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x += 2) {
          const bits = () => Math.floor(rnd() * 0x10000);
          words[(k * h + y) * rowWords + (x >> 1)] = (bits() | (bits() << 16)) >>> 0;
        }
      }
    }
    const buffer = device.createBuffer({
      label: "histogram-test:rasters",
      size: words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, words);
    const resident = { buffer, rowWords, precision: "f16" as const, mean: new Float64Array(K), sd: new Float64Array(K) };
    const plane = residentRasterPlane(resident, 1, w, h);
    const values = decodePlane(words, 1, w, h, rowWords, true);
    const cpu = histogramCpu(values, { bins: 128 });
    expect(cpu.totals[3]).toBeGreaterThan(0);

    expect(diff(await histogramGpu(plane, { bins: 128 }), cpu)).toBeUndefined();
    // And through the graph, as a raster-plane source.
    const g = new Graph();
    const outs = g.op("histogram", { field: g.source(rasterPlaneValue(plane)) }, { bins: 128 });
    expect(mismatch((await pull(g, outs[0]!)).data!, cpu.counts)).toBeUndefined();
  });

  it("a real gramMatrixGpu f16 plane", async () => {
    const device = await getDevice();
    const xs: number[] = [];
    const ys: number[] = [];
    const ids: number[] = [];
    for (let i = 0; i < 300; i++) {
      xs.push(20 + ((i * 37) % 160));
      ys.push(20 + ((i * 53) % 160));
      ids.push(i % 2);
    }
    const res = await gramMatrixGpu(channelsFromLabels(xs, ys, ids), {
      bbox: [0, 0, 200, 200],
      width: 70,
      height: 40,
      radius: 12,
      kernel: EPANECHNIKOV,
      precision: "f16",
    });
    const r = res.resident;
    expect(r.rowWords * 2).toBeGreaterThan(70); // padded
    const words = new Uint32Array(await readBackBytes(device, "histogram-test:gram", r.buffer, 0, 2 * 40 * r.rowWords * 4));
    const cpu = histogramCpu(decodePlane(words, 1, 70, 40, r.rowWords, true), { bins: 256 });
    const gpu = await histogramGpu(residentRasterPlane(r, 1, 70, 40), { bins: 256 });
    expect(diff(gpu, cpu)).toBeUndefined();
    expect(cpu.hi).toBeGreaterThan(cpu.lo);
  });
});
