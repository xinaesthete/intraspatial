import { describe, expect, it } from "vite-plus/test";
import { getDevice, readBackBytes, writeView } from "../device";
import { fieldMaxResident } from "./fieldMax";
import { splatDensityGpu, splatDensityResident } from "./splatDensity";

// The resident path is what a deck layer drives: points already on the GPU, field left on the GPU.
// The host path (`splatDensityGpu`) is its oracle — same shader, same rasterisation — so these
// compare the two rather than re-deriving the KDE.

const bbox: [number, number, number, number] = [0, 0, 20, 20];
const W = 40;
const H = 40;
const SIGMA = 1.2;

async function upload(device: GPUDevice, data: Float32Array): Promise<GPUBuffer> {
  const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  writeView(device.queue, buf, data);
  return buf;
}

function fieldBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({ size: W * H * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
}

async function read(device: GPUDevice, buf: GPUBuffer): Promise<Float32Array> {
  return new Float32Array(await readBackBytes(device, "splatResident:test", buf, 0, W * H * 4));
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe("splatDensityResident", () => {
  it("honours per-point weights at stride 3, matching the host path", async () => {
    const device = await getDevice();
    const xs = [4, 10, 15.5];
    const ys = [5, 12, 3];
    const ws = [1, 2.5, 0.5];
    const xyw = new Float32Array(xs.flatMap((x, i) => [x, ys[i]!, ws[i]!]));
    const dst = fieldBuffer(device);
    await splatDensityResident(await upload(device, xyw), xs.length, dst, { width: W, height: H, sigma: SIGMA, bbox, stride: 3 });
    const host = await splatDensityGpu(xs, ys, { width: W, height: H, sigma: SIGMA, bbox, weights: ws });
    expect(maxAbsDiff(await read(device, dst), host.data)).toBeLessThan(1e-5);
  });

  it("keeps concurrent callers' fields apart", async () => {
    // Two layers updating in the same frame: both calls are in flight at once. Before the splat,
    // staging copy and de-pad shared one submit, the second call's splat could land in the shared
    // staging buffer before the first had de-padded it.
    const device = await getDevice();
    const a = new Float32Array([5, 5]);
    const b = new Float32Array([15, 15, 14, 16]);
    const dstA = fieldBuffer(device);
    const dstB = fieldBuffer(device);
    const opts = { width: W, height: H, sigma: SIGMA, bbox };
    // Upload first: an await between the two calls would let the first finish before the second
    // starts, and the test would pass whether or not the calls can interleave.
    const [pa, pb] = [await upload(device, a), await upload(device, b)];
    await Promise.all([splatDensityResident(pa, 1, dstA, opts), splatDensityResident(pb, 2, dstB, opts)]);
    const hostA = await splatDensityGpu([5], [5], opts);
    const hostB = await splatDensityGpu([15, 14], [15, 16], opts);
    const errA = maxAbsDiff(await read(device, dstA), hostA.data);
    const errB = maxAbsDiff(await read(device, dstB), hostB.data);
    expect(Math.max(errA, errB)).toBeLessThan(1e-5);
  });
});

describe("atomic accumulation (no float32-blendable)", () => {
  it("matches the blend path: same cells, same kernel, fixed-point instead of blending", async () => {
    // Forced here; on a device without float32-blendable (Firefox) it is chosen automatically.
    const device = await getDevice();
    const pts: number[] = [];
    let s = 7;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    for (let i = 0; i < 2000; i++) pts.push(2 + rnd() * 16, 2 + rnd() * 16, 0.25 + rnd() * 2.75);
    const xyw = new Float32Array(pts);
    const src = await upload(device, xyw);
    const opts = { width: W, height: H, sigma: SIGMA, bbox, stride: 3 as const, device };
    const blend = fieldBuffer(device);
    const atomic = fieldBuffer(device);
    await splatDensityResident(src, 2000, blend, { ...opts, accumulate: "blend" });
    await splatDensityResident(src, 2000, atomic, { ...opts, accumulate: "atomic", maxWeight: 3 });
    const a = await read(device, blend);
    const b = await read(device, atomic);
    let peak = 0;
    for (const v of a) peak = Math.max(peak, v);
    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < a.length; i++) {
      sumA += a[i]!;
      sumB += b[i]!;
    }
    // Mass agrees closely: neither path is biased (measured: 3e-6).
    expect(Math.abs(sumA - sumB) / sumA).toBeLessThan(1e-5);
    // Cells differ by unbiased f32 noise: blending rounds every add, in raster order, and the two
    // paths evaluate exp() in different shader stages. With ~2000 overlapping points per cell here
    // that is ~1e-4 of the peak (measured 8.2e-5; not the support edge — a 6σ support changes nothing).
    expect(maxAbsDiff(a, b) / peak).toBeLessThan(2e-4);
  });
});

describe("explicit device", () => {
  it("splats and reduces on the device it is handed", async () => {
    // What a deck layer does: pass its own device rather than relying on getDevice().
    const device = await getDevice();
    const xyw = new Float32Array([6, 7, 2, 12, 9, 1]);
    const dst = fieldBuffer(device);
    await splatDensityResident(await upload(device, xyw), 2, dst, { width: W, height: H, sigma: SIGMA, bbox, stride: 3, device });
    const max = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    await fieldMaxResident(dst, W * H, max, device);
    const field = await read(device, dst);
    const got = new Float32Array(await readBackBytes(device, "fieldMax:explicit", max, 0, 4))[0];
    expect(got).toBe(Math.max(...field));
  });
});

describe("fieldMaxResident", () => {
  it("leaves the buffer's maximum on the GPU as f32 bits, across many workgroups", async () => {
    const device = await getDevice();
    // Larger than one workgroup's stride so the grid-stride loop and the cross-group atomic both run.
    const n = 300_000;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = (i * 7919) % 1000;
    data[123_457] = 4321.5;
    const src = await upload(device, data);
    const dst = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    await fieldMaxResident(src, n, dst);
    const got = new Float32Array(await readBackBytes(device, "fieldMax:test", dst, 0, 4))[0];
    expect(got).toBe(4321.5);
  });
});
