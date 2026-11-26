// `TcmRenderer`: TCMs that coexist, A and B as weightings of one point set, outputs a caller owns,
// and a Γ window smaller than the data. `tcmRender.gpu.test.ts` pins the statistic itself (oracle
// parity); this file pins what a deck layer relies on.
import { describe, expect, it } from "vite-plus/test";
import { QUARTIC } from "#src/spatial/kernels";
import type { CellCloud } from "#src/spatial/tcm";
import { getDevice, readBackBytes } from "../device";
import { type ResidentPoints, TcmRenderer, tcmRowFloats } from "./tcmRender";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cloud(n: number, seed: number): CellCloud {
  const rnd = mulberry32(seed);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(2 + rnd() * 16);
    ys.push(2 + rnd() * 16);
  }
  return { xs, ys };
}

const BBOX = [0, 0, 20, 20] as const;
const P = { width: 64, height: 64, radius: 3, sigma: 1.5, alpha: 5, kernel: QUARTIC, bbox: BBOX } as const;

function storage(device: GPUDevice, data: Float32Array, label: string): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size: Math.max(16, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

function peak(a: Float32Array): number {
  let m = 0;
  for (const v of a) m = Math.max(m, Math.abs(v));
  return m;
}

describe("TcmRenderer", () => {
  it("keeps its own outputs: two renderers' fields coexist", async () => {
    const device = await getDevice();
    const r1 = new TcmRenderer(device, "test:r1");
    const r2 = new TcmRenderer(device, "test:r2");
    const a = cloud(120, 0x51);
    const b = cloud(150, 0x9e);
    const c = cloud(90, 0x33);
    // Render both before reading either: with module-level pooled textures, the second render
    // overwrote the first's.
    const t1 = r1.render(a, b, P);
    const t2 = r2.render(b, c, P);
    const [g1, g2] = await Promise.all([r1.read(t1), r2.read(t2)]);
    const solo1 = await r1.read(r1.render(a, b, P));
    const solo2 = await r2.read(r2.render(b, c, P));
    expect(peak(g1)).toBeGreaterThan(0);
    expect(maxAbsDiff(g1, solo1)).toBe(0);
    expect(maxAbsDiff(g2, solo2)).toBe(0);
    expect(maxAbsDiff(g1, g2)).toBeGreaterThan(0);
  });

  it("A and B as 0/1 weightings of one point set give the TCM of the two subsets", async () => {
    const device = await getDevice();
    const r = new TcmRenderer(device, "test:weights");
    const a = cloud(120, 0x51);
    const b = cloud(150, 0x9e);
    const n = a.xs.length + b.xs.length;
    // One point set: A's cells then B's, interleaved with a third type both weightings leave out.
    const xy = new Float32Array(2 * (n + 40));
    const wA = new Float32Array(n + 40);
    const wB = new Float32Array(n + 40);
    const other = cloud(40, 0x77);
    let k = 0;
    const put = (x: number, y: number, inA: boolean, inB: boolean) => {
      xy[2 * k] = x;
      xy[2 * k + 1] = y;
      wA[k] = inA ? 1 : 0;
      wB[k] = inB ? 1 : 0;
      k++;
    };
    for (let i = 0; i < 40; i++) put(other.xs[i]!, other.ys[i]!, false, false);
    for (let i = 0; i < a.xs.length; i++) put(a.xs[i]!, a.ys[i]!, true, false);
    for (let i = 0; i < b.xs.length; i++) put(b.xs[i]!, b.ys[i]!, false, true);
    const points = storage(device, xy, "test:xy");
    const A: ResidentPoints = { points, weights: storage(device, wA, "test:wA"), count: k };
    const B: ResidentPoints = { points, weights: storage(device, wB, "test:wB"), count: k };
    const rhoB = b.xs.length / (20 * 20);
    const grid = { bbox: BBOX, width: P.width, height: P.height };
    const weighted = await r.read(r.renderResident({ a: A, b: B, rhoB }, { ...P, grid }));
    const clouds = await r.read(r.render(a, b, P));
    expect(peak(clouds)).toBeGreaterThan(0);
    // Same points, same order of blending within each subset: equal to f32 rounding.
    expect(maxAbsDiff(weighted, clouds) / peak(clouds)).toBeLessThan(1e-5);
  });

  it("copies Γ and max |Γ| into caller-owned buffers in the same submit", async () => {
    const device = await getDevice();
    const r = new TcmRenderer(device, "test:out");
    const w = 50; // not a multiple of 64, so the rows are padded
    const h = 40;
    const rowFloats = tcmRowFloats(w);
    expect(rowFloats).toBe(64);
    const out = device.createBuffer({
      label: "test:field",
      size: rowFloats * h * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const max = device.createBuffer({
      label: "test:max",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // A stale value in `max` must not survive: it is cleared before the reduction.
    device.queue.writeBuffer(max, 0, new Float32Array([1e9]));
    const params = { ...P, width: w, height: h };
    const tex = r.render(cloud(120, 0x51), cloud(150, 0x9e), params, { buffer: out, max });
    const fromTexture = await r.read(tex);
    const padded = new Float32Array(await readBackBytes(device, "test:field", out, 0, rowFloats * h * 4));
    const maxBits = new Float32Array(await readBackBytes(device, "test:max", max, 0, 4));
    // `read` flips to row 0 at minY; the buffer keeps the texture's row 0 at maxY.
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        expect(padded[row * rowFloats + col]).toBe(fromTexture[(h - 1 - row) * w + col]);
      }
    }
    expect(maxBits[0]).toBe(peak(fromTexture));
    expect(maxBits[0]).toBeGreaterThan(0);
  });

  it("renders a window of Γ: a mark grid padded by the Γ support gives the full field's crop", async () => {
    const device = await getDevice();
    const r = new TcmRenderer(device, "test:window");
    const a = cloud(160, 0x51);
    const b = cloud(200, 0x9e);
    const cell = 20 / 80;
    const full = await r.read(r.render(a, b, { ...P, width: 80, height: 80 }));
    // The window [6, 8]–[12, 14], at the same cell, with the marks over the window padded by
    // σ·radiusSigma (= 6) and clipped to the data, also on the same lattice.
    const win = [6, 8, 12, 14] as const;
    const ww = (win[2] - win[0]) / cell;
    const wh = (win[3] - win[1]) / cell;
    const markBox = [0, 2, 18, 20] as const;
    const B: ResidentPoints = {
      points: storage(device, pack(b), "test:b"),
      weights: storage(device, ones(b.xs.length), "test:bw"),
      count: b.xs.length,
    };
    const A: ResidentPoints = {
      points: storage(device, pack(a), "test:a"),
      weights: storage(device, ones(a.xs.length), "test:aw"),
      count: a.xs.length,
    };
    const part = await r.read(
      r.renderResident(
        { a: A, b: B, rhoB: b.xs.length / 400 },
        {
          ...P,
          grid: { bbox: win, width: ww, height: wh },
          markGrid: { bbox: markBox, width: (markBox[2] - markBox[0]) / cell, height: (markBox[3] - markBox[1]) / cell },
        },
      ),
    );
    const scale = peak(full);
    let err = 0;
    for (let row = 0; row < wh; row++) {
      for (let col = 0; col < ww; col++) {
        const fx = col + win[0] / cell;
        const fy = row + win[1] / cell;
        err = Math.max(err, Math.abs(part[row * ww + col]! - full[fy * 80 + fx]!) / scale);
      }
    }
    expect(scale).toBeGreaterThan(0);
    expect(err).toBeLessThan(1e-4);
  });

  it("refuses work after destroy()", async () => {
    const r = new TcmRenderer(await getDevice(), "test:destroyed");
    await r.read(r.render(cloud(20, 1), cloud(20, 2), P));
    r.destroy();
    r.destroy(); // idempotent
    expect(() => r.render(cloud(20, 1), cloud(20, 2), P)).toThrow(/after destroy/);
  });
});

function pack(c: CellCloud): Float32Array {
  const out = new Float32Array(2 * c.xs.length);
  for (let i = 0; i < c.xs.length; i++) {
    out[2 * i] = c.xs[i]!;
    out[2 * i + 1] = c.ys[i]!;
  }
  return out;
}

function ones(n: number): Float32Array {
  return new Float32Array(n).fill(1);
}
