import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import { Graph, pull } from "#src/gpu/graph/index";
import { patchSupportCpu } from "#src/gpu/graph/ops/patchSupport";
import { maskCountsInRects, maskRuns, patchGrid, rectsArray } from "#src/spatial/patchGrid";
import { patchSupportGpu } from "./patchSupport";

// GPU support counts must equal the host run counts EXACTLY (both are integer sums), for every
// encoding, on a canvas whose width is not a multiple of anything, with rectangles touching
// the right edge and the last row (the P[n] = total read). One aggregated `expect` per result.

const W = 97;
const H = 61;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

function blobs(seed: number): Uint8Array {
  const rnd = lcg(seed);
  const m = new Uint8Array(W * H);
  for (let k = 0; k < 15; k++) {
    const cx = rnd() * W;
    const cy = rnd() * H;
    const r = 2 + rnd() * 14;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x - cx) ** 2 + (y - cy) ** 2 < r * r) m[y * W + x] = 1;
  }
  return m;
}

function randomRects(seed: number, n: number): Uint32Array {
  const rnd = lcg(seed);
  const out = new Uint32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x0 = Math.floor(rnd() * W);
    const y0 = Math.floor(rnd() * H);
    out.set([x0, y0, x0 + Math.floor(rnd() * (W - x0 + 1)), y0 + Math.floor(rnd() * (H - y0 + 1))], i * 4);
  }
  // The whole canvas, the last pixel, and an empty rectangle.
  out.set([0, 0, W, H, W - 1, H - 1, W, H, 5, 5, 5, 9], (n - 3) * 4);
  return out;
}

function firstMismatch(a: ArrayLike<number>, b: ArrayLike<number>): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return `rect ${i}: got ${a[i]}, want ${b[i]}`;
  return undefined;
}

describe("patchSupportGpu", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("u8, u32 and f32 masks give the run counts exactly", async () => {
    const m = blobs(3);
    const rects = randomRects(11, 400);
    const boxes = Array.from({ length: rects.length / 4 }, (_, i) => ({
      x0: rects[i * 4]!,
      y0: rects[i * 4 + 1]!,
      x1: rects[i * 4 + 2]!,
      y1: rects[i * 4 + 3]!,
    }));
    const want = Uint32Array.from(maskCountsInRects(maskRuns(m, W, H), boxes));
    // f32: NaN, +Inf and negatives are not support; a W*H not divisible by 4 exercises the u8 tail.
    const f = Float32Array.from(m, (v, i) => (v ? 0.25 + (i % 7) : i % 5 === 0 ? Number.NaN : i % 5 === 1 ? -1 : 0));
    f[0] = Infinity;
    const wantF = patchSupportCpu(f, W, "f32", rects);
    const got = {
      u8: await patchSupportGpu(m, W, H, rects),
      u32: await patchSupportGpu(
        Uint32Array.from(m, (v) => v * 7),
        W,
        H,
        rects,
      ),
      f32: await patchSupportGpu(f, W, H, rects),
    };
    expect(firstMismatch(got.u8, want)).toBeUndefined();
    expect(firstMismatch(got.u32, want)).toBeUndefined();
    expect(firstMismatch(got.f32, wantF)).toBeUndefined();
    expect(want[want.length - 3]).toBe(m.reduce((s, v) => s + v, 0));
  });

  it("concurrent host calls on one device each count their own mask", async () => {
    // The host form pools its mask, rectangle and staging buffers per device: without its queue, a
    // second call uploads while the first awaits, and the first counts the second's mask.
    const rects = randomRects(4, 120);
    const masks = [blobs(21), blobs(22), blobs(23), blobs(24)];
    const got = await Promise.all(masks.map((m) => patchSupportGpu(m, W, H, rects)));
    const want = masks.map((m) => patchSupportCpu(m, W, "u32", rects));
    expect(got.map((g, i) => firstMismatch(g, want[i]!))).toEqual(masks.map(() => undefined));
    // A call that fails inside the queue (a canvas past the 33.5 Mpx flags binding) does not stall it.
    const bigW = 6000;
    const bigH = 5600;
    await expect(patchSupportGpu(new Uint8Array(bigW * bigH), bigW, bigH, Uint32Array.of(0, 0, 1, 1))).rejects.toThrow(
      /maxStorageBufferBindingSize/,
    );
    expect(firstMismatch(await patchSupportGpu(masks[1]!, W, H, rects), want[1]!)).toBeUndefined();
  });

  it("a patch grid's support fractions agree with the host grid", async () => {
    const m = blobs(8);
    const runs = maskRuns(m, W, H);
    const grid = patchGrid({ footprint: runs, size: 12, stride: 6, rules: [{ kind: "fraction", name: "tissue", mask: runs, min: 0.5 }] });
    const rects = rectsArray(grid.candidates);
    const counts = await patchSupportGpu(m, W, H, rects);
    const fractions = Array.from(counts, (c) => c / 144);
    expect(fractions).toEqual(grid.candidates.map((p) => p.support[0]));
  });

  it("graph op: resident result equals cpuGolden, n × 1 u32", async () => {
    const m = blobs(5);
    const rects = randomRects(2, 50);
    const g = new Graph();
    const mask = g.grid(m, W, H);
    const r = g.source({ shape: { kind: "matrix", rows: 50, cols: 4 }, dtype: "u32", data: rects }, "rects");
    const out = g.op1("patchSupport", { mask, rects: r });
    const gpu = await pull(g, out);
    const cpu = await pull(g, out, { mode: "cpu" });
    expect(gpu.shape).toEqual({ kind: "matrix", rows: 50, cols: 1 });
    expect(firstMismatch(gpu.data!, cpu.data!)).toBeUndefined();
    expect(firstMismatch(cpu.data!, patchSupportCpu(m, W, "f32", rects))).toBeUndefined();
  });

  it("graph op: a multi-lane mask is refused at build", () => {
    const g = new Graph();
    const vec = g.source(
      { shape: { kind: "grid", width: 4, height: 3 }, dtype: "f32", element: { kind: "vec", n: 4 }, data: new Float32Array(48) },
      "vec4",
    );
    const r = g.source({ shape: { kind: "matrix", rows: 1, cols: 4 }, dtype: "u32", data: Uint32Array.of(0, 0, 4, 3) }, "rects");
    expect(() => g.op1("patchSupport", { mask: vec, rects: r })).toThrow(/must be scalar/);
  });
});
