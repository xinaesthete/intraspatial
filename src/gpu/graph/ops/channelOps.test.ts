// The channel ops' references, and the wand's field subgraph built from them.
import { beforeAll, describe, expect, it } from "vite-plus/test";
import type { FieldValue } from "../handle";
import { Graph, pull } from "../index";
import { registerWandOps, wandField } from "../wand";
import { type GramRasters, channelMixOp, channelStructureTensorOp, gramChannelsOp } from "./channelOps";
import { convolveSeparableOp } from "./convolveSeparable";

beforeAll(() => registerWandOps());

const channels = (data: ArrayLike<number>, width: number, height: number, K: number): FieldValue => ({
  shape: { kind: "grid", width, height },
  dtype: "f32",
  axes: [{ name: "c", type: "channel", length: K }],
  data: Float32Array.from(data),
});
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);

describe("gramChannels", () => {
  it("unpacks padded f32 rows into planes", () => {
    // 2 channels, 3×2, rows padded to 4 words.
    const f = new Float32Array([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0]);
    const payload: GramRasters = { kind: "gramRasters", words: new Uint32Array(f.buffer), rowWords: 4, half: false };
    const v: FieldValue = {
      shape: { kind: "grid", width: 3, height: 2 },
      dtype: "f32",
      axes: [{ name: "c", type: "channel", length: 2 }],
      payload,
    };
    expect(Array.from(gramChannelsOp.cpuGolden!([v], {})[0]!.data!)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it("unpacks f16 pairs, even column in the low half", () => {
    // One channel, 3×1: 1.0 (0x3c00), −2 (0xc000), 0.5 (0x3800), padding.
    const words = new Uint32Array([0x3c00 | (0xc000 << 16), 0x3800]);
    const payload: GramRasters = { kind: "gramRasters", words, rowWords: 2, half: true };
    const v: FieldValue = {
      shape: { kind: "grid", width: 3, height: 1 },
      dtype: "f32",
      axes: [{ name: "c", type: "channel", length: 1 }],
      payload,
    };
    expect(Array.from(gramChannelsOp.cpuGolden!([v], {})[0]!.data!)).toEqual([1, -2, 0.5]);
  });
});

describe("channelMix", () => {
  it("is M·x + b per pixel, with as many output channels as M has rows", () => {
    const x = channels([1, 2, 10, 20], 2, 1, 2); // pixel 0 = (1, 10), pixel 1 = (2, 20)
    const out = channelMixOp.cpuGolden!([x], { matrix: [1, 1, 2, -1, 0, 0.5], offset: [0, 1, -1] })[0]!;
    expect(Array.from(out.data!)).toEqual([11, 22, -7, -15, 4, 9]);
    expect(out.axes?.[0]?.length).toBe(3);
  });
});

describe("channelStructureTensor", () => {
  it("measures a ramp's slope, one-sided at the edges, per `scale` pixels", () => {
    // u = 2x in one channel, 0 in the other: the steepest change is 2 per pixel everywhere.
    const w = 5;
    const h = 3;
    const u = new Float32Array(2 * w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) u[y * w + x] = 2 * x;
    const out = channelStructureTensorOp.cpuGolden!([channels(u, w, h, 2)], { scale: 3 })[0]!.data!;
    for (let i = 0; i < w * h; i++) {
      expect(out[i * 4]).toBeCloseTo(6, 6); // 2 per pixel × 3 pixels
      expect(out[i * 4 + 1]).toBeCloseTo(36, 5);
      expect(out[i * 4 + 2]).toBe(0);
      expect(out[i * 4 + 3]).toBe(0);
    }
  });

  it("takes the larger eigenvalue across channels (Di Zenzo), not a sum of magnitudes", () => {
    // Two channels changing along x and along y: G = diag(1, 1), so √λmax = 1, not √2.
    const w = 4;
    const n = w * w;
    const u = new Float32Array(2 * n);
    for (let y = 0; y < w; y++)
      for (let x = 0; x < w; x++) {
        u[y * w + x] = x;
        u[n + y * w + x] = y;
      }
    const out = channelStructureTensorOp.cpuGolden!([channels(u, w, w, 2)], { scale: 1 })[0]!.data!;
    expect(out[5 * 4]).toBeCloseTo(1, 6);
  });
});

describe("convolveSeparable over a channel axis", () => {
  it("blurs each plane on its own", () => {
    const w = 7;
    const h = 5;
    const n = w * h;
    const u = Float32Array.from({ length: 3 * n }, (_, i) => Math.sin(i * 0.7) + (i >= n ? 5 : 0));
    const p = { kernel: "gaussian", sigma: 1.2, radius: 4 };
    const all = convolveSeparableOp.cpuGolden!([channels(u, w, h, 3)], p)[0]!.data!;
    for (let k = 0; k < 3; k++) {
      const one = convolveSeparableOp.cpuGolden!(
        [{ shape: { kind: "grid", width: w, height: h }, dtype: "f32", data: u.slice(k * n, (k + 1) * n) }],
        p,
      )[0]!.data!;
      expect(maxDiff(all.subarray(k * n, (k + 1) * n), one)).toBe(0);
    }
  });
});

describe("wandField", () => {
  const w = 9;
  const h = 7;
  const n = w * h;
  const K = 3;
  const x = Float32Array.from({ length: K * n }, (_, i) => Math.cos(i * 0.31) * (1 + (i % 5)) + Math.floor(i / n));
  const mean = [0.5, 1, 2];
  const sd = [2, 0.5, 0];
  const A = [1, 0, 0.5, 0, 2, 0, 0.3, 0.3, 0.3];

  async function run(q: number, residualSigma: number) {
    const g = new Graph();
    const src = g.source(channels(x, w, h, K), "x");
    const { u, tensor } = wandField(g, src, { A, q, mean, sd, perR: 3, residualSigma });
    return { u: (await pull(g, u, { mode: "cpu" })).data!, tensor: (await pull(g, tensor, { mode: "cpu" })).data! };
  }

  it("u is A applied to the standardised channels (a zero spread drops that channel)", async () => {
    const { u } = await run(K, 0);
    for (const i of [0, 17, n - 1]) {
      const z = [0, 1, 2].map((a) => (sd[a]! > 0 ? (x[a * n + i]! - mean[a]!) / sd[a]! : 0));
      for (let k = 0; k < K; k++) expect(u[k * n + i]).toBeCloseTo(A[k * K]! * z[0]! + A[k * K + 1]! * z[1]! + A[k * K + 2]! * z[2]!, 5);
    }
  });

  it("the tensor is u's, unless the modes after q are blurred first", async () => {
    const plain = await run(1, 0);
    const direct = channelStructureTensorOp.cpuGolden!([channels(plain.u, w, h, K)], { scale: 3 })[0]!.data!;
    expect(maxDiff(plain.tensor, direct)).toBeLessThan(1e-5);
    // Every mode main: nothing to blur, so the blur changes nothing.
    expect(maxDiff((await run(K, 2)).tensor, (await run(K, 0)).tensor)).toBe(0);
    // Modes after the first blurred: a smoother field, so a different one.
    expect(maxDiff((await run(1, 2)).tensor, plain.tensor)).toBeGreaterThan(1e-3);
  });
});
