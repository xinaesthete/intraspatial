// The channel ops' kernels against their references, through the wand's field subgraph on the GPU.
// House rules: one small scene, aggregated assertions.
import { describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import type { FieldValue } from "../handle";
import { Graph, pull } from "../index";
import { registerWandOps, wandField } from "../wand";
import type { GramRasters } from "./channelOps";

registerWandOps();

describe("wandField on the GPU", () => {
  it("matches the CPU references from packed f32 rasters, with the residual blur", async () => {
    const w = 11;
    const h = 7;
    const K = 3;
    const rowWords = 64; // padded, as the Gram's texture copies are
    const f = new Float32Array(K * h * rowWords);
    for (let a = 0; a < K; a++)
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) f[(a * h + y) * rowWords + x] = Math.sin(0.4 * x + a) * Math.cos(0.5 * y) + a;
    const words = new Uint32Array(f.buffer);
    const device = await getDevice();
    const buffer = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, words);

    const run = async (payload: GramRasters, mode: "gpu" | "cpu") => {
      const g = new Graph();
      const src: FieldValue = {
        shape: { kind: "grid", width: w, height: h },
        dtype: "f32",
        axes: [{ name: "c", type: "channel", length: K }],
        payload,
      };
      const x = g.op("gramChannels", { rasters: g.source(src, "rasters") })[0]!;
      const p = { A: [1, 0, 0.5, 0, 2, 0, 0.3, -0.3, 0.3], q: 1, mean: [0, 1, 2], sd: [1, 0.5, 2], perR: 3, residualSigma: 1.5 };
      const { u, tensor } = wandField(g, x, p);
      return { u: (await pull(g, u, { mode })).data!, tensor: (await pull(g, tensor, { mode })).data! };
    };
    const gpu = await run({ kind: "gramRasters", buffer, rowWords, half: false }, "gpu");
    const cpu = await run({ kind: "gramRasters", words, rowWords, half: false }, "cpu");
    const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]!)), 0);
    expect(gpu.u.length).toBe(K * w * h);
    expect(gpu.tensor.length).toBe(4 * w * h);
    expect(maxDiff(gpu.u, cpu.u)).toBeLessThan(1e-4);
    expect(maxDiff(gpu.tensor, cpu.tensor)).toBeLessThan(1e-3);
  });
});
