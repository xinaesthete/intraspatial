import { beforeAll, describe, expect, it } from "vitest";
import { getDevice } from "../../device";
import { Graph, pull, pullData } from "../index";

// The graph node must agree with its `cpuGolden` (min/max bit-exact, mean to f32 rounding), and a
// resident chain — morphology → decimate → threshold — must run with the mid-chain size change,
// the output buffer leased at the SMALLER size. Mirrors the LIDAR chain: open → block-mean to a
// coarser canopy-cover fraction.

const W = 50;
const H = 34;

function firstMismatch(a: Float32Array, b: Float32Array, tol: number): string | undefined {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i]! - b[i]!) > tol) return `index ${i}: got ${a[i]}, want ${b[i]}`;
  return undefined;
}

describe("decimate graph op", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("every mode matches cpuGolden on a non-divisible grid and reports the decimated shape", async () => {
    const field = new Float32Array(W * H);
    for (let i = 0; i < field.length; i++) field[i] = ((i * 2654435761) % 89) / 89;
    for (const mode of ["mean", "min", "max"]) {
      const g = new Graph();
      const out = g.op1("decimate", { grid: g.grid(field, W, H) }, { factor: 4, mode });
      const v = await pull(g, out);
      expect(v.shape).toEqual({ kind: "grid", width: 13, height: 9 });
      const gpu = await pullData(g, out);
      const cpu = await pullData(g, out, { mode: "cpu" });
      expect(gpu.length).toBe(13 * 9);
      expect(firstMismatch(gpu, cpu, mode === "mean" ? 1e-5 : 0), mode).toBeUndefined();
      expect(new Set(Array.from(gpu)).size).toBeGreaterThan(1);
    }
  });

  it("resident chain: opening -> block mean -> threshold gives canopy-cover fraction per block", async () => {
    // 0/1 canopy mask with a solid 8x8 blob at (8..15, 8..15) on a 50x34 grid; factor-8 block
    // mean of the opened mask is 1 in block (1,1) and 0 elsewhere.
    const mask = new Float32Array(W * H);
    for (let y = 8; y < 16; y++) for (let x = 8; x < 16; x++) mask[y * W + x] = 1;
    const g = new Graph();
    const opened = g.op1("morphology", { grid: g.grid(mask, W, H) }, { op: "open", radius: 1 });
    const cover = g.op1("decimate", { grid: opened }, { factor: 8, mode: "mean" });
    const dense = g.op1("threshold", { in: cover }, { thresh: 0.5, soft: false });
    const gpu = await pullData(g, dense);
    const cpu = await pullData(g, dense, { mode: "cpu" });
    const ow = Math.ceil(W / 8);
    expect(gpu.length).toBe(ow * Math.ceil(H / 8));
    expect(firstMismatch(gpu, cpu, 0)).toBeUndefined();
    let total = 0;
    for (const v of gpu) total += v;
    expect(total).toBe(1);
    expect(gpu[1 * ow + 1]).toBe(1);
  });
});
