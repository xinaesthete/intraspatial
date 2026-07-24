import { describe, expect, it } from "vitest";
import { crossPCF, crossPCFMatrix, type LabelledCells } from "../../spatial/pcf";
import type { CellCloud } from "../../spatial/tcm";
import { crossPCFGpu, crossPCFMatrixGpu } from "./crossPcf";

// Two standing constraints of Dawn-on-Node, both already paid for elsewhere in this suite:
//   • Assertions are AGGREGATED (max diff, totals) rather than per-element expect() loops — a loop
//     of thousands of expect() calls kills the fork in a way that mimics the known teardown
//     flakiness but isn't.
//   • Point counts are kept SMALL. At ~500 A / ~1000 B across three submissions in one process the
//     fork dies at teardown and takes the (already passing) results with it. Correctness here is
//     size-independent — the counts are integers, so parity is parity — and the large-N runs
//     belong in the browser harness.
function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("crossPCFGpu", () => {
  it("bins the exact-ring case identically to the CPU path", async () => {
    const A: CellCloud = { xs: [100], ys: [100] };
    const bx: number[] = [];
    const by: number[] = [];
    for (let k = 0; k < 8; k++) {
      const th = (k / 8) * 2 * Math.PI;
      bx.push(100 + 11 * Math.cos(th));
      by.push(100 + 11 * Math.sin(th));
    }
    const gpu = await crossPCFGpu(A, { xs: bx, ys: by }, { bbox: [0, 0, 200, 200], rMax: 20, nBins: 10 });
    expect(gpu.counts[5]).toBe(8); // all 8 in [10,12)
    expect(gpu.counts.reduce((s, c, i) => (i === 5 ? s : s + c), 0)).toBe(0);
  });

  it("matches the CPU crossPCF on a random cloud", async () => {
    const rnd = rng(0x9e3779b9);
    const cloud = (n: number): CellCloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(rnd() * 400);
        ys.push(rnd() * 400);
      }
      return { xs, ys };
    };
    const A = cloud(120);
    const B = cloud(200);
    const p = { bbox: [0, 0, 400, 400] as [number, number, number, number], rMax: 40, nBins: 8 };
    const cpu = crossPCF(A, B, p);
    const gpu = await crossPCFGpu(A, B, p);

    const total = (c: readonly number[]) => c.reduce((s, v) => s + v, 0);
    // The counts are integers on both sides, so this is the SAME arithmetic — the only slack is
    // f32-vs-f64 classification of a pair sitting within a float ulp of a bin edge or of rMax.
    expect(Math.abs(total(gpu.counts) - total(cpu.counts))).toBeLessThanOrEqual(2);
    const maxCount = gpu.counts.reduce((m, v, i) => Math.max(m, Math.abs(v - cpu.counts[i]!)), 0);
    expect(maxCount).toBeLessThanOrEqual(2);
    const maxG = gpu.g.reduce((m, v, i) => Math.max(m, Math.abs(v - cpu.g[i]!)), 0);
    expect(maxG).toBeLessThan(0.02);
  });
});

describe("crossPCFMatrixGpu", () => {
  it("matches the CPU N-way matrix over three overlapping type clouds", async () => {
    const rnd = rng(12345);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (const [id, n, cx, cy] of [
      [1, 90, 120, 150],
      [2, 70, 180, 150],
      [3, 60, 150, 220],
    ] as const) {
      for (let i = 0; i < n; i++) {
        xs.push(cx + (rnd() - 0.5) * 120);
        ys.push(cy + (rnd() - 0.5) * 120);
        typeId.push(id);
      }
    }
    const cells: LabelledCells = { xs, ys, typeId };
    const p = { bbox: [0, 0, 300, 300] as [number, number, number, number], radius: 25 };
    const cpu = crossPCFMatrix(cells, p);
    const gpu = await crossPCFMatrixGpu(cells, p);

    expect(gpu.types).toEqual(cpu.types);
    expect(gpu.counts).toEqual(cpu.counts);
    let maxG = 0;
    for (let i = 0; i < cpu.g.length; i++) maxG = Math.max(maxG, Math.abs(gpu.g[i]! - cpu.g[i]!));
    expect(maxG).toBeLessThan(0.01);
  });
});
