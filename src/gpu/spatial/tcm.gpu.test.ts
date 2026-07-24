import { describe, expect, it } from "vitest";
import { type CellCloud, computeTcmReference, crossMarks, type TcmParams } from "../../spatial/tcm";
import { computeTcmGpu, crossMarksGpu } from "./tcm";

// Same two Dawn-on-Node constraints as the cross-PCF tests next door: aggregated assertions (no
// per-element expect() loops) and small clouds/grids. The CPU reference in `src/spatial/tcm.ts` is
// the oracle for both paths, so parity at this size is the whole claim.
function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rnd = rng(0xc0ffee);
const cloud = (n: number, cx: number, cy: number, sp: number): CellCloud => {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(cx + (rnd() - 0.5) * sp);
    ys.push(cy + (rnd() - 0.5) * sp);
  }
  return { xs, ys };
};

describe("crossMarksGpu (eqs 9–13)", () => {
  it("reproduces the CPU marks — the neighbour counts are integers, so this is exact", async () => {
    const A = cloud(40, 10, 10, 18);
    const B = cloud(60, 8, 12, 14);
    const p: TcmParams = { width: 16, height: 16, bbox: [0, 0, 20, 20], radius: 3, sigma: 1.5, alpha: 5 };
    const cpu = crossMarks(A, B, p);
    const gpu = await crossMarksGpu(A, B, p);

    // The CPU array is the raw mark m; the GPU one is the TRANSFORMED mark M — apply the same
    // transform here rather than exporting a second kernel just to compare.
    const M = (m: number) => (m >= 5 ? 1 : m > 1 ? (m - 1) / 4 : m <= 1 / 5 ? -1 : (1 - 1 / m) / 4);
    let maxAbs = 0;
    for (let i = 0; i < cpu.length; i++) maxAbs = Math.max(maxAbs, Math.abs(gpu[i]! - M(cpu[i]!)));
    expect(maxAbs).toBeLessThan(1e-5);
  });
});

describe("computeTcmGpu (eq 14)", () => {
  it("reproduces the exact CPU reference Γ_AB grid", async () => {
    const A = cloud(60, 10, 10, 18);
    const B = cloud(90, 8, 12, 14);
    const p: TcmParams = { width: 16, height: 16, bbox: [0, 0, 20, 20], radius: 3, sigma: 1.5, alpha: 5 };
    const ref = computeTcmReference(A, B, p);
    const gpu = await computeTcmGpu(A, B, p);

    let peak = 0;
    let maxAbs = 0;
    for (let i = 0; i < ref.length; i++) {
      peak = Math.max(peak, Math.abs(ref[i]!));
      maxAbs = Math.max(maxAbs, Math.abs(gpu[i]! - ref[i]!));
    }
    expect(peak).toBeGreaterThan(0.1); // the case is non-trivial
    // Same window, same sample points, same normalisation — only the SUMMATION ORDER differs
    // (bucket order on the GPU, A-index order on the CPU), so the gap is f32 rounding.
    expect(maxAbs / peak).toBeLessThan(1e-4);
  });

  it("gets the sign right: positive where A sits in B, negative where A is excluded", async () => {
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 7; i++)
      for (let j = 0; j < 7; j++) {
        bx.push(4 + i / 3);
        by.push(4 + j / 3);
      }
    const A: CellCloud = { xs: [5, 15], ys: [5, 15] };
    const p: TcmParams = { width: 20, height: 20, bbox: [0, 0, 20, 20], radius: 5, sigma: 1, alpha: 5 };
    const grid = await computeTcmGpu(A, { xs: bx, ys: by }, p);
    const at = (x: number, y: number) => grid[y * p.width + x]!;
    expect(at(5, 5)).toBeGreaterThan(0);
    expect(at(15, 15)).toBeLessThan(0);
    expect(Math.abs(at(5, 5))).toBeCloseTo(Math.abs(at(15, 15)), 5);
  });
});
