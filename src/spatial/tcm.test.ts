import { describe, expect, it } from "vitest";
import { type CellCloud, computeTcm, computeTcmReference, crossMarks, markToM, type TcmParams } from "./tcm";

describe("markToM (eqs 10–13, α=5)", () => {
  const a = 5;
  it("is 0 at CSR and clamps at ±1", () => {
    expect(markToM(1, a)).toBe(0);
    expect(markToM(5, a)).toBe(1);
    expect(markToM(10, a)).toBe(1);
    expect(markToM(1 / 5, a)).toBe(-1);
    expect(markToM(0.05, a)).toBe(-1);
    expect(markToM(0, a)).toBe(-1); // no neighbours ⇒ maximal exclusion
  });
  it("is linear-in-m above CSR, reciprocal below (M(m) = −M(1/m))", () => {
    expect(markToM(2, a)).toBeCloseTo(0.25, 12); // (2−1)/(5−1)
    expect(markToM(0.5, a)).toBeCloseTo(-0.25, 12); // = −M(2), the reciprocal branch
    for (const m of [1.25, 1.5, 2, 3, 4]) {
      expect(markToM(m, a)).toBeCloseTo(-markToM(1 / m, a), 12);
    }
  });
  it("rejects α ≤ 1", () => {
    expect(() => markToM(2, 1)).toThrow();
  });
});

describe("crossMarks / computeTcmReference (Mode 1, global ρ_B)", () => {
  // ROI [0,20]², a tight B cluster (7×7 grid) around (5,5) so every B is within radius 5 of it.
  const bx: number[] = [];
  const by: number[] = [];
  for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
    bx.push(4 + i / 3);
    by.push(4 + j / 3);
  }
  const B: CellCloud = { xs: bx, ys: by };
  const p: TcmParams = { width: 20, height: 20, bbox: [0, 0, 20, 20], radius: 5, sigma: 1, alpha: 5 };

  it("m_ab: strong clustering inside the B cluster, zero far away", () => {
    const A: CellCloud = { xs: [5, 15], ys: [5, 15] };
    const m = crossMarks(A, B, p);
    expect(m[0]!).toBeGreaterThan(5); // (5,5) sits in the cluster ⇒ ≥α-fold over CSR
    expect(m[1]!).toBe(0); // (15,15) has no B within radius ⇒ 0
  });

  it("Γ_ab is positive where A co-locates with B, negative where it is excluded", () => {
    const A: CellCloud = { xs: [5, 15], ys: [5, 15] };
    const grid = computeTcmReference(A, B, p);
    const at = (x: number, y: number) => grid[y * p.width + x]!;
    expect(at(5, 5)).toBeGreaterThan(0); // A@(5,5), M=+1 ⇒ positive bump
    expect(at(15, 15)).toBeLessThan(0); // A@(15,15), M=−1 ⇒ negative bump
    // Antisymmetry of the two equal-and-opposite kernels: peak magnitudes match.
    expect(Math.abs(at(5, 5))).toBeCloseTo(Math.abs(at(15, 15)), 6);
  });

  it("grid-accelerated computeTcm reproduces the reference exactly", () => {
    // Deterministic pseudo-random two clouds over the ROI.
    let s = 987654321;
    const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    const cloud = (n: number, cx: number, cy: number, sp: number): CellCloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(cx + (rnd() - 0.5) * sp);
        ys.push(cy + (rnd() - 0.5) * sp);
      }
      return { xs, ys };
    };
    const A = cloud(120, 10, 10, 18);
    const B = cloud(200, 8, 12, 14);
    const q: TcmParams = { width: 48, height: 48, bbox: [0, 0, 20, 20], radius: 3, sigma: 1.5, alpha: 5 };
    const fast = computeTcm(A, B, q);
    const ref = computeTcmReference(A, B, q);
    let maxAbs = 0;
    for (let i = 0; i < ref.length; i++) maxAbs = Math.max(maxAbs, Math.abs(fast[i]! - ref[i]!));
    expect(maxAbs).toBeLessThan(1e-5); // same counts, same splat ⇒ identical up to fp
  });
});
