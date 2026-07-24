import { describe, expect, it } from "vitest";
import { crossPCF } from "./pcf";
import type { CellCloud } from "./tcm";

describe("crossPCF (eq 8, Mode 1)", () => {
  it("bins pairs into the correct annulus with the correct normalisation", () => {
    // One A at the centre of a large ROI (so no edge effect for r ≤ rMax), and 8 B on a ring at
    // radius 11 — every pair must land in bin 5 = [10,12) with an exactly computable g.
    const A: CellCloud = { xs: [100], ys: [100] };
    const bx: number[] = [];
    const by: number[] = [];
    for (let k = 0; k < 8; k++) {
      const th = (k / 8) * 2 * Math.PI;
      bx.push(100 + 11 * Math.cos(th));
      by.push(100 + 11 * Math.sin(th));
    }
    const B: CellCloud = { xs: bx, ys: by };
    const res = crossPCF(A, B, { bbox: [0, 0, 200, 200], rMax: 20, nBins: 10 }); // dr = 2

    expect(res.counts[5]).toBe(8); // all 8 pairs in [10,12)
    expect(res.counts.reduce((s, c, i) => (i === 5 ? s : s + c), 0)).toBe(0); // and nowhere else

    const rhoB = 8 / (200 * 200);
    const annulus = Math.PI * (12 * 12 - 10 * 10);
    expect(res.g[5]!).toBeCloseTo(8 / (1 * rhoB * annulus), 6);
    expect(res.g[0]!).toBe(0);
  });

  it("CSR: g(r) ≈ 1 for uniform A and B (rMax ≪ ROI so edge bias is small)", () => {
    // mulberry32 — a good-quality PRNG; a simple LCG's consecutive values lie on 2D lattice planes,
    // which fabricates spatial structure and breaks the CSR premise.
    let a = 0x9e3779b9;
    const rnd = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const uniform = (n: number): CellCloud => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        xs.push(rnd() * 500);
        ys.push(rnd() * 500);
      }
      return { xs, ys };
    };
    const res = crossPCF(uniform(2500), uniform(5000), { bbox: [0, 0, 500, 500], rMax: 15, nBins: 8 });
    const mean = res.g.reduce((a, b) => a + b, 0) / res.g.length;
    expect(mean).toBeGreaterThan(0.85); // slight downward drift is the deferred edge correction
    expect(mean).toBeLessThan(1.15);
  });

  it("clustering: g(small r) ≫ 1 when B tightly surrounds A", () => {
    // A and B are the same tight blob ⇒ strong short-range co-location.
    let s = 99;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < 400; i++) {
      xs.push(250 + (rnd() - 0.5) * 10);
      ys.push(250 + (rnd() - 0.5) * 10);
    }
    const cloud: CellCloud = { xs, ys };
    const res = crossPCF(cloud, cloud, { bbox: [0, 0, 500, 500], rMax: 20, nBins: 10 });
    expect(res.g[0]!).toBeGreaterThan(3); // strongly clustered at the shortest range
  });
});
