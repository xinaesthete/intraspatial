import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import type { LabelledCells } from "./pcf";
import { benjaminiHochberg, quadratCorrelation, quadratCounts, rowCorrelation } from "./quadratCorrelation";

const BBOX = [0, 0, 400, 400] as const;

describe("quadratCounts", () => {
  it("bins into the stated grid", () => {
    const cells: LabelledCells = { xs: [5, 15, 105, 395], ys: [5, 5, 5, 395], typeId: [0, 0, 1, 1] };
    const q = quadratCounts(cells, { bbox: BBOX, quadratSize: 100, nTypes: 2 });
    expect(q.cols).toBe(4);
    expect(q.rows).toBe(4);
    expect(q.counts[0 * 16 + 0]).toBe(2); // both type-0 cells in quadrat (0,0)
    expect(q.counts[1 * 16 + 1]).toBe(1); // type 1 at x=105 → quadrat (1,0)
    expect(q.counts[1 * 16 + 15]).toBe(1); // and at (3,3)
  });

  it("clamps a cell just outside the bbox into the edge quadrat", () => {
    const cells: LabelledCells = { xs: [-0.001, 400.001], ys: [200, 200], typeId: [0, 0] };
    const q = quadratCounts(cells, { bbox: BBOX, quadratSize: 100, nTypes: 1 });
    let total = 0;
    for (const v of q.counts) total += v;
    expect(total).toBe(2);
  });
});

describe("rowCorrelation", () => {
  it("is 1 on the diagonal and −1 for an exact anti-correlation", () => {
    const m = Float64Array.from([1, 2, 3, 4, 4, 3, 2, 1]);
    const c = rowCorrelation(m, 2, 4);
    expect(c[0]!).toBeCloseTo(1, 12);
    expect(c[3]!).toBeCloseTo(1, 12);
    expect(c[1]!).toBeCloseTo(-1, 12);
  });

  it("returns NaN for a row with no variance rather than 0", () => {
    // 0 would read as "no association"; the truth is that the correlation is undefined.
    const m = Float64Array.from([1, 2, 3, 5, 5, 5]);
    const c = rowCorrelation(m, 2, 3);
    expect(Number.isNaN(c[1]!)).toBe(true);
  });
});

describe("quadratCorrelation", () => {
  /** Two types filling the same quadrats (co-located) or complementary ones (segregated).
   *
   *  The per-quadrat TOTALS are deliberately unequal between the two modes and between hot and cold
   *  quadrats. With two types and a constant total the correlation is forced to exactly −1 on every
   *  labelling, so the null has zero variance and the effect size is undefined — see the degenerate
   *  case pinned below. Varying totals is what makes this a test of the statistic rather than of an
   *  arithmetic identity. */
  function build(mode: "together" | "apart", seed: number): LabelledCells {
    const rnd = mulberry32(seed);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let qy = 0; qy < 4; qy++) {
      for (let qx = 0; qx < 4; qx++) {
        const hot = (qx + qy) % 2 === 0;
        const nA = hot ? 40 : 4;
        const nB = mode === "together" ? (hot ? 36 : 6) : hot ? 4 : 30;
        for (let i = 0; i < nA; i++) {
          xs.push(qx * 100 + rnd() * 100);
          ys.push(qy * 100 + rnd() * 100);
          typeId.push(0);
        }
        for (let i = 0; i < nB; i++) {
          xs.push(qx * 100 + rnd() * 100);
          ys.push(qy * 100 + rnd() * 100);
          typeId.push(1);
        }
      }
    }
    return { xs, ys, typeId };
  }

  it("is positive for co-located types and negative for segregated ones", () => {
    const p = { bbox: BBOX, quadratSize: 100, nTypes: 2 } as const;
    expect(quadratCorrelation(build("together", 3), p).r[1]!).toBeGreaterThan(0.8);
    expect(quadratCorrelation(build("apart", 3), p).r[1]!).toBeLessThan(-0.8);
  });

  it("scores against the null, which is NOT centred at zero", () => {
    // The property that makes SES necessary rather than decorative. Shuffling labels holds each
    // quadrat's TOTAL fixed, so both types inherit the tissue's density pattern and the null
    // correlation is already strongly positive wherever density varies. A co-located pair therefore
    // scores a modest SES despite r ≈ 1 — it is barely more co-located than sharing the same dense
    // quadrats already implies — while a segregated pair scores hugely negative, because it is
    // fighting that shared density. Reading r alone would call both "strong".
    const p = { bbox: BBOX, quadratSize: 100, nTypes: 2, simulations: 199, seed: 1 } as const;
    const together = quadratCorrelation(build("together", 5), p);
    const apart = quadratCorrelation(build("apart", 5), p);
    expect(together.r[1]!).toBeGreaterThan(0.8);
    expect(apart.r[1]!).toBeLessThan(-0.8);
    expect(together.ses[1]!).toBeGreaterThan(1);
    expect(apart.ses[1]!).toBeLessThan(-3);
    expect(apart.p[1]!).toBeLessThanOrEqual(0.01);
    expect(apart.q[1]!).toBeLessThanOrEqual(0.01);
    expect(together.quadrats).toBe(16);
  });

  it("does not find association where there is none", () => {
    // Both types uniform: the labelling carries no spatial information.
    const rnd = mulberry32(11);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let i = 0; i < 4000; i++) {
      xs.push(rnd() * 400);
      ys.push(rnd() * 400);
      typeId.push(i % 2);
    }
    const res = quadratCorrelation({ xs, ys, typeId }, { bbox: BBOX, quadratSize: 100, nTypes: 2, simulations: 199, seed: 2 });
    expect(Math.abs(res.ses[1]!)).toBeLessThan(3);
    expect(res.p[1]!).toBeGreaterThan(0.05);
  });

  it("has a diagonal of 1 and stays symmetric", () => {
    const res = quadratCorrelation(build("together", 7), { bbox: BBOX, quadratSize: 100, nTypes: 2 });
    expect(res.r[0]!).toBeCloseTo(1, 12);
    expect(res.r[3]!).toBeCloseTo(1, 12);
    expect(res.r[1]!).toBeCloseTo(res.r[2]!, 12);
  });

  it("reports an undefined effect size when the null is degenerate", () => {
    // Two types with the SAME total in every quadrat: A + B is constant, so corr(A,B) = −1 for every
    // labelling including the observed one. The null has no spread, so there is no scale to measure
    // an effect against and SES is NaN — which is the honest answer. Returning 0 would say "no
    // effect" and returning ±∞ would say "certain"; both are wrong.
    const rnd = mulberry32(41);
    const xs: number[] = [];
    const ys: number[] = [];
    const typeId: number[] = [];
    for (let qy = 0; qy < 4; qy++) {
      for (let qx = 0; qx < 4; qx++) {
        // A varies between quadrats (so its row is not constant and r is defined), but A + B is 40
        // everywhere — which is what forces r = −1 and flattens the null.
        const nA = (qx + qy) % 2 === 0 ? 30 : 10;
        for (let i = 0; i < 40; i++) {
          xs.push(qx * 100 + rnd() * 100);
          ys.push(qy * 100 + rnd() * 100);
          typeId.push(i < nA ? 0 : 1);
        }
      }
    }
    const res = quadratCorrelation({ xs, ys, typeId }, { bbox: BBOX, quadratSize: 100, nTypes: 2, simulations: 99, seed: 3 });
    expect(res.r[1]!).toBeCloseTo(-1, 9);
    expect(Number.isNaN(res.ses[1]!)).toBe(true);
  });

  it("skips inference when no simulations are asked for", () => {
    const res = quadratCorrelation(build("apart", 9), { bbox: BBOX, quadratSize: 100, nTypes: 2 });
    expect(res.simulations).toBe(0);
    expect(res.ses.length).toBe(0);
    expect(res.q.length).toBe(0);
  });
});

describe("benjaminiHochberg", () => {
  it("matches the textbook worked example", () => {
    // p = .01 .02 .03 .04 .05 with m = 5 → q = .05 .05 .05 .05 .05
    const q = benjaminiHochberg([0.01, 0.02, 0.03, 0.04, 0.05]);
    for (const v of q) expect(v).toBeCloseTo(0.05, 12);
  });

  it("is monotone in p", () => {
    const q = benjaminiHochberg([0.001, 0.2, 0.02, 0.9, 0.3]);
    const pairs = [
      [0.001, q[0]!],
      [0.02, q[2]!],
      [0.2, q[1]!],
      [0.3, q[4]!],
      [0.9, q[3]!],
    ].sort((a, b) => a[0]! - b[0]!);
    for (let i = 1; i < pairs.length; i++) expect(pairs[i]![1]!).toBeGreaterThanOrEqual(pairs[i - 1]![1]! - 1e-12);
  });

  it("never exceeds 1, and carries NaN through without counting it", () => {
    const q = benjaminiHochberg([0.9, 0.95, Number.NaN, 1]);
    expect(Number.isNaN(q[2]!)).toBe(true);
    for (const v of q) if (Number.isFinite(v)) expect(v).toBeLessThanOrEqual(1);
    // m is 3, not 4 — the NaN was never a test.
    expect(q[0]!).toBeCloseTo(Math.min(1, (0.9 * 3) / 1), 12);
  });

  it("controls the false discovery rate under a global null", () => {
    // 200 independent null tests, uniform p. At q ≤ 0.1 the expected number of discoveries is small;
    // a broken implementation (e.g. forgetting the rank factor) floods.
    const rnd = mulberry32(31);
    const p: number[] = [];
    for (let i = 0; i < 200; i++) p.push(rnd());
    const q = benjaminiHochberg(p);
    let disc = 0;
    for (const v of q) if (v <= 0.1) disc++;
    expect(disc).toBeLessThan(10);
  });
});
