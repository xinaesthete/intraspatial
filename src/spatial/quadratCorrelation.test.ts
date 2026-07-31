import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import type { LabelledCells } from "./pcf";
import { benjaminiHochberg, partialCorrelation, quadratCorrelation, quadratCounts, rowCorrelation } from "./quadratCorrelation";

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

describe("partialCorrelation", () => {
  // Build a correlation matrix from data, so the fixtures are honest matrices rather than
  // hand-written ones that might not be positive-definite.
  function corrOf(rows: number[][]): { r: Float64Array; k: number } {
    const k = rows.length;
    const q = rows[0]!.length;
    const m = new Float64Array(k * q);
    for (let a = 0; a < k; a++) for (let j = 0; j < q; j++) m[a * q + j] = rows[a]![j]!;
    return { r: rowCorrelation(m, k, q), k };
  }

  it("kills an association that a shared driver fully explains", () => {
    // THE reason the paper uses this statistic. X drives both Y and Z; Y and Z are conditionally
    // independent given X. Their plain correlation is large and entirely induced — the partial
    // correlation is what says so.
    const rnd = mulberry32(3);
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];
    for (let i = 0; i < 400; i++) {
      const xi = rnd() * 10;
      x.push(xi);
      y.push(xi + (rnd() - 0.5) * 0.6);
      z.push(xi + (rnd() - 0.5) * 0.6);
    }
    const { r, k } = corrOf([x, y, z]);
    const pc = partialCorrelation(r, k);
    expect(r[1 * 3 + 2]!).toBeGreaterThan(0.9); // Y–Z look strongly associated…
    expect(Math.abs(pc[1 * 3 + 2]!)).toBeLessThan(0.2); // …but not once X is accounted for.
    expect(pc[0 * 3 + 1]!).toBeGreaterThan(0.5); // the real X–Y link survives
  });

  it("leaves a genuinely direct association standing", () => {
    const rnd = mulberry32(5);
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];
    for (let i = 0; i < 400; i++) {
      x.push(rnd() * 10);
      y.push(rnd() * 10);
      z.push(rnd() * 10);
    }
    // Y tied to X directly; Z independent of both.
    for (let i = 0; i < y.length; i++) y[i] = x[i]! + (rnd() - 0.5) * 2;
    const { r, k } = corrOf([x, y, z]);
    const pc = partialCorrelation(r, k);
    expect(pc[0 * 3 + 1]!).toBeGreaterThan(0.8);
    expect(Math.abs(pc[0 * 3 + 2]!)).toBeLessThan(0.2);
  });

  it("is symmetric with a unit diagonal", () => {
    const rnd = mulberry32(9);
    const rows = Array.from({ length: 5 }, () => Array.from({ length: 200 }, () => rnd()));
    const { r, k } = corrOf(rows);
    const pc = partialCorrelation(r, k);
    for (let a = 0; a < k; a++) {
      expect(pc[a * k + a]!).toBeCloseTo(1, 12);
      for (let b = 0; b < k; b++) expect(pc[a * k + b]!).toBeCloseTo(pc[b * k + a]!, 12);
    }
  });

  it("equals the plain correlation when there is nothing to condition on", () => {
    // With exactly two variables the conditioning set is empty, so the two statistics must coincide.
    const rnd = mulberry32(13);
    const x = Array.from({ length: 200 }, () => rnd());
    const y = x.map((v) => v * 0.7 + rnd() * 0.5);
    const { r, k } = corrOf([x, y]);
    const pc = partialCorrelation(r, k);
    expect(pc[1]!).toBeCloseTo(r[1]!, 10);
  });

  it("drops a variance-free type instead of poisoning the whole matrix", () => {
    // A type absent from the ROI makes its correlations NaN. Feeding that row to the inverse would
    // return NaN for every pair, not just its own — so it is excluded and only its row comes back NaN.
    const rnd = mulberry32(29);
    const x = Array.from({ length: 200 }, () => rnd());
    const y = x.map((v) => v + rnd() * 0.3);
    const dead = new Array(200).fill(0);
    const { r, k } = corrOf([x, y, dead]);
    const pc = partialCorrelation(r, k);
    expect(Number.isNaN(pc[2 * 3 + 0]!)).toBe(true);
    expect(Number.isFinite(pc[0 * 3 + 1]!)).toBe(true);
  });

  it("returns all-NaN rather than a fabricated answer when the matrix is singular", () => {
    // Z is an exact copy of X: conditioning on a set that already determines the row has no answer,
    // and a ridge term would invent one that looks like a measurement.
    const rnd = mulberry32(37);
    const x = Array.from({ length: 100 }, () => rnd());
    const y = Array.from({ length: 100 }, () => rnd());
    const { r, k } = corrOf([x, y, [...x]]);
    const pc = partialCorrelation(r, k);
    expect(pc.every((v) => Number.isNaN(v))).toBe(true);
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
