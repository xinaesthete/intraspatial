import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import { crossPCF } from "./pcf";
import { crossPCFEnvelope } from "./pcfEnvelope";

// Three things have to be true for the envelope to mean anything, and each is checked separately.
//
//   1. The curve the band surrounds is the SAME curve the rest of the UI draws. If the observed g
//      here disagreed with `crossPCF`, the picture would be a band around one statistic and a line
//      through another, and it would look entirely fine.
//   2. The test has its stated size. Under a null pattern it must reject at about α, not more —
//      this is the property pointwise bands only appear to have, and the reason this file exists.
//   3. It has power. A pattern built to violate the null must actually be caught, or (2) is
//      satisfiable by a test that never rejects anything.

const BBOX = [0, 0, 400, 400] as const;

/** Two independent uniform populations — the null is true by construction. */
function independent(nA: number, nB: number, seed: number) {
  const rnd = mulberry32(seed);
  const ax: number[] = [];
  const ay: number[] = [];
  const bx: number[] = [];
  const by: number[] = [];
  for (let i = 0; i < nA; i++) {
    ax.push(rnd() * 400);
    ay.push(rnd() * 400);
  }
  for (let i = 0; i < nB; i++) {
    bx.push(rnd() * 400);
    by.push(rnd() * 400);
  }
  return { a: { xs: ax, ys: ay }, b: { xs: bx, ys: by } };
}

/** Segregated: A in the left half, B in the right. Random labelling must reject this. */
function segregated(n: number, seed: number) {
  const rnd = mulberry32(seed);
  const ax: number[] = [];
  const ay: number[] = [];
  const bx: number[] = [];
  const by: number[] = [];
  for (let i = 0; i < n; i++) {
    ax.push(rnd() * 190);
    ay.push(rnd() * 400);
    bx.push(210 + rnd() * 190);
    by.push(rnd() * 400);
  }
  return { a: { xs: ax, ys: ay }, b: { xs: bx, ys: by } };
}

describe("crossPCFEnvelope — the observed curve", () => {
  it("is exactly what crossPCF computes, so the band and the line are one statistic", () => {
    const { a, b } = independent(500, 400, 11);
    const p = { bbox: BBOX, rMax: 60, nBins: 6 } as const;
    const env = crossPCFEnvelope(a, b, { ...p, simulations: 19, seed: 1 });
    const direct = crossPCF(a, b, p);
    for (let k = 0; k < 6; k++) expect(env.observed[k]!).toBeCloseTo(direct.g[k]!, 10);
    expect(env.r).toEqual(direct.r);
  });

  it("honours edgeCorrected, and the correction raises the curve", () => {
    const { a, b } = independent(800, 800, 12);
    const p = { bbox: BBOX, rMax: 60, nBins: 3, simulations: 19, seed: 1 } as const;
    const plain = crossPCFEnvelope(a, b, p);
    const corrected = crossPCFEnvelope(a, b, { ...p, edgeCorrected: true });
    for (let k = 0; k < 3; k++) expect(corrected.observed[k]!).toBeGreaterThan(plain.observed[k]!);
  });

  it("counts each unordered pair once", () => {
    const { a, b } = independent(200, 150, 13);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 40, nBins: 4, simulations: 19, seed: 1 });
    // Brute force over the union, i < j, within range.
    const xs = [...a.xs, ...b.xs];
    const ys = [...a.ys, ...b.ys];
    let brute = 0;
    for (let i = 0; i < xs.length; i++) {
      for (let j = i + 1; j < xs.length; j++) {
        const dx = xs[j]! - xs[i]!;
        const dy = ys[j]! - ys[i]!;
        if (dx * dx + dy * dy < 1600) brute++;
      }
    }
    expect(env.pairs).toBe(brute);
  });
});

describe("crossPCFEnvelope — size and power", () => {
  it("rejects an independent pattern at about α, not more", () => {
    // The property the whole construction exists for. 60 trials at α=0.2 (a level coarse enough to
    // be measurable with few trials): the expected count is 12, and a badly-sized test — the
    // pointwise band this replaces — would land far above.
    let rejects = 0;
    const trials = 60;
    for (let t = 0; t < trials; t++) {
      const { a, b } = independent(300, 300, 1000 + t);
      const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 39, alpha: 0.2, seed: 7000 + t });
      if (env.envelope.exits) rejects++;
    }
    // Binomial(60, 0.2) has sd ≈ 3.1; allow a generous ±3.5 sd so this is not flaky, while still
    // failing loudly on a test that rejects half the time.
    expect(rejects).toBeGreaterThan(2);
    expect(rejects).toBeLessThan(24);
  });

  it("catches a segregated pattern", () => {
    const { a, b } = segregated(400, 21);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 199, seed: 3 });
    expect(env.envelope.exits).toBe(true);
    expect(env.envelope.p).toBeLessThanOrEqual(0.01);
    // Segregation means fewer A–B pairs at short range than labelling at random would give.
    expect(env.observed[0]!).toBeLessThan(env.envelope.lo[0]!);
  });

  it("does not reject when A and B are interleaved copies of one pattern", () => {
    // Alternate labels along a common point set: as close to the null being exactly true as a
    // constructed example gets.
    const rnd = mulberry32(31);
    const ax: number[] = [];
    const ay: number[] = [];
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 800; i++) {
      const x = rnd() * 400;
      const y = rnd() * 400;
      if (i % 2 === 0) {
        ax.push(x);
        ay.push(y);
      } else {
        bx.push(x);
        by.push(y);
      }
    }
    const env = crossPCFEnvelope({ xs: ax, ys: ay }, { xs: bx, ys: by }, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 99, seed: 5 });
    expect(env.envelope.exits).toBe(false);
    expect(env.envelope.p).toBeGreaterThan(0.05);
  });
});

describe("crossPCFEnvelope — guards", () => {
  it("refuses a self-pair, where the union null has no content", () => {
    const xs = [1, 2, 3];
    const ys = [1, 2, 3];
    expect(() => crossPCFEnvelope({ xs, ys }, { xs, ys }, { bbox: BBOX, rMax: 10, nBins: 2, simulations: 19 })).toThrow(/self-pair/);
  });

  it("refuses an empty population", () => {
    expect(() => crossPCFEnvelope({ xs: [], ys: [] }, { xs: [1], ys: [1] }, { bbox: BBOX, rMax: 10, nBins: 2 })).toThrow(/non-empty/);
  });

  it("refuses rather than hangs when the neighbour structure is too large", () => {
    const { a, b } = independent(400, 400, 41);
    expect(() => crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 200, nBins: 4, simulations: 19, maxPairs: 100 })).toThrow(/exceeds maxPairs/);
  });

  it("is reproducible for a given seed and varies without one", () => {
    const { a, b } = independent(300, 300, 51);
    const p = { bbox: BBOX, rMax: 50, nBins: 5, simulations: 39 } as const;
    const one = crossPCFEnvelope(a, b, { ...p, seed: 99 });
    const two = crossPCFEnvelope(a, b, { ...p, seed: 99 });
    const other = crossPCFEnvelope(a, b, { ...p, seed: 100 });
    expect(one.envelope.p).toBe(two.envelope.p);
    expect([...one.envelope.lo]).toEqual([...two.envelope.lo]);
    // A different seed gives a different band (it would be a broken RNG otherwise).
    expect([...other.envelope.lo]).not.toEqual([...one.envelope.lo]);
  });
});
