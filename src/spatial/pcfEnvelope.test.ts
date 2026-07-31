import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import { crossPCF } from "./pcf";
import { crossPCFEnvelope } from "./pcfEnvelope";

// Four things have to be true for the envelope to mean anything.
//
//   1. The curve the band surrounds is the SAME curve the rest of the UI draws. If the observed g
//      here disagreed with `crossPCF`, the picture would be a band around one statistic and a line
//      through another, and it would look entirely fine.
//   2. The test has its stated size — under a true null it rejects at about α, not more. This is
//      the property pointwise bands only appear to have.
//   3. It has power: a pattern built to violate the null is caught, or (2) is satisfiable by a test
//      that never rejects.
//   4. **It does not reject when the two patterns are independent but each self-clustered.** This is
//      the one that matters most here, because it is what the first implementation got wrong — it
//      rejected 20/20 on exactly this input, having tested "are A and B one interleaved population"
//      instead of "is A associated with B". `independentClustered` is a permanent guard against
//      that null coming back as the default.

const BBOX = [0, 0, 400, 400] as const;

/** Two independent uniform populations. */
function independent(nA: number, nB: number, seed: number) {
  const rnd = mulberry32(seed);
  const mk = (n: number) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      xs.push(rnd() * 400);
      ys.push(rnd() * 400);
    }
    return { xs, ys };
  };
  return { a: mk(nA), b: mk(nB) };
}

/** Clustered children around a given set of parents. */
function around(parents: { xs: number[]; ys: number[] }, perParent: number, spread: number, rnd: () => number) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let p = 0; p < parents.xs.length; p++) {
    for (let k = 0; k < perParent; k++) {
      xs.push(parents.xs[p]! + (rnd() - 0.5) * spread * 2);
      ys.push(parents.ys[p]! + (rnd() - 0.5) * spread * 2);
    }
  }
  return { xs, ys };
}

function parents(n: number, rnd: () => number) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(rnd() * 400);
    ys.push(rnd() * 400);
  }
  return { xs, ys };
}

/** Each type clustered around its OWN parents. Strongly self-clustered, mutually independent — so
 *  there is no association to find, and a correct association test must not report one. */
function independentClustered(seed: number) {
  const rnd = mulberry32(seed);
  return { a: around(parents(40, rnd), 25, 12, rnd), b: around(parents(40, rnd), 25, 12, rnd) };
}

/** Both types clustered around the SAME parents — genuine positive association. */
function coClustered(seed: number) {
  const rnd = mulberry32(seed);
  const shared = parents(40, rnd);
  return { a: around(shared, 25, 12, rnd), b: around(shared, 25, 12, rnd) };
}

describe("crossPCFEnvelope — the observed curve", () => {
  it.each(["shift", "label"] as const)("is exactly what crossPCF computes (%s null)", (nullModel) => {
    const { a, b } = independent(500, 400, 11);
    const p = { bbox: BBOX, rMax: 60, nBins: 6 } as const;
    const env = crossPCFEnvelope(a, b, { ...p, nullModel, simulations: 19, seed: 1 });
    const direct = crossPCF(a, b, p);
    for (let k = 0; k < 6; k++) expect(env.observed[k]!).toBeCloseTo(direct.g[k]!, 10);
    expect(env.r).toEqual(direct.r);
  });

  it.each(["shift", "label"] as const)("honours edgeCorrected, which raises the curve (%s null)", (nullModel) => {
    const { a, b } = independent(800, 800, 12);
    const p = { bbox: BBOX, rMax: 60, nBins: 3, simulations: 19, seed: 1, nullModel } as const;
    const plain = crossPCFEnvelope(a, b, p);
    const corrected = crossPCFEnvelope(a, b, { ...p, edgeCorrected: true });
    for (let k = 0; k < 3; k++) expect(corrected.observed[k]!).toBeGreaterThan(plain.observed[k]!);
  });
});

describe("crossPCFEnvelope — the shift null tests association", () => {
  it("does NOT reject independent patterns that are each self-clustered", () => {
    // The regression guard. The random-labelling null rejected this 20/20; the shift null must not,
    // because a rigid translation preserves each pattern's own clustering and destroys only their
    // relative position — which is the only thing that differs from independence here.
    let rejects = 0;
    const trials = 20;
    for (let t = 0; t < trials; t++) {
      const { a, b } = independentClustered(700 + t);
      const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 99, seed: 3 + t });
      if (env.envelope.exits) rejects++;
    }
    expect(rejects).toBeLessThanOrEqual(4); // ~1 expected at α=0.05; 20 was the bug
  });

  it("rejects genuine co-clustering", () => {
    const { a, b } = coClustered(23);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 199, seed: 3 });
    expect(env.envelope.exits).toBe(true);
    expect(env.envelope.p).toBeLessThanOrEqual(0.01);
    expect(env.observed[0]!).toBeGreaterThan(env.envelope.hi[0]!); // attraction, not exclusion
  });

  it("rejects segregation", () => {
    // A checkerboard, NOT two half-planes. The distinction is a genuine property of this null and
    // worth pinning: a rigid shift can reproduce a two-block split (translate B onto its own half
    // and the pattern is unchanged), so the null distribution there is enormously wide and includes
    // the observed. A fine checkerboard is only reproduced by shifts near a multiple of its period,
    // which are a vanishing fraction of the torus — so the observed really is extreme.
    const rnd = mulberry32(21);
    const ax: number[] = [];
    const ay: number[] = [];
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < 2000; i++) {
      const x = rnd() * 400;
      const y = rnd() * 400;
      const even = ((Math.floor(x / 40) + Math.floor(y / 40)) & 1) === 0;
      (even ? ax : bx).push(x);
      (even ? ay : by).push(y);
    }
    const env = crossPCFEnvelope({ xs: ax, ys: ay }, { xs: bx, ys: by }, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 199, seed: 3 });
    expect(env.envelope.exits).toBe(true);
    expect(env.observed[0]!).toBeLessThan(env.envelope.lo[0]!);
  });

  it("has its stated size on a true null", () => {
    let rejects = 0;
    const trials = 60;
    for (let t = 0; t < trials; t++) {
      const { a, b } = independent(300, 300, 1000 + t);
      const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, simulations: 39, alpha: 0.2, seed: 7000 + t });
      if (env.envelope.exits) rejects++;
    }
    // Binomial(60, 0.2): mean 12, sd ≈ 3.1. Generous bounds so this is not flaky, tight enough to
    // fail loudly on a test that rejects half the time or never.
    expect(rejects).toBeGreaterThan(2);
    expect(rejects).toBeLessThan(24);
  });
});

describe("crossPCFEnvelope — the label null tests exchangeability", () => {
  it("does not reject when A and B are an arbitrary split of ONE population", () => {
    // A COIN FLIP per point, not alternating indices. Alternating looks exchangeable and is not:
    // `around` emits each parent's children consecutively, so `i % 2` splits every cluster exactly
    // in half, which is more balanced than chance. A random labelling sometimes leaves a cluster
    // mostly A or mostly B, and that reduces cross pairs — so the stratified split sits
    // systematically above the null and this test rejected until the construction was fixed.
    const rnd = mulberry32(31);
    const all = around(parents(40, rnd), 25, 12, rnd);
    const ax: number[] = [];
    const ay: number[] = [];
    const bx: number[] = [];
    const by: number[] = [];
    for (let i = 0; i < all.xs.length; i++) {
      const toA = rnd() < 0.5;
      (toA ? ax : bx).push(all.xs[i]!);
      (toA ? ay : by).push(all.ys[i]!);
    }
    const env = crossPCFEnvelope(
      { xs: ax, ys: ay },
      { xs: bx, ys: by },
      {
        bbox: BBOX,
        rMax: 60,
        nBins: 6,
        nullModel: "label",
        simulations: 99,
        seed: 5,
      },
    );
    expect(env.envelope.exits).toBe(false);
    expect(env.envelope.p).toBeGreaterThan(0.05);
  });

  it("rejects independent self-clustered patterns — the reason it is not the default", () => {
    // Documented, not lamented: this null asks whether A ∪ B is one interleaved population, and for
    // two separately-clustered types the answer is trivially no. Pinned so that the difference
    // between the two nulls stays a deliberate choice rather than a surprise.
    const { a, b } = independentClustered(700);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 6, nullModel: "label", simulations: 99, seed: 3 });
    expect(env.envelope.exits).toBe(true);
    expect(env.observed[0]!).toBeLessThan(env.envelope.lo[0]!); // null expects MORE contact than independence
  });

  it("counts each unordered pair once", () => {
    const { a, b } = independent(200, 150, 13);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 40, nBins: 4, nullModel: "label", simulations: 19, seed: 1 });
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

describe("crossPCFEnvelope — guards", () => {
  it("refuses a self-pair", () => {
    const xs = [1, 2, 3];
    const ys = [1, 2, 3];
    expect(() => crossPCFEnvelope({ xs, ys }, { xs, ys }, { bbox: BBOX, rMax: 10, nBins: 2, simulations: 19 })).toThrow(/self-pair/);
  });

  it("refuses an empty population", () => {
    expect(() => crossPCFEnvelope({ xs: [], ys: [] }, { xs: [1], ys: [1] }, { bbox: BBOX, rMax: 10, nBins: 2 })).toThrow(/non-empty/);
  });

  it("refuses rather than hangs when the label null's structure is too large", () => {
    const { a, b } = independent(400, 400, 41);
    expect(() => crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 200, nBins: 4, nullModel: "label", simulations: 19, maxPairs: 100 })).toThrow(
      /exceeds maxPairs/,
    );
  });

  it.each(["shift", "label"] as const)("is reproducible for a given seed (%s null)", (nullModel) => {
    const { a, b } = independent(300, 300, 51);
    const p = { bbox: BBOX, rMax: 50, nBins: 5, simulations: 39, nullModel } as const;
    const one = crossPCFEnvelope(a, b, { ...p, seed: 99 });
    const two = crossPCFEnvelope(a, b, { ...p, seed: 99 });
    const other = crossPCFEnvelope(a, b, { ...p, seed: 100 });
    expect(one.envelope.p).toBe(two.envelope.p);
    expect([...one.envelope.lo]).toEqual([...two.envelope.lo]);
    expect([...other.envelope.lo]).not.toEqual([...one.envelope.lo]);
  });

  it("the shift null wraps: every simulation keeps every B point inside the ROI", () => {
    // Indirect but decisive — if the wrap leaked, B would drift out of the ROI and the simulated
    // curves would collapse toward zero as the shift grew.
    const { a, b } = independent(400, 400, 61);
    const env = crossPCFEnvelope(a, b, { bbox: BBOX, rMax: 60, nBins: 4, simulations: 99, seed: 12 });
    // Under a uniform null every simulated curve should sit near 1, not near 0.
    expect(env.envelope.lo[0]!).toBeGreaterThan(0.5);
    expect(env.envelope.hi[0]!).toBeLessThan(1.8);
  });
});
