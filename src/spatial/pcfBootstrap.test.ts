import { describe, expect, it } from "vitest";
import { mulberry32 } from "./kernelAnalysis";
import { crossPCF } from "./pcf";
import { crossPCFBootstrap } from "./pcfBootstrap";

// The CI is only worth drawing if (1) the line it straddles is the same line the rest of the UI
// shows, and (2) the interval covers the truth at its stated rate. (2) is the one that a plausible
// but wrong resampling scheme would fail while still producing a pretty band.

const BBOX = [0, 0, 400, 400] as const;

function uniform(n: number, rnd: () => number) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(rnd() * 400);
    ys.push(rnd() * 400);
  }
  return { xs, ys };
}

describe("crossPCFBootstrap", () => {
  const p = { bbox: BBOX, rMax: 60, nBins: 6 } as const;

  it("returns exactly the crossPCF curve", () => {
    const rnd = mulberry32(3);
    const a = uniform(600, rnd);
    const b = uniform(500, rnd);
    const boot = crossPCFBootstrap(a, b, p, { resamples: 99, seed: 1 });
    const direct = crossPCF(a, b, p);
    for (let k = 0; k < 6; k++) expect(boot.g[k]!).toBeCloseTo(direct.g[k]!, 10);
    expect(boot.r).toEqual(direct.r);
  });

  it("returns exactly the crossPCF curve when edge-corrected too", () => {
    const rnd = mulberry32(4);
    const a = uniform(600, rnd);
    const b = uniform(500, rnd);
    const q = { ...p, edgeCorrected: true } as const;
    const boot = crossPCFBootstrap(a, b, q, { resamples: 99, seed: 1 });
    const direct = crossPCF(a, b, q);
    for (let k = 0; k < 6; k++) expect(boot.g[k]!).toBeCloseTo(direct.g[k]!, 10);
  });

  it("straddles the estimate — it is an interval for g, not for a null", () => {
    const rnd = mulberry32(5);
    const boot = crossPCFBootstrap(uniform(800, rnd), uniform(800, rnd), p, { resamples: 299, seed: 2 });
    for (let k = 0; k < 6; k++) {
      expect(boot.lo[k]!).toBeLessThanOrEqual(boot.g[k]!);
      expect(boot.hi[k]!).toBeGreaterThanOrEqual(boot.g[k]!);
    }
  });

  it("narrows as the anchor sample grows, roughly as 1/√N_A", () => {
    const width = (nA: number, seed: number) => {
      const rnd = mulberry32(seed);
      const boot = crossPCFBootstrap(uniform(nA, rnd), uniform(4000, rnd), p, { resamples: 299, seed: 2 });
      return boot.hi[3]! - boot.lo[3]!;
    };
    const small = width(250, 11);
    const large = width(4000, 11);
    expect(large).toBeLessThan(small);
    // 16× the anchors should be about 4× tighter; allow a wide factor for sampling noise.
    const ratio = small / large;
    expect(ratio).toBeGreaterThan(2);
    expect(ratio).toBeLessThan(8);
  });

  it("covers the truth at about its stated rate — WITH the edge correction", () => {
    // Under independence the true g(r) is 1 in every bin, so coverage is directly measurable, and a
    // resampling scheme that got the unit wrong (pairs instead of anchors, say) would produce a band
    // that looks fine and covers at the wrong rate.
    //
    // `edgeCorrected` is not optional here, and finding that out is the useful part: uncorrected,
    // coverage measured 0/120. A confidence interval is an interval for what the ESTIMATOR converges
    // to, and the uncorrected estimator converges low — by perimeter·r/area, which at r ∈ [40,50) on
    // a 400×400 ROI is far larger than the width of a 700-anchor CI. So the band was correct and
    // simply centred on the wrong number. Anything reading a CI as "the truth is in here" needs the
    // bias gone first; that is why the panel draws the corrected estimator.
    let covered = 0;
    const trials = 120;
    for (let t = 0; t < trials; t++) {
      const rnd = mulberry32(400 + t);
      const boot = crossPCFBootstrap(
        uniform(700, rnd),
        uniform(700, rnd),
        { ...p, edgeCorrected: true },
        { resamples: 199, alpha: 0.1, seed: 900 + t },
      );
      // One bin, well inside the range, so this is a pointwise coverage check at 90%.
      if (boot.lo[4]! <= 1 && 1 <= boot.hi[4]!) covered++;
    }
    // Binomial(120, 0.9): mean 108, sd ≈ 3.3. Generous bounds — the bootstrap is approximate.
    expect(covered).toBeGreaterThan(94);
    expect(covered).toBeLessThanOrEqual(120);
  });

  it("is reproducible for a seed and responds to alpha", () => {
    const rnd = mulberry32(7);
    const a = uniform(500, rnd);
    const b = uniform(500, rnd);
    const one = crossPCFBootstrap(a, b, p, { resamples: 199, seed: 42 });
    const two = crossPCFBootstrap(a, b, p, { resamples: 199, seed: 42 });
    expect([...one.lo]).toEqual([...two.lo]);
    const wide = crossPCFBootstrap(a, b, p, { resamples: 199, seed: 42, alpha: 0.5 });
    expect(wide.hi[2]! - wide.lo[2]!).toBeLessThan(one.hi[2]! - one.lo[2]!);
  });

  it("survives an empty population without throwing", () => {
    const rnd = mulberry32(9);
    const empty = crossPCFBootstrap({ xs: [], ys: [] }, uniform(50, rnd), p, { resamples: 19 });
    expect(empty.nA).toBe(0);
    expect([...empty.g]).toEqual([0, 0, 0, 0, 0, 0]);
  });
});
