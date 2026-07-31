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

  it("narrows as the anchor sample grows, roughly as 1/√N_A — under the ANCHOR scheme", () => {
    const width = (nA: number, seed: number) => {
      const rnd = mulberry32(seed);
      const boot = crossPCFBootstrap(uniform(nA, rnd), uniform(4000, rnd), p, { resamples: 299, seed: 2, scheme: "anchor" });
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

  it("is governed by the TILE count, not the anchor count, under the block scheme", () => {
    // The property that distinguishes the two schemes, and the reason the test above had to be
    // pinned to `scheme: "anchor"`. A block bootstrap's effective sample size is the number of
    // resampling units — tiles — so pouring 16× more anchors into the same 16 tiles buys far less
    // than 1/√N_A. Measured, the same change that tightens the anchor band by ~4× moves this one by
    // about 1.3×. Reading a block band as if it narrowed like √N_A would badly overstate what a
    // dense ROI tells you.
    const width = (nA: number) => {
      const rnd = mulberry32(11);
      const boot = crossPCFBootstrap(uniform(nA, rnd), uniform(4000, rnd), p, { resamples: 299, seed: 2 });
      return boot.hi[3]! - boot.lo[3]!;
    };
    const ratio = width(250) / width(4000);
    expect(ratio).toBeGreaterThan(1);
    expect(ratio).toBeLessThan(2);
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

  it("beats the anchor scheme on a CLUSTERED pattern, which is the whole reason for it", () => {
    // Under independence both schemes are fine (the test above). The block bootstrap earns its keep
    // when anchors are NOT independent, which is the case the cross-PCF exists to measure.
    //
    // A Thomas cluster process: parents scattered uniformly, offspring scattered tightly around
    // them. Nearby A anchors then see largely the SAME B cells, so their contributions are
    // correlated — and an anchor bootstrap, which draws them as if independent, understates the
    // variance and reports a band that is too narrow. Truth is still g = 1 in every bin because A
    // and B are generated independently of one another; only the within-type clustering is shared.
    function thomas(nParents: number, perParent: number, spread: number, rnd: () => number) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < nParents; i++) {
        const px = rnd() * 400;
        const py = rnd() * 400;
        for (let j = 0; j < perParent; j++) {
          // Box-Muller, clamped back into the ROI so the density stays roughly uniform.
          const u = Math.max(1e-9, rnd());
          const rad = spread * Math.sqrt(-2 * Math.log(u));
          const th = 2 * Math.PI * rnd();
          xs.push(Math.min(399.99, Math.max(0, px + rad * Math.cos(th))));
          ys.push(Math.min(399.99, Math.max(0, py + rad * Math.sin(th))));
        }
      }
      return { xs, ys };
    }

    const trials = 100;
    const hit = { block: 0, anchor: 0 };
    for (let t = 0; t < trials; t++) {
      const rnd = mulberry32(9000 + t);
      const a = thomas(40, 18, 12, rnd);
      const b = thomas(40, 18, 12, rnd);
      for (const scheme of ["block", "anchor"] as const) {
        const boot = crossPCFBootstrap(a, b, { ...p, edgeCorrected: true }, { resamples: 199, alpha: 0.05, seed: 1200 + t, scheme });
        if (boot.lo[4]! <= 1 && 1 <= boot.hi[4]!) hit[scheme]++;
      }
    }
    // The block scheme must be near its nominal 95% and must beat the anchor scheme outright. The
    // bounds are loose because both are approximations and 100 trials is a coarse instrument; the
    // ORDERING is the claim, and it is not marginal — over 300 realisations it is 98% against 40%.
    expect(hit.block).toBeGreaterThan(80);
    expect(hit.block).toBeGreaterThan(hit.anchor + 10);
  });

  it("reports the tile pool it used, empty tiles included", () => {
    // The pool size is a property of the ROI and the tile side, not of where the cells happen to be;
    // dropping empty tiles would narrow the interval and is exactly the bug to guard against.
    const rnd = mulberry32(31);
    const res = crossPCFBootstrap(uniform(300, rnd), uniform(300, rnd), p, { resamples: 49, blockSize: 100 });
    expect(res.scheme).toBe("block");
    expect(res.blocks).toBe(16); // 400 × 400 ROI at 100 µm tiles
    const anchor = crossPCFBootstrap(uniform(300, rnd), uniform(300, rnd), p, { resamples: 49, scheme: "anchor" });
    expect(anchor.blocks).toBe(0);
  });

  it("is insensitive to the tile side on INDEPENDENT points, and that is not a bug", () => {
    // Worth pinning because the intuition "bigger tiles → fewer of them → wider band" is wrong here,
    // and I asserted it before measuring. The ratio estimator's variance is the between-tile
    // variance divided by the tile count; on uniform points, doubling the tile side averages away
    // that variance at the same rate as it removes tiles, so the two cancel and the width barely
    // moves. Tile size only matters once there is dependence for a bigger tile to capture — which is
    // exactly the clustered case the scheme exists for.
    const rnd = mulberry32(37);
    const a = uniform(900, rnd);
    const b = uniform(900, rnd);
    const width = (blockSize: number) => {
      const res = crossPCFBootstrap(a, b, { ...p, edgeCorrected: true }, { resamples: 299, blockSize, seed: 5 });
      return res.hi[4]! - res.lo[4]!;
    };
    const ratio = width(200) / width(50);
    expect(ratio).toBeGreaterThan(0.6);
    expect(ratio).toBeLessThan(1.7);
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
