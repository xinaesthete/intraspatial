import { describe, expect, it } from "vitest";
import { crossPCF } from "./pcf";
import type { PointPattern } from "./pointPatterns";
import {
  colocalised,
  csr,
  gradient,
  guilds,
  hardcore,
  independentClustered,
  makePointPattern,
  POINT_PATTERNS,
  patternClouds,
  segregated,
  thomas,
} from "./pointPatterns";

// The module's whole claim is that `truth.crossG` is the answer a correct estimator recovers. These
// tests are that claim, checked against `crossPCF` — which is independently validated against the
// published gr10/gr20 columns (`pnpm mdv:parity`). So a disagreement here means the closed form is
// wrong, and that is exactly what these need to catch: a plausible-looking formula that is off by a
// factor of 2 in σ would sail through any "is it bigger than 1" assertion.

const BINS = 20;
const R_MAX = 250;
const dr = R_MAX / BINS;

/** Estimate g_ab(r) with the edge correction on — the closed forms are for the plane, so an
 *  uncorrected estimate would be low by the ROI's perimeter-to-area ratio and every comparison
 *  below would fail for a reason that has nothing to do with the formulae. */
function estimate(p: PointPattern, a: number, b: number) {
  const clouds = patternClouds(p);
  return crossPCF(clouds[a]!, clouds[b]!, { bbox: p.bbox, rMax: R_MAX, nBins: BINS, edgeCorrected: true });
}

/** Average g over `seeds` realisations — one realisation of a point process is a noisy thing, and
 *  the alternative to averaging is a tolerance so wide it would accept a wrong formula. */
function meanG(make: (seed: number) => PointPattern, a: number, b: number, seeds: number): number[] {
  const acc = new Array<number>(BINS).fill(0);
  for (let s = 0; s < seeds; s++) {
    const g = estimate(make(s + 1), a, b).g;
    for (let k = 0; k < BINS; k++) acc[k]! += g[k]! / seeds;
  }
  return acc;
}

describe("pointPatterns — the registry", () => {
  it("makes every registered pattern, with dense type ids and a stated truth", () => {
    for (const spec of POINT_PATTERNS) {
      const p = makePointPattern(spec.key, { n: 400, seed: 3 });
      expect(p.xs.length, spec.key).toBe(p.ys.length);
      expect(p.typeId.length, spec.key).toBe(p.xs.length);
      expect(p.xs.length, spec.key).toBeGreaterThan(50);
      const ids = new Set(p.typeId);
      expect(
        [...ids].sort((x, y) => x - y),
        spec.key,
      ).toEqual(p.typeNames.map((_, i) => i));
      expect(p.truth.note.length, spec.key).toBeGreaterThan(20);
      // Every point inside the stated window: the bbox is what the statistics divide by, so a point
      // outside it would inflate ρ and quietly bias everything computed from the pattern.
      for (let i = 0; i < p.xs.length; i++) {
        expect(p.xs[i]! >= p.bbox[0] && p.xs[i]! < p.bbox[2], spec.key).toBe(true);
        expect(p.ys[i]! >= p.bbox[1] && p.ys[i]! < p.bbox[3], spec.key).toBe(true);
      }
    }
  });

  it("honours `n` as a per-type count, on every pattern", () => {
    // `n` drifting is invisible in g for the thinned and split patterns — thinning does not change
    // the pcf — so nothing else here would catch it. Two have already got this wrong: `gradient`
    // returned 3,097 per type for 1,200 asked (a mis-derived thinning factor) and the cluster
    // patterns lost a third of their points outside the expanded window.
    const N = 900;
    for (const spec of POINT_PATTERNS) {
      const p = makePointPattern(spec.key, { n: N, seed: 5 });
      for (let id = 0; id < p.typeNames.length; id++) {
        const got = p.typeId.filter((t) => t === id).length;
        // `hardcore` is the one that may legitimately fall short — SSI jams before it packs a window
        // — so it gets a floor rather than a band, and the exactness it DOES promise is tested
        // separately.
        const lo = spec.key === "hardcore" ? 0.4 : 0.8;
        expect(got / N, `${spec.key} type ${id}: ${got} for n=${N}`).toBeGreaterThan(lo);
        expect(got / N, `${spec.key} type ${id}: ${got} for n=${N}`).toBeLessThan(1.25);
      }
    }
  });

  it("is deterministic in the seed, and not in name only", () => {
    const a = makePointPattern("thomas", { n: 300, seed: 7 });
    const b = makePointPattern("thomas", { n: 300, seed: 7 });
    const c = makePointPattern("thomas", { n: 300, seed: 8 });
    expect(a.xs).toEqual(b.xs);
    expect(a.xs).not.toEqual(c.xs);
  });

  it("refuses an unknown key with the list of known ones", () => {
    expect(() => makePointPattern("banana")).toThrow(/banana.*csr/s);
  });
});

describe("pointPatterns — the closed forms are the answer crossPCF recovers", () => {
  it("csr: g ≡ 1, on both the cross and the auto pair", () => {
    const cross = meanG((s) => csr({ n: 1200, seed: s }), 0, 1, 6);
    for (let k = 0; k < BINS; k++) expect(cross[k]!, `bin ${k}`).toBeCloseTo(1, 1);
    // Auto: bin 0 carries the self-pairs crossPCF counts when given one cloud twice, so it is
    // skipped here rather than pretended away — see the module header.
    const auto = meanG((s) => csr({ n: 1200, seed: s }), 0, 0, 6);
    for (let k = 1; k < BINS; k++) expect(auto[k]!, `bin ${k}`).toBeCloseTo(1, 1);
  });

  it("thomas: the estimate tracks 1 + exp(−r²/4σ²)/(4πκσ²)", () => {
    // 150 parents, not the default 40: at 40 the estimator's own O(1/n_clusters) deficit reaches 7%
    // mid-range and would be indistinguishable from a wrong formula. That deficit is real and is
    // pinned by its own test below; this one is about the closed form, so it runs where the
    // estimator is unbiased.
    // Cross pair (0, 1), not the auto pair: the two labels are a coin flip over one process, so the
    // closed form applies to the cross-pcf too — and reading it there sidesteps the self-pair spike
    // in bin 0 entirely rather than skipping the bin and hoping.
    const make = (s: number) => thomas({ n: 1200, seed: s, sigma: 30, parents: 150 });
    const g = meanG(make, 0, 1, 10);
    const truth = make(1).truth;
    for (let k = 0; k <= 15; k++) {
      const r = (k + 0.5) * dr;
      const want = truth.crossG(0, 1, r)!;
      // A ratio, because g runs from ~1.8 at short range to 1.0 — an absolute band would be vacuous
      // at one end and impossible at the other.
      expect(Math.abs(g[k]! / want - 1), `bin ${k} (r≈${r.toFixed(0)}, want ${want.toFixed(3)}, got ${g[k]!.toFixed(3)})`).toBeLessThan(
        0.06,
      );
    }
    // Not merely "tracks a curve": it must be a DECREASING curve well above 1 at short range, or a
    // flat estimator would pass the ratio test on the tail, where the truth is 1 anyway.
    expect(g[1]!).toBeGreaterThan(1.5); // closed form says 1.82 here
    expect(g[BINS - 1]!).toBeLessThan(1.1);
  });

  it("colocalised: the cross estimate tracks 1 + exp(−r²/2σ²)/(2πσ²λ_A)", () => {
    const make = (s: number) => colocalised({ n: 1200, seed: s, sigma: 25 });
    const g = meanG(make, 0, 1, 10);
    const truth = make(1).truth;
    for (let k = 0; k < BINS; k++) {
      const r = (k + 0.5) * dr;
      const want = truth.crossG(0, 1, r)!;
      expect(Math.abs(g[k]! / want - 1), `bin ${k} (r≈${r.toFixed(0)}, want ${want.toFixed(3)}, got ${g[k]!.toFixed(3)})`).toBeLessThan(
        0.06,
      );
    }
    // The elevation is 1/(2πσ²λ_A) at r → 0, which at these settings is 0.21 — modest, and stated
    // rather than guessed at. An earlier version asserted g[0] > 2 on no derivation at all and only
    // passed because a broken Poisson deviate was halving λ_A.
    expect(g[0]!).toBeGreaterThan(1.15);
    expect(g[BINS - 1]!).toBeLessThan(1.05);
  });

  it("independentClustered: each type clumps, the pair does not", () => {
    const make = (s: number) => independentClustered({ n: 1200, seed: s, sigma: 30, parents: 200 });
    const cross = meanG(make, 0, 1, 6);
    for (let k = 0; k < BINS; k++) expect(cross[k]!, `cross bin ${k}`).toBeCloseTo(1, 1);
    const auto = meanG(make, 0, 0, 6);
    expect(auto[1]!).toBeGreaterThan(1.5); // and the auto-pcf is emphatically not 1
  });

  it("independentClustered: the estimator's short-range shortfall is O(1/clusters), not a bug here", () => {
    // The truth is 1 at every r, and `crossPCF` does not say so when the clusters are few: it
    // normalises by the observed global ρ̂_B, and a realisation with 40 clumps is a lumpy intensity
    // surface rather than a flat one. Asserting the CONVERGENCE pins the cause — a generator that
    // secretly correlated the two types would be biased at every cluster count, and a broken
    // estimator would not improve either. Both alternatives fail this; a fat tolerance around 1
    // would have let both through.
    const at = (parents: number) => meanG((s) => independentClustered({ n: 1200, seed: s, sigma: 30, parents }), 0, 1, 8)[0]!;
    const few = at(40);
    const many = at(400);
    expect(few).toBeLessThan(0.97); // the shortfall is real and worth knowing about
    expect(Math.abs(many - 1)).toBeLessThan(Math.abs(few - 1)); // and it is the cluster count causing it
    expect(many).toBeCloseTo(1, 1);
  });

  it("segregated: cross-g is exactly 0 below the gap, not merely small", () => {
    // The one assertion in this file that needs no tolerance at all. No estimator, however biased,
    // can report a pair that does not exist.
    const p = segregated({ n: 900, seed: 4, gap: 120 });
    const res = estimate(p, 0, 1);
    for (let k = 0; k < BINS; k++) {
      const rOuter = (k + 1) * dr;
      if (rOuter <= 120) expect(res.counts[k]!, `bin ${k}`).toBe(0);
    }
    expect(res.counts.some((c) => c > 0)).toBe(true); // ... and it is not empty everywhere
  });

  it("hardcore: no pair closer than the core, exactly — across labels too", () => {
    const p = hardcore({ n: 350, seed: 5, core: 25 });
    let closest = Infinity;
    for (let i = 0; i < p.xs.length; i++) {
      for (let j = i + 1; j < p.xs.length; j++) {
        closest = Math.min(closest, Math.hypot(p.xs[i]! - p.xs[j]!, p.ys[i]! - p.ys[j]!));
      }
    }
    expect(closest).toBeGreaterThanOrEqual(25);
    expect(p.xs.length).toBeGreaterThan(300); // the attempt budget did not starve it
  });

  it("gradient: the TRUE cross-g is 1, and the homogeneous estimator says otherwise", () => {
    // This is the fixture's whole purpose, so the test states both halves. The types are built
    // independently, so no association exists; a Mode-1 estimator divides by a single global ρ_B and
    // therefore reports one anyway. If this ever starts passing at g ≈ 1, either the estimator grew
    // an inhomogeneous null or the generator stopped ramping — both worth knowing.
    const make = (s: number) => gradient({ n: 1500, seed: s, contrast: 8 });
    expect(make(1).truth.crossG(0, 1, 50)).toBe(1);
    const g = meanG(make, 0, 1, 6);
    expect(g[2]!).toBeGreaterThan(1.05);
  });
});

describe("patternClouds", () => {
  it("splits without losing or duplicating a point", () => {
    const p = makePointPattern("colocalised", { n: 500, seed: 2 });
    const clouds = patternClouds(p);
    expect(clouds.length).toBe(p.typeNames.length);
    expect(clouds.reduce((a, c) => a + c.xs.length, 0)).toBe(p.xs.length);
    for (const c of clouds) expect(c.xs.length).toBe(c.ys.length);
  });
});

describe("pointPatterns — beyond a pair of types", () => {
  it("every pattern honours `types`, with dense ids and a truth for every pair", () => {
    for (const spec of POINT_PATTERNS) {
      const K = 5;
      const p = makePointPattern(spec.key, { n: 300, seed: 4, types: K });
      expect(p.typeNames.length, spec.key).toBe(K);
      expect(new Set(p.typeId).size, spec.key).toBe(K);
      // The truth has to answer for the whole matrix, not just (0,1) — a pair-shaped closure that
      // ignored its arguments would pass every test written against two types.
      for (let a = 0; a < K; a++) {
        for (let b = 0; b < K; b++) {
          const v = p.truth.crossG(a, b, 30);
          expect(v === undefined || (Number.isFinite(v) && v >= 0), `${spec.key} (${a},${b}) → ${v}`).toBe(true);
        }
      }
    }
  });

  it("guilds: the true matrix is block diagonal, and the estimate recovers the blocks", () => {
    const K = 6;
    const G = 2;
    const make = (s: number) => guilds({ n: 900, seed: s, types: K, guilds: G, sigma: 40, parents: 150 });
    const p = make(1);
    // The truth itself must be blocked — same guild elevated, different guild exactly 1.
    expect(p.truth.crossG(0, 1, 20)!).toBeGreaterThan(1.2); // 0 and 1 share guild 1
    expect(p.truth.crossG(0, 5, 20)!).toBe(1); // 0 and 5 do not
    expect(p.truth.crossG(4, 5, 20)!).toBeGreaterThan(1.2);

    // And the ESTIMATE must reproduce that split. Averaged over seeds because a clustered pattern is
    // noisy; the claim is the separation between within- and between-guild, not either value.
    const within: number[] = [];
    const between: number[] = [];
    for (let s = 1; s <= 4; s++) {
      const q = make(s);
      const clouds = patternClouds(q);
      for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
          const g = crossPCF(clouds[a]!, clouds[b]!, { bbox: q.bbox, rMax: 120, nBins: 4, edgeCorrected: true }).g[0]!;
          (q.truth.crossG(a, b, 15)! > 1 ? within : between).push(g);
        }
      }
    }
    const mean = (v: number[]) => v.reduce((x, y) => x + y, 0) / v.length;
    // Every within-guild pair must beat every between-guild pair — a clean separation, not just a
    // difference of means, since the point of the fixture is that the blocks are RECOVERABLE.
    expect(Math.min(...within), `within ${mean(within).toFixed(2)} vs between ${mean(between).toFixed(2)}`).toBeGreaterThan(
      Math.max(...between),
    );
    expect(mean(between)).toBeCloseTo(1, 0.7);
  });

  it("colocalised: recruited↔recruited uses 4σ², anchor↔recruited uses 2σ²", () => {
    // The two legs have DIFFERENT widths and it is easy to use one formula for both: an
    // anchor-offspring pair is one Gaussian displacement, a pair of offspring sharing an anchor is
    // the difference of two. Getting it wrong inflates the correlation length by √2 and still looks
    // like a decaying curve, so it is checked against the estimate on both legs.
    const make = (s: number) => colocalised({ n: 1200, seed: s, types: 3, sigma: 25 });
    const anchorLeg = meanG(make, 0, 1, 10);
    const sharedLeg = meanG(make, 1, 2, 10);
    const t = make(1).truth;
    for (let k = 0; k < 8; k++) {
      const r = (k + 0.5) * dr;
      expect(Math.abs(anchorLeg[k]! / t.crossG(0, 1, r)! - 1), `anchor leg bin ${k}`).toBeLessThan(0.08);
      expect(Math.abs(sharedLeg[k]! / t.crossG(1, 2, r)! - 1), `shared leg bin ${k}`).toBeLessThan(0.08);
    }
    // And they are genuinely different curves, so the test above is not vacuous.
    expect(t.crossG(0, 1, 25)!).toBeGreaterThan(t.crossG(1, 2, 25)!);
  });

  it("segregated: the exclusion is graded by strip distance, so the ordering is recoverable", () => {
    const K = 4;
    const p = segregated({ n: 500, seed: 6, types: K, gap: 60 });
    const clouds = patternClouds(p);
    // Neighbouring strips can be 60 apart; the two ends cannot be closer than 3 gaps + 2 widths.
    const near = p.truth.crossG(0, 1, 59)!;
    expect(near).toBe(0);
    expect(p.truth.crossG(0, 3, 59)).toBe(0);
    // Brute-force the real minimum separation and check the truth never over-claims.
    for (let a = 0; a < K; a++) {
      for (let b = a + 1; b < K; b++) {
        let closest = Infinity;
        for (let i = 0; i < clouds[a]!.xs.length; i++) {
          for (let j = 0; j < clouds[b]!.xs.length; j++) {
            closest = Math.min(closest, Math.hypot(clouds[a]!.xs[i]! - clouds[b]!.xs[j]!, clouds[a]!.ys[i]! - clouds[b]!.ys[j]!));
          }
        }
        // Whatever radius the truth claims is empty must really be empty.
        let claimed = 0;
        for (let r = 1; r < 600; r++) if (p.truth.crossG(a, b, r) === 0) claimed = r;
        expect(closest, `strips ${a},${b}: closest ${closest.toFixed(1)}, claimed empty below ${claimed}`).toBeGreaterThanOrEqual(claimed);
      }
    }
  });
});
