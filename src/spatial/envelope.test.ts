import { describe, expect, it } from "vitest";
import { globalRankEnvelope } from "./envelope";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Box-Muller, so the coverage test runs against a continuous distribution with no ties. */
function normals(n: number, r: () => number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(r(), 1e-12);
    const v = r();
    const mag = Math.sqrt(-2 * Math.log(u));
    out[i] = mag * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = mag * Math.sin(2 * Math.PI * v);
  }
  return out;
}

describe("globalRankEnvelope", () => {
  it("has the coverage it claims — the whole reason it exists", () => {
    // The load-bearing property, and the one a pointwise band fails. Draw s+1 exchangeable curves,
    // call one of them "observed", and the test must reject at α. Anything else means the envelope
    // is decorative. Both readings are checked: the ERL p-value (the test) and leaving the drawn
    // band (what a reader sees), since either being wrong is a wrong claim.
    const r = rng(7);
    const D = 8;
    const S = 99;
    const REPS = 1000;
    let byP = 0;
    let byBand = 0;
    for (let rep = 0; rep < REPS; rep++) {
      const observed = normals(D, r);
      const simulated = Array.from({ length: S }, () => normals(D, r));
      const env = globalRankEnvelope(observed, simulated, { alpha: 0.05 });
      if (env.p <= 0.05) byP++;
      if (env.exits) byBand++;
    }
    // sd of a 5% rate over 1000 reps is 0.0069, so these are ±3.5σ bounds.
    expect(byP / REPS).toBeGreaterThan(0.026);
    expect(byP / REPS).toBeLessThan(0.074);
    // The band is the deepest rank level with size <= alpha, so it is conservative by construction
    // and never anti-conservative.
    expect(byBand / REPS).toBeLessThan(0.074);
  });

  it("a pointwise band over the same curves rejects far too often — the mistake being avoided", () => {
    // Not testing our code: constructing the naive alternative to show the size of the error, so
    // the choice in envelope.ts is justified by a number rather than by assertion.
    const r = rng(11);
    const D = 8;
    const S = 99;
    const REPS = 600;
    let rejected = 0;
    for (let rep = 0; rep < REPS; rep++) {
      const observed = normals(D, r);
      const simulated = Array.from({ length: S }, () => normals(D, r));
      let exits = false;
      for (let d = 0; d < D; d++) {
        const col = simulated.map((c) => c[d]!).sort((a, b) => a - b);
        // 2.5% / 97.5% pointwise, the textbook-looking band.
        if ((observed[d] ?? 0) < col[Math.floor(0.025 * S)]! || (observed[d] ?? 0) > col[Math.ceil(0.975 * S) - 1]!) {
          exits = true;
        }
      }
      if (exits) rejected++;
    }
    // Nominally 5%; measured around 20% at D = 8. That gap is the multiplicity being ignored.
    expect(rejected / REPS).toBeGreaterThan(0.12);
  });

  it("gives the smallest attainable p to a curve that is extreme everywhere", () => {
    const r = rng(3);
    const simulated = Array.from({ length: 99 }, () => normals(5, r));
    const observed = new Float64Array(5).fill(50);
    const env = globalRankEnvelope(observed, simulated);
    expect(env.observedRank).toBe(1);
    expect(env.p).toBeCloseTo(1 / 100, 12);
    expect(env.exits).toBe(true);
  });

  it("does not reject a curve sitting in the middle of the simulations", () => {
    const r = rng(5);
    const simulated = Array.from({ length: 99 }, () => normals(5, r));
    const observed = new Float64Array(5);
    for (let d = 0; d < 5; d++) {
      const col = simulated.map((c) => c[d]!).sort((a, b) => a - b);
      observed[d] = col[49]!; // the median at every point
    }
    const env = globalRankEnvelope(observed, simulated);
    expect(env.exits).toBe(false);
    expect(env.p).toBeGreaterThan(0.05);
    for (let d = 0; d < 5; d++) {
      expect(observed[d]!).toBeGreaterThanOrEqual(env.lo[d]!);
      expect(observed[d]!).toBeLessThanOrEqual(env.hi[d]!);
    }
  });

  it("the band and the p-value are one statement, not two", () => {
    // Leaving the drawn band must imply significance, or a reader could see a curve outside the
    // envelope next to a non-significant p — the failure mode that makes an envelope worse than
    // nothing. The converse is not asserted: the band is the range of the retained curves, so a
    // curve can be the most extreme by ERL and still sit just inside their hull.
    const r = rng(13);
    let agree = 0;
    const reps = 300;
    for (let rep = 0; rep < reps; rep++) {
      const simulated = Array.from({ length: 39 }, () => normals(6, r));
      const env = globalRankEnvelope(normals(6, r), simulated, { alpha: 0.1 });
      if (env.exits) expect(env.p).toBeLessThanOrEqual(env.alpha);
      if (env.exits === env.p <= env.alpha) agree++;
    }
    expect(agree / reps).toBeGreaterThan(0.95);
  });

  it("refuses a level it cannot attain rather than returning a band that is not one", () => {
    const r = rng(17);
    const simulated = Array.from({ length: 10 }, () => normals(4, r));
    // alpha·(s+1) = 0.55 < 1: no rank threshold excludes anything.
    expect(() => globalRankEnvelope(normals(4, r), simulated, { alpha: 0.05 })).toThrow(/at least/);
    // Same simulations, an attainable level.
    expect(() => globalRankEnvelope(normals(4, r), simulated, { alpha: 0.2 })).not.toThrow();
  });

  it("is invariant to the order the simulations arrive in", () => {
    const r = rng(23);
    const simulated = Array.from({ length: 49 }, () => normals(5, r));
    const observed = normals(5, r);
    const a = globalRankEnvelope(observed, simulated);
    const b = globalRankEnvelope(observed, [...simulated].reverse());
    expect(b.p).toBe(a.p);
    expect([...b.lo]).toEqual([...a.lo]);
    expect([...b.hi]).toEqual([...a.hi]);
  });
});
