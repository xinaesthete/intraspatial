// Targeted correctness for the fiddly HsPf per-cell math + a few structural invariants
// (ADR-0011, decision 5) — the silent-wrong-math failure mode (offspring-table indexing,
// fitness blend, LD denominator, gather normalisation) caught without a whole-field golden.
import { describe, expect, it } from "vitest";
import { blendFitness, DEFAULT_FITNESS, gatherCell, ld, OFFSPRING, reactBite, recombine, selectionWeights, sum4, type Vec4 } from "./math";
import { betaOneC, makeNeighbourhood } from "./neighbourhood";

const close = (a: number, b: number, eps = 1e-9) => Math.abs(a - b) < eps;
/** Relabel the two loci: -+ ↔ +- (indices 1 ↔ 2); a genuine symmetry of the two-locus system. */
const swap = (v: Vec4): Vec4 => [v[0], v[2], v[1], v[3]];

describe("offspring table", () => {
  it("has 16 rows, each a probability distribution", () => {
    expect(OFFSPRING).toHaveLength(16);
    for (const row of OFFSPRING) expect(close(sum4(row), 1)).toBe(true);
  });
});

describe("recombine", () => {
  it("preserves total mass as (Σpf)² — ⇒ 1 for a normalised vector", () => {
    const pf: Vec4 = [0.4, 0.1, 0.3, 0.2];
    expect(close(sum4(recombine(pf)), 1)).toBe(true);
    const un: Vec4 = [0.4, 0.2, 0.6, 0.8]; // Σ = 2
    expect(close(sum4(recombine(un)), 4)).toBe(true);
  });
  it("fixes a pure genotype", () => {
    expect(recombine([1, 0, 0, 0])).toEqual([1, 0, 0, 0]);
    expect(recombine([0, 0, 0, 1])).toEqual([0, 0, 0, 1]);
  });
  it("commutes with the locus-swap relabelling (offspring-table symmetry)", () => {
    const pf: Vec4 = [0.4, 0.1, 0.3, 0.2];
    const a = recombine(swap(pf));
    const b = swap(recombine(pf));
    for (let i = 0; i < 4; i++) expect(close(a[i] ?? 0, b[i] ?? 0)).toBe(true);
  });
});

describe("selection + fitness blend", () => {
  it("s goes 0→1 as fs goes 0→1, with a = 1 − s", () => {
    expect(selectionWeights(0)).toEqual({ a: 1, s: 0 });
    expect(selectionWeights(1)).toEqual({ a: 0, s: 1 });
    const mid = selectionWeights(0.5);
    expect(close(mid.a + mid.s, 1)).toBe(true);
    expect(mid.s).toBeGreaterThan(0);
    expect(mid.s).toBeLessThan(1);
  });
  it("blends to A at a=1 and to S at s=1", () => {
    expect(blendFitness(DEFAULT_FITNESS, 1, 0)).toEqual(DEFAULT_FITNESS.A);
    expect(blendFitness(DEFAULT_FITNESS, 0, 1)).toEqual(DEFAULT_FITNESS.S);
  });
});

describe("reactBite", () => {
  it("with twoBiteRate 0 and unit fitness is the identity", () => {
    const pf: Vec4 = [0.4, 0.1, 0.3, 0.2];
    expect(reactBite(pf, [1, 1, 1, 1], 0)).toEqual(pf);
  });
  it("with twoBiteRate 0 is the pointwise fitness product", () => {
    const pf: Vec4 = [0.4, 0.1, 0.3, 0.2];
    const fit: Vec4 = [1, 0.9, 0.9, 0.8];
    const got = reactBite(pf, fit, 0);
    const want: Vec4 = [0.4, 0.09, 0.27, 0.16];
    for (let i = 0; i < 4; i++) expect(close(got[i] ?? 0, want[i] ?? 0)).toBe(true);
  });
});

describe("ld", () => {
  it("is 0 at linkage equilibrium", () => {
    expect(close(ld([0.25, 0.25, 0.25, 0.25]), 0)).toBe(true);
  });
  it("saturates to ±1 at complete association", () => {
    expect(close(ld([0.5, 0, 0, 0.5]), 1)).toBe(true);
    expect(close(ld([0, 0.5, 0.5, 0]), -1)).toBe(true);
  });
  it("returns 0 for a degenerate marginal (no division by zero)", () => {
    expect(ld([1, 0, 0, 0])).toBe(0);
  });
});

describe("gatherCell invariants", () => {
  const bites = [
    { pf: [0.5, 0.2, 0.2, 0.1] as Vec4, weight: 0.7 },
    { pf: [0.1, 0.3, 0.4, 0.2] as Vec4, weight: 1.3 },
    { pf: [0.25, 0.25, 0.25, 0.25] as Vec4, weight: 0.5 },
  ];
  it("normalises to a genotype distribution (sums to 1)", () => {
    const fit: Vec4 = [1, 0.9, 0.9, 0.8];
    expect(close(sum4(gatherCell(bites, fit, 0.3)), 1)).toBe(true);
  });
  it("a uniform field with unit fitness and no recombination is a fixed point", () => {
    const p: Vec4 = [0.4, 0.1, 0.3, 0.2];
    const uniform = [
      { pf: p, weight: 0.7 },
      { pf: p, weight: 1.1 },
    ];
    const out = gatherCell(uniform, [1, 1, 1, 1], 0);
    for (let i = 0; i < 4; i++) expect(close(out[i] ?? 0, p[i] ?? 0)).toBe(true);
  });
  it("commutes with the locus-swap relabelling", () => {
    const swappedBites = bites.map((b) => ({ pf: swap(b.pf), weight: b.weight }));
    const a = gatherCell(swappedBites, swap([1, 0.9, 0.85, 0.8]), 0.3);
    const b = swap(gatherCell(bites, [1, 0.9, 0.85, 0.8], 0.3));
    for (let i = 0; i < 4; i++) expect(close(a[i] ?? 0, b[i] ?? 0)).toBe(true);
  });
});

describe("seeded neighbourhood", () => {
  it("betaOneC maps 0→0 and is increasing", () => {
    expect(betaOneC(0, 6)).toBe(0);
    expect(betaOneC(0.9, 6)).toBeGreaterThan(betaOneC(0.1, 6));
  });
  it("is deterministic in the seed and bounded by maxDistance", () => {
    const p = { mapWidthInKm: 10000, maxDistanceInKm: 2000, concentration: 6, count: 500, gridWidth: 500, seed: 42 };
    const a = makeNeighbourhood(p);
    const b = makeNeighbourhood(p);
    expect(a.data).toEqual(b.data);
    expect(makeNeighbourhood({ ...p, seed: 43 }).data).not.toEqual(a.data);
    const maxCells = p.maxDistanceInKm / (p.mapWidthInKm / p.gridWidth) + 1; // +1 for rounding
    for (let i = 0; i < a.count; i++) {
      const dx = a.data[i * 3] ?? 0;
      const dy = a.data[i * 3 + 1] ?? 0;
      expect(Math.hypot(dx, dy)).toBeLessThanOrEqual(maxCells + 1e-6);
    }
  });
});
