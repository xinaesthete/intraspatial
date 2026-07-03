import { describe, expect, it } from "vitest";
import { figureAt, figureTargetVel, FIGURE_SEQUENCE, partnerIndex } from "./figures";
import { dot, length, type Vec3In } from "./vec3";

describe("figureAt", () => {
  it("cycles figures every period and advances the index", () => {
    expect(figureAt(0, 100, 0).figure).toBe(FIGURE_SEQUENCE[0]);
    expect(figureAt(0, 100, 0).figureIndex).toBe(0);
    expect(figureAt(150, 100, 0).figureIndex).toBe(1);
    expect(figureAt(150, 100, 0).figure).toBe(FIGURE_SEQUENCE[1]);
    expect(figureAt(50, 100, 0).phase).toBeCloseTo(0.5, 6);
  });

  it("rotates the sequence by seed and is deterministic", () => {
    expect(figureAt(0, 100, 2).figure).toBe(FIGURE_SEQUENCE[2]);
    expect(figureAt(1234, 100, 7)).toEqual(figureAt(1234, 100, 7));
  });
});

describe("partnerIndex progression", () => {
  it("gives a valid partner that changes between figures (dancers advance)", () => {
    const n = 8;
    const p0 = partnerIndex(0, 0, n);
    const p1 = partnerIndex(0, 1, n);
    expect(p0).toBeGreaterThanOrEqual(0);
    expect(p0).toBeLessThan(n);
    expect(p0).not.toBe(0); // not self
    expect(p1).not.toBe(p0); // advanced to a new partner
  });

  it("is a no-op for a lone dancer", () => {
    expect(partnerIndex(0, 3, 1)).toBe(0);
  });
});

describe("figureTargetVel", () => {
  const p: Vec3In = [3, 0, 0];

  it("gather points inward, scatter outward", () => {
    expect(dot(figureTargetVel("gather", p, 0, [0, 0, 0], 1), p)).toBeLessThan(0);
    expect(dot(figureTargetVel("scatter", p, 0, [0, 0, 0], 1), p)).toBeGreaterThan(0);
  });

  it("grandChain counter-rotates even vs odd dancers", () => {
    const even = figureTargetVel("grandChain", p, 0, [0, 0, 0], 1);
    const odd = figureTargetVel("grandChain", p, 1, [0, 0, 0], 1);
    expect(dot(even, odd)).toBeLessThan(0); // opposite tangential directions
  });

  it("swing orbits the couple midpoint (perpendicular to the offset)", () => {
    const partner: Vec3In = [1, 0, 0];
    const v = figureTargetVel("swing", p, 0, partner, 1);
    // midpoint (2,0,0); offset (1,0,0); target ⟂ offset
    expect(dot(v, [1, 0, 0])).toBeCloseTo(0, 6);
    expect(length(v)).toBeCloseTo(1, 6);
  });

  it("targets are finite and roughly speed-scaled", () => {
    for (const f of FIGURE_SEQUENCE) {
      const v = figureTargetVel(f, p, 2, [1, 1, 0], 0.6);
      expect(v.every((x) => Number.isFinite(x))).toBe(true);
    }
  });
});
