import { describe, expect, it } from "vitest";
import { bodyTaper, noseAxial, noseRadial } from "./sweptCreature";

describe("nose profile", () => {
  it("grows the radius from tip to shoulder and shrinks the reach", () => {
    for (const e of [0.5, 0.78, 1, 1.4]) {
      expect(noseRadial(0, e)).toBeCloseTo(0, 12); // tip: zero radius
      expect(noseRadial(1, e)).toBeCloseTo(1, 12); // shoulder: full radius
      expect(noseAxial(0, e)).toBeCloseTo(1, 12); // tip: max forward reach
      expect(noseAxial(1, e)).toBeCloseTo(0, 6); // shoulder: no reach (cos(π/2)^e ≈ 0, fp-noisy)
    }
  });

  it("is monotonic in radius along the nose", () => {
    let prev = -1;
    for (let i = 0; i <= 10; i++) {
      const r = noseRadial(i / 10, 0.78);
      expect(r).toBeGreaterThanOrEqual(prev);
      prev = r;
    }
  });

  it("e = 1 traces a quarter circle (radius² + reach² = 1)", () => {
    for (let i = 0; i <= 10; i++) {
      const s = i / 10;
      const r = noseRadial(s, 1);
      const a = noseAxial(s, 1);
      expect(r * r + a * a).toBeCloseTo(1, 10);
    }
  });
});

describe("body taper", () => {
  it("falls from full at the shoulder to zero at the tail", () => {
    expect(bodyTaper(0, 1.4)).toBeCloseTo(1, 12);
    expect(bodyTaper(1, 1.4)).toBeCloseTo(0, 12);
  });

  it("clamps out-of-range input", () => {
    expect(bodyTaper(-0.5, 2)).toBeCloseTo(1, 12);
    expect(bodyTaper(1.5, 2)).toBeCloseTo(0, 12);
  });

  it("is monotonically decreasing", () => {
    let prev = 2;
    for (let i = 0; i <= 10; i++) {
      const r = bodyTaper(i / 10, 1.4);
      expect(r).toBeLessThanOrEqual(prev);
      prev = r;
    }
  });
});

describe("shoulder continuity", () => {
  it("nose and body radius agree where they meet", () => {
    for (const e of [0.5, 0.78, 1, 1.4]) {
      expect(noseRadial(1, e)).toBeCloseTo(bodyTaper(0, 1.4), 12);
    }
  });
});
