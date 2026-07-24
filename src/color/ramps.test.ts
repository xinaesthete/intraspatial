import { describe, expect, it } from "vitest";
import { inGamut, oklabToLinear, oklchToOklab, srgbToOklab, srgbToOklch } from "./oklab";
import { categoricalHues, deg, diverging, oklchToSrgbMapped, sequential } from "./ramps";

const lightnessOf = (c: readonly [number, number, number]) => srgbToOklab(c)[0];
/** Smallest angle between two hues (radians), wrapped into [0, π]. */
const hueGap = (a: number, b: number) => {
  const TAU = 2 * Math.PI;
  return Math.abs(((((a - b) % TAU) + TAU + Math.PI) % TAU) - Math.PI);
};

describe("diverging ramp", () => {
  it("is lightness-symmetric about zero — the property an sRGB ramp cannot have", () => {
    // The whole reason for building this in OKLCh: equal-magnitude clustering and exclusion must
    // look equally strong. Perceived lightness is the dominant cue, so it must depend on |t| only.
    for (const t of [0.1, 0.25, 0.5, 0.75, 1]) {
      expect(lightnessOf(diverging(t)), `|t| = ${t}`).toBeCloseTo(lightnessOf(diverging(-t)), 6);
    }
  });

  it("for contrast: the naive sRGB blue→white→red ramp is markedly asymmetric", () => {
    // Same construction done by lerping sRGB endpoints — the thing we are avoiding. At t = ±1 the
    // two arms differ in perceived lightness by a wide margin, so exclusion reads as "stronger".
    const naiveNeg = lightnessOf([0, 0, 1]);
    const naivePos = lightnessOf([1, 0, 0]);
    expect(Math.abs(naivePos - naiveNeg)).toBeGreaterThan(0.15);
    // ...whereas ours differ by nothing at all.
    expect(Math.abs(lightnessOf(diverging(1)) - lightnessOf(diverging(-1)))).toBeLessThan(1e-6);
  });

  it("is monotone in |t| and neutral at 0", () => {
    const centre = srgbToOklch(diverging(0));
    expect(centre[1]).toBeLessThan(0.01); // no chroma at the centre
    expect(centre[0]).toBeGreaterThan(0.9); // ...and it is the light end
    let prev = lightnessOf(diverging(0));
    for (const t of [0.2, 0.4, 0.6, 0.8, 1]) {
      const L = lightnessOf(diverging(t));
      expect(L).toBeLessThan(prev); // magnitude darkens, monotonically
      prev = L;
    }
  });

  it("separates the arms by hue", () => {
    // Hues are radians here (oklab.ts's convention); the two arms sit most of a half-turn apart.
    expect(hueGap(srgbToOklch(diverging(0.8))[2], srgbToOklch(diverging(-0.8))[2])).toBeGreaterThan(1.5);
  });
});

describe("sequential ramp", () => {
  it("is monotone in lightness at any hue, so two types stay comparable by brightness", () => {
    for (const hue of categoricalHues(4)) {
      let prev = -1;
      for (let i = 0; i <= 10; i++) {
        const L = lightnessOf(sequential(i / 10, hue));
        expect(L, `hue ${hue}`).toBeGreaterThan(prev);
        prev = L;
      }
    }
  });

  it("gives every hue the same lightness profile — brightness means density, not hue", () => {
    const hues = categoricalHues(6);
    for (const t of [0.25, 0.5, 0.75]) {
      const Ls = hues.map((h) => lightnessOf(sequential(t, h)));
      expect(Math.max(...Ls) - Math.min(...Ls), `t = ${t}`).toBeLessThan(0.02);
    }
  });
});

describe("gamut mapping", () => {
  it("reduces chroma instead of clipping channels, preserving L and h", () => {
    // A wildly out-of-gamut request: high chroma at high lightness.
    const want = [0.9, 0.4, deg(150)] as const;
    const got = oklchToSrgbMapped(want);
    expect(got.every((c) => c >= -1e-6 && c <= 1 + 1e-6)).toBe(true);
    const back = srgbToOklch(got);
    expect(back[0]).toBeCloseTo(want[0], 3); // lightness preserved
    expect(hueGap(back[2], want[2])).toBeLessThan(0.01); // hue preserved
    expect(back[1]).toBeLessThan(want[1]); // chroma is what gave way
  });

  it("leaves in-gamut colours untouched", () => {
    const lch = srgbToOklch([0.4, 0.6, 0.8]);
    expect(inGamut(oklabToLinear(oklchToOklab(lch)))).toBe(true);
    const got = oklchToSrgbMapped(lch);
    expect(got[0]).toBeCloseTo(0.4, 5);
    expect(got[1]).toBeCloseTo(0.6, 5);
    expect(got[2]).toBeCloseTo(0.8, 5);
  });

  it("every ramp sample is in gamut", () => {
    for (let i = -10; i <= 10; i++) {
      const c = diverging(i / 10);
      expect(
        c.every((v) => v >= -1e-6 && v <= 1 + 1e-6),
        `diverging ${i / 10}`,
      ).toBe(true);
    }
    for (const hue of categoricalHues(8)) {
      for (let i = 0; i <= 10; i++) {
        const c = sequential(i / 10, hue);
        expect(
          c.every((v) => v >= -1e-6 && v <= 1 + 1e-6),
          `sequential ${hue}`,
        ).toBe(true);
      }
    }
  });
});
