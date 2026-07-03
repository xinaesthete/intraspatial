import { describe, expect, it } from "vitest";
import {
  inGamut,
  type LinearRgb,
  linearChannelToSrgb,
  linearToOklab,
  mixOklab,
  mixOklch,
  oklabToLinear,
  oklabToOklch,
  oklchToOklab,
  type Srgb,
  srgbChannelToLinear,
  srgbToOklab,
  srgbToOklch,
  oklchToSrgb,
} from "./oklab";

const TAU = Math.PI * 2;

describe("okLab", () => {
  it("matches Ottosson's linear-sRGB → okLab reference triples", () => {
    // Björn Ottosson, "A perceptual color space for image processing" — the published table
    // maps LINEAR sRGB to okLab, isolating the LMS-cbrt matrices from the gamma transfer.
    const cases: Array<[LinearRgb, [number, number, number]]> = [
      [[1, 1, 1], [1.0, 0.0, 0.0]],
      [[1, 0, 0], [0.627955, 0.224863, 0.125846]],
      [[0, 1, 0], [0.86644, -0.233888, 0.179498]],
      [[0, 0, 1], [0.452014, -0.032457, -0.311528]],
    ];
    for (const [lin, want] of cases) {
      const got = linearToOklab(lin);
      expect(got[0]).toBeCloseTo(want[0], 4);
      expect(got[1]).toBeCloseTo(want[1], 4);
      expect(got[2]).toBeCloseTo(want[2], 4);
    }
  });

  it("sRGB transfer function inverts and is continuous at the knee", () => {
    // Round-trips to ~ppm; not tighter, because c = 0.04045 sits exactly on the sRGB knee,
    // where the forward (0.04045) and inverse (0.0031308) thresholds aren't perfect inverses —
    // an artefact of the standard's piecewise definition, not of these functions.
    for (const c of [0, 0.001, 0.04045, 0.05, 0.2, 0.5, 0.9, 1]) {
      expect(linearChannelToSrgb(srgbChannelToLinear(c))).toBeCloseTo(c, 6);
    }
    // continuity across the piecewise boundary (linear ↔ power segments meet)
    const eps = 1e-7;
    expect(srgbChannelToLinear(0.04045 + eps)).toBeCloseTo(srgbChannelToLinear(0.04045 - eps), 6);
  });

  it("okLab → linear → okLab is identity (in and out of gamut)", () => {
    const cols: LinearRgb[] = [
      [0.1, 0.4, 0.8],
      [0.9, 0.2, 0.05],
      [0.5, 0.5, 0.5],
      [1.2, -0.1, 0.3], // deliberately out of gamut — cbrt sign-carry must hold
    ];
    for (const lin of cols) {
      const back = oklabToLinear(linearToOklab(lin));
      expect(back[0]).toBeCloseTo(lin[0], 6);
      expect(back[1]).toBeCloseTo(lin[1], 6);
      expect(back[2]).toBeCloseTo(lin[2], 6);
    }
  });

  it("achromatic colours have ~zero chroma", () => {
    for (const v of [0.1, 0.5, 0.9]) {
      const lab = srgbToOklab([v, v, v]);
      expect(lab[1]).toBeCloseTo(0, 6);
      expect(lab[2]).toBeCloseTo(0, 6);
      expect(oklabToOklch(lab)[1]).toBeCloseTo(0, 6); // C ≈ 0
    }
  });

  it("sRGB → okLCH → sRGB round-trips", () => {
    const cols: Srgb[] = [
      [0.8, 0.1, 0.3],
      [0.2, 0.7, 0.9],
      [0.05, 0.05, 0.05],
    ];
    for (const c of cols) {
      const back = oklchToSrgb(srgbToOklch(c));
      expect(back[0]).toBeCloseTo(c[0], 6);
      expect(back[1]).toBeCloseTo(c[1], 6);
      expect(back[2]).toBeCloseTo(c[2], 6);
    }
  });

  it("okLab ⇄ okLCH round-trips and normalises hue to [0, τ)", () => {
    const lab: [number, number, number] = [0.6, -0.08, 0.12];
    const lch = oklabToOklch(lab);
    expect(lch[2]).toBeGreaterThanOrEqual(0);
    expect(lch[2]).toBeLessThan(TAU);
    const back = oklchToOklab(lch);
    expect(back[0]).toBeCloseTo(lab[0], 12);
    expect(back[1]).toBeCloseTo(lab[1], 12);
    expect(back[2]).toBeCloseTo(lab[2], 12);
  });

  it("mixOklab midpoint is the componentwise average", () => {
    const a: [number, number, number] = [0.2, -0.1, 0.05];
    const b: [number, number, number] = [0.8, 0.1, -0.05];
    const m = mixOklab(a, b, 0.5);
    expect(m[0]).toBeCloseTo(0.5, 12);
    expect(m[1]).toBeCloseTo(0, 12);
    expect(m[2]).toBeCloseTo(0, 12);
  });

  it("mixOklch takes hue the short way across the 0/τ seam", () => {
    // hues 0.1 and (τ − 0.1) straddle 0; the midpoint should sit at ~0, not ~π.
    const a: [number, number, number] = [0.5, 0.1, 0.1];
    const b: [number, number, number] = [0.5, 0.1, TAU - 0.1];
    const h = mixOklch(a, b, 0.5)[2];
    const distToZero = Math.min(h, TAU - h);
    expect(distToZero).toBeCloseTo(0, 6);
  });

  it("inGamut flags displayable vs clipped linear colours", () => {
    expect(inGamut([0.2, 0.4, 0.6])).toBe(true);
    expect(inGamut([0, 1, 1])).toBe(true);
    expect(inGamut([1.2, 0.1, 0.1])).toBe(false);
    expect(inGamut([-0.05, 0.5, 0.5])).toBe(false);
  });
});
