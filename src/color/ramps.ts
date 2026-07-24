// Colour ramps for the statistical rasters, built in OKLCh on top of `oklab.ts`.
//
// Why the space matters here rather than being a nicety: Γ_ab is a SIGNED field and is read by
// comparing its two arms. A blue→white→red ramp interpolated in sRGB has arms of markedly
// different perceived lightness — sRGB blue is far darker than sRGB red at the same nominal
// distance from white — so equal-magnitude clustering and exclusion do not look equally strong,
// and the reader sees an asymmetry that is not in the data. Deriving the ramp in OKLCh lets
// lightness and chroma depend ONLY on |value| while the sign picks the hue, which makes that
// failure impossible by construction rather than by taste. `ramps.test.ts` asserts it.
//
// Out-of-gamut requests are resolved by CHROMA REDUCTION, not per-channel clipping. Clipping a
// channel moves both hue and lightness, i.e. it corrupts exactly the two dimensions the ramps use
// to carry meaning; backing off chroma sacrifices only saturation, which carries none.

import { inGamut, type Oklch, oklabToLinear, oklchToOklab, oklchToSrgb, type Srgb } from "./oklab";

const clamp01 = (c: Srgb): Srgb => [Math.min(1, Math.max(0, c[0])), Math.min(1, Math.max(0, c[1])), Math.min(1, Math.max(0, c[2]))];

/** OKLCh → sRGB (0..1), gamut-mapped by bisection on chroma at fixed L and h.
 *
 *  The final clamp is not the gamut mapping — the bisection is. `inGamut` admits a small epsilon,
 *  and a linear channel a hair over 1 becomes an sRGB channel a hair over 1; the clamp only sweeps
 *  up that sliver so callers can rely on a true 0..1 range. */
export function oklchToSrgbMapped(lch: Oklch): Srgb {
  if (inGamut(oklabToLinear(oklchToOklab(lch)))) return clamp01(oklchToSrgb(lch));
  const [L, C, h] = lch;
  let lo = 0;
  let hi = C;
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2;
    if (inGamut(oklabToLinear(oklchToOklab([L, mid, h])))) lo = mid;
    else hi = mid;
  }
  return clamp01(oklchToSrgb([L, lo, h]));
}

/** sRGB 0..1 → a `rgb(…)` string for canvas/CSS. */
export const cssRgb = (c: Srgb): string => `rgb(${c.map((v) => Math.round(255 * Math.min(1, Math.max(0, v)))).join(",")})`;

/** sRGB 0..1 → packed bytes, for writing straight into an ImageData buffer. */
export const rgbBytes = (c: Srgb): [number, number, number] => [
  Math.round(255 * Math.min(1, Math.max(0, c[0]))),
  Math.round(255 * Math.min(1, Math.max(0, c[1]))),
  Math.round(255 * Math.min(1, Math.max(0, c[2]))),
];

/** Hues are in RADIANS throughout, matching `oklab.ts`'s polar convention. */
export const deg = (d: number) => (d * Math.PI) / 180;

export interface DivergingOpts {
  /** Hue (rad) for negative values. Default `deg(250)` — blue. */
  negativeHue?: number;
  /** Hue (rad) for positive values. Default `deg(29)` — red. */
  positiveHue?: number;
  /** Lightness at the neutral centre and at the extremes. */
  centreL?: number;
  endL?: number;
  /** Chroma at the extremes. */
  endC?: number;
}

/** Signed diverging ramp over `t ∈ [−1, 1]`.
 *
 *  L and C are functions of |t| alone; only the hue knows the sign. So +x and −x are necessarily
 *  equally prominent — the property a diverging map for a signed statistic has to have. */
export function diverging(t: number, opts: DivergingOpts = {}): Srgb {
  const negativeHue = opts.negativeHue ?? deg(250);
  const positiveHue = opts.positiveHue ?? deg(29);
  const centreL = opts.centreL ?? 0.97;
  const endL = opts.endL ?? 0.55;
  const endC = opts.endC ?? 0.15;
  const clamped = Math.max(-1, Math.min(1, t));
  const mag = Math.abs(clamped);
  return oklchToSrgbMapped([centreL + (endL - centreL) * mag, endC * mag, clamped < 0 ? negativeHue : positiveHue]);
}

export interface SequentialOpts {
  minL?: number;
  maxL?: number;
  peakC?: number;
}

/** Sequential ramp over `t ∈ [0, 1]` at a fixed hue (rad) — for a single type's density.
 *
 *  Lightness is monotone in t, so two types drawn at different hues remain comparable by
 *  brightness, which is the comparison the side-by-side KDE panels ask for. Chroma peaks in the
 *  middle: near-black and near-white cannot hold saturation in any gamut, so asking for it there
 *  only gets mapped away again. */
export function sequential(t: number, hue: number, opts: SequentialOpts = {}): Srgb {
  const minL = opts.minL ?? 0.14;
  const maxL = opts.maxL ?? 0.95;
  const peakC = opts.peakC ?? 0.15;
  const v = Math.max(0, Math.min(1, t));
  return oklchToSrgbMapped([minL + (maxL - minL) * v, peakC * Math.sin(Math.PI * v) ** 0.7, hue]);
}

/** Evenly spaced hues (rad) for `n` categories. Equal steps on the OKLCh hue circle are the ones
 *  that look equal, which is the whole reason to pick categorical colours here rather than in HSL. */
export function categoricalHues(n: number, offset = deg(200)): number[] {
  const k = Math.max(n, 1);
  return Array.from({ length: k }, (_, i) => (offset + (2 * Math.PI * i) / k) % (2 * Math.PI));
}
