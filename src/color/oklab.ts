// okLab / okLCH — a perceptually-uniform colour space (Björn Ottosson, 2020,
// "A perceptual color space for image processing"). Scalar sim fields map to colour
// that reads correctly and interpolates without the muddy midpoints and hue skew of
// sRGB/HSL lerps — so it underpins every colour mapping in the visualisation (trails,
// instance colour, the distance-matrix heatmap, later the trait system).
//
// The chain is sRGB (gamma) ⇄ linear sRGB ⇄ okLab ⇄ okLCH. Each link is a pure function
// on immutable tuples (matching vec3.ts); colour "spaces" are kept as distinct types so a
// conversion is always explicit — the same space-tagging the expression IR will make into
// nodes (see docs/render-traits-and-expression-dsl.md). This is the CPU reference; a TSL
// mirror (oklab.tsl.ts) implements the identical maths for material shaders, kept honest
// by a golden cross-check the same way forces.ts / dancerGpu are.
//
// Matrices and coefficients are Ottosson's canonical values; the okLab test asserts them
// against his published linear-sRGB → okLab reference triples.

/** Gamma-encoded sRGB, nominal 0..1 (the space CSS colours and `THREE.Color` live in). */
export type Srgb = readonly [number, number, number];
/** Linear-light sRGB primaries, 0..1 in-gamut. */
export type LinearRgb = readonly [number, number, number];
/** okLab: perceptual lightness `L` (0..1) + opponent axes `a` (green→red), `b` (blue→yellow). */
export type Oklab = readonly [number, number, number];
/** okLCH: okLab in polar form — `L`, chroma `C` (≥0), hue `h` in **radians** (atan2 range). */
export type Oklch = readonly [number, number, number];

const TAU = Math.PI * 2;

// ── sRGB transfer function (per channel) ────────────────────────────────────────────────

/** One gamma-encoded sRGB channel → linear light. */
export function srgbChannelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** One linear-light channel → gamma-encoded sRGB. */
export function linearChannelToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
}

export const srgbToLinear = (c: Srgb): LinearRgb => [
  srgbChannelToLinear(c[0]),
  srgbChannelToLinear(c[1]),
  srgbChannelToLinear(c[2]),
];

export const linearToSrgb = (c: LinearRgb): Srgb => [
  linearChannelToSrgb(c[0]),
  linearChannelToSrgb(c[1]),
  linearChannelToSrgb(c[2]),
];

// ── linear sRGB ⇄ okLab (Ottosson's LMS-cbrt model) ─────────────────────────────────────

/** Linear-light sRGB → okLab. `Math.cbrt` carries sign, so out-of-gamut negatives stay sane. */
export function linearToOklab(c: LinearRgb): Oklab {
  const r = c[0];
  const g = c[1];
  const b = c[2];
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);
  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

/** okLab → linear-light sRGB (may fall outside [0,1] when the colour is out of gamut). */
export function oklabToLinear(lab: Oklab): LinearRgb {
  const L = lab[0];
  const a = lab[1];
  const b = lab[2];
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

// ── okLab ⇄ okLCH (polar) ───────────────────────────────────────────────────────────────

/** okLab → okLCH. Hue in radians; chroma is the opponent-plane magnitude. */
export function oklabToOklch(lab: Oklab): Oklch {
  const a = lab[1];
  const b = lab[2];
  const h = Math.atan2(b, a);
  return [lab[0], Math.hypot(a, b), h < 0 ? h + TAU : h];
}

/** okLCH → okLab. */
export function oklchToOklab(lch: Oklch): Oklab {
  const C = lch[1];
  const h = lch[2];
  return [lch[0], C * Math.cos(h), C * Math.sin(h)];
}

// ── convenience compositions (sRGB ⇄ okLab/okLCH) ───────────────────────────────────────

export const srgbToOklab = (c: Srgb): Oklab => linearToOklab(srgbToLinear(c));
export const oklabToSrgb = (lab: Oklab): Srgb => linearToSrgb(oklabToLinear(lab));
export const srgbToOklch = (c: Srgb): Oklch => oklabToOklch(srgbToOklab(c));
export const oklchToSrgb = (lch: Oklch): Srgb => oklabToSrgb(oklchToOklab(lch));

// ── perceptual interpolation ────────────────────────────────────────────────────────────

/** Linear blend in okLab (perceptual straight-line mix — no muddy midpoint). `t` in [0,1]. */
export const mixOklab = (a: Oklab, b: Oklab, t: number): Oklab => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/** Blend in okLCH, taking hue the **short way** around the wheel (so red→magenta doesn't
 *  detour through green). Chroma/lightness lerp linearly. `t` in [0,1]. */
export function mixOklch(a: Oklch, b: Oklch, t: number): Oklch {
  let dh = b[2] - a[2];
  if (dh > Math.PI) dh -= TAU;
  else if (dh < -Math.PI) dh += TAU;
  let h = a[2] + dh * t;
  if (h < 0) h += TAU;
  else if (h >= TAU) h -= TAU;
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, h];
}

/** True when a linear-sRGB colour sits inside the displayable gamut (all channels in [0,1],
 *  within `eps`). Handy for probing okLCH ramps before feeding a shader. */
export function inGamut(c: LinearRgb, eps = 1e-4): boolean {
  return c[0] >= -eps && c[0] <= 1 + eps && c[1] >= -eps && c[1] <= 1 + eps && c[2] >= -eps && c[2] <= 1 + eps;
}
