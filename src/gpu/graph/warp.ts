// Warps: how a position in [0, 1] (a slider, a MIDI controller, a Mutator's normalised trait)
// maps to a value in a `ParamSpec`'s [min, max], as SuperCollider's ControlSpec warps do
// (Warp.sc: LinearWarp, ExponentialWarp, FaderWarp, DbFaderWarp, CurveWarp), so that a spec
// written in SuperCollider means the same here. A port of inorganic's
// `packages/live-av-core/src/warp.ts`, checked against values sclang 3.13 computes.
//
// - "lin": evenly.
// - "exp": evenly in ratio (min and max non-zero and of the same sign): frequencies, times.
//   Over Hz this is exactly a linear range of MIDI pitch.
// - "amp": squared, for linear amplitudes (a fader).
// - "db": for decibels, squared in amplitude (a fader in dB).
// - a number: a curve (sclang's CurveWarp): 0 is linear, positive starts slowly, negative quickly.

/** A warp, as SuperCollider names them; a number is a curve. */
export type Warp = "lin" | "exp" | "amp" | "db" | number;

/** A range and how to move through it. */
export interface WarpSpec {
  min: number;
  max: number;
  /** Default "lin". */
  warp?: Warp;
  /** Round mapped values to a multiple of this (0 or absent: don't). */
  step?: number;
}

const dbamp = (db: number) => 10 ** (db / 20);
const ampdb = (amp: number) => 20 * Math.log10(amp);
const clip01 = (u: number) => Math.min(1, Math.max(0, u));
const roundTo = (v: number, step: number | undefined) => (step ? Math.round(v / step) * step : v);

/** Whether `warp` can traverse [min, max]: an exp warp needs both ends non-zero and of the
 *  same sign; a curve needs `exp(curve)` finite (past about 709 it overflows, and the map
 *  turns to NaN); the others take any finite range. Unknown names are invalid. */
export function warpValid(warp: Warp | undefined, min: number, max: number): boolean {
  if (warp === undefined) return true;
  if (typeof warp === "number") return Number.isFinite(warp) && Number.isFinite(Math.exp(warp));
  if (warp === "exp") return min !== 0 && max !== 0 && Math.sign(min) === Math.sign(max);
  return warp === "lin" || warp === "amp" || warp === "db";
}

/** Whether `warp` is the identity (absent, "lin", or a curve too flat to differ from it). */
export function warpIsLinear(warp: Warp | undefined): boolean {
  return warp === undefined || warp === "lin" || (typeof warp === "number" && Math.abs(warp) < 0.001);
}

/** The value at position `u` (clipped to [0, 1]) of the range, as ControlSpec.map. */
export function warpMap(u: number, { min, max, warp = "lin", step }: WarpSpec): number {
  const t = clip01(u);
  const range = max - min;
  let v: number;
  if (typeof warp === "number") {
    if (Math.abs(warp) < 0.001) v = min + t * range;
    else {
      const grow = Math.exp(warp);
      const a = range / (1 - grow);
      v = min + a - a * grow ** t;
    }
  } else if (warp === "exp") v = min * (max / min) ** t;
  else if (warp === "amp") v = range >= 0 ? t * t * range + min : (1 - (1 - t) ** 2) * range + min;
  else if (warp === "db") {
    const lo = dbamp(min);
    const r = dbamp(max) - lo;
    v = ampdb(r >= 0 ? t * t * r + lo : (1 - (1 - t) ** 2) * r + lo);
  } else v = min + t * range;
  return roundTo(v, step);
}

/** The position in [0, 1] of value `v` (rounded to step, clipped to the range), as
 *  ControlSpec.unmap. */
export function warpUnmap(v: number, { min, max, warp = "lin", step }: WarpSpec): number {
  const x = Math.min(Math.max(roundTo(v, step), Math.min(min, max)), Math.max(min, max));
  const range = max - min;
  if (range === 0) return 0;
  let u: number;
  if (typeof warp === "number") {
    if (Math.abs(warp) < 0.001) u = (x - min) / range;
    else {
      const grow = Math.exp(warp);
      const a = range / (1 - grow);
      u = Math.log((min + a - x) / a) / warp;
    }
  } else if (warp === "exp") u = Math.log(x / min) / Math.log(max / min);
  else if (warp === "amp") u = range >= 0 ? Math.sqrt((x - min) / range) : 1 - Math.sqrt(1 - (x - min) / range);
  else if (warp === "db") {
    const lo = dbamp(min);
    const r = dbamp(max) - lo;
    u = r >= 0 ? Math.sqrt((dbamp(x) - lo) / r) : 1 - Math.sqrt(1 - (dbamp(x) - lo) / r);
  } else u = (x - min) / range;
  return clip01(u + 0); // (+ 0: no -0)
}
