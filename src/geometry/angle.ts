// Angle — the typed rotation-magnitude value (ADR-0010). A rotation angle is a *value with a
// unit*, never a bare number whose meaning depends on an ambient mode. Explicit constructors
// (`deg` / `rad` / `turns`) canonicalise to radians internally; bare numbers are governed by a
// **catalogue-level default unit** passed in explicitly at construction — never a hidden global
// that makes the same expression mean two things.

/** A rotation magnitude, canonicalised to radians. Construct via {@link deg}/{@link rad}/
 *  {@link turns}; the `radians` field is the single source of truth. */
export interface Angle {
  readonly radians: number;
  /** Brand so a bare number can't be passed where an Angle is required. */
  readonly __angle: true;
}

/** The unit a catalogue interprets bare-number angles in. */
export type AngleUnit = "deg" | "rad" | "turns";

function angle(radians: number): Angle {
  return { radians, __angle: true };
}

/** An angle given in degrees. */
export function deg(value: number): Angle {
  return angle((value * Math.PI) / 180);
}
/** An angle given in radians. */
export function rad(value: number): Angle {
  return angle(value);
}
/** An angle given in turns (1 turn = 2π). */
export function turns(value: number): Angle {
  return angle(value * 2 * Math.PI);
}

/** Runtime guard: is `x` an {@link Angle} (vs a bare number / expression)? */
export function isAngle(x: unknown): x is Angle {
  return typeof x === "object" && x !== null && (x as { __angle?: unknown }).__angle === true;
}

/** The radians-per-unit factor for a bare-number angle in `unit`. */
export function unitToRadians(unit: AngleUnit): number {
  switch (unit) {
    case "deg":
      return Math.PI / 180;
    case "rad":
      return 1;
    case "turns":
      return 2 * Math.PI;
  }
}
