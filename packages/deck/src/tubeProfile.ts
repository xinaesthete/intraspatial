// The cross-section of a tube along its chain, and the change from one to another over time.
// Host side of `TubeLayer`'s `profile`; no GPU here.
//
// A profile says, per particle, how wide and thick the section is, how square, what colour, and
// which way its broad side faces. A round tube is the profile where width and thickness are 1.
//
// Width and thickness each have two values at a particle: one where the section arrives from the
// particle before (`In`) and one where it leaves for the next (`Out`). When they differ the section
// steps there, which is how an arrowhead gets its flat back.
//
// The broad side's direction is not stored as a vector, since the chain moves on the device every
// frame. It is stored as four particle indices `a0 a1 b0 b1`, and the direction is
// `(P[a1] - P[a0]) + (P[b1] - P[b0])` from the positions as they are when drawn. A helix names
// particles one turn apart (its axis); a strand of a sheet names its neighbour strands.

/** Floats per particle in `TubeProfile.shapes`. */
export const SHAPE_FLOATS = 12;
/** Indices per particle in `TubeProfile.orients`. */
export const ORIENT_WORDS = 4;
/** In `orients`: this particle names no direction (its section turns to face the viewer). */
export const NO_ORIENT = 0xffffffff;

/** Offsets within a particle's `SHAPE_FLOATS`. Sizes are multiples of the layer's `radius`. */
export const SHAPE = {
  widthIn: 0,
  widthOut: 1,
  thickIn: 2,
  thickOut: 3,
  /** rgb in 0..1, and at +3 how far it replaces the layer's ramp colour (0 none, 1 all). */
  colour: 4,
  /** 0 an ellipse, 1 a rectangle. */
  square: 8,
} as const;

export interface TubeProfile {
  /** `SHAPE_FLOATS` per particle. */
  readonly shapes: Float32Array;
  /** `ORIENT_WORDS` per particle. */
  readonly orients: Uint32Array;
}

/** The round tube: every size 1, the ramp's colour, no direction. */
export function roundProfile(count: number): TubeProfile {
  const shapes = new Float32Array(count * SHAPE_FLOATS);
  for (let i = 0; i < count; i++) shapes.fill(1, i * SHAPE_FLOATS, i * SHAPE_FLOATS + 4);
  return { shapes, orients: new Uint32Array(count * ORIENT_WORDS).fill(NO_ORIENT) };
}

/** Slow at both ends: the fraction of a change done at fraction `t` of its time. */
export function ease(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
}

/** The profile part-way from `from` to `to`. Sizes and colours are mixed; a direction cannot be,
 *  so each particle takes the nearer end's, or whichever end has one. */
export function mixProfiles(from: TubeProfile, to: TubeProfile, t: number): TubeProfile {
  const shapes = new Float32Array(to.shapes.length);
  for (let k = 0; k < shapes.length; k++) shapes[k] = from.shapes[k]! + (to.shapes[k]! - from.shapes[k]!) * t;
  const orients = new Uint32Array(to.orients.length);
  for (let i = 0; i < orients.length; i += ORIENT_WORDS) {
    const near = t < 0.5 ? from.orients : to.orients;
    const far = t < 0.5 ? to.orients : from.orients;
    const source = near[i] === NO_ORIENT ? far : near;
    for (let k = 0; k < ORIENT_WORDS; k++) orients[i + k] = source[i + k]!;
  }
  return { shapes, orients };
}

/**
 * Both ends of a change as the shader reads them: per particle `from` then `to`, for shapes and for
 * directions. Where only one end names a direction both get it: the other end is round or nearly
 * so there, and this way the section flattens in place instead of turning as it changes.
 */
export function packTransition(from: TubeProfile, to: TubeProfile): { shapes: Float32Array; orients: Uint32Array } {
  const count = to.shapes.length / SHAPE_FLOATS;
  const shapes = new Float32Array(count * SHAPE_FLOATS * 2);
  const orients = new Uint32Array(count * ORIENT_WORDS * 2);
  for (let i = 0; i < count; i++) {
    shapes.set(from.shapes.subarray(i * SHAPE_FLOATS, (i + 1) * SHAPE_FLOATS), i * SHAPE_FLOATS * 2);
    shapes.set(to.shapes.subarray(i * SHAPE_FLOATS, (i + 1) * SHAPE_FLOATS), i * SHAPE_FLOATS * 2 + SHAPE_FLOATS);
    const a = from.orients.subarray(i * ORIENT_WORDS, (i + 1) * ORIENT_WORDS);
    const b = to.orients.subarray(i * ORIENT_WORDS, (i + 1) * ORIENT_WORDS);
    orients.set(a[0] === NO_ORIENT ? b : a, i * ORIENT_WORDS * 2);
    orients.set(b[0] === NO_ORIENT ? a : b, i * ORIENT_WORDS * 2 + ORIENT_WORDS);
  }
  return { shapes, orients };
}
