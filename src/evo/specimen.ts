// A specimen: a point in trait-space, plus a velocity through it. The velocity is the
// load-bearing idea for this artefact — it makes "gestures heritable". A live
// performance gesture is an impulse to `vel`; steering is `vel`'s inertia carrying the
// playhead onward; and `marry` averages parents' `vel`, so a parent passes on its
// *motion through trait-space*, not just its position. `pos`/`vel` are normalised
// [0,1] coords (one per NUMBER trait); `enable` is one bit per ENABLE trait; `seed`
// fixes the individual's own randomness (e.g. its swarm seeding) so its phenotype
// replays deterministically.
//
// (Naming: "specimen" not "genome" — this repo also handles real biological data; see
// the note in ./traitSpace.)

import { mulberry32, type Rng } from "./rng";
import type { TraitSpace } from "./traitSpace";

export interface Specimen {
  /** Normalised [0,1] coordinate per NUMBER trait. */
  pos: Float64Array;
  /** Per-NUMBER-trait velocity through trait-space (heritable gesture momentum). */
  vel: Float64Array;
  /** One bit per ENABLE trait (on/off of an influence). */
  enable: Uint8Array;
  /** The individual's own RNG seed — fixes its phenotype's randomness. */
  seed: number;
}

/** A specimen with all NUMBER traits at the centre of their range, all influences on,
 *  zero velocity — a neutral starting individual. */
export function neutralSpecimen(space: TraitSpace, seed: number): Specimen {
  return {
    pos: new Float64Array(space.numCount).fill(0.5),
    vel: new Float64Array(space.numCount),
    enable: new Uint8Array(space.enableCount).fill(1),
    seed,
  };
}

/** A random individual: every NUMBER trait uniform in [0,1], every influence on with
 *  probability 0.5, zero velocity. Deterministic in `seed`. */
export function randomSpecimen(space: TraitSpace, seed: number): Specimen {
  const rng: Rng = mulberry32(seed);
  const pos = new Float64Array(space.numCount);
  for (let i = 0; i < pos.length; i++) pos[i] = rng();
  const enable = new Uint8Array(space.enableCount);
  for (let i = 0; i < enable.length; i++) enable[i] = rng() < 0.5 ? 1 : 0;
  return { pos, vel: new Float64Array(space.numCount), enable, seed };
}

export function cloneSpecimen(sp: Specimen): Specimen {
  return { pos: Float64Array.from(sp.pos), vel: Float64Array.from(sp.vel), enable: Uint8Array.from(sp.enable), seed: sp.seed };
}

export interface SerializedSpecimen {
  pos: number[];
  vel: number[];
  enable: number[];
  seed: number;
}

export function serializeSpecimen(sp: Specimen): SerializedSpecimen {
  return { pos: Array.from(sp.pos), vel: Array.from(sp.vel), enable: Array.from(sp.enable), seed: sp.seed };
}

export function deserializeSpecimen(s: SerializedSpecimen): Specimen {
  return { pos: Float64Array.from(s.pos), vel: Float64Array.from(s.vel), enable: Uint8Array.from(s.enable), seed: s.seed };
}
