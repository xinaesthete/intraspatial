// The Mutator operators — Todd & Latham's aesthetic-selection primitives made
// generic over any trait-space. All pure functions of `(specimen, …, rng)`, so a
// lineage replays exactly.
//
// The unifying model (the conceptual spine of this artefact): a specimen is a playhead
// in trait-space.
//   • mutate — stochastic noise on position (Latham's mutation rate)
//   • marry  — blend two parents' position AND velocity (heritable momentum)
//   • steer  — add an impulse to velocity (a performance gesture / directed evolution)
//   • advance — integrate position by velocity with inertia (steering carries on)
// Live "playing the field" = steer + advance every frame; breeding = mutate + marry.
// Both write the same specimen, so gestures and traits are the same substance.

import { gauss, hashSeed, mulberry32, type Rng } from "./rng";
import { cloneSpecimen, type Specimen } from "./specimen";
import type { TraitSpace } from "./traitSpace";

/** Stochastic perturbation. NUMBER traits get Gaussian noise scaled by `rate`
 *  (clamped back into [0,1]); ENABLE traits flip with probability `rate`. Locked
 *  traits are untouched. Velocity is inherited unchanged (mutation moves where you
 *  are, not how you're moving). */
export function mutate(space: TraitSpace, sp: Specimen, rate: number, rng: Rng): Specimen {
  const out = cloneSpecimen(sp);
  for (const trait of space.traits) {
    if (trait.locked) continue;
    if (trait.kind === "number") {
      out.pos[trait.slot] = clamp01(out.pos[trait.slot]! + gauss(rng) * rate);
    } else if (trait.kind === "enable") {
      if (rng() < rate) out.enable[trait.slot] = out.enable[trait.slot] ? 0 : 1;
    }
  }
  out.seed = hashSeed(sp.seed, u32(rng));
  return out;
}

/** Marriage — per-trait arithmetic crossover. Each NUMBER trait blends the parents by
 *  a fresh random weight; the same weight blends their velocities, so a lively parent
 *  passes on its motion through trait-space. ENABLE traits are inherited from one parent
 *  by coin flip. Locked traits are taken from `a` (the primary parent). */
export function marry(space: TraitSpace, a: Specimen, b: Specimen, rng: Rng): Specimen {
  const out = cloneSpecimen(a);
  for (const trait of space.traits) {
    if (trait.kind === "number") {
      if (trait.locked) continue; // keep a's value + velocity
      const t = rng();
      out.pos[trait.slot] = a.pos[trait.slot]! * (1 - t) + b.pos[trait.slot]! * t;
      out.vel[trait.slot] = a.vel[trait.slot]! * (1 - t) + b.vel[trait.slot]! * t;
    } else if (trait.kind === "enable") {
      if (trait.locked) continue;
      out.enable[trait.slot] = rng() < 0.5 ? a.enable[trait.slot]! : b.enable[trait.slot]!;
    }
  }
  out.seed = hashSeed(a.seed, b.seed, u32(rng));
  return out;
}

/** Steer — add an impulse to velocity along `dir` (a vector in NUMBER-trait-space).
 *  This is a performance gesture, or directed evolution toward a chosen target. Pure:
 *  position is unchanged; `advance` later turns this momentum into motion. Locked
 *  traits receive no impulse. */
export function steer(space: TraitSpace, sp: Specimen, dir: Float64Array, amount: number): Specimen {
  const out = cloneSpecimen(sp);
  for (const trait of space.traits) {
    if (trait.kind !== "number" || trait.locked) continue;
    out.vel[trait.slot] = out.vel[trait.slot]! + dir[trait.slot]! * amount;
  }
  return out;
}

/** Integrate the playhead one tick: position advances by velocity, velocity decays by
 *  `damping`. At a [0,1] wall the position clamps and that component of velocity
 *  reflects, so the playhead bounces rather than sticking. Locked traits hold still. */
export function advance(space: TraitSpace, sp: Specimen, dt: number, damping: number): Specimen {
  const out = cloneSpecimen(sp);
  for (const trait of space.traits) {
    if (trait.kind !== "number" || trait.locked) continue;
    const i = trait.slot;
    let p = out.pos[i]! + out.vel[i]! * dt;
    if (p < 0) {
      p = -p;
      out.vel[i] = -out.vel[i]!;
    } else if (p > 1) {
      p = 2 - p;
      out.vel[i] = -out.vel[i]!;
    }
    out.pos[i] = clamp01(p);
    out.vel[i] = out.vel[i]! * damping;
  }
  return out;
}

/** A direction in NUMBER-trait-space from `a` toward `b` (for steering toward a chosen
 *  individual — e.g. when a selection blends the playhead toward an offspring). */
export function toward(a: Specimen, b: Specimen): Float64Array {
  const dir = new Float64Array(a.pos.length);
  for (let i = 0; i < dir.length; i++) dir[i] = b.pos[i]! - a.pos[i]!;
  return dir;
}

export interface BreedOptions {
  /** Mutation rate applied after marriage (and as the sole operator for one parent). */
  rate: number;
  rng: Rng;
  /** Keep the (first) parent unchanged as element 0 of the generation. */
  keepElite?: boolean;
}

/** Produce a generation of `n` specimens from selected parents — the engine behind the
 *  3×3 breeding grid. No parents ⇒ random individuals; one parent ⇒ mutations; two or
 *  more ⇒ each child is a mutated marriage of two randomly chosen parents. Fully
 *  determined by `opts.rng`, so a generation replays. */
export function breed(space: TraitSpace, parents: Specimen[], n: number, opts: BreedOptions): Specimen[] {
  const { rate, rng } = opts;
  const out: Specimen[] = [];
  let start = 0;
  if (opts.keepElite && parents.length > 0) {
    out.push(cloneSpecimen(parents[0]!));
    start = 1;
  }
  for (let i = start; i < n; i++) {
    if (parents.length === 0) {
      out.push(randomFrom(space, u32(rng)));
    } else if (parents.length === 1) {
      out.push(mutate(space, parents[0]!, rate, rng));
    } else {
      const a = parents[(rng() * parents.length) | 0]!;
      let b = parents[(rng() * parents.length) | 0]!;
      if (b === a) b = parents[(parents.indexOf(a) + 1) % parents.length]!;
      out.push(mutate(space, marry(space, a, b, rng), rate, rng));
    }
  }
  return out;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// A fresh uint32 drawn from the deterministic stream — used to derive child seeds so
// each individual gets its own (reproducible) phenotype randomness.
function u32(rng: Rng): number {
  return (rng() * 0x100000000) >>> 0;
}

function randomFrom(space: TraitSpace, seed: number): Specimen {
  const r = mulberry32(seed);
  const pos = new Float64Array(space.numCount);
  for (let i = 0; i < pos.length; i++) pos[i] = r();
  const enable = new Uint8Array(space.enableCount);
  for (let i = 0; i < enable.length; i++) enable[i] = r() < 0.5 ? 1 : 0;
  return { pos, vel: new Float64Array(space.numCount), enable, seed };
}
