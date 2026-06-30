// Deterministic pseudo-randomness for the evolutionary layer. Same seed ⇒ same
// stream, so a bred generation, a mutation, or a whole pedigree replays bit-for-bit
// — the property that lets `src/evo` unit-test like the repo's CPU goldens (no
// `Math.random`, no `Date`). The core generator is the repo's own `mulberry32`
// (also used by `playground/src/sources.ts`); keeping one implementation here means
// the swarm seeder and the breeder draw from the same well-behaved stream.

/** A deterministic stream of uniforms in [0, 1). */
export type Rng = () => number;

/** mulberry32 — a small, fast, good-enough PRNG. A given seed yields a fixed
 *  sequence of uniforms in [0, 1) drawn sequentially. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A uniform draw in [lo, hi). */
export function uniform(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

/** One standard normal (mean 0, variance 1) via Box–Muller. We take a fresh pair
 *  each call and keep one — stateless w.r.t. the caller, which keeps the mutation
 *  ops pure functions of `(specimen, rng)`. */
export function gauss(rng: Rng): number {
  const u1 = 1 - rng(); // in (0, 1], avoids log(0)
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Combine integers into a single uint32 seed, deterministically. Used to derive a
 *  child's RNG seed from `(parentSeed, generation, slot)` so a given lineage step is
 *  reproducible without any global counter. */
export function hashSeed(...parts: number[]): number {
  let h = 0x811c9dc5; // FNV-ish offset basis
  for (const p of parts) {
    let x = p | 0;
    x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
    x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
    x ^= x >>> 16;
    h = Math.imul(h ^ x, 0x01000193);
  }
  return h >>> 0;
}
