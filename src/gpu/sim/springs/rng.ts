// Counter-based random numbers for the spring model. Every draw is a pure function of integer
// keys (particle, step, salt), so the CPU golden and the GPU kernel produce the same stream and a
// run is reproducible without carrying generator state. The hash is the PCG output function
// CSynth's sampled-pairs code uses (`pushpcg` in organic/TS/springsfs.ts).

/** PCG hash of a u32. */
export function pcg(v: number): number {
  const state = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/** Hash of three u32 keys. */
export function hash3(a: number, b: number, c: number): number {
  return pcg((a ^ pcg((b ^ pcg(c >>> 0)) >>> 0)) >>> 0);
}

/** Uniform in [0, 1) from three keys. */
export function rand01(a: number, b: number, c: number): number {
  return hash3(a, b, c) / 4294967296;
}

/** Hash of an unordered particle pair and a step: the same value whichever end asks. This is
 *  what lets both ends of a bond reach the same random decision without communicating. */
export function pairHash(i: number, j: number, step: number, salt: number): number {
  const lo = Math.min(i, j);
  const hi = Math.max(i, j);
  return hash3(lo, (Math.imul(hi, 0x9e3779b1) ^ salt) >>> 0, step);
}

/** `pairHash` as a uniform in [0, 1). */
export function pairRand01(i: number, j: number, step: number, salt: number): number {
  return pairHash(i, j, step, salt) / 4294967296;
}

/** The u32 a hash must be below for an event of probability `prob`. Decisions compare integers,
 *  so the CPU and the GPU agree exactly. */
export function probThreshold(prob: number): number {
  return Math.min(0xffffffff, Math.max(0, Math.floor(prob * 4294967296)));
}

/** Salts that keep the model's independent random streams apart. */
export const SALT = {
  noiseGate: 0x51ed270b,
  noiseX: 0x1b873593,
  noiseY: 0x85ebca6b,
  noiseZ: 0xc2b2ae35,
  motileX: 0x27d4eb2f,
  motileY: 0x165667b1,
  motileZ: 0xd3a2646c,
  bondBreak: 0xfd7046c5,
  bondForm: 0x7feb352d,
  divide: 0xb55a4f09,
  divideDir: 0x2545f491,
  leader: 0x68e31da4,
} as const;
