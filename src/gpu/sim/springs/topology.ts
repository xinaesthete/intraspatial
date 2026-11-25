// The spring topology: CSynth's `topologybuff` (organic/TS/springs.ts), as typed arrays a GPU
// kernel can bind directly. Design: docs/bond-primitive.md.
//
// Each particle has `K` bond slots. A bond is stored twice, once under each end, so a force
// kernel only ever reads its own row. A slot is four f32 words:
//
//   [0] partner index and kind, as a u32 bit pattern: `partner | kind << 24`
//   [1] rest length    [2] strength    [3] distance power
//
// An empty slot has word 0 = 0xFFFFFFFF. Rod, fix and pull are one per particle and live in
// their own arrays rather than in special slots.

/** Word 0 of an empty slot. */
export const NO_BOND = 0xffffffff;
/** Words per bond slot. */
export const SLOT_WORDS = 4;
const PARTNER_MASK = 0x00ffffff;

export interface SpringTopology {
  /** Particle capacity. */
  readonly n: number;
  /** Bond slots per particle. */
  readonly K: number;
  /** `n * K * 4` words; see the file header. */
  readonly slots: Float32Array;
  /** The same memory as `slots`, for reading and writing word 0. */
  readonly slotBits: Uint32Array;
  /** Per particle `[partner, min, max, damp]`; partner < 0 is no rod, damp < 0 is disabled. */
  readonly rod: Float32Array;
  /** Per particle `[on, x, y, z]`; `on` 1 pins the particle to the position. */
  readonly fix: Float32Array;
  /** Per particle `[x, y, z, weight]`; weight 0 is no pull. */
  readonly pull: Float32Array;
  /** Per particle `[angle, stiffness]` of the torsion spring over particles `i .. i + 3`;
   *  stiffness 0 is none. Indexed by the first of the four, as in FoldSynth. */
  readonly torsion: Float32Array;
}

export interface Bond {
  readonly partner: number;
  readonly kind: number;
  readonly rest: number;
  readonly strength: number;
  readonly pow: number;
}

/** An empty topology for `n` particles with `K` bond slots each (CSynth's default is 8). */
export function createTopology(n: number, K = 8): SpringTopology {
  if (n > PARTNER_MASK) throw new Error(`springs: at most ${PARTNER_MASK} particles (got ${n})`);
  const slots = new Float32Array(n * K * SLOT_WORDS);
  const slotBits = new Uint32Array(slots.buffer);
  for (let s = 0; s < n * K; s++) slotBits[s * SLOT_WORDS] = NO_BOND;
  const rod = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) rod[i * 4] = -1;
  return { n, K, slots, slotBits, rod, fix: new Float32Array(n * 4), pull: new Float32Array(n * 4), torsion: new Float32Array(n * 2) };
}

/** Partner index held in slot word 0. */
export function slotPartner(bits: number): number {
  return bits & PARTNER_MASK;
}

/** Bond kind held in slot word 0. */
export function slotKind(bits: number): number {
  return bits >>> 24;
}

/** Word 0 for a bond to `partner` of `kind`. */
export function slotWord(partner: number, kind: number): number {
  return ((kind << 24) | partner) >>> 0;
}

/** Offset of the slot in row `i` bonded to `j`, or -1. */
export function findSlot(t: SpringTopology, i: number, j: number): number {
  const base = i * t.K * SLOT_WORDS;
  for (let k = 0; k < t.K; k++) {
    const o = base + k * SLOT_WORDS;
    const bits = t.slotBits[o]!;
    if (bits !== NO_BOND && slotPartner(bits) === j) return o;
  }
  return -1;
}

/** Offset of the first empty slot in row `i`, or -1 when the row is full. */
export function freeSlot(t: SpringTopology, i: number): number {
  const base = i * t.K * SLOT_WORDS;
  for (let k = 0; k < t.K; k++) {
    const o = base + k * SLOT_WORDS;
    if (t.slotBits[o] === NO_BOND) return o;
  }
  return -1;
}

function writeSlot(t: SpringTopology, o: number, partner: number, kind: number, rest: number, strength: number, pow: number): void {
  t.slotBits[o] = slotWord(partner, kind);
  t.slots[o + 1] = rest;
  t.slots[o + 2] = strength;
  t.slots[o + 3] = pow;
}

/**
 * Add a spring between `a` and `b`, or update it when one exists. Returns false, and changes
 * nothing, when either row is full. CSynth's `addspring`.
 */
export function addSpring(t: SpringTopology, a: number, b: number, rest = 1, strength = 1, pow = 0, kind = 0): boolean {
  if (a === b) return false;
  const oa = findSlot(t, a, b);
  const ob = findSlot(t, b, a);
  const sa = oa >= 0 ? oa : freeSlot(t, a);
  const sb = ob >= 0 ? ob : freeSlot(t, b);
  if (sa < 0 || sb < 0) return false;
  writeSlot(t, sa, b, kind, rest, strength, pow);
  writeSlot(t, sb, a, kind, rest, strength, pow);
  return true;
}

/** Remove the spring between `a` and `b` from both rows. */
export function removeSpring(t: SpringTopology, a: number, b: number): void {
  const oa = findSlot(t, a, b);
  if (oa >= 0) t.slotBits[oa] = NO_BOND;
  const ob = findSlot(t, b, a);
  if (ob >= 0) t.slotBits[ob] = NO_BOND;
}

/** Remove every spring of particle `i`, from both ends. */
export function removeAllSprings(t: SpringTopology, i: number): void {
  const base = i * t.K * SLOT_WORDS;
  for (let k = 0; k < t.K; k++) {
    const bits = t.slotBits[base + k * SLOT_WORDS]!;
    if (bits !== NO_BOND) removeSpring(t, i, slotPartner(bits));
  }
}

/** The spring between `a` and `b`, if any. */
export function getSpring(t: SpringTopology, a: number, b: number): Bond | undefined {
  const o = findSlot(t, a, b);
  if (o < 0) return undefined;
  const bits = t.slotBits[o]!;
  return { partner: slotPartner(bits), kind: slotKind(bits), rest: t.slots[o + 1]!, strength: t.slots[o + 2]!, pow: t.slots[o + 3]! };
}

/** The bonds of particle `i`. */
export function bondsOf(t: SpringTopology, i: number): Bond[] {
  const out: Bond[] = [];
  const base = i * t.K * SLOT_WORDS;
  for (let k = 0; k < t.K; k++) {
    const o = base + k * SLOT_WORDS;
    const bits = t.slotBits[o]!;
    if (bits !== NO_BOND) {
      out.push({
        partner: slotPartner(bits),
        kind: slotKind(bits),
        rest: t.slots[o + 1]!,
        strength: t.slots[o + 2]!,
        pow: t.slots[o + 3]!,
      });
    }
  }
  return out;
}

/** Number of bonds, each counted once. */
export function bondCount(t: SpringTopology): number {
  let c = 0;
  for (let s = 0; s < t.n * t.K; s++) if (t.slotBits[s * SLOT_WORDS] !== NO_BOND) c++;
  return c / 2;
}

/**
 * The first violation of the table's invariant, or `undefined` when it holds: `j` is in row `i`
 * exactly when `i` is in row `j`, with the same kind, rest length, strength and power.
 */
export function topologyDefect(t: SpringTopology): string | undefined {
  for (let i = 0; i < t.n; i++) {
    for (const b of bondsOf(t, i)) {
      const back = getSpring(t, b.partner, i);
      if (!back) return `bond ${i}->${b.partner} has no return`;
      if (back.kind !== b.kind || back.rest !== b.rest || back.strength !== b.strength || back.pow !== b.pow) {
        return `bond ${i}<->${b.partner} differs between its ends`;
      }
    }
  }
  return undefined;
}

/** Hold `a` between `min` and `max` from `b`. One rod per particle. CSynth's `addrod`. */
export function addRod(t: SpringTopology, a: number, b: number, min = 1, max = min, damp = 1): void {
  t.rod.set([b, min, max, damp], a * 4);
}

export function removeRod(t: SpringTopology, a: number): void {
  t.rod[a * 4] = -1;
}

/** Pin particle `i` at a position. CSynth's `setfix`. */
export function setFix(t: SpringTopology, i: number, x: number, y: number, z: number): void {
  t.fix.set([1, x, y, z], i * 4);
}

export function removeFix(t: SpringTopology, i: number): void {
  t.fix[i * 4] = 0;
}

/** Pull particle `i` towards a position with a weight. CSynth's `addpull`. */
export function setPull(t: SpringTopology, i: number, x: number, y: number, z: number, weight = 1): void {
  t.pull.set([x, y, z, weight], i * 4);
}

export function removePull(t: SpringTopology, i: number): void {
  t.pull[i * 4 + 3] = 0;
}

/** Hold the dihedral angle of particles `i, i + 1, i + 2, i + 3` at `angle` radians. FoldSynth's
 *  `addTorsionSpring`. */
export function setTorsion(t: SpringTopology, i: number, angle: number, stiffness = 1): void {
  t.torsion.set([angle, stiffness], i * 2);
}

export function removeTorsion(t: SpringTopology, i: number): void {
  t.torsion[i * 2 + 1] = 0;
}
