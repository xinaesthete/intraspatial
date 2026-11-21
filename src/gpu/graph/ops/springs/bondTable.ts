// The bond table as a graph value: a bundle of two parts, `link` and `spring`, each `n × K`
// entries (docs/bond-primitive.md, docs/spring-model-as-ops.md). Row `i` holds the bonds of
// particle `i`; every bond is stored under both of its ends, so a force op gathers from its own
// row and writes nothing else.
//
//   link    u32    partner index in the low 24 bits, bond kind in the high 8; `NO_LINK` = empty
//   spring  vec4f  rest length, strength, distance power, unused
//
// Two parts rather than the legacy four-word slot: the partner word of an empty slot is a NaN bit
// pattern, which an f32 buffer need not preserve, and with the integer word on its own no kernel
// has to bitcast.
import { NO_BOND, SLOT_WORDS, type SpringTopology } from "#src/gpu/sim/springs/topology";
import type { ElementType, FieldValue, Shape } from "../../handle";
import { type BundleSpec, bundleValue, partNames } from "../bundleOps";

export const BOND_TABLE: BundleSpec = {
  name: "bondTable",
  label: "Bond table",
  parts: [
    { name: "link", kind: "points", describe: "Per slot: partner index and bond kind, or empty." },
    { name: "spring", kind: "points", describe: "Per slot: rest length, strength and distance power." },
  ],
};

/** The empty-slot marker in `link`. */
export const NO_LINK = NO_BOND;
/** Kinds a bond can have: the high byte of `link` holds one, and each has a strength multiplier. */
export const BOND_KINDS = 8;

export const VEC4: ElementType = { kind: "vec", n: 4 };

/** The static description of a bond-table port. */
export const BOND_TABLE_PORT = { name: BOND_TABLE.name, parts: partNames(BOND_TABLE) } as const;

/** Slots per particle of a bond table, given how many particles it is for. Checked at build time. */
export function slotsPerParticle(table: Shape, n: number, who: string): number {
  if (table.kind !== "bundle" || table.name !== BOND_TABLE.name) throw new Error(`${who}: expected a bond table`);
  const link = table.parts.link;
  const spring = table.parts.spring;
  if (!link || !spring || link.kind !== "points" || spring.kind !== "points" || link.n !== spring.n) {
    throw new Error(`${who}: bond table parts do not agree`);
  }
  if (n === 0 || link.n % n !== 0) throw new Error(`${who}: bond table of ${link.n} slots does not divide among ${n} particles`);
  return link.n / n;
}

/** A host bond table from separate arrays. */
export function bondTableValue(link: Uint32Array, spring: Float32Array): FieldValue {
  const slots = link.length;
  if (spring.length !== slots * 4) throw new Error(`bondTable: ${slots} links need ${slots * 4} spring floats, got ${spring.length}`);
  return bundleValue(BOND_TABLE, [
    { shape: { kind: "points", n: slots }, dtype: "u32", data: link },
    { shape: { kind: "points", n: slots }, dtype: "f32", element: VEC4, data: spring },
  ]);
}

/** The bond table of a `SpringTopology` (the fused model's four-word slots), as a graph value. */
export function bondTableOf(t: SpringTopology): FieldValue {
  const slots = t.n * t.K;
  const link = new Uint32Array(slots);
  const spring = new Float32Array(slots * 4);
  for (let s = 0; s < slots; s++) {
    const o = s * SLOT_WORDS;
    link[s] = t.slotBits[o]!;
    if (link[s] === NO_BOND) continue;
    spring[s * 4] = t.slots[o + 1]!;
    spring[s * 4 + 1] = t.slots[o + 2]!;
    spring[s * 4 + 2] = t.slots[o + 3]!;
  }
  return bondTableValue(link, spring);
}

/** Write a bond table's two parts back into a `SpringTopology`'s four-word slots. */
export function writeBondTable(t: SpringTopology, link: ArrayLike<number>, spring: ArrayLike<number>): void {
  const slots = t.n * t.K;
  if (link.length !== slots) throw new Error(`bondTable: ${link.length} links do not fit a topology of ${slots} slots`);
  for (let s = 0; s < slots; s++) {
    const o = s * SLOT_WORDS;
    t.slots[o + 1] = spring[s * 4]!;
    t.slots[o + 2] = spring[s * 4 + 1]!;
    t.slots[o + 3] = spring[s * 4 + 2]!;
    // Last: the integer word shares the buffer with the floats, and its bits must be kept.
    t.slotBits[o] = link[s]!;
  }
}
