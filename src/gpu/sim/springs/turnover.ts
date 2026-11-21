// Bond turnover: breaking bonds and forming new ones while the simulation runs. This is what
// CSynth does not have (its topology is edited by the host), and it is where viscoelasticity
// comes from: a network whose bonds never break is elastic, and one whose bonds turn over
// relaxes its stress. Design: docs/bond-primitive.md.
//
// Both functions are written the way a GPU kernel has to work. A particle decides about its own
// row only, from values both ends of a bond hold identically, so the two ends always agree and
// the table stays symmetric without either end writing the other's row.
import type { SpringParams } from "./params";
import { pairHash, probThreshold, SALT } from "./rng";
import { DEAD, gatherNeighbours, neighbourGrid, type SpringState } from "./state";
import { findSlot, freeSlot, NO_BOND, SLOT_WORDS, type SpringTopology, slotKind, slotPartner, slotWord } from "./topology";

export interface BreakRule {
  /** Bitmask of the bond kinds this rule may break. */
  readonly kinds: number;
  /** Break when the strain `|len / rest - 1|` exceeds this, stretched or compressed. The
   *  Elosegui-Artola rule. */
  readonly maxStrain?: number;
  /** With `maxStrain`: only a stretched bond breaks, so a compressed network keeps its bonds and
   *  fails by cracking where it is pulled apart. */
  readonly tensionOnly?: boolean;
  /** Break with probability `1 / tau` per step: a Maxwell element with relaxation time `tau`. */
  readonly tau?: number;
}

/** Whether the bond in slot `o` of row `i` breaks this step. Symmetric in the two ends. */
function breaks(s: SpringState, t: SpringTopology, p: SpringParams, rule: BreakRule, i: number, o: number): boolean {
  const bits = t.slotBits[o]!;
  const j = slotPartner(bits);
  if (s.species[j] === DEAD || s.species[i] === DEAD) return true;
  if (((rule.kinds >> slotKind(bits)) & 1) === 0) return false;
  if (rule.maxStrain !== undefined) {
    const len = Math.hypot(s.pos[j * 4]! - s.pos[i * 4]!, s.pos[j * 4 + 1]! - s.pos[i * 4 + 1]!, s.pos[j * 4 + 2]! - s.pos[i * 4 + 2]!);
    const rest = t.slots[o + 1]! * p.springlen + p.springlenfix;
    const strain = len / rest - 1;
    if ((rule.tensionOnly ? strain : Math.abs(strain)) > rule.maxStrain) return true;
  }
  if (rule.tau !== undefined && pairHash(i, j, s.step, p.seed ^ SALT.bondBreak) < probThreshold(1 / rule.tau)) return true;
  return false;
}

/**
 * Clear every bond the rule breaks, and every bond to a dead particle. Returns bonds broken.
 * `onBreak` is called once per broken bond, before it is cleared.
 */
export function breakBonds(
  s: SpringState,
  t: SpringTopology,
  p: SpringParams,
  rule: BreakRule,
  onBreak?: (i: number, j: number) => void,
): number {
  // Decide first, clear second: a decision must not see another row's clears from this step.
  const doomed: number[] = [];
  for (let i = 0; i < t.n; i++) {
    const base = i * t.K * SLOT_WORDS;
    for (let k = 0; k < t.K; k++) {
      const o = base + k * SLOT_WORDS;
      if (t.slotBits[o] !== NO_BOND && breaks(s, t, p, rule, i, o)) doomed.push(o);
    }
  }
  for (const o of doomed) {
    if (onBreak) {
      const i = Math.floor(o / (t.K * SLOT_WORDS));
      const j = slotPartner(t.slotBits[o]!);
      if (i < j) onBreak(i, j);
    }
  }
  for (const o of doomed) t.slotBits[o] = NO_BOND;
  return doomed.length / 2;
}

export interface FormRule {
  /** Largest distance at which a bond forms. */
  readonly range: number;
  /** Species × species (`MAX_SPECIES` square) bond kind to form, or -1 for none. */
  readonly kindOf: readonly number[];
  /** Strength of a new bond, per kind. */
  readonly strength: readonly number[];
  /** Rest length of a new bond: its current length (it forms stress-free), or a fixed value. */
  readonly rest: "current" | number;
  /** Chance per step that a given pair is allowed to bond; default 1. */
  readonly prob?: number;
  /** Number of species per side of `kindOf`. */
  readonly speciesCount: number;
}

/**
 * Form bonds by mutual choice. Each particle with a free slot proposes to its nearest eligible
 * neighbour; a bond forms only where two particles propose to each other. Each particle gains
 * at most one bond per call. Returns bonds formed.
 */
export function formBonds(s: SpringState, t: SpringTopology, p: SpringParams, rule: FormRule): number {
  const n = s.n;
  const grid = neighbourGrid(s, rule.range, p.dims);
  const cand = new Int32Array(n);
  const proposal = new Int32Array(n).fill(-1);
  const threshold = probThreshold(rule.prob ?? 1);

  for (let i = 0; i < n; i++) {
    const si = s.species[i]!;
    if (si === DEAD || freeSlot(t, i) < 0) continue;
    const x = s.pos[i * 4]!;
    const y = s.pos[i * 4 + 1]!;
    const z = s.pos[i * 4 + 2]!;
    const count = gatherNeighbours(grid, x, y, z, cand);
    let best = -1;
    let bestLen = Infinity;
    for (let c = 0; c < count; c++) {
      const j = cand[c]!;
      const sj = s.species[j]!;
      if (j === i || sj === DEAD) continue;
      if (rule.kindOf[si * rule.speciesCount + sj]! < 0) continue;
      const len = Math.hypot(s.pos[j * 4]! - x, s.pos[j * 4 + 1]! - y, s.pos[j * 4 + 2]! - z);
      if (len > rule.range || len === 0) continue;
      if (len > bestLen || (len === bestLen && j > best)) continue;
      if (findSlot(t, i, j) >= 0 || freeSlot(t, j) < 0) continue;
      if (pairHash(i, j, s.step, p.seed ^ SALT.bondForm) >= threshold) continue;
      best = j;
      bestLen = len;
    }
    proposal[i] = best;
  }

  let formed = 0;
  for (let i = 0; i < n; i++) {
    const j = proposal[i]!;
    if (j < 0 || proposal[j] !== i) continue;
    const o = freeSlot(t, i);
    const kind = rule.kindOf[s.species[i]! * rule.speciesCount + s.species[j]!]!;
    const len = Math.hypot(s.pos[j * 4]! - s.pos[i * 4]!, s.pos[j * 4 + 1]! - s.pos[i * 4 + 1]!, s.pos[j * 4 + 2]! - s.pos[i * 4 + 2]!);
    t.slotBits[o] = slotWord(j, kind);
    t.slots[o + 1] = rule.rest === "current" ? (len - p.springlenfix) / p.springlen : rule.rest;
    t.slots[o + 2] = rule.strength[kind]!;
    t.slots[o + 3] = 0;
    formed++;
  }
  return formed / 2;
}
