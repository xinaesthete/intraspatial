// Growth: cells that divide, and particles that die. The first part of the simulation toolbox
// where the particle count changes. Capacity is fixed up front; a division brings a dead slot
// to life, so no index is ever renumbered and bonds stay valid.
//
// These are the host versions, and the reference. The device versions are graph ops in
// src/gpu/graph/ops/springs/ (`division.ts`, `leaders.ts`).
import type { SpringParams } from "./params";
import { hash3, rand01, rateThreshold, SALT } from "./rng";
import { DEAD, firstDead, type SpringState } from "./state";
import { removeAllSprings, type SpringTopology } from "./topology";

export interface DivisionRule {
  /** Species that divides. */
  readonly species: number;
  /** Mean number of steps between divisions of an uncrowded cell. */
  readonly tau: number;
  /** A cell whose density (`pos.w`, the sum of `1 / len³` over its neighbours) is above this
   *  does not divide. This is the pressure feedback: crowded cells stop growing. */
  readonly maxDensity: number;
  /** Distance between the two daughters. */
  readonly offset: number;
}

/** Divide the cells the rule selects this step. Returns the number of divisions. `onDivide` is
 *  called with the parent and the slot its daughter was born into. */
export function divideCells(
  s: SpringState,
  p: SpringParams,
  rule: DivisionRule,
  onDivide?: (parent: number, daughter: number) => void,
): number {
  const key = (p.seed ^ Math.imul(s.step, 0x9e3779b1)) >>> 0;
  let slot = firstDead(s);
  let divisions = 0;
  // A daughter born into a later slot must not divide again in the same pass.
  const born = new Set<number>();
  for (let i = 0; i < s.n && slot >= 0; i++) {
    if (s.species[i] !== rule.species || born.has(i)) continue;
    if (s.pos[i * 4 + 3]! > rule.maxDensity) continue;
    if (rand01(i, key, SALT.divide) >= 1 / rule.tau) continue;
    const a = rand01(i, key, SALT.divideDir) * 2 * Math.PI;
    const cz = p.dims === 3 ? rand01(i, key, SALT.divideZ) * 2 - 1 : 0;
    const sxy = Math.sqrt(1 - cz * cz);
    const h = rule.offset / 2;
    const dx = Math.cos(a) * sxy * h;
    const dy = Math.sin(a) * sxy * h;
    const dz = cz * h;
    const x = s.pos[i * 4]!;
    const y = s.pos[i * 4 + 1]!;
    const z = s.pos[i * 4 + 2]!;
    const w = s.pos[i * 4 + 3]!;
    // Both daughters start at rest.
    s.pos.set([x - dx, y - dy, z - dz, w], i * 4);
    s.prev.set([x - dx, y - dy, z - dz, w], i * 4);
    s.pos.set([x + dx, y + dy, z + dz, w], slot * 4);
    s.prev.set([x + dx, y + dy, z + dz, w], slot * 4);
    s.species[slot] = rule.species;
    s.radius[slot] = s.radius[i]!;
    born.add(slot);
    onDivide?.(i, slot);
    divisions++;
    slot = firstDead(s, slot + 1);
  }
  return divisions;
}

/** Remove particle `i`: mark it dead and drop its bonds from both ends. */
export function killParticle(s: SpringState, t: SpringTopology, i: number): void {
  removeAllSprings(t, i);
  t.fix[i * 4] = 0;
  t.pull[i * 4 + 3] = 0;
  t.rod[i * 4] = -1;
  s.species[i] = DEAD;
}

export interface LeaderRule {
  /** Species that can become a leader, and the species a leader is. */
  readonly from: number;
  readonly to: number;
  /** Mean number of steps before an eligible cell becomes a leader. */
  readonly tau: number;
  /** A cell is eligible only with at most this many tissue neighbours within 1.3 diameters:
   *  it must be on an exposed part of the surface. */
  readonly maxNeighbours: number;
  /** No new leader within this distance of an existing one. This is lateral inhibition, and it
   *  sets how far apart branches start. */
  readonly spacing: number;
  /** A leader with at least this many tissue neighbours has been overgrown and goes back. */
  readonly revertNeighbours: number;
}

/** `leaderStatus` values. */
export const LEADER_STATUS = { none: 0, candidate: 1, stays: 2, reverts: 3 } as const;
/** How near a tissue neighbour is, in particle diameters. */
export const LEADER_NEIGHBOUR_REACH = 1.3;

/** The status of every particle, on the host. Shared with `selectLeaders`. */
export function leaderStatuses(
  pos: ArrayLike<number>,
  species: ArrayLike<number>,
  n: number,
  rule: { from: number; to: number; tau: number; maxNeighbours: number; revertNeighbours: number; seed: number; step: number },
): Uint32Array {
  const out = new Uint32Array(n);
  const key = (rule.seed ^ Math.imul(rule.step, 0x9e3779b1)) >>> 0;
  const threshold = rateThreshold(rule.tau);
  const neighbours = (i: number) => {
    let c = 0;
    for (let j = 0; j < n; j++) {
      if (j === i || (species[j] !== rule.from && species[j] !== rule.to)) continue;
      // In single precision, as the kernel measures it.
      const len = Math.fround(Math.hypot(pos[j * 4]! - pos[i * 4]!, pos[j * 4 + 1]! - pos[i * 4 + 1]!, pos[j * 4 + 2]! - pos[i * 4 + 2]!));
      if (len < Math.fround(LEADER_NEIGHBOUR_REACH)) c++;
    }
    return c;
  };
  for (let i = 0; i < n; i++) {
    if (species[i] === rule.to) out[i] = neighbours(i) >= rule.revertNeighbours ? LEADER_STATUS.reverts : LEADER_STATUS.stays;
    else if (species[i] === rule.from && hash3(i, key, SALT.leader) < threshold) {
      const c = neighbours(i);
      if (c >= 1 && c <= rule.maxNeighbours) out[i] = LEADER_STATUS.candidate;
    }
  }
  return out;
}

/** The new species from the statuses, on the host. Shared with `selectLeaders`. Returns how many
 *  particles changed. */
export function applyLeaderStatuses(
  pos: ArrayLike<number>,
  species: { [k: number]: number },
  status: ArrayLike<number>,
  n: number,
  rule: { from: number; to: number; spacing: number },
): number {
  const within = (i: number, j: number) =>
    Math.fround(Math.hypot(pos[j * 4]! - pos[i * 4]!, pos[j * 4 + 1]! - pos[i * 4 + 1]!, pos[j * 4 + 2]! - pos[i * 4 + 2]!)) <
    Math.fround(rule.spacing);
  let changed = 0;
  for (let i = 0; i < n; i++) {
    if (status[i] === LEADER_STATUS.reverts) {
      species[i] = rule.from;
      changed++;
    } else if (status[i] === LEADER_STATUS.candidate) {
      let blocked = false;
      for (let j = 0; j < n && !blocked; j++) {
        if (j === i) continue;
        if ((status[j] === LEADER_STATUS.stays || (status[j] === LEADER_STATUS.candidate && j < i)) && within(i, j)) blocked = true;
      }
      if (!blocked) {
        species[i] = rule.to;
        changed++;
      }
    }
  }
  return changed;
}

/**
 * Tip-cell selection, as in sprouting blood vessels: exposed surface cells turn into leaders at
 * random, but never close to an existing leader, and a leader that ends up buried turns back.
 * Returns how many particles changed species.
 *
 * The rule is the one the device runs (`leaderStatus` and `leaderApply`), so the two agree: a
 * candidate becomes a leader unless a leader that is staying, or a candidate with a lower index,
 * is within `spacing`. An earlier version walked the cells in order and let each new leader
 * inhibit the later ones, which cannot be done in parallel. The difference is rare: it needs two
 * candidates near each other in the same round.
 */
export function selectLeaders(s: SpringState, p: SpringParams, rule: LeaderRule): number {
  const status = leaderStatuses(s.pos, s.species, s.n, { ...rule, seed: p.seed, step: s.step });
  return applyLeaderStatuses(s.pos, s.species, status, s.n, rule);
}

/** Turn every `from` particle back into `to`: for when leader selection is switched off, since
 *  nothing else would revert the leaders already chosen. Returns how many changed. */
export function clearLeaders(s: SpringState, from: number, to: number): number {
  let changed = 0;
  for (let i = 0; i < s.n; i++) {
    if (s.species[i] !== from) continue;
    s.species[i] = to;
    changed++;
  }
  return changed;
}
