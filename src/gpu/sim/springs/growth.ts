// Growth: cells that divide, and particles that die. The first part of the simulation toolbox
// where the particle count changes. Capacity is fixed up front; a division brings a dead slot
// to life, so no index is ever renumbered and bonds stay valid.
//
// Runs on the host. A device version needs the k-th dividing cell matched to the k-th free slot,
// which is two prefix sums over `src/gpu/scan`.
import type { SpringParams } from "./params";
import { rand01, SALT } from "./rng";
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
    const cz = p.dims === 3 ? rand01(i, key, SALT.motileZ) * 2 - 1 : 0;
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
