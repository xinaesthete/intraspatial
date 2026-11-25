// Start a spring-model tissue from measured cell positions: pick the cells nearest a point on a
// map, put them in model units (one unit is about one cell diameter), and plant them in the gel in
// place of the scenario's own ball of cells.
import type { Matrix16 } from "./springsLayers";
import { DEAD, MATRIX, removeAllSprings, removeFix, type SpheroidScenario, spawn } from "../../../src/gpu/sim/springs/index";

export interface CellPatch {
  /** Indices into the table of the chosen cells, nearest first. */
  readonly indices: Int32Array;
  /** The patch's centre, in the table's units. */
  readonly centre: readonly [number, number];
  /** Table units per model unit: the typical spacing between neighbouring cells in the patch. */
  readonly spacing: number;
  /** Radius of the patch, in the table's units. */
  readonly radius: number;
}

/** The `count` cells nearest `(cx, cy)`. The spacing is the side of the square each cell would
 *  have if they tiled the patch's disc evenly: sqrt(disc area / count). */
export function nearestCells(xs: ArrayLike<number>, ys: ArrayLike<number>, cx: number, cy: number, count: number): CellPatch {
  const n = Math.min(xs.length, ys.length);
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) d2[i] = (xs[i]! - cx) ** 2 + (ys[i]! - cy) ** 2;
  const order = new Int32Array(n).map((_, i) => i).sort((a, b) => d2[a]! - d2[b]!);
  const indices = order.slice(0, Math.min(count, n));
  const radius = indices.length > 0 ? Math.sqrt(d2[indices[indices.length - 1]!]!) : 0;
  const spacing = indices.length > 1 && radius > 0 ? Math.sqrt((Math.PI * radius * radius) / indices.length) : 1;
  return { indices, centre: [cx, cy], spacing, radius };
}

/** The patch in model units, on the plane z = 0. The table's y runs down the screen and the
 *  model's runs up, so y is negated. */
export function patchToModel(xs: ArrayLike<number>, ys: ArrayLike<number>, patch: CellPatch): [number, number, number][] {
  const [cx, cy] = patch.centre;
  return Array.from(patch.indices, (i): [number, number, number] => [(xs[i]! - cx) / patch.spacing, -(ys[i]! - cy) / patch.spacing, 0]);
}

/** The matrix that undoes `patchToModel`, column-major, for drawing the model over the table. */
export function modelToTable(patch: CellPatch): Matrix16 {
  const s = patch.spacing;
  const [cx, cy] = patch.centre;
  // z is flattened: over a map the model is seen from above.
  return [s, 0, 0, 0, 0, -s, 0, 0, 0, 0, 0, 0, cx, cy, 0, 1];
}

/**
 * Replace a spheroid scenario's cells with `cells` (model units). Gel particles closer than
 * `clearance` to a new cell are removed with their bonds, so no cell starts inside the gel.
 * Call before the scenario is stepped or uploaded. Returns how many cells were planted.
 */
export function seedCells(sc: SpheroidScenario, cells: readonly (readonly [number, number, number])[], clearance = 0.8): number {
  const { state, topology } = sc;
  const firstMatrix = state.species.indexOf(MATRIX);
  const capacity = firstMatrix < 0 ? state.n : firstMatrix;
  for (let i = 0; i < capacity; i++) {
    if (state.species[i] === DEAD) continue;
    removeAllSprings(topology, i);
    state.species[i] = DEAD;
  }
  const planted = cells.slice(0, capacity);
  const reach2 = clearance * clearance;
  for (let i = capacity; i < state.n; i++) {
    if (state.species[i] !== MATRIX) continue;
    const x = state.pos[i * 4]!;
    const y = state.pos[i * 4 + 1]!;
    const z = state.pos[i * 4 + 2]!;
    if (!planted.some(([px, py, pz]) => (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2 < reach2)) continue;
    removeAllSprings(topology, i);
    removeFix(topology, i);
    state.species[i] = DEAD;
  }
  planted.forEach(([x, y, z], k) => spawn(state, k, x, y, z));
  return planted.length;
}
