// Particle state of the spring model, and the neighbour lookup the short-range paths share.
import { type BucketGrid, buildBucketGrid } from "#src/spatial/bucketGrid";

/** Species value of an unused particle slot. Dead particles take no part in any force. */
export const DEAD = 255;

export interface SpringState {
  /** Particle capacity; live particles are those whose species is not `DEAD`. */
  readonly n: number;
  /** `[x, y, z, density]` per particle. Density is CSynth's `w`: the sum of `1 / len³` over the
   *  pairs evaluated last step. */
  pos: Float32Array;
  /** Positions one step earlier. Velocity is `(pos - prev) / springrate`, as in CSynth. */
  prev: Float32Array;
  /** Third buffer the step writes into before the three rotate. */
  scratch: Float32Array;
  readonly species: Uint8Array;
  readonly radius: Float32Array;
  /** Steps taken; a key of every random draw. */
  step: number;
}

/** A state of `n` dead particles. */
export function createState(n: number): SpringState {
  return {
    n,
    pos: new Float32Array(n * 4),
    prev: new Float32Array(n * 4),
    scratch: new Float32Array(n * 4),
    species: new Uint8Array(n).fill(DEAD),
    radius: new Float32Array(n).fill(0.5),
    step: 0,
  };
}

/** Bring particle `i` to life at rest at a position. */
export function spawn(s: SpringState, i: number, x: number, y: number, z: number, species = 0, radius = 0.5): void {
  s.pos.set([x, y, z, 0], i * 4);
  s.prev.set([x, y, z, 0], i * 4);
  s.species[i] = species;
  s.radius[i] = radius;
}

/** Number of live particles, optionally of one species. */
export function liveCount(s: SpringState, species?: number): number {
  let c = 0;
  for (let i = 0; i < s.n; i++) {
    const sp = s.species[i]!;
    if (sp !== DEAD && (species === undefined || sp === species)) c++;
  }
  return c;
}

/** Index of the first dead slot, or -1 when the state is full. */
export function firstDead(s: SpringState, from = 0): number {
  for (let i = from; i < s.n; i++) if (s.species[i] === DEAD) return i;
  return -1;
}

/** Target matrices for the chain made of the first `size` particles: CSynth's `contactbuff` and
 *  `distbuff`. Contact values ≤ -9 mark missing data and < -1000 a chain boundary. */
export interface SpringMatrices {
  readonly size: number;
  readonly contact?: Float32Array;
  readonly dist?: Float32Array;
}

/** Bucket grid over the particles, for queries out to `range`. Dead particles sit in it at
 *  their parked position and are skipped by species. */
export function neighbourGrid(s: SpringState, range: number, dims: 2 | 3): BucketGrid {
  const n = s.n;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const zs = dims === 3 ? new Float32Array(n) : undefined;
  for (let i = 0; i < n; i++) {
    xs[i] = s.pos[i * 4]!;
    ys[i] = s.pos[i * 4 + 1]!;
    if (zs) zs[i] = s.pos[i * 4 + 2]!;
  }
  return buildBucketGrid(xs, ys, range, undefined, zs);
}

/** Write the ids in the 3×3(×3) stencil around a position into `out`; returns how many. */
export function gatherNeighbours(g: BucketGrid, x: number, y: number, z: number, out: Int32Array): number {
  const depth = g.depth ?? 1;
  const cx = Math.min(g.cols - 1, Math.max(0, Math.floor((x - g.minX) / g.cell)));
  const cy = Math.min(g.rows - 1, Math.max(0, Math.floor((y - g.minY) / g.cell)));
  const cz = g.depth === undefined ? 0 : Math.min(depth - 1, Math.max(0, Math.floor((z - (g.minZ ?? 0)) / g.cell)));
  let count = 0;
  for (let dz = -1; dz <= 1; dz++) {
    const lz = cz + dz;
    if (lz < 0 || lz >= depth) continue;
    for (let dy = -1; dy <= 1; dy++) {
      const ly = cy + dy;
      if (ly < 0 || ly >= g.rows) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const lx = cx + dx;
        if (lx < 0 || lx >= g.cols) continue;
        const b = lx + g.cols * (ly + g.rows * lz);
        for (let k = g.cellOffsets[b]!; k < g.cellOffsets[b + 1]!; k++) out[count++] = g.pointIds[k]!;
      }
    }
  }
  return count;
}
