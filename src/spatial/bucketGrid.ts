// Uniform bucket grid in CSR (compressed) form — the neighbourhood index the cell-stats kernels
// walk instead of an O(N_A·N_B) all-pairs loop.
//
// A bucket grid with cell size `r` has the property the whole cell-stats front leans on: every
// point within distance `r` of a query lies in the query's own bucket or one of the 8 around it,
// so an exact per-pair distance test over just that 3×3 neighbourhood reproduces the all-pairs
// count. (`src/spatial/tcm.ts` and `pcf.ts` build the same structure inline as `number[][]`; this
// is the flat form, which is what a GPU kernel can actually bind.)
//
// CSR layout, two Int32Arrays, no per-bucket allocation:
//   start[b] .. start[b+1]   — the slice of `items` holding bucket b's point indices
//   items[k]                 — a point index
// Built by counting sort: one pass to count, a prefix sum, one pass to place. O(N), typed arrays
// throughout, so the build stays negligible next to the pair work it accelerates.
//
// Points outside `bounds` are CLAMPED into the edge buckets rather than dropped. That is safe for
// the 3×3 query: a point clamped from outside is at least as far away as the bucket it lands in
// implies, so the exact distance test still rejects it, and a genuinely in-range point can never
// be clamped past one bucket of its true position.
//
// Third axis: pass `zs` (and optionally 6-number bounds) and the lattice gains `depth`/`minZ`,
// the stencil becomes 3×3×3, and `cellId = cx + cols*(cy + rows*cz)` — x-fastest, so in 2D it is
// the existing `row*cols + col`. One implementation, not two: `depth === undefined` is 2D.

/** The lattice a bucket grid is laid over. `src/gpu/spatial/gridIndex.ts` builds the same
 *  structure on the device from exactly this, so the two builds agree cell for cell. */
export interface GridLattice {
  readonly cols: number;
  readonly rows: number;
  /** Bucket side length in world units (= the query radius the grid was built for). */
  readonly cell: number;
  readonly minX: number;
  readonly minY: number;
  /** Cells along z. Absent (with `minZ`) on a 2D lattice. */
  readonly depth?: number;
  readonly minZ?: number;
}

/** `[minX, minY, maxX, maxY]`. */
export type Bounds2 = readonly [number, number, number, number];
/** `[minX, minY, minZ, maxX, maxY, maxZ]`. */
export type Bounds3 = readonly [number, number, number, number, number, number];

export interface BucketGrid extends GridLattice {
  /** `numCells(lattice) + 1` prefix offsets into `items`. */
  readonly start: Int32Array;
  /** Point indices grouped by bucket; `items[start[b] .. start[b+1])` is bucket b. */
  readonly items: Int32Array;
}

/** `[min, max]` of one coordinate array; `[0, 0]` for no points. */
function rangeOf(vs: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < vs.length; i++) {
    const v = vs[i]!;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? [lo, hi] : [0, 0];
}

/** Cells in the lattice: `cols * rows * (depth ?? 1)`. */
export function numCells(lattice: GridLattice): number {
  return lattice.cols * lattice.rows * (lattice.depth ?? 1);
}

/** The lattice `buildBucketGrid` lays over `(xs, ys[, zs])` with the given cell size: origin at
 *  the min corner, `ceil(extent / cell) + 1` buckets per axis (the `+ 1` keeps a point on the max
 *  edge in range without a clamp). `bounds` fixes origin/extent — `[minX, minY, maxX, maxY]` in
 *  2D, `[minX, minY, minZ, maxX, maxY, maxZ]` when `zs` is given; without it the points' own
 *  bounds are used. The lattice has a third axis (`depth`/`minZ`) exactly when `zs` is given. */
export function latticeFor(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds?: Bounds2 | Bounds3,
  zs?: ArrayLike<number>,
): GridLattice {
  const c = Math.max(cell, 1e-9);
  const axis = (lo: number, hi: number) => Math.max(1, Math.ceil((hi - lo) / c) + 1);
  if (zs === undefined) {
    if (bounds !== undefined && bounds.length !== 4) throw new Error(`latticeFor: 2D lattice needs 4 bounds, got ${bounds.length}`);
    const [minX, maxX] = bounds ? [bounds[0], bounds[2]] : rangeOf(xs);
    const [minY, maxY] = bounds ? [bounds[1], bounds[3]] : rangeOf(ys);
    return { cols: axis(minX, maxX), rows: axis(minY, maxY), cell: c, minX, minY };
  }
  if (zs.length !== xs.length) throw new Error(`latticeFor: zs.length ${zs.length} !== xs.length ${xs.length}`);
  if (bounds !== undefined && bounds.length !== 6) throw new Error(`latticeFor: 3D lattice needs 6 bounds, got ${bounds.length}`);
  const [minX, maxX] = bounds ? [bounds[0], bounds[3]] : rangeOf(xs);
  const [minY, maxY] = bounds ? [bounds[1], bounds[4]] : rangeOf(ys);
  const [minZ, maxZ] = bounds ? [bounds[2], bounds[5]] : rangeOf(zs);
  return { cols: axis(minX, maxX), rows: axis(minY, maxY), cell: c, minX, minY, depth: axis(minZ, maxZ), minZ };
}

/** Build a CSR bucket grid over `(xs, ys[, zs])` with the given cell size; see `latticeFor` for
 *  `bounds` and the third axis. */
export function buildBucketGrid(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds?: Bounds2 | Bounds3,
  zs?: ArrayLike<number>,
): BucketGrid {
  const n = xs.length;
  const lattice = latticeFor(xs, ys, cell, bounds, zs);
  const { cols, rows, cell: c, minX, minY } = lattice;
  const depth = lattice.depth ?? 1;
  const minZ = lattice.minZ ?? 0;
  const nb = cols * rows * depth;

  const start = new Int32Array(nb + 1);
  const items = new Int32Array(n);
  const bucketOf = (i: number) => {
    const col = Math.min(cols - 1, Math.max(0, Math.floor((xs[i]! - minX) / c)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor((ys[i]! - minY) / c)));
    const lay = zs === undefined ? 0 : Math.min(depth - 1, Math.max(0, Math.floor((zs[i]! - minZ) / c)));
    return col + cols * (row + rows * lay);
  };

  for (let i = 0; i < n; i++) start[bucketOf(i) + 1]!++;
  for (let b = 0; b < nb; b++) start[b + 1]! += start[b]!;
  const cursor = start.slice(0, nb); // running write head per bucket
  for (let i = 0; i < n; i++) {
    const b = bucketOf(i);
    items[cursor[b]!++] = i;
  }
  return { ...lattice, start, items };
}
