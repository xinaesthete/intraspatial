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

/** The lattice a bucket grid is laid over. `src/gpu/spatial/gridIndex.ts` builds the same
 *  structure on the device from exactly this, so the two builds agree cell for cell. */
export interface GridLattice {
  readonly cols: number;
  readonly rows: number;
  /** Bucket side length in world units (= the query radius the grid was built for). */
  readonly cell: number;
  readonly minX: number;
  readonly minY: number;
}

export interface BucketGrid extends GridLattice {
  /** `cols*rows + 1` prefix offsets into `items`. */
  readonly start: Int32Array;
  /** Point indices grouped by bucket; `items[start[b] .. start[b+1])` is bucket b. */
  readonly items: Int32Array;
}

/** `[minX, minY, maxX, maxY]` of the points; all zeros for no points. */
function boundsOf(xs: ArrayLike<number>, ys: ArrayLike<number>): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!;
    const y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : [0, 0, 0, 0];
}

/** The lattice `buildBucketGrid` lays over `(xs, ys)` with the given cell size: origin at the
 *  min corner, `ceil(extent / cell) + 1` buckets per axis (the `+ 1` keeps a point on the max
 *  edge in range without a clamp). `bounds` (`[minX, minY, maxX, maxY]`) fixes origin/extent;
 *  without it the points' own bounds are used. */
export function latticeFor(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds?: readonly [number, number, number, number],
): GridLattice {
  const [minX, minY, maxX, maxY] = bounds ?? boundsOf(xs, ys);
  const c = Math.max(cell, 1e-9);
  return {
    cols: Math.max(1, Math.ceil((maxX - minX) / c) + 1),
    rows: Math.max(1, Math.ceil((maxY - minY) / c) + 1),
    cell: c,
    minX,
    minY,
  };
}

/** Build a CSR bucket grid over `(xs, ys)` with the given cell size; see `latticeFor` for `bounds`. */
export function buildBucketGrid(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds?: readonly [number, number, number, number],
): BucketGrid {
  const n = xs.length;
  const lattice = latticeFor(xs, ys, cell, bounds);
  const { cols, rows, cell: c, minX, minY } = lattice;
  const nb = cols * rows;

  const start = new Int32Array(nb + 1);
  const items = new Int32Array(n);
  const bucketOf = (i: number) => {
    const col = Math.min(cols - 1, Math.max(0, Math.floor((xs[i]! - minX) / c)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor((ys[i]! - minY) / c)));
    return row * cols + col;
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
