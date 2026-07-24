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

export interface BucketGrid {
  readonly cols: number;
  readonly rows: number;
  /** Bucket side length in world units (= the query radius the grid was built for). */
  readonly cell: number;
  readonly minX: number;
  readonly minY: number;
  /** `cols*rows + 1` prefix offsets into `items`. */
  readonly start: Int32Array;
  /** Point indices grouped by bucket; `items[start[b] .. start[b+1])` is bucket b. */
  readonly items: Int32Array;
}

/** Build a CSR bucket grid over `(xs, ys)` with the given cell size. `bounds`
 *  (`[minX, minY, maxX, maxY]`) fixes the grid origin/extent; without it the points' own bounds
 *  are used. */
export function buildBucketGrid(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  cell: number,
  bounds?: readonly [number, number, number, number],
): BucketGrid {
  const n = xs.length;
  let minX: number;
  let minY: number;
  let maxX: number;
  let maxY: number;
  if (bounds) {
    [minX, minY, maxX, maxY] = bounds;
  } else {
    minX = Infinity;
    minY = Infinity;
    maxX = -Infinity;
    maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xs[i]!;
      const y = ys[i]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (!Number.isFinite(minX)) {
      minX = 0;
      minY = 0;
      maxX = 0;
      maxY = 0;
    }
  }
  const c = Math.max(cell, 1e-9);
  const cols = Math.max(1, Math.ceil((maxX - minX) / c) + 1);
  const rows = Math.max(1, Math.ceil((maxY - minY) / c) + 1);
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
  return { cols, rows, cell: c, minX, minY, start, items };
}
