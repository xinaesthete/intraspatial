// Internal: shared by the PCF estimators. Deliberately not re-exported from `index.ts`.

import type { Coords } from "./tcm";

/** A uniform grid over a point set as flat typed arrays (CSR): the points of bucket `b` are
 *  `items[start[b] .. start[b+1])`, in ascending input order. Counting sort in two passes over the
 *  points, so a large set costs three allocations rather than one array per bucket. `colOf`/`rowOf`
 *  clamp, so a query point outside the set's own extent lands in the nearest edge bucket. */
export interface PointGrid {
  readonly cols: number;
  readonly rows: number;
  readonly start: Int32Array;
  readonly items: Int32Array;
  readonly colOf: (x: number) => number;
  readonly rowOf: (y: number) => number;
}

/** Grid of cell size `cell` over the `n` points `idx` selects (all of them when `idx` is null), built
 *  over THEIR extent. Items are indices into the original `xs`/`ys`. */
export function buildGrid(xs: Coords, ys: Coords, idx: Int32Array | null, n: number, cell: number): PointGrid {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k]! : k;
    const x = xs[i]!;
    const y = ys[i]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const cols = n > 0 ? Math.max(1, Math.ceil((maxX - minX) / cell) + 1) : 1;
  const rows = n > 0 ? Math.max(1, Math.ceil((maxY - minY) / cell) + 1) : 1;
  const colOf = (x: number) => Math.min(cols - 1, Math.max(0, Math.floor((x - minX) / cell)));
  const rowOf = (y: number) => Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / cell)));
  const start = new Int32Array(cols * rows + 1);
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k]! : k;
    start[rowOf(ys[i]!) * cols + colOf(xs[i]!) + 1]!++;
  }
  for (let b = 0; b < cols * rows; b++) start[b + 1]! += start[b]!;
  const cursor = start.slice(0, cols * rows);
  const items = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k]! : k;
    items[cursor[rowOf(ys[i]!) * cols + colOf(xs[i]!)]!++] = i;
  }
  return { cols, rows, start, items, colOf, rowOf };
}
