// Internal: row masks and domain membership as an index list. Shared by the PCF estimators and
// deliberately not re-exported from `index.ts`.

import { type Domain, domainContains } from "./domain";
import type { Coords } from "./tcm";

/** Per-cell row filter: index-aligned to the cloud, nonzero keeps the cell. */
export type RowMask = ArrayLike<number>;

export function checkMask(who: string, mask: RowMask | undefined, n: number): void {
  if (mask && mask.length !== n) throw new Error(`${who}: row mask has ${mask.length} entries for ${n} cells`);
}

/** The rows a call uses. `idx` lists them in order, or is `null` when every row is used — the
 *  common case allocates nothing. `included` counts rows `include` keeps, before the domain test. */
export interface RowSelection {
  readonly idx: Int32Array | null;
  readonly kept: number;
  readonly included: number;
}

/** One pass, one preallocated buffer, no column copies: callers read the original columns at `idx[k]`. */
export function selectRows(d: Domain | undefined, include: RowMask | undefined, xs: Coords, ys: Coords): RowSelection {
  const n = xs.length;
  if (!d && !include) return { idx: null, kept: n, included: n };
  const buf = new Int32Array(n);
  let kept = 0;
  let included = 0;
  for (let i = 0; i < n; i++) {
    if (include && !include[i]) continue;
    included++;
    if (d && !domainContains(d, xs[i]!, ys[i]!)) continue;
    buf[kept++] = i;
  }
  return { idx: kept === n ? null : buf.subarray(0, kept), kept, included };
}
