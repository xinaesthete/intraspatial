// A TCM layer's points as the renderer reads them, and the one place its host data is sanitised.
//
// One point set, two weightings: `xy` is shared by A and B, so a layer per cell-type pair costs two
// weight arrays rather than two copies of the coordinates. As in `packPoints`:
//
// - a point with a non-finite coordinate is left out of both (weight 0), and kept out of every bound;
// - a non-finite weight counts as 0. So does a NEGATIVE one: a weight is a mass here (ρ̂_B and ρ_B
//   are weight per unit area), and a negative mass has no TCM meaning.

import type { Bounds } from "./fieldPlan";

export interface TcmPoints {
  /** [x, y] per point; at least one point's worth, so no buffer is ever zero-sized. */
  xy: Float32Array;
  /** A's weight per point (same length as the points, at least 1). */
  wA: Float32Array;
  /** B's weight per point. */
  wB: Float32Array;
  /** Points packed, including left-out ones. */
  n: number;
  /** Bounds of the points with A weight > 0 — where Γ can be non-zero, before its support — or
   *  null when A is empty. */
  boundsA: Bounds | null;
  /** Bounds of every point with finite coordinates: the default region ρ_B is taken over. */
  boundsAll: Bounds | null;
  /** Total B weight. */
  totalB: number;
}

function clean(w: number | undefined): number {
  return w !== undefined && Number.isFinite(w) && w > 0 ? w : 0;
}

export function packTcmPoints(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  weightsA?: ArrayLike<number> | null,
  weightsB?: ArrayLike<number> | null,
): TcmPoints {
  const n = Math.min(x.length, y.length);
  const m = Math.max(1, n);
  const xy = new Float32Array(2 * m);
  const wA = new Float32Array(m);
  const wB = new Float32Array(m);
  const a = new Extent();
  const all = new Extent();
  let totalB = 0;
  for (let i = 0; i < n; i++) {
    const px = x[i]!;
    const py = y[i]!;
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue; // left out: zeros already
    xy[2 * i] = px;
    xy[2 * i + 1] = py;
    all.add(px, py);
    const ai = weightsA ? clean(weightsA[i]) : 1;
    const bi = weightsB ? clean(weightsB[i]) : 1;
    wA[i] = ai;
    wB[i] = bi;
    totalB += bi;
    if (ai > 0) a.add(px, py);
  }
  return { xy, wA, wB, n, boundsA: a.bounds(), boundsAll: all.bounds(), totalB };
}

class Extent {
  minX = Infinity;
  minY = Infinity;
  maxX = -Infinity;
  maxY = -Infinity;
  add(x: number, y: number): void {
    if (x < this.minX) this.minX = x;
    if (x > this.maxX) this.maxX = x;
    if (y < this.minY) this.minY = y;
    if (y > this.maxY) this.maxY = y;
  }
  bounds(): Bounds | null {
    return this.minX <= this.maxX ? [this.minX, this.minY, this.maxX, this.maxY] : null;
  }
}

/** ρ_B: total B weight per unit area of `region`. 0 for an empty or degenerate region. A region of
 *  zero extent along one axis (all points on a line) has no area to take a density over. */
export function densityOver(totalB: number, region: Bounds | null): number {
  if (!region) return 0;
  const area = (region[2] - region[0]) * (region[3] - region[1]);
  return area > 0 ? totalB / area : 0;
}
