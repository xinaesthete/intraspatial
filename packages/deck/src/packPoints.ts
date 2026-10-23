// Points as the splat reads them, [x, y, w] per point — and the one place host data is sanitised.
//
// A single NaN weight poisons every grid cell its kernel touches: additive blending turns the whole
// footprint (±4σ) into NaN, which draws as a solid block. That is what a normalised sparse score
// gives for every row without a value ((value − min) / range), and it looks like a rendering bug,
// not bad input. The shader can't be trusted to catch it — WGSL compilers may assume NaN never
// occurs and fold `w == w` to true — so it is done here, on the host, in the loop that packs the
// points anyway.
//
// - a non-finite weight counts as 0;
// - a point with a non-finite coordinate is left out (weight 0), and kept out of the bounds, where an
//   Infinity would otherwise stretch the grid over nothing.
//
// The bounds cover only points that contribute (weight ≠ 0), so a category or filter that selects a
// corner of the data gets a grid over that corner, not over everything.

import type { Bounds } from "./fieldPlan";

export interface PackedPoints {
  /** [x, y, w] per point; at least one point's worth, so the buffer is never zero-sized. */
  xyw: Float32Array;
  /** Points packed (including left-out ones, which carry weight 0). */
  n: number;
  /** Bounds of the contributing points, or null if none contribute. */
  bounds: Bounds | null;
  /** The largest weight (0 if none contribute) — the fixed-point scale of the compute splat. */
  maxWeight: number;
}

export function packPoints(x: ArrayLike<number>, y: ArrayLike<number>, weights?: ArrayLike<number> | null): PackedPoints {
  const n = Math.min(x.length, y.length);
  const xyw = new Float32Array(Math.max(1, n) * 3);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxWeight = 0;
  for (let i = 0; i < n; i++) {
    const px = x[i]!;
    const py = y[i]!;
    const raw = weights ? weights[i]! : 1;
    const placed = Number.isFinite(px) && Number.isFinite(py);
    const w = placed && Number.isFinite(raw) ? raw : 0;
    xyw[3 * i] = placed ? px : 0;
    xyw[3 * i + 1] = placed ? py : 0;
    xyw[3 * i + 2] = w;
    if (w === 0) continue;
    if (w > maxWeight) maxWeight = w;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  }
  return { xyw, n, bounds: minX <= maxX ? [minX, minY, maxX, maxY] : null, maxWeight };
}
