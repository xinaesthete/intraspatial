// Edge correction: the area of a disk or annulus centred on a cell, clipped to a rectangular ROI.
//
// This is the piece `docs/muspan-cell-stats-plan.md` §4 calls "the fiddly geometric piece — needed
// by *both* stats", and it is what separates a cross-PCF that agrees with the published numbers
// from one that is systematically low.
//
// ## Why it is not optional
//
// The cross-PCF (eq 8) normalises each anchor's neighbour count by the CSR expectation
// `ρ_B · A_r(x_a)`. Take `A_r` to be the full annulus `π(r₁²−r₀²)` and every anchor within `r` of
// the ROI boundary is over-normalised: part of its annulus lies outside the ROI, where by
// construction there are no cells to find, so the observed count is compared against an expectation
// that includes area the data never covered. The estimator comes out low, and — this is the part
// that matters — it comes out low *by an amount that depends on the ROI's shape*, since the
// affected fraction scales as perimeter·r/area. Measured on the covid project: g(20) runs 0.9% low
// on a 2 mm × 2 mm ROI and 11.7% low on a 648,000 µm² one. A cross-ROI comparison built on the
// uncorrected estimator is therefore comparing ROI geometry as much as biology.
//
// ## The construction
//
// Everything reduces to one quantity: `cornerArea(r, x, y)` = the area of the disk of radius `r`
// centred at the origin intersected with the axis-aligned box `[0,x] × [0,y]`. Given that, the
// intersection with any rectangle follows by inclusion–exclusion over its four corners, with the
// sign convention `h(x,y) = sgn(x)·sgn(y)·cornerArea(r,|x|,|y|)` extending it to all quadrants:
//
//     area = h(x₂,y₂) − h(x₁,y₂) − h(x₂,y₁) + h(x₁,y₁)
//
// and an annulus is the difference of two disks. `cornerArea` itself splits into three cases on
// where the corner sits relative to the circle, integrating `√(r²−u²)` over the part of the span
// where the circle, rather than the box, is the binding constraint.
//
// Exact, not sampled: these are closed forms, so the correction adds no error of its own.

/** Antiderivative of `√(r²−u²)`, i.e. `∫₀ᵘ √(r²−t²) dt`. Clamped because `u` may reach `r` and
 *  round a hair past it, where the naive form yields NaN from `√(negative)` / `asin(>1)`. */
function integralSqrt(r: number, u: number): number {
  const r2 = r * r;
  const s = Math.sqrt(Math.max(0, r2 - u * u));
  const t = Math.min(1, Math.max(-1, u / r));
  return 0.5 * (u * s + r2 * Math.asin(t));
}

/** Area of `disk(0, r) ∩ [0,x] × [0,y]` for `x, y ≥ 0`. */
export function cornerArea(r: number, x: number, y: number): number {
  if (x <= 0 || y <= 0 || r <= 0) return 0;
  const X = Math.min(x, r);
  const Y = Math.min(y, r);
  // Corner inside the disk: the box is wholly contained, so it *is* the intersection.
  if (X * X + Y * Y <= r * r) return X * Y;
  // Otherwise the circle crosses the top edge at u = xs. Left of xs the box's own top edge y binds;
  // right of it the circle does.
  const xs = Math.sqrt(Math.max(0, r * r - Y * Y));
  return xs * Y + (integralSqrt(r, X) - integralSqrt(r, xs));
}

/** Signed corner term for the inclusion–exclusion, valid in every quadrant. */
function signedCorner(r: number, x: number, y: number): number {
  return Math.sign(x) * Math.sign(y) * cornerArea(r, Math.abs(x), Math.abs(y));
}

/**
 * Area of the disk of radius `r` centred at `(cx, cy)` intersected with the axis-aligned rectangle
 * `[minX, minY] × [maxX, maxY]`. Exact.
 */
export function diskRectArea(cx: number, cy: number, r: number, minX: number, minY: number, maxX: number, maxY: number): number {
  if (r <= 0) return 0;
  const x1 = minX - cx;
  const x2 = maxX - cx;
  const y1 = minY - cy;
  const y2 = maxY - cy;
  return signedCorner(r, x2, y2) - signedCorner(r, x1, y2) - signedCorner(r, x2, y1) + signedCorner(r, x1, y1);
}

/**
 * Area of the annulus `[rInner, rOuter)` centred at `(cx, cy)`, clipped to the rectangle. The
 * difference of two clipped disks — exact for the same reason each of them is.
 */
export function annulusRectArea(
  cx: number,
  cy: number,
  rInner: number,
  rOuter: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): number {
  return diskRectArea(cx, cy, rOuter, minX, minY, maxX, maxY) - diskRectArea(cx, cy, rInner, minX, minY, maxX, maxY);
}

/**
 * Clipped areas of every annulus in a uniform binning, for one anchor, in one pass.
 *
 * `bins[k]` is the area of `[k·dr, (k+1)·dr)` clipped to the rectangle. Written into `out` (length
 * `nBins`) to keep this allocation-free in the per-cell loop that calls it once per anchor.
 */
export function annulusAreasInto(
  out: Float64Array,
  cx: number,
  cy: number,
  dr: number,
  nBins: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): void {
  let prev = 0;
  for (let k = 0; k < nBins; k++) {
    const a = diskRectArea(cx, cy, (k + 1) * dr, minX, minY, maxX, maxY);
    out[k] = a - prev;
    prev = a;
  }
}
