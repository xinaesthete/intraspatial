// The mark grid behind a TCM view: where ρ̂_B must be rendered so that every A cell whose Gaussian
// reaches the Γ grid can read its mark, and how finely.
//
// Γ at a pixel sums the A cells within σ·radiusSigma of it, and each of those reads ρ̂_B at its own
// position. So the marks are needed over the Γ grid padded by that support, and only where A cells
// are: the result is clipped to A's bounds, then padded by one cell on every side. The padding is
// for the bilinear fetch: an A cell within half a cell of the grid's edge would otherwise fall
// outside the stencil between texel centres and read the clamped edge texel instead of
// interpolating. ρ̂_B itself needs no padding — every B cell's kernel quad is drawn and clipped by
// the rasterizer, so B cells just outside still count.
//
// The cell is the Γ grid's, refined to `radius / cellsPerRadius` when that is finer: the mark is
// sampled bilinearly from this grid, and its error falls with the cell relative to the radius
// (fastest for smooth kernels; the top-hat's step is what a raster cannot represent). It is snapped to
// multiples of the cell, as `fieldPlan` snaps the Γ grid, so a re-render at the same cell lands on
// the same lattice.

import type { Bounds, FieldPlan } from "./fieldPlan";

export interface MarkPlanInput {
  /** The Γ grid. */
  plan: FieldPlan;
  /** Bounds of the A cells (weight > 0). */
  boundsA: Bounds;
  /** Γ's support: σ · radiusSigma, in data units. */
  support: number;
  /** The mark neighbourhood radius r, in data units. */
  radius: number;
  /** Mark cells per radius, at least. */
  cellsPerRadius: number;
  /** Most cells the grid may have; the cell grows to fit. */
  maxCells: number;
  /** Most cells per side (the device's `maxTextureDimension2D`). */
  maxSide: number;
}

/** The mark grid, or null when no A cell reaches the Γ grid (Γ is 0 there). */
export function planMarks(input: MarkPlanInput): FieldPlan | null {
  const [gx0, gy0, gx1, gy1] = input.plan.bbox;
  const s = input.support;
  const a = input.boundsA;
  const region: Bounds = [Math.max(gx0 - s, a[0]), Math.max(gy0 - s, a[1]), Math.min(gx1 + s, a[2]), Math.min(gy1 + s, a[3])];
  if (region[0] > region[2] || region[1] > region[3]) return null;
  let cell = Math.min(input.plan.cell, input.radius / input.cellsPerRadius);
  for (;;) {
    // Snapped outwards, then a cell of padding each side: every A cell is at least a cell from the
    // edge, so its four bilinear texels all lie inside.
    const bbox: Bounds = [
      (Math.floor(region[0] / cell) - 1) * cell,
      (Math.floor(region[1] / cell) - 1) * cell,
      (Math.floor(region[2] / cell) + 2) * cell,
      (Math.floor(region[3] / cell) + 2) * cell,
    ];
    const width = Math.max(1, Math.round((bbox[2] - bbox[0]) / cell));
    const height = Math.max(1, Math.round((bbox[3] - bbox[1]) / cell));
    if (width * height <= input.maxCells && width <= input.maxSide && height <= input.maxSide) return { bbox, width, height, cell };
    const byCells = Math.sqrt((width * height) / input.maxCells);
    const bySide = Math.max(width, height) / input.maxSide;
    cell *= Math.max(byCells, bySide, 1) * 1.01;
  }
}

/** `plan` with its cell grown until neither side exceeds `maxSide` (snapped as `planField` snaps). */
export function fitSide(plan: FieldPlan, maxSide: number): FieldPlan {
  let p = plan;
  while (p.width > maxSide || p.height > maxSide) {
    const cell = p.cell * (Math.max(p.width, p.height) / maxSide) * 1.01;
    const [x0, y0, x1, y1] = p.bbox;
    const bbox: Bounds = [
      Math.floor(x0 / cell) * cell,
      Math.floor(y0 / cell) * cell,
      Math.ceil(x1 / cell) * cell,
      Math.ceil(y1 / cell) * cell,
    ];
    p = {
      bbox,
      width: Math.max(1, Math.round((bbox[2] - bbox[0]) / cell)),
      height: Math.max(1, Math.round((bbox[3] - bbox[1]) / cell)),
      cell,
    };
  }
  return p;
}
