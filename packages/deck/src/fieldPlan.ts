// Which grid to splat a density field into, for what is on screen — and when the one already made
// will still do.
//
// A KDE with bandwidth σ is band-limited: detail finer than a fraction of σ isn't there to resolve.
// So the grid's cell never needs to be finer than `σ / cellsPerSigma`, however far the view zooms
// in, and never finer than about a screen pixel when zoomed out. The larger of the two is the cell,
// which bounds the work at both ends: zoomed in, each point covers at most `(2·radius·cellsPerSigma)²`
// cells; zoomed out, the grid is no bigger than the screen.
//
// The grid covers the visible region plus a margin, clipped to where the field can be non-zero (the
// data's bounds padded by the kernel support), and is SNAPPED to multiples of the cell: panning
// within the margin reuses it outright, and a re-splat at the same cell lands on the same lattice, so
// contours don't shimmer as the view moves.
//
// Everything here is in the layer's own data units — the units of the positions and of σ.

export type Bounds = [minX: number, minY: number, maxX: number, maxY: number];

export interface FieldPlan {
  /** The grid's extent, in data units; cells tile it exactly. */
  bbox: Bounds;
  width: number;
  height: number;
  /** Cell size, in data units (square). */
  cell: number;
}

export interface PlanInput {
  /** What is on screen, in data units. */
  visible: Bounds;
  /** The viewport's size in screen pixels. */
  pixels: [width: number, height: number];
  /** The points' bounds. */
  data: Bounds;
  /** Kernel σ, in data units. */
  sigma: number;
  /** Kernel support, in σ. */
  radiusSigma: number;
  /** The finest grid needed: cells per σ. */
  cellsPerSigma: number;
  /** The finest grid wanted when zoomed out: screen pixels per cell. */
  pixelsPerCell: number;
  /** Extra coverage beyond the visible region on each side, as a fraction of its span. */
  margin: number;
  /** Most cells the grid may have; the cell grows to fit. */
  maxCells: number;
}

/** Where the field can be non-zero: the data's bounds padded by the kernel support. */
export function support(data: Bounds, sigma: number, radiusSigma: number): Bounds {
  const r = sigma * radiusSigma;
  return [data[0] - r, data[1] - r, data[2] + r, data[3] + r];
}

function intersect(a: Bounds, b: Bounds): Bounds | null {
  const out: Bounds = [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
  return out[0] < out[2] && out[1] < out[3] ? out : null;
}

/** The cell this view wants, before any cap on the grid's size. */
export function idealCell(input: PlanInput): number {
  const [minX, minY, maxX, maxY] = input.visible;
  const perPixel = Math.max((maxX - minX) / Math.max(1, input.pixels[0]), (maxY - minY) / Math.max(1, input.pixels[1]));
  return Math.max(perPixel * input.pixelsPerCell, input.sigma / input.cellsPerSigma);
}

/** The grid to splat for this view, or null when nothing of the field is on screen. */
export function planField(input: PlanInput): FieldPlan | null {
  const sup = support(input.data, input.sigma, input.radiusSigma);
  const [vx0, vy0, vx1, vy1] = input.visible;
  const mx = (vx1 - vx0) * input.margin;
  const my = (vy1 - vy0) * input.margin;
  const region = intersect([vx0 - mx, vy0 - my, vx1 + mx, vy1 + my], sup);
  if (!region) return null;

  let cell = idealCell(input);
  for (;;) {
    const bbox: Bounds = [
      Math.floor(region[0] / cell) * cell,
      Math.floor(region[1] / cell) * cell,
      Math.ceil(region[2] / cell) * cell,
      Math.ceil(region[3] / cell) * cell,
    ];
    const width = Math.max(1, Math.round((bbox[2] - bbox[0]) / cell));
    const height = Math.max(1, Math.round((bbox[3] - bbox[1]) / cell));
    if (width * height <= input.maxCells) return { bbox, width, height, cell };
    // Too many cells: grow the cell by the shortfall (a little extra, since snapping can add a row).
    cell *= Math.sqrt((width * height) / input.maxCells) * 1.01;
  }
}

/** Whether `plan` still serves this view: it covers the visible part of the field, and its cell is
 *  within a factor of √2 of the cell a fresh plan would choose — so panning inside the margin reuses
 *  it, and zooming re-splats about every half-octave. */
export function planServes(plan: FieldPlan, input: PlanInput): boolean {
  const needed = intersect(input.visible, support(input.data, input.sigma, input.radiusSigma));
  if (!needed) return true; // nothing of the field on screen, so nothing to redo
  const [bx0, by0, bx1, by1] = plan.bbox;
  if (needed[0] < bx0 || needed[1] < by0 || needed[2] > bx1 || needed[3] > by1) return false;
  const fresh = planField(input);
  if (!fresh) return true;
  const ratio = plan.cell / fresh.cell;
  return ratio > Math.SQRT1_2 && ratio < Math.SQRT2;
}
