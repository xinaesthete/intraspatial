// Points as a tiled datasource (ADR-0008, amended for points).
//
// A points element is cut into a uniform grid of square tiles at one level. A tile OWNS the points
// in its half-open rectangle `[minX, maxX) × [minY, maxY)` — closed on the grid's far edges — so a
// union of tiles never double-counts a point on a shared edge. That matters because readers filter
// to CLOSED bounds (sd.js `filterPointsToBounds`), so a raw per-tile read returns edge points twice.
//
// Selection is by region, not camera: the analysis window plus an apron (the splat radius), since
// a point outside the window still deposits kernel mass inside it. Tiles are keyed by `ChunkId`, so
// a moving window re-selects and the `TileCache` serves every tile it already holds.

import type { ChunkId, SelectedChunk, Selection } from "./types";

/** Axis-aligned rectangle in the element's own coordinates — sd.js's `SpatialBounds` shape. */
export interface Rect {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

/** A uniform grid of square tiles over a points element's extent. */
export interface PointsGrid {
  readonly bounds: Rect;
  readonly level: number;
  /** Tile side, in element units: the longer span of `bounds` divided by `2^level`. */
  readonly tileSize: number;
  readonly cols: number;
  readonly rows: number;
  /** Mean rows per tile at the element's average density — a budgeting estimate, not a bound. */
  readonly rowsPerTile: number;
}

/** One tile's owned points. `codes` are feature codes; `columns` are passthrough per-point values. */
export interface PointsTile {
  readonly id: ChunkId;
  readonly rect: Rect;
  readonly count: number;
  readonly xs: Float32Array;
  readonly ys: Float32Array;
  readonly codes: Int32Array;
  readonly columns: Readonly<Record<string, Float32Array>>;
}

/** Points as a reader returns them for a tile's (closed) bounds, before ownership is applied. */
export interface RawPoints {
  readonly xs: ArrayLike<number>;
  readonly ys: ArrayLike<number>;
  readonly codes: ArrayLike<number>;
  readonly columns?: Readonly<Record<string, ArrayLike<number>>>;
}

export interface PointsGridOptions {
  /** Finest tile allowed, in expected rows. Reads round up to whole row groups, so a tile smaller
   *  than a few groups refetches the same bytes in more requests. Default 100 000. */
  readonly minRowsPerTile?: number;
}

/** Choose the finest grid whose tiles still hold `minRowsPerTile` at the element's mean density. */
export function pointsGrid(bounds: Rect, totalRows: number, opts: PointsGridOptions = {}): PointsGrid {
  const spanX = bounds.maxX - bounds.minX;
  const spanY = bounds.maxY - bounds.minY;
  if (!(spanX > 0 && spanY > 0)) throw new Error(`pointsGrid: degenerate bounds ${JSON.stringify(bounds)}`);
  const minRows = opts.minRowsPerTile ?? 100_000;
  const density = totalRows / (spanX * spanY);
  const maxSpan = Math.max(spanX, spanY);
  const minSide = Math.sqrt(minRows / Math.max(density, Number.MIN_VALUE));
  const level = Math.max(0, Math.floor(Math.log2(maxSpan / minSide)));
  const tileSize = maxSpan / 2 ** level;
  return {
    bounds,
    level,
    tileSize,
    cols: Math.max(1, Math.ceil(spanX / tileSize)),
    rows: Math.max(1, Math.ceil(spanY / tileSize)),
    rowsPerTile: density * tileSize * tileSize,
  };
}

// Edges are computed by ONE function so neighbouring tiles agree on a shared edge bit-for-bit; the
// last edge is the bounds itself, so a point on the far boundary is still owned.
const edgeX = (g: PointsGrid, i: number): number => (i >= g.cols ? g.bounds.maxX : g.bounds.minX + i * g.tileSize);
const edgeY = (g: PointsGrid, j: number): number => (j >= g.rows ? g.bounds.maxY : g.bounds.minY + j * g.tileSize);

/** The rectangle tile `(x, y)` owns. */
export function tileRect(grid: PointsGrid, x: number, y: number): Rect {
  return { minX: edgeX(grid, x), minY: edgeY(grid, y), maxX: edgeX(grid, x + 1), maxY: edgeY(grid, y + 1) };
}

/** Whether tile `id` owns the point `(px, py)`: half-open, closed on the grid's far edges. */
export function ownsPoint(grid: PointsGrid, id: ChunkId, px: number, py: number): boolean {
  const r = tileRect(grid, id.x, id.y);
  const inX = px >= r.minX && (px < r.maxX || (id.x === grid.cols - 1 && px <= r.maxX));
  const inY = py >= r.minY && (py < r.maxY || (id.y === grid.rows - 1 && py <= r.maxY));
  return inX && inY;
}

/** Keep only the points tile `id` owns, as a compact `PointsTile`. */
export function ownedTile(grid: PointsGrid, id: ChunkId, raw: RawPoints): PointsTile {
  const n = Math.min(raw.xs.length, raw.ys.length, raw.codes.length);
  const keep: number[] = [];
  for (let i = 0; i < n; i++) if (ownsPoint(grid, id, raw.xs[i] ?? Number.NaN, raw.ys[i] ?? Number.NaN)) keep.push(i);
  const m = keep.length;
  const xs = new Float32Array(m);
  const ys = new Float32Array(m);
  const codes = new Int32Array(m);
  for (let k = 0; k < m; k++) {
    const i = keep[k] ?? 0;
    xs[k] = raw.xs[i] ?? 0;
    ys[k] = raw.ys[i] ?? 0;
    codes[k] = raw.codes[i] ?? 0;
  }
  const columns: Record<string, Float32Array> = {};
  for (const [name, col] of Object.entries(raw.columns ?? {})) {
    if (col.length < n) throw new Error(`ownedTile: column "${name}" has ${col.length} values for ${n} points`);
    const out = new Float32Array(m);
    for (let k = 0; k < m; k++) out[k] = col[keep[k] ?? 0] ?? 0;
    columns[name] = out;
  }
  return { id, rect: tileRect(grid, id.x, id.y), count: m, xs, ys, codes, columns };
}

/** Resident bytes of a tile — the `TileCache` ceiling's unit. */
export function pointsTileBytes(tile: PointsTile): number {
  let bytes = tile.xs.byteLength + tile.ys.byteLength + tile.codes.byteLength;
  for (const c of Object.values(tile.columns)) bytes += c.byteLength;
  return bytes;
}

/** `window` grown by `margin` on every side. */
export function expandRect(r: Rect, margin: number): Rect {
  return { minX: r.minX - margin, minY: r.minY - margin, maxX: r.maxX + margin, maxY: r.maxY + margin };
}

/**
 * `want` made into a window the analysis can afford: cut to `extent`, then, if its area is over
 * `maxArea`, shrunk about its own centre to that area at the same aspect ratio. Pure.
 *
 * What a viewport-driven window needs: zoomed out, the view covers more tissue than one Gram
 * should — the points loaded and the raster both grow with area — so the window becomes the
 * middle of the view rather than all of it. `undefined` when the view misses the extent.
 */
export function clampWindow(want: Rect, extent: Rect, maxArea: number): Rect | undefined {
  const minX = Math.max(want.minX, extent.minX);
  const minY = Math.max(want.minY, extent.minY);
  const maxX = Math.min(want.maxX, extent.maxX);
  const maxY = Math.min(want.maxY, extent.maxY);
  if (!(maxX > minX && maxY > minY)) return undefined;
  const area = (maxX - minX) * (maxY - minY);
  if (area <= maxArea) return { minX, minY, maxX, maxY };
  const k = Math.sqrt(maxArea / area);
  const hw = ((maxX - minX) * k) / 2;
  const hh = ((maxY - minY) * k) / 2;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return { minX: cx - hw, minY: cy - hh, maxX: cx + hw, maxY: cy + hh };
}

export interface SelectPointsOptions {
  /** Bytes per resident point, for `approxBytes`. Default 16: x, y, code and one passthrough column. */
  readonly bytesPerPoint?: number;
}

/**
 * The tiles covering `window ⊕ apron`, nearest to the window's centre first. Pure.
 *
 * `nearestDepth` carries the distance from the window centre to the tile (0 when the tile contains
 * it), so `resolve` callers that load in order fill the middle of the view first.
 */
export function selectPointsTiles(grid: PointsGrid, window: Rect, apron: number, opts: SelectPointsOptions = {}): Selection {
  const need = expandRect(window, apron);
  const { bounds, tileSize } = grid;
  const clampI = (v: number, hi: number): number => Math.min(hi, Math.max(0, v));
  const x0 = clampI(Math.floor((need.minX - bounds.minX) / tileSize), grid.cols - 1);
  const x1 = clampI(Math.floor((need.maxX - bounds.minX) / tileSize), grid.cols - 1);
  const y0 = clampI(Math.floor((need.minY - bounds.minY) / tileSize), grid.rows - 1);
  const y1 = clampI(Math.floor((need.maxY - bounds.minY) / tileSize), grid.rows - 1);
  const disjoint = need.maxX < bounds.minX || need.minX > bounds.maxX || need.maxY < bounds.minY || need.minY > bounds.maxY;
  const cx = (window.minX + window.maxX) / 2;
  const cy = (window.minY + window.maxY) / 2;
  const approxBytes = Math.round(grid.rowsPerTile * (opts.bytesPerPoint ?? 16));
  const chunks: SelectedChunk[] = [];
  if (!disjoint) {
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const r = tileRect(grid, x, y);
        const dx = Math.max(r.minX - cx, 0, cx - r.maxX);
        const dy = Math.max(r.minY - cy, 0, cy - r.maxY);
        chunks.push({ id: { level: grid.level, x, y, z: 0 }, nearestDepth: Math.hypot(dx, dy), approxBytes });
      }
    }
  }
  chunks.sort((a, b) => a.nearestDepth - b.nearestDepth);
  const countByLevel = new Array<number>(grid.level + 1).fill(0);
  countByLevel[grid.level] = chunks.length;
  return { chunks, totalApproxBytes: approxBytes * chunks.length, countByLevel };
}
