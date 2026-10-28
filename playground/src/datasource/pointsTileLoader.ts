// A SpatialData points element as a tiled `Loader<PointsTile>` (ADR-0008 amended; ADR-0010-loader).
//
// sd.js owns the I/O: each tile is one `loadPointsInBounds` over the tile's rectangle, which on a
// Morton-sorted element range-reads only the row groups the rectangle's Z-cover touches. The
// reader's bounds filter is closed, so `ownedTile` re-applies half-open ownership — without it a
// point on a shared tile edge is counted twice.
//
// A points-zarr element (sd.js 0.13) is tiled too, through `loadPointsZarrInBounds` instead: its row
// index plans each read, with no parquet involved.
//
// An element with no tiled layout has nothing to range-read, so it is read whole — if it fits a
// row budget — and bucketed into the same grid in memory (`bucketPoints`). Everything downstream
// sees the same tiles either way. What it cannot get is extra columns: sd.js's whole-element read
// returns coordinates and feature codes only.

import type { SpatialData } from "@spatialdata/core";
import {
  bucketPoints,
  expandRect,
  ownedTile,
  type PointsGrid,
  type PointsTile,
  pointsGrid,
  pointsGridAtLevel,
  type Rect,
  tileRect,
} from "../../../src/datasource/points";
import type { ChunkId, Loader } from "../../../src/datasource/types";
import { type ResolvedSpace, resolveNgffXY } from "../../../src/spatial/ngffTransform";
import { symbolAttrs } from "./cellTable";

export interface PointsFeature {
  readonly code: number;
  readonly name: string;
  readonly count?: number;
}

export interface PointsSource {
  readonly element: string;
  /** The grid tiles are cut from — for a multiscale element, its finest. */
  readonly grid: PointsGrid;
  /** For a multiscale element, the grid to cut `window ⊕ apron` with: bigger tiles for a bigger
   *  window, each read at the level its budget allows. Absent: always {@link grid}. */
  readonly gridFor?: (window: Rect, apron: number) => PointsGrid;
  readonly loader: Loader<PointsTile>;
  /** Feature catalog, in the code space the tiles carry. */
  readonly features: readonly PointsFeature[];
  /** Passthrough columns every tile carries. */
  readonly columns: readonly string[];
  /** Columns asked for that this element cannot supply (an untiled read has none to give). */
  readonly missingColumns: readonly string[];
  /** False when the element has no tiled layout and was read whole into memory. */
  readonly tiled: boolean;
  /** Per level, the share of the element's points it holds (`[1, ~0.25, …]`), for a multiscale
   *  element whose tiles may be served from a sample; absent when every tile holds every point. */
  readonly levelFractions?: readonly number[];
  readonly totalRows: number;
  /** Element space → its coordinate system, when the element declares one. */
  readonly space?: ResolvedSpace;
}

export interface OpenPointsOptions {
  /** Passthrough numeric columns to carry per point, e.g. `["qv"]`. */
  readonly columns?: readonly string[];
  /** Finest tile, in expected rows. Default: two row groups. */
  readonly minRowsPerTile?: number;
  /** Most rows an element with no tiled layout may have and still be read whole. */
  readonly untiledBudget?: number;
  /** Most points one tile of a multiscale (points-zarr) element should read. Past it, sd.js serves
   *  the tile from a coarser level — a uniform random sample, tagged with its `sampleFraction`.
   *  Default sd.js's own `POINTS_TILE_TARGET_ROWS`; `Infinity` reads every point at any zoom. */
  readonly tilePointBudget?: number;
}

/** Default `untiledBudget`: about 100 MB of coordinates and codes, a few seconds to read and bucket. */
export const DEFAULT_UNTILED_BUDGET = 5_000_000;

/** Names of the points elements in a store. */
export function listPointsElements(sdata: SpatialData): string[] {
  return Object.keys(sdata.points ?? {});
}

type PointsElement = NonNullable<SpatialData["points"]>[string];

/**
 * Open a points element as a tiled source: range-read tile by tile when it is Morton-tiled, else
 * read whole within `untiledBudget` rows. Throws on an untiled element over the budget.
 */
export async function openPointsSource(sdata: SpatialData, element: string, opts: OpenPointsOptions = {}): Promise<PointsSource> {
  const el = sdata.points?.[element];
  if (!el) throw new Error(`No points element "${element}" (have: ${listPointsElements(sdata).join(", ") || "none"})`);
  const meta = await el.getPointsTilingMetadata();
  if (meta?.kind === "points-zarr") return openPointsZarr(sdata, element, el, meta, opts);
  if (!meta?.bounds) return openUntiled(sdata, element, el, opts);
  const grid = pointsGrid(meta.bounds, meta.totalRows, { minRowsPerTile: opts.minRowsPerTile ?? 2 * meta.maxRowsPerGroup });
  const catalog = await el.listFeatures();
  const features = (catalog?.entries ?? []).map((e) => ({ code: e.code, name: e.name, count: e.count }));
  const columns = [...(opts.columns ?? [])];
  const empty = (id: ChunkId): PointsTile =>
    ownedTile(grid, id, { xs: [], ys: [], codes: [], columns: Object.fromEntries(columns.map((c) => [c, []])) });

  const loader: Loader<PointsTile> = {
    async getChunk(id) {
      const res = await el.loadPointsInBounds({ bounds: tileRect(grid, id.x, id.y), columns });
      // An empty tile may omit the requested columns (sd.js 0.11.1, when no row group is selected).
      if ((res.data[0]?.length ?? 0) === 0) return empty(id);
      const codes = res.featureCodes ?? res.featureIndices;
      if (!codes) throw new Error(`points/${element}: the tiled read returned no feature codes`);
      const got = res.columns ?? {};
      const missing = columns.filter((c) => !got[c]);
      if (missing.length) throw new Error(`points/${element}: sd.js did not return column(s) ${missing.join(", ")}`);
      const [xs, ys] = res.data;
      if (!xs || !ys) throw new Error(`points/${element}: the tiled read returned no coordinates`);
      return ownedTile(grid, id, { xs, ys, codes, columns: got });
    },
  };

  return {
    element,
    grid,
    loader,
    features,
    columns,
    missingColumns: [],
    tiled: true,
    totalRows: meta.totalRows,
    space: elementSpace(sdata, element),
  };
}

type PointsZarrMeta = Extract<NonNullable<Awaited<ReturnType<PointsElement["getPointsTilingMetadata"]>>>, { kind: "points-zarr" }>;

/** Tiles the window's long side spans, the way deck's 256–512 px tiles span a view. */
const TILES_ACROSS = 8;

/**
 * A points-zarr element (sd.js 0.13, experimental), or a standard one with a points-zarr copy. The
 * same scheme as sd.js's own tiled layer (`multiscaleTileGrid` + `createPointsZarrTiledPointsLoader`):
 * the window picks the tile size — a quadtree level over the extent, from one tile for the whole of
 * it down to the format's finest — and every tile reads within one point budget, so sd.js serves a
 * big tile from a coarser level. A coarser level is a uniform random sample, and the tile says what
 * share it holds. The format copies no passthrough columns, so none are offered.
 */
async function openPointsZarr(
  sdata: SpatialData,
  element: string,
  el: PointsElement,
  meta: PointsZarrMeta,
  opts: OpenPointsOptions,
): Promise<PointsSource> {
  const { bounds: extent, totalRows } = meta;
  const span = Math.max(extent.maxX - extent.minX, extent.maxY - extent.minY);
  // Finest: the first quadtree level at or under the format's own finest tile. Rounding the other way
  // can leave it up to twice the side — four times the points — and out of budget for level 0.
  const finest = Math.max(0, Math.ceil(Math.log2(span / meta.finestTileSize)));
  const grids = new Map<number, PointsGrid>();
  const gridAt = (level: number): PointsGrid => {
    let g = grids.get(level);
    if (!g) grids.set(level, (g = pointsGridAtLevel(extent, totalRows, level)));
    return g;
  };
  const gridFor = (window: Rect, apron: number): PointsGrid => {
    const need = expandRect(window, apron);
    const long = Math.max(need.maxX - need.minX, need.maxY - need.minY);
    const level = long > 0 ? Math.ceil(Math.log2((span * TILES_ACROSS) / long)) : finest;
    return gridAt(Math.min(finest, Math.max(0, level)));
  };
  const catalog = await el.listFeatures();
  const features = (catalog?.entries ?? []).map((e) => ({ code: e.code, name: e.name, count: e.count }));
  const levelFractions = pointsZarrLevelFractions(sdata, element);
  const pointBudget = opts.tilePointBudget ?? (await import("@spatialdata/core")).POINTS_TILE_TARGET_ROWS;
  const loader: Loader<PointsTile> = {
    async getChunk(id) {
      const grid = gridAt(id.level);
      // sd.js filters `[min, max)`, so the far edges are pushed out a hair: `ownedTile` keeps a point
      // sitting exactly on the element's max edge, and sd.js must not have dropped it first.
      const pad = grid.tileSize * 1e-6;
      const rect = tileRect(grid, id.x, id.y);
      const lastX = id.x + 1 >= grid.cols;
      const lastY = id.y + 1 >= grid.rows;
      const bounds = { ...rect, maxX: lastX ? rect.maxX + pad : rect.maxX, maxY: lastY ? rect.maxY + pad : rect.maxY };
      // The level is chosen for the UNCLIPPED tile, so an edge tile samples like its neighbours.
      const levelBounds = { ...rect, maxX: rect.minX + grid.tileSize, maxY: rect.minY + grid.tileSize };
      const res = await el.loadPointsZarrInBounds({ bounds, levelBounds, pointBudget });
      const [xs, ys] = res.data;
      if (!xs || !ys) throw new Error(`points/${element}: the points-zarr read returned no coordinates`);
      const tile = ownedTile(grid, id, { xs, ys, codes: res.featureCodes });
      // An all-features read is served from one level.
      const level = res.levels[0] ?? 0;
      if (level === 0) return tile;
      const sampleFraction = levelFractions?.[level];
      if (!sampleFraction) throw new Error(`points/${element}: tile served from level ${level}, whose share of the points is unknown`);
      return { ...tile, sampleFraction };
    },
  };
  return {
    element,
    grid: gridAt(finest),
    gridFor,
    loader,
    features,
    columns: [],
    missingColumns: [...(opts.columns ?? [])],
    tiled: true,
    ...(levelFractions ? { levelFractions } : {}),
    totalRows: meta.totalRows,
    space: elementSpace(sdata, element),
  };
}

/** Each level's `points:n_points` over level 0's, from the element's `multiscales` layout — on the
 *  element itself, or on its points-zarr copy (`POINTS_ZARR_SIDECAR` in sd.js). */
function pointsZarrLevelFractions(sdata: SpatialData, element: string): number[] | undefined {
  // UPSTREAM(sd.js): `PointsZarrSource.levels` carries these, but the element keeps its source private.
  const node = pointsNode(sdata, element);
  const sidecar = node && typeof node === "object" ? (node as Record<string, unknown>)["experimental.points-zarr"] : undefined;
  for (const n of [node, sidecar]) {
    const attrs = n ? symbolAttrs(n, (v) => "multiscales" in v) : undefined;
    const layout: unknown = (attrs?.multiscales as { layout?: unknown } | undefined)?.layout;
    if (!Array.isArray(layout)) continue;
    const counts = layout.map((l: unknown) => Number((l as Record<string, unknown> | null)?.["points:n_points"]));
    const base = counts[0];
    if (base && counts.every((c) => c > 0)) return counts.map((c) => c / base);
  }
  return undefined;
}

/** An element with no tiled layout: read whole, then served from memory through the same grid. */
async function openUntiled(sdata: SpatialData, element: string, el: PointsElement, opts: OpenPointsOptions): Promise<PointsSource> {
  const budget = opts.untiledBudget ?? DEFAULT_UNTILED_BUDGET;
  const total = await el.getParquetRowCount();
  if (total > budget) {
    throw new Error(
      `points/${element} has no tiled layout, and its ${total.toLocaleString()} rows are over the ${budget.toLocaleString()}-row budget ` +
        "for reading it whole. Raise the budget, or write a tiled copy with `spatialdata-js-util points index-permutations`.",
    );
  }
  const res = await el.loadPoints({ memoryCap: budget, includeFeatureCodes: true });
  if (res.preloadTruncated) throw new Error(`points/${element}: the whole read stopped short of ${total.toLocaleString()} rows`);
  const [xs, ys] = res.data;
  const codes = res.featureCodes;
  if (!xs || !ys) throw new Error(`points/${element}: the whole read returned no coordinates`);
  if (!codes) throw new Error(`points/${element}: the whole read returned no feature codes, so there is nothing to make channels from`);
  const bounds = extentOf(xs, ys);
  if (!bounds) throw new Error(`points/${element} has no points with finite coordinates`);
  const grid = pointsGrid(bounds, xs.length, { minRowsPerTile: opts.minRowsPerTile ?? 100_000 });
  const tiles = bucketPoints(grid, { xs, ys, codes });
  const empty = (id: ChunkId): PointsTile => ownedTile(grid, id, { xs: [], ys: [], codes: [] });
  const loader: Loader<PointsTile> = { getChunk: async (id) => tiles.get(`${id.x},${id.y}`) ?? empty(id) };
  const counts = res.featureCodeCounts;
  const features = (res.featureCatalog?.entries ?? []).map((e) => ({ code: e.code, name: e.name, count: counts?.get(e.code) ?? e.count }));
  return {
    element,
    grid,
    loader,
    features,
    columns: [],
    missingColumns: [...(opts.columns ?? [])],
    tiled: false,
    totalRows: xs.length,
    space: elementSpace(sdata, element),
  };
}

/** The bounding box of the points with finite coordinates, or undefined when there are none. */
function extentOf(xs: ArrayLike<number>, ys: ArrayLike<number>): Rect | undefined {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    const x = xs[i] ?? Number.NaN;
    const y = ys[i] ?? Number.NaN;
    if (!(Number.isFinite(x) && Number.isFinite(y))) continue;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return maxX > minX && maxY > minY ? { minX, minY, maxX, maxY } : undefined;
}

/** The element's own `coordinateTransformations`, resolved the way `cellTable` resolves a region. */
function elementSpace(sdata: SpatialData, element: string): ResolvedSpace | undefined {
  // UPSTREAM(sd.js): the typed element API does not expose resolved transforms; read the tree.
  const node = pointsNode(sdata, element);
  const attrs = node ? symbolAttrs(node, (v) => "coordinateTransformations" in v) : undefined;
  return resolveNgffXY(attrs?.coordinateTransformations);
}

/** The element's node in the store's zarr tree. */
function pointsNode(sdata: SpatialData, element: string): unknown {
  const tree: unknown = sdata.rootStore.tree;
  const points = tree && typeof tree === "object" ? (tree as Record<string, unknown>).points : undefined;
  return points && typeof points === "object" ? (points as Record<string, unknown>)[element] : undefined;
}
