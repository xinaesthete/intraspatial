// A SpatialData points element as a tiled `Loader<PointsTile>` (ADR-0008 amended; ADR-0010-loader).
//
// sd.js owns the I/O: each tile is one `loadPointsInBounds` over the tile's rectangle, which on a
// Morton-sorted element range-reads only the row groups the rectangle's Z-cover touches. The
// reader's bounds filter is closed, so `ownedTile` re-applies half-open ownership — without it a
// point on a shared tile edge is counted twice.
//
// An element with no tiled layout has nothing to range-read, so it is read whole — if it fits a
// row budget — and bucketed into the same grid in memory (`bucketPoints`). Everything downstream
// sees the same tiles either way. What it cannot get is extra columns: sd.js's whole-element read
// returns coordinates and feature codes only.

import type { SpatialData } from "@spatialdata/core";
import { bucketPoints, ownedTile, type PointsGrid, type PointsTile, pointsGrid, type Rect, tileRect } from "../../../src/datasource/points";
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
  readonly grid: PointsGrid;
  readonly loader: Loader<PointsTile>;
  /** Feature catalog, in the code space the tiles carry. */
  readonly features: readonly PointsFeature[];
  /** Passthrough columns every tile carries. */
  readonly columns: readonly string[];
  /** Columns asked for that this element cannot supply (an untiled read has none to give). */
  readonly missingColumns: readonly string[];
  /** False when the element has no tiled layout and was read whole into memory. */
  readonly tiled: boolean;
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
  const tree: unknown = sdata.rootStore.tree;
  const points = tree && typeof tree === "object" ? (tree as Record<string, unknown>).points : undefined;
  const node = points && typeof points === "object" ? (points as Record<string, unknown>)[element] : undefined;
  const attrs = node ? symbolAttrs(node, (v) => "coordinateTransformations" in v) : undefined;
  return resolveNgffXY(attrs?.coordinateTransformations);
}
