// A SpatialData points element as a tiled `Loader<PointsTile>` (ADR-0008 amended; ADR-0010-loader).
//
// sd.js owns the I/O: each tile is one `loadPointsInBounds` over the tile's rectangle, which on a
// Morton-sorted element range-reads only the row groups the rectangle's Z-cover touches. The
// reader's bounds filter is closed, so `ownedTile` re-applies half-open ownership — without it a
// point on a shared tile edge is counted twice.

import type { SpatialData } from "@spatialdata/core";
import { ownedTile, type PointsGrid, type PointsTile, pointsGrid, tileRect } from "../../../src/datasource/points";
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
  readonly totalRows: number;
  /** Element space → its coordinate system, when the element declares one. */
  readonly space?: ResolvedSpace;
}

export interface OpenPointsOptions {
  /** Passthrough numeric columns to carry per point, e.g. `["qv"]`. */
  readonly columns?: readonly string[];
  /** Finest tile, in expected rows. Default: two row groups. */
  readonly minRowsPerTile?: number;
}

/** Names of the points elements in a store. */
export function listPointsElements(sdata: SpatialData): string[] {
  return Object.keys(sdata.points ?? {});
}

/** Open a Morton-tiled points element as a tiled source. Throws on an element that is not tiled. */
export async function openPointsSource(sdata: SpatialData, element: string, opts: OpenPointsOptions = {}): Promise<PointsSource> {
  const el = sdata.points?.[element];
  if (!el) throw new Error(`No points element "${element}" (have: ${listPointsElements(sdata).join(", ") || "none"})`);
  const meta = await el.getPointsTilingMetadata();
  if (!meta?.bounds) {
    throw new Error(`points/${element} is not Morton-tiled; write a tiled copy with \`spatialdata-js-util points index-permutations\``);
  }
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

  return { element, grid, loader, features, columns, totalRows: meta.totalRows, space: elementSpace(sdata, element) };
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
