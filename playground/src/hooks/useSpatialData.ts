// Store-level hooks over `@spatialdata/core`: the store, a tiled points element, and the tiles a
// window needs. Each is keyed so a change upstream re-derives everything below it.

import type { SpatialData } from "@spatialdata/core";
import { useMemo, useState } from "react";
import { type PointsTile, pointsTileBytes, type Rect, selectPointsTiles } from "../../../src/datasource/points";
import { chunkKey, resolveWith, TileCache } from "../../../src/datasource/tileCache";
import type { ChunkId } from "../../../src/datasource/types";
import { type OpenPointsOptions, openPointsSource, type PointsSource } from "../datasource/pointsTileLoader";
import { openSpatialData } from "../datasource/spatialDataStore";
import { type AsyncState, useAsync } from "./useAsync";

/** The store at `url`. Scoped to the URL: while another store opens, or if it fails, there is none —
 *  never the previous one standing in for it. */
export function useSpatialData(url: string): AsyncState<SpatialData> {
  return useAsync(() => (url ? openSpatialData(url) : undefined), [url], url);
}

export function usePointsSource(
  sdata: SpatialData | undefined,
  element: string | undefined,
  opts: OpenPointsOptions = {},
): AsyncState<PointsSource> {
  const columns = (opts.columns ?? []).join(",");
  // Scoped to the store and element: another element's source is no stand-in for this one's.
  const scope = useMemo(() => ({ sdata, element }), [sdata, element]);
  // `columns` stands for opts.columns, so a fresh options object does not reopen the element.
  return useAsync(
    () => (sdata && element ? openPointsSource(sdata, element, opts) : undefined),
    [sdata, element, columns, opts.minRowsPerTile],
    scope,
  );
}

export interface WindowTiles {
  readonly tiles: readonly PointsTile[];
  readonly window: Rect;
  /** Tiles fetched for this window, as opposed to served from the cache. */
  readonly fetched: number;
}

export interface WindowTilesState extends AsyncState<WindowTiles> {
  readonly progress?: { readonly done: number; readonly total: number };
}

/**
 * The tiles covering `window ⊕ apron`, through a per-source `TileCache` — so a panned window only
 * fetches the tiles that newly came into view.
 */
export function usePointsTiles(
  source: PointsSource | undefined,
  window: Rect | undefined,
  apron: number,
  maxBytes = 512e6,
): WindowTilesState {
  const cache = useMemo(() => (source ? new TileCache<PointsTile>({ maxBytes }) : undefined), [source, maxBytes]);
  const [progress, setProgress] = useState<{ done: number; total: number }>();
  const state = useAsync(
    (signal) => {
      if (!source || !window || !cache) return undefined;
      const selection = selectPointsTiles(source.grid, window, apron);
      // Progress counts the tiles that must be FETCHED; cached ones cost nothing.
      const total = selection.chunks.filter((c) => !cache.has(chunkKey(c.id))).length;
      let done = 0;
      let fetched = 0;
      setProgress({ done, total });
      const loader = {
        getChunk: async (id: ChunkId) => {
          fetched++;
          const t = await source.loader.getChunk(id);
          if (!signal.aborted) setProgress({ done: ++done, total });
          return t;
        },
      };
      return resolveWith(selection, loader, pointsTileBytes, cache).then((set) => ({ tiles: [...set.values()], window, fetched }));
    },
    [source, cache, window, apron],
    // Tiles stand in for the next window's only while they come from the same source.
    source,
  );
  // Derived, not cleared by an effect: an all-cached load can finish before `loading` ever renders true.
  return { ...state, progress: state.loading ? progress : undefined };
}
