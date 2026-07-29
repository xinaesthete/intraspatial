// The SpatialData store, opened once at the high level and threaded by reference.
//
// One home for `readZarr`, so the whole datasource layer — cell tables, expression columns, and the
// image context / scene loaders — shares a single parse per store instead of each re-opening it.
// This module depends on NOTHING but `@spatialdata/core`, deliberately: importing it (e.g. from the
// image loader) must not drag cell-table ingestion into an image-only bundle.

import type { SpatialData } from "@spatialdata/core";

const storeCache = new Map<string, Promise<SpatialData>>();

/**
 * Open a store ONCE, at the high level, and hand back the `SpatialData` to pass around.
 *
 * `@spatialdata/core`'s `readZarr`, not `zarrextra`'s `openExtraConsolidated` scattered across every
 * function. Cached per URL, so every reader — `listCellTables`, `readCellTable`, `listVars`,
 * `readVarColumns`, and the image loader — shares one parse and one consolidated store.
 */
export function openSpatialData(url: string): Promise<SpatialData> {
  let cached = storeCache.get(url);
  if (!cached) {
    cached = (async () => {
      const sd = await import("@spatialdata/core");
      return sd.readZarr(url);
    })();
    storeCache.set(url, cached);
  }
  return cached;
}

/**
 * The consolidated zarr tree behind a `SpatialData`.
 *
 * UPSTREAM(sd.js): we reach into `rootStore.tree` because the typed element API does not yet cover
 * the reads the datasource layer needs — decoded var-name strings, obs-column dtypes/cardinality,
 * layer enumeration, and arbitrary sparse-column selection (each marked at its call site). `readZarr`
 * is the high-level entry and `SpatialData` is what we thread; this is the *one* documented drop to
 * the raw tree. As those accessors land upstream, each marked block swaps to the accessor and the
 * surface area of this bridge shrinks. `ConsolidatedStore.tree` is the same `ZarrTree` the low-level
 * `cellTable`/`varMatrix` helpers already consume, so nothing downstream changes.
 */
export function storeTree(sdata: SpatialData): { tables?: Record<string, unknown> } {
  return sdata.rootStore.tree as unknown as { tables?: Record<string, unknown> };
}
