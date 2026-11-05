// Stores opened from the user's disk — a picked `.zarr` folder or `.zip` — under a string key.
//
// Every page holds its store as a string (a text input, a `useState`, the per-URL cache in
// `spatialDataStore`). A picked store is an object (zarrextra's `FolderStore` / `ZipStore`), so
// rather than widen every one of those to `StoreReference`, a pick is registered here and handed
// back as a `local:` key. The readers resolve the key with `storeReference` at the one place a
// store is opened; everything above it stays a string.

import type { StoreReference } from "@spatialdata/core";

const LOCAL_SCHEME = "local:";
const stores = new Map<string, Exclude<StoreReference, string>>();
let picks = 0;
// Differs per page load, so a key left over from before a reload (a restored form input, say) never
// matches a new pick that happens to get the same number.
const session = Math.random().toString(36).slice(2, 7);

/** Whether `location` names a store picked from disk rather than a URL. */
function isLocalStore(location: string): boolean {
  return location.startsWith(LOCAL_SCHEME);
}

/**
 * What to hand `readZarr` / `openExtraConsolidated` for a page's store string: the picked store for
 * a `local:` key, the URL otherwise. A `local:` key outlives its store only across a reload (the
 * files are not ours to reopen), so that case says to pick again rather than fetching `local:…`.
 */
export function storeReference(location: string): StoreReference {
  if (!isLocalStore(location)) return location;
  const store = stores.get(location);
  if (!store) throw new Error(`${location} is no longer open (picked files do not survive a reload) — pick it again`);
  return store;
}

function register(store: Exclude<StoreReference, string>, label: string): string {
  // Numbered, so picking the same folder again (after it changed on disk, say) opens it afresh
  // instead of hitting the per-location cache.
  const key = `${LOCAL_SCHEME}${session}.${++picks}/${label}`;
  stores.set(key, store);
  return key;
}

/**
 * The files of a picked folder (`<input type="file" webkitdirectory>`) as a store key. Pick the
 * `.zarr` folder itself, or a folder whose only child it is. Nothing is read until a key is asked
 * for — each read is a `Blob.slice`.
 */
export async function openLocalFolder(files: readonly File[]): Promise<string> {
  const root = files[0]?.webkitRelativePath.split("/")[0];
  if (!root) throw new Error("no files in the picked folder");
  const { FolderStore } = await import("zarrextra");
  return register(FolderStore.fromFiles(files), `${root}/`);
}

/** A zipped store (`.zarr.zip`, written STORED — DEFLATE entries are inflated whole on every read). */
export async function openLocalZip(file: File): Promise<string> {
  const { ZipStore } = await import("zarrextra");
  return register(await ZipStore.open(file), file.name);
}
