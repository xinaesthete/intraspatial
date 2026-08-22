// Tileset, chunk-identity cache, and Resolve (ADR-0008 §6–7). Framework-free: the
// TileCache is generic over its payload so the render backend can cache GPU textures
// while a cluster/analysis path caches decoded Tiles — same LRU, keyed by chunk
// identity (not the graph memo), bounded by the Resource ceiling.

import type { MemoryReporting } from "../gpu/graph/memory";
import type { ChunkId, Loader, Selection, Tile } from "./types";
import { tileBytes } from "./types";

/** Stable identity of a chunk — the TileCache key. */
export const chunkKey = (id: ChunkId): string => `${id.level}:${id.x}:${id.y}:${id.z}`;

/** The keyed collection Resolve produces — the currency between Resolve and consumers. */
export type Tileset = ReadonlyMap<string, Tile>;

interface Entry<V> {
  v: V;
  bytes: number;
}

/**
 * LRU cache keyed by chunk identity, bounded by a byte ceiling. Insertion order in the
 * backing Map is the recency order; `get`/`set` move the entry to most-recent, and
 * `set` evicts the least-recent while over the ceiling. `dispose` frees an evicted
 * payload (e.g. a GPU texture). Deterministic — pure memoisation, no effect on values.
 */
export class TileCache<V> implements MemoryReporting {
  private map = new Map<string, Entry<V>>();
  private total = 0;
  private readonly maxBytes: number;
  private readonly onEvict?: (v: V) => void;

  constructor(opts: { maxBytes: number; dispose?: (v: V) => void }) {
    this.maxBytes = opts.maxBytes;
    this.onEvict = opts.dispose;
  }

  /** Resident bytes held (MemoryReporting) — the actual working set, since callers `set` each
   *  entry's real payload size (the render backend's fp16 texture bytes, a decode's Tile bytes). */
  get byteLength(): number {
    return this.total;
  }
  get bytes(): number {
    return this.total;
  }
  get size(): number {
    return this.map.size;
  }
  has(key: string): boolean {
    return this.map.has(key);
  }

  /** Fetch, marking the entry most-recently-used. */
  get(key: string): V | undefined {
    const e = this.map.get(key);
    if (!e) return undefined;
    this.map.delete(key);
    this.map.set(key, e);
    return e.v;
  }

  /** Insert/replace, then evict least-recent entries until within the ceiling. */
  set(key: string, v: V, bytes: number): void {
    const prev = this.map.get(key);
    if (prev) {
      this.total -= prev.bytes;
      this.map.delete(key);
    }
    this.map.set(key, { v, bytes });
    this.total += bytes;
    while (this.total > this.maxBytes && this.map.size > 1) {
      const oldest = this.map.keys().next().value;
      if (oldest === undefined) break;
      const e = this.map.get(oldest);
      this.map.delete(oldest);
      if (e) {
        this.total -= e.bytes;
        this.onEvict?.(e.v);
      }
    }
  }

  /** Evict everything not in `keys` (e.g. chunks no longer in view). */
  keep(keys: ReadonlySet<string>): void {
    for (const k of [...this.map.keys()]) {
      if (keys.has(k)) continue;
      const e = this.map.get(k);
      this.map.delete(k);
      if (e) {
        this.total -= e.bytes;
        this.onEvict?.(e.v);
      }
    }
  }

  clear(): void {
    for (const e of this.map.values()) this.onEvict?.(e.v);
    this.map.clear();
    this.total = 0;
  }
}

/**
 * Resolve a Selection to a Tileset by pulling the Loader, memoised through an optional
 * decoded-Tile cache. The one effectful step (ADR-0008 §2): its only effect is
 * `loader.getChunk`. Cache hits never re-fetch.
 */
export async function resolve(selection: Selection, loader: Loader, cache?: TileCache<Tile>): Promise<Tileset> {
  const out = new Map<string, Tile>();
  const pending = new Set<string>();
  await Promise.all(
    selection.chunks.map(async (sc) => {
      const key = chunkKey(sc.id);
      if (out.has(key) || pending.has(key)) return; // dedupe within one selection
      const cached = cache?.get(key);
      if (cached) {
        out.set(key, cached);
        return;
      }
      pending.add(key);
      const tile = await loader.getChunk(sc.id);
      cache?.set(key, tile, tileBytes(tile));
      out.set(key, tile);
    }),
  );
  return out;
}
