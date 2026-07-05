// TileCache LRU behaviour and Resolve memoisation (ADR-0008 §6–7).
import { describe, expect, it } from "vitest";
import { chunkKey, resolve, TileCache } from "./tileCache";
import type { ChunkId, Loader, Selection, Tile } from "./types";

const sel = (ids: ChunkId[]): Selection => ({
  chunks: ids.map((id) => ({ id, nearestDepth: 1, approxBytes: 0 })),
  totalApproxBytes: 0,
  countByLevel: [],
});

const fakeTile = (id: ChunkId): Tile => ({ id, dims: [1, 1, 1], element: { kind: "scalar" }, dtype: "f32", data: new Float32Array([1]) });

function countingLoader(): { loader: Loader; calls: () => number } {
  let n = 0;
  return {
    loader: {
      getChunk: async (id) => {
        n++;
        return fakeTile(id);
      },
    },
    calls: () => n,
  };
}

describe("TileCache — LRU under a byte ceiling", () => {
  it("evicts the least-recently-used when over the ceiling", () => {
    const evicted: string[] = [];
    const c = new TileCache<string>({ maxBytes: 25, dispose: (v) => evicted.push(v) });
    c.set("A", "a", 10);
    c.set("B", "b", 10);
    c.set("C", "c", 10); // 30 > 25 → evict oldest (A)
    expect(c.has("A")).toBe(false);
    expect(c.has("B")).toBe(true);
    expect(c.has("C")).toBe(true);
    expect(evicted).toEqual(["a"]);
    expect(c.bytes).toBe(20);
  });

  it("get() marks recency so the untouched entry is evicted", () => {
    const c = new TileCache<string>({ maxBytes: 25 });
    c.set("A", "a", 10);
    c.set("B", "b", 10);
    expect(c.get("A")).toBe("a"); // A now most-recent
    c.set("C", "c", 10); // evict oldest = B
    expect(c.has("A")).toBe(true);
    expect(c.has("B")).toBe(false);
  });

  it("keep() drops everything not in the working set", () => {
    const c = new TileCache<string>({ maxBytes: 1000 });
    c.set("A", "a", 10);
    c.set("B", "b", 10);
    c.set("C", "c", 10);
    c.keep(new Set(["B"]));
    expect(c.size).toBe(1);
    expect(c.has("B")).toBe(true);
  });
});

describe("resolve — pulls the loader, memoised through the cache", () => {
  const ids: ChunkId[] = [
    { level: 0, x: 0, y: 0, z: 0 },
    { level: 0, x: 1, y: 0, z: 0 },
    { level: 1, x: 0, y: 0, z: 0 },
  ];

  it("fetches every chunk without a cache", async () => {
    const { loader, calls } = countingLoader();
    const ts = await resolve(sel(ids), loader);
    expect(ts.size).toBe(3);
    expect(calls()).toBe(3);
    expect(ts.has(chunkKey(ids[0] as ChunkId))).toBe(true);
  });

  it("a second resolve over the same chunks hits the cache (no new fetches)", async () => {
    const { loader, calls } = countingLoader();
    const cache = new TileCache<Tile>({ maxBytes: 1_000_000 });
    await resolve(sel(ids), loader, cache);
    expect(calls()).toBe(3);
    await resolve(sel(ids), loader, cache);
    expect(calls()).toBe(3); // all served from cache
  });
});
