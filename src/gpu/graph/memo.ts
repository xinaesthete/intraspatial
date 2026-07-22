// Content-addressed memoisation for the executor.
//
// A node's cache key is a hash of (op name, params, the keys of its inputs) — a
// Merkle DAG over the computation. Two pulls that share a structurally identical
// sub-computation get identical keys, so unchanged upstream nodes (e.g. an expensive
// splat) are reused even when the graph is rebuilt from scratch each pull and when a
// downstream param changes. Change one param and only that node and its dependents
// get fresh keys; everything else is a cache hit.
//
// The cache is caller-owned (passed via PullOptions.cache), so it persists across
// pulls in a live UI but never leaks into the Node test path, which omits it.
import type { FieldValue } from "./handle";

/** cyrb53 string hash → base36 digest (fixed length, so parent keys don't grow). */
export function hashString(str: string, seed = 0): string {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  const n = 4294967296 * (2097151 & h2) + (h1 >>> 0);
  return n.toString(36);
}

/** Hash the raw bytes of a typed array (exact; fast even for large grids). */
export function hashBytes(view: ArrayBufferView): string {
  const b = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < b.length; i++) {
    const ch = b[i]!;
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

/** Stable JSON (sorted keys) so param object key order doesn't affect the hash. */
export function stableJSON(obj: unknown): string {
  if (obj === null || typeof obj !== "object") return JSON.stringify(obj) ?? "null";
  if (Array.isArray(obj)) return "[" + obj.map(stableJSON).join(",") + "]";
  const keys = Object.keys(obj as Record<string, unknown>).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableJSON((obj as Record<string, unknown>)[k])).join(",") + "}";
}

/** Per-object identity for source values we cannot hash by content. Keyed weakly, so tagging a
 *  value never keeps it alive. Two *structurally equal* resident sources get different tags —
 *  that costs a cache miss, which is the safe direction to be wrong in. */
const identityTags = new WeakMap<object, string>();
let identitySeq = 0;

function identityTag(v: FieldValue): string {
  let tag = identityTags.get(v);
  if (tag === undefined) {
    tag = `id${identitySeq++}`;
    identityTags.set(v, tag);
  }
  return tag;
}

/** Identity key for a source value (shape + its data bytes / opaque payload).
 *
 *  A GPU-resident source (ADR-0017) has no host bytes to hash, so it keys off object identity
 *  instead. This is not merely a perf question: hashing such a value by content would fall
 *  through to a single constant, making *every* resident source collide with every other and
 *  silently serve wrong cache hits. Identity keying is forced, per ADR-0017's `hashSource`
 *  note — a resident source that is mutated in place on the GPU must be re-`source`d (or its
 *  `version` bumped) to be seen as new, exactly as a mutated host array must be. */
export function hashSource(v: FieldValue): string {
  const shape = stableJSON(v.shape);
  const body = v.data ? hashBytes(v.data) : v.payload !== undefined ? stableJSON(v.payload) : v.buffer ? identityTag(v) : "empty";
  return "src:" + hashString(shape + "|" + body);
}

export interface GraphMemo {
  get(key: string): FieldValue | undefined;
  set(key: string, value: FieldValue): void;
  size(): number;
  clear(): void;
}

/** A simple LRU memo (insertion-order Map, get-promotes, oldest evicted). */
export function createMemo(maxEntries = 512): GraphMemo {
  const map = new Map<string, FieldValue>();
  return {
    get(key) {
      const v = map.get(key);
      if (v !== undefined) {
        map.delete(key);
        map.set(key, v); // promote (LRU)
      }
      return v;
    },
    set(key, value) {
      map.set(key, value);
      if (map.size > maxEntries) {
        const oldest = map.keys().next().value;
        if (oldest !== undefined) map.delete(oldest);
      }
    },
    size: () => map.size,
    clear: () => map.clear(),
  };
}
