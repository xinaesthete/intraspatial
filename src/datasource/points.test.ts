// Points tiling: the grid, window selection, and half-open ownership. The load-bearing claim is
// that a window streamed tile by tile holds EXACTLY the points a single load of it would — no edge
// point lost or doubled, even though the reader's own bounds filter is closed.
import { describe, expect, it } from "vitest";
import {
  clampWindow,
  expandRect,
  ownedTile,
  ownsPoint,
  type PointsTile,
  pointsGrid,
  pointsTileBytes,
  type RawPoints,
  type Rect,
  selectPointsTiles,
  tileRect,
} from "./points";
import { resolveWith, TileCache } from "./tileCache";
import type { ChunkId, Loader } from "./types";

const BOUNDS: Rect = { minX: 0, minY: 0, maxX: 1000, maxY: 400 };

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** Random points, plus points sitting exactly on tile edges and on the far bounds. */
function scene(grid: ReturnType<typeof pointsGrid>): { xs: number[]; ys: number[]; codes: number[]; qv: number[] } {
  const rnd = lcg(7);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < 4000; i++) {
    xs.push(BOUNDS.minX + rnd() * (BOUNDS.maxX - BOUNDS.minX));
    ys.push(BOUNDS.minY + rnd() * (BOUNDS.maxY - BOUNDS.minY));
  }
  for (let i = 0; i <= grid.cols; i++) {
    for (let j = 0; j <= grid.rows; j++) {
      const r = tileRect(grid, Math.min(i, grid.cols - 1), Math.min(j, grid.rows - 1));
      xs.push(i === grid.cols ? r.maxX : r.minX);
      ys.push(j === grid.rows ? r.maxY : r.minY);
    }
  }
  return { xs, ys, codes: xs.map((_, i) => i % 7), qv: xs.map((_, i) => (i % 40) + 1) };
}

/** A reader like sd.js's: returns every point in the CLOSED bounds, edge points included. */
function closedReader(s: ReturnType<typeof scene>, r: Rect): RawPoints {
  const keep = s.xs
    .map((_x, i) => i)
    .filter((i) => {
      const x = s.xs[i] ?? 0;
      const y = s.ys[i] ?? 0;
      return x >= r.minX && x <= r.maxX && y >= r.minY && y <= r.maxY;
    });
  return {
    xs: keep.map((i) => s.xs[i] ?? 0),
    ys: keep.map((i) => s.ys[i] ?? 0),
    codes: keep.map((i) => s.codes[i] ?? 0),
    columns: { qv: keep.map((i) => s.qv[i] ?? 0) },
  };
}

const key = (x: number, y: number, c: number): string => `${Math.fround(x)},${Math.fround(y)},${c}`;

describe("pointsGrid", () => {
  it("picks the finest level whose tiles still hold minRowsPerTile", () => {
    const g = pointsGrid(BOUNDS, 400_000, { minRowsPerTile: 10_000 });
    expect(g.rowsPerTile).toBeGreaterThanOrEqual(10_000);
    // one level finer would drop below the floor
    expect(g.rowsPerTile / 4).toBeLessThan(10_000);
    expect(g.cols * g.tileSize).toBeGreaterThanOrEqual(1000);
    expect(g.rows * g.tileSize).toBeGreaterThanOrEqual(400);
  });

  it("refuses degenerate bounds", () => {
    expect(() => pointsGrid({ minX: 0, minY: 0, maxX: 0, maxY: 1 }, 10)).toThrow(/degenerate/);
  });
});

describe("ownership", () => {
  const grid = pointsGrid(BOUNDS, 4000, { minRowsPerTile: 300 });

  it("every point in the bounds is owned by exactly one tile — edges and far corners included", () => {
    const s = scene(grid);
    let multiple = 0;
    let none = 0;
    for (let i = 0; i < s.xs.length; i++) {
      let owners = 0;
      for (let y = 0; y < grid.rows; y++) {
        for (let x = 0; x < grid.cols; x++) if (ownsPoint(grid, { level: grid.level, x, y, z: 0 }, s.xs[i] ?? 0, s.ys[i] ?? 0)) owners++;
      }
      if (owners > 1) multiple++;
      if (owners === 0) none++;
    }
    expect({ multiple, none }).toEqual({ multiple: 0, none: 0 });
  });

  it("a window streamed tile by tile equals one closed read of the whole extent", async () => {
    const s = scene(grid);
    const loader: Loader<PointsTile> = {
      getChunk: async (id: ChunkId) => ownedTile(grid, id, closedReader(s, tileRect(grid, id.x, id.y))),
    };
    const selection = selectPointsTiles(grid, BOUNDS, 0);
    expect(selection.chunks.length).toBe(grid.cols * grid.rows);
    const tiles = await resolveWith(selection, loader, pointsTileBytes);
    const streamed: string[] = [];
    const qvSum = { streamed: 0, whole: 0 };
    for (const t of tiles.values()) {
      for (let i = 0; i < t.count; i++) {
        streamed.push(key(t.xs[i] ?? 0, t.ys[i] ?? 0, t.codes[i] ?? 0));
        qvSum.streamed += t.columns.qv?.[i] ?? 0;
      }
    }
    const whole = closedReader(s, BOUNDS);
    const once: string[] = [];
    for (let i = 0; i < whole.xs.length; i++) {
      once.push(key(whole.xs[i] ?? 0, whole.ys[i] ?? 0, whole.codes[i] ?? 0));
      qvSum.whole += whole.columns?.qv?.[i] ?? 0;
    }
    expect(streamed.length).toBe(once.length);
    expect(streamed.sort().join("|") === once.sort().join("|")).toBe(true);
    expect(qvSum.streamed).toBe(qvSum.whole);
  });
});

describe("selectPointsTiles", () => {
  const grid = pointsGrid(BOUNDS, 4000, { minRowsPerTile: 300 });
  const window: Rect = { minX: 300, minY: 100, maxX: 500, maxY: 200 };

  it("covers window ⊕ apron, nearest-first, and nothing when disjoint", () => {
    const sel = selectPointsTiles(grid, window, 60);
    const need = expandRect(window, 60);
    for (const c of sel.chunks) {
      const r = tileRect(grid, c.id.x, c.id.y);
      expect(r.maxX >= need.minX && r.minX <= need.maxX && r.maxY >= need.minY && r.minY <= need.maxY).toBe(true);
    }
    const depths = sel.chunks.map((c) => c.nearestDepth);
    expect(depths).toEqual([...depths].sort((a, b) => a - b));
    expect(depths[0]).toBe(0);
    expect(selectPointsTiles(grid, { minX: 5000, minY: 5000, maxX: 6000, maxY: 6000 }, 10).chunks).toHaveLength(0);
  });

  it("a moving window reuses cached tiles instead of reloading them", async () => {
    const s = scene(grid);
    let loads = 0;
    const loader: Loader<PointsTile> = {
      getChunk: async (id) => {
        loads++;
        return ownedTile(grid, id, closedReader(s, tileRect(grid, id.x, id.y)));
      },
    };
    const cache = new TileCache<PointsTile>({ maxBytes: 1e9 });
    const first = selectPointsTiles(grid, window, 20);
    await resolveWith(first, loader, pointsTileBytes, cache);
    const afterFirst = loads;
    const shifted = selectPointsTiles(grid, { ...window, minX: 320, maxX: 520 }, 20);
    await resolveWith(shifted, loader, pointsTileBytes, cache);
    const fresh = shifted.chunks.filter((c) => !first.chunks.some((f) => f.id.x === c.id.x && f.id.y === c.id.y)).length;
    expect(loads - afterFirst).toBe(fresh);
  });
});

describe("clampWindow", () => {
  const extent: Rect = { minX: 0, minY: 0, maxX: 100, maxY: 100 };
  it("cuts the view to the extent", () => {
    expect(clampWindow({ minX: -50, minY: 20, maxX: 30, maxY: 60 }, extent, 1e9)).toEqual({ minX: 0, minY: 20, maxX: 30, maxY: 60 });
  });
  it("shrinks an oversized view about its centre, keeping its aspect", () => {
    const w = clampWindow({ minX: 0, minY: 0, maxX: 80, maxY: 20 }, extent, 400);
    expect(w).toEqual({ minX: 20, minY: 5, maxX: 60, maxY: 15 }); // 40×10 = 400, centred on (40, 10)
  });
  it("shrinks to the longest side when that is the tighter limit", () => {
    const w = clampWindow({ minX: 0, minY: 0, maxX: 80, maxY: 20 }, extent, 1e9, 40);
    expect(w).toEqual({ minX: 20, minY: 5, maxX: 60, maxY: 15 }); // 40 long, aspect kept, centred on (40, 10)
  });
  it("is undefined when the view misses the extent", () => {
    expect(clampWindow({ minX: 200, minY: 0, maxX: 300, maxY: 50 }, extent, 1e9)).toBeUndefined();
  });
});
