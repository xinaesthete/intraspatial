import { describe, expect, it } from "vite-plus/test";
import {
  type Box,
  axisStarts,
  componentBoxes,
  maskCountsInRects,
  maskRuns,
  maskRunsFromRows,
  nonZero,
  patchGrid,
  patchLabelImage,
  patchLayers,
  pointCountsInRects,
  pointPixels,
  rectsArray,
  roundHalfEven,
  toPixels,
  unionRuns,
} from "./patchGrid";

// Goldens: SIMBA's grid on four small masks (SIMBA_GOLDEN, below, made by running TNBC-MMB's
// `generate_patch_bounds` + `tabulate_patch_support` on exactly these CASES), and brute-force
// dense references for the run arithmetic on random masks.

interface Case {
  height: number;
  width: number;
  rects: [number, number, number, number][];
  size: number;
  stride: number;
  /** [row, col], as SIMBA's centroids. */
  centroids: [number, number][];
}

const CASES: Record<string, Case> = {
  "two-blobs": {
    height: 30,
    width: 40,
    rects: [
      [3, 2, 25, 20],
      [22, 15, 30, 24],
      [33, 4, 36, 7],
      [36, 7, 38, 9], // touches the speck above only at a corner: its own 4-connected component
    ],
    size: 10,
    stride: 5,
    centroids: [
      [5, 5],
      [6.5, 6.5],
      [7.5, 7.5],
      [12, 20],
      [19.5, 24.5],
      [29.9, 39.9],
      [2.5, 3.5],
    ],
  },
  "small-image": {
    height: 6,
    width: 8,
    rects: [
      [1, 1, 4, 3],
      [6, 4, 8, 6],
    ],
    size: 10,
    stride: 5,
    centroids: [
      [0.5, 0.5],
      [5, 7],
    ],
  },
  "edge-clamp": {
    height: 25,
    width: 33,
    rects: [
      [20, 0, 33, 25],
      [0, 10, 4, 12],
    ],
    size: 8,
    stride: 3,
    centroids: [
      [24, 32],
      [10, 1.5],
    ],
  },
  "stride-gt-size": { height: 20, width: 50, rects: [[0, 0, 50, 20]], size: 6, stride: 9, centroids: [] },
};

function rasterOf(c: Case): Uint8Array {
  const m = new Uint8Array(c.width * c.height);
  for (const [x0, y0, x1, y1] of c.rects) for (let y = y0; y < y1; y++) m.fill(1, y * c.width + x0, y * c.width + x1);
  return m;
}

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

/** Blobby random mask: thresholded sum of a few random discs. */
function randomMask(w: number, h: number, seed: number, discs = 12): Uint8Array {
  const rnd = lcg(seed);
  const m = new Uint8Array(w * h);
  for (let k = 0; k < discs; k++) {
    const cx = rnd() * w;
    const cy = rnd() * h;
    const r = 1 + rnd() * Math.min(w, h) * 0.2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if ((x - cx) ** 2 + (y - cy) ** 2 < r * r) m[y * w + x] = 1;
  }
  return m;
}

/** Dense 4-connected flood-fill boxes: the reference for `componentBoxes`. */
function denseBoxes(m: Uint8Array, w: number, h: number): Box[] {
  const seen = new Uint8Array(w * h);
  const out: Box[] = [];
  for (let i = 0; i < w * h; i++) {
    if (!m[i] || seen[i]) continue;
    const stack = [i];
    seen[i] = 1;
    let x0 = w;
    let y0 = h;
    let x1 = 0;
    let y1 = 0;
    while (stack.length) {
      const p = stack.pop()!;
      const x = p % w;
      const y = (p - x) / w;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x + 1);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y + 1);
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        const nx = x + dx;
        const ny = y + dy;
        const q = ny * w + nx;
        if (nx >= 0 && ny >= 0 && nx < w && ny < h && m[q] && !seen[q]) {
          seen[q] = 1;
          stack.push(q);
        }
      }
    }
    out.push({ x0, y0, x1, y1 });
  }
  return out;
}

const boxKey = (b: Box) => `${b.y0},${b.x0},${b.y1},${b.x1}`;

describe("patchGrid matches SIMBA", () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const mask = maskRuns(rasterOf(c), c.width, c.height);
      const cells = pointPixels(
        c.centroids.map((p) => p[1]),
        c.centroids.map((p) => p[0]),
        c.width,
        c.height,
      );
      const grid = patchGrid({
        footprint: mask,
        size: c.size,
        stride: c.stride,
        rules: [
          { kind: "fraction", name: "acquired", mask, min: 0.5 },
          { kind: "count", name: "cells", points: cells, min: 1 },
        ],
      });
      const got = grid.candidates.map((p) => [p.x0, p.y0, p.x1, p.y1, p.edge, p.support[0], p.support[1]]);
      expect(got).toEqual(SIMBA_GOLDEN[name]);
      // ids are candidate order; kept = SIMBA's valid_imc_patch (>= 0.5 acquired, >= 1 cell).
      expect(grid.candidates.map((p) => p.id)).toEqual(got.map((_, i) => i));
      const want = SIMBA_GOLDEN[name]!.flatMap((p, i) => (p[5] >= 0.5 && p[6] >= 1 ? [i] : []));
      expect(grid.patches.map((p) => p.id)).toEqual(want);
    });
  }
});

describe("run arithmetic vs dense references", () => {
  it("componentBoxes = dense 4-connected flood fill", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const w = 37 + seed;
      const h = 29;
      const m = randomMask(w, h, seed, 4 + seed * 3);
      const got = componentBoxes(maskRuns(m, w, h))
        .map(boxKey)
        .sort();
      expect(got).toEqual(denseBoxes(m, w, h).map(boxKey).sort());
    }
  });

  it("maskCountsInRects = dense counting, every rectangle", () => {
    const w = 41;
    const h = 33;
    const m = randomMask(w, h, 9, 20);
    const runs = maskRuns(m, w, h);
    const rnd = lcg(77);
    const rects: Box[] = Array.from({ length: 300 }, () => {
      const x0 = Math.floor(rnd() * w);
      const y0 = Math.floor(rnd() * h);
      return { x0, y0, x1: x0 + 1 + Math.floor(rnd() * (w - x0)), y1: y0 + 1 + Math.floor(rnd() * (h - y0)) };
    });
    const want = rects.map((r) => {
      let s = 0;
      for (let y = r.y0; y < r.y1; y++) for (let x = r.x0; x < r.x1; x++) s += m[y * w + x]!;
      return s;
    });
    expect(Array.from(maskCountsInRects(runs, rects))).toEqual(want);
  });

  it("pointCountsInRects = dense counting of rounded, clamped pixels", () => {
    const w = 30;
    const h = 20;
    const rnd = lcg(5);
    const xs = Array.from({ length: 500 }, () => rnd() * (w + 2) - 1);
    const ys = Array.from({ length: 500 }, (_, i) => (i % 7 === 0 ? Math.floor(rnd() * h) + 0.5 : rnd() * (h + 2) - 1));
    const pts = pointPixels(xs, ys, w, h);
    const rects: Box[] = [
      { x0: 0, y0: 0, x1: w, y1: h },
      { x0: 3, y0: 2, x1: 11, y1: 9 },
      { x0: 29, y0: 19, x1: 30, y1: 20 },
      { x0: 10, y0: 5, x1: 11, y1: 6 },
    ];
    const clamp = (v: number, n: number) => Math.min(Math.max(roundHalfEven(v), 0), n - 1);
    const want = rects.map((r) =>
      xs.reduce((s, x, i) => {
        const c = clamp(x, w);
        const rr = clamp(ys[i]!, h);
        return s + (c >= r.x0 && c < r.x1 && rr >= r.y0 && rr < r.y1 ? 1 : 0);
      }, 0),
    );
    expect(Array.from(pointCountsInRects(pts, rects))).toEqual(want);
  });

  it("roundHalfEven is numpy's round", () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4999, 2.5001, 3].map(roundHalfEven)).toEqual([0, 2, 2, 0, -2, 2, 3, 3]);
  });

  it("streamed rows, union and nonZero (NaN counts, as numpy's != 0)", () => {
    const w = 6;
    const a = [0, 1, 1, 0, Number.NaN, 0];
    const b = [1, 0, 0, 0, 0, -2];
    const ra = maskRunsFromRows(w, 1, () => a, nonZero);
    expect([...ra.x0, ...ra.x1]).toEqual([1, 4, 3, 5]);
    const u = unionRuns([ra, maskRunsFromRows(w, 1, () => b, nonZero)]);
    expect([...u.x0, ...u.x1]).toEqual([0, 4, 3, 6]);
    // The default test is finite and > 0: NaN and negatives are not support.
    const pos = maskRuns(Float32Array.from([...a, ...b]), w, 2);
    expect([...pos.x0, ...pos.x1, ...pos.rowStart]).toEqual([1, 0, 3, 1, 0, 1, 2]);
  });
});

describe("units, layers and labels", () => {
  it("axisStarts: narrow, wide with trailing edge, exact fit", () => {
    expect(axisStarts(10, 14, 100, 10, 5)).toEqual([[4, false]]);
    expect(axisStarts(10, 33, 100, 10, 5)).toEqual([
      [10, false],
      [15, false],
      [20, false],
      [23, true],
    ]);
    expect(axisStarts(0, 30, 30, 10, 10)).toEqual([
      [0, false],
      [10, false],
      [20, false],
    ]);
  });

  it("µm convert by the pixel size", () => {
    expect(toPixels(300, "um", 1)).toBe(300);
    expect(toPixels(300, "um", 2)).toBe(150);
    expect(toPixels(150, "px")).toBe(150);
    expect(() => toPixels(300, "um")).toThrow(/pixelSize/);
    expect(() => toPixels(0.2, "px")).toThrow(/less than one pixel/);
  });

  it("half-overlap lattice: four disjoint layers that cover every patch once", () => {
    const w = 64;
    const h = 48;
    const full = maskRuns(new Uint8Array(w * h).fill(1), w, h);
    const grid = patchGrid({ footprint: full, size: 16, stride: 8 });
    const layers = patchLayers(grid.patches);
    expect(layers.length).toBe(4);
    expect(layers.flat().sort((a, b) => a - b)).toEqual(grid.patches.map((_, i) => i));
    // Each layer's label image holds each member's full area and nothing else.
    for (const layer of layers) {
      const img = patchLabelImage(grid.patches, layer, w, h);
      const area = new Map<number, number>();
      for (const v of img) if (v) area.set(v, (area.get(v) ?? 0) + 1);
      expect([...area.keys()].sort((a, b) => a - b)).toEqual(layer.map((i) => i + 1));
      expect([...area.values()].every((a) => a === 256)).toBe(true);
    }
  });

  it("edge patches and differently anchored components add layers, never overlap within one", () => {
    const c = CASES["two-blobs"]!;
    const grid = patchGrid({ footprint: maskRuns(rasterOf(c), c.width, c.height), size: c.size, stride: c.stride });
    const layers = patchLayers(grid.candidates);
    expect(layers.length).toBeGreaterThan(4);
    const ok = layers.every((l) =>
      l.every((i, a) =>
        l.slice(a + 1).every((j) => {
          const p = grid.candidates[i]!;
          const q = grid.candidates[j]!;
          return !(p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1);
        }),
      ),
    );
    expect(ok).toBe(true);
  });

  it("a coarser label image takes the rectangle holding each label pixel's centre", () => {
    const rects: Box[] = [
      { x0: 0, y0: 0, x1: 4, y1: 4 },
      { x0: 4, y0: 0, x1: 10, y1: 3 },
    ];
    // 10x4 at scale 2 -> 5x2; centres at 1, 3, 5, 7, 9 (x) and 1, 3 (y).
    expect(Array.from(patchLabelImage(rects, [0, 1], 10, 4, 2))).toEqual([1, 1, 2, 2, 2, 1, 1, 0, 0, 0]);
  });

  it("rectsArray packs [x0, y0, x1, y1]", () => {
    expect(Array.from(rectsArray([{ x0: 1, y0: 2, x1: 3, y1: 4 }]))).toEqual([1, 2, 3, 4]);
  });
});

// SIMBA's own output (generate_patch_bounds + tabulate_patch_support, TNBC-MMB) on the CASES masks:
// [x0, y0, x1, y1, edge, acquiredFraction, nCells] per candidate, in SIMBA's patch order.
export const SIMBA_GOLDEN: Record<string, [number, number, number, number, boolean, number, number][]> = {
  "two-blobs": [
    [26, 0, 36, 10, false, 0.09, 0],
    [28, 0, 38, 10, false, 0.13, 0],
    [3, 2, 13, 12, false, 1.0, 4],
    [8, 2, 18, 12, false, 1.0, 1],
    [13, 2, 23, 12, false, 1.0, 0],
    [18, 2, 28, 12, false, 0.7, 0],
    [20, 2, 30, 12, true, 0.5, 0],
    [3, 7, 13, 17, false, 1.0, 1],
    [8, 7, 18, 17, false, 1.0, 1],
    [13, 7, 23, 17, false, 1.0, 1],
    [18, 7, 28, 17, false, 0.76, 1],
    [20, 7, 30, 17, true, 0.6, 1],
    [3, 12, 13, 22, false, 0.8, 0],
    [8, 12, 18, 22, false, 0.8, 0],
    [13, 12, 23, 22, false, 0.82, 1],
    [18, 12, 28, 22, false, 0.83, 2],
    [20, 12, 30, 22, true, 0.81, 2],
    [3, 14, 13, 24, true, 0.6, 0],
    [8, 14, 18, 24, true, 0.6, 0],
    [13, 14, 23, 24, true, 0.64, 0],
    [18, 14, 28, 24, true, 0.81, 1],
    [20, 14, 30, 24, true, 0.87, 1],
  ],
  "small-image": [[0, 0, 8, 6, true, 0.20833333333333334, 2]],
  "edge-clamp": [
    [20, 0, 28, 8, false, 1.0, 0],
    [23, 0, 31, 8, false, 1.0, 0],
    [25, 0, 33, 8, true, 1.0, 0],
    [20, 3, 28, 11, false, 1.0, 0],
    [23, 3, 31, 11, false, 1.0, 0],
    [25, 3, 33, 11, true, 1.0, 0],
    [0, 4, 8, 12, false, 0.125, 1],
    [20, 6, 28, 14, false, 1.0, 0],
    [23, 6, 31, 14, false, 1.0, 0],
    [25, 6, 33, 14, true, 1.0, 0],
    [20, 9, 28, 17, false, 1.0, 0],
    [23, 9, 31, 17, false, 1.0, 0],
    [25, 9, 33, 17, true, 1.0, 0],
    [20, 12, 28, 20, false, 1.0, 0],
    [23, 12, 31, 20, false, 1.0, 0],
    [25, 12, 33, 20, true, 1.0, 0],
    [20, 15, 28, 23, false, 1.0, 0],
    [23, 15, 31, 23, false, 1.0, 0],
    [25, 15, 33, 23, true, 1.0, 0],
    [20, 17, 28, 25, true, 1.0, 0],
    [23, 17, 31, 25, true, 1.0, 0],
    [25, 17, 33, 25, true, 1.0, 1],
  ],
  "stride-gt-size": [
    [0, 0, 6, 6, false, 1.0, 0],
    [9, 0, 15, 6, false, 1.0, 0],
    [18, 0, 24, 6, false, 1.0, 0],
    [27, 0, 33, 6, false, 1.0, 0],
    [36, 0, 42, 6, false, 1.0, 0],
    [44, 0, 50, 6, true, 1.0, 0],
    [0, 9, 6, 15, false, 1.0, 0],
    [9, 9, 15, 15, false, 1.0, 0],
    [18, 9, 24, 15, false, 1.0, 0],
    [27, 9, 33, 15, false, 1.0, 0],
    [36, 9, 42, 15, false, 1.0, 0],
    [44, 9, 50, 15, true, 1.0, 0],
    [0, 14, 6, 20, true, 1.0, 0],
    [9, 14, 15, 20, true, 1.0, 0],
    [18, 14, 24, 20, true, 1.0, 0],
    [27, 14, 33, 20, true, 1.0, 0],
    [36, 14, 42, 20, true, 1.0, 0],
    [44, 14, 50, 20, true, 1.0, 0],
  ],
};
