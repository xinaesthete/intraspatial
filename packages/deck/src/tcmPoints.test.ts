import { describe, expect, it } from "vite-plus/test";
import { planMarks } from "./tcmPlan";
import { densityOver, packTcmPoints } from "./tcmPoints";

describe("packTcmPoints", () => {
  it("shares coordinates and keeps both weightings", () => {
    const p = packTcmPoints([0, 1, 2], [0, 1, 4], [1, 0, 1], [0, 2, 1]);
    expect(Array.from(p.xy)).toEqual([0, 0, 1, 1, 2, 4]);
    expect(Array.from(p.wA)).toEqual([1, 0, 1]);
    expect(Array.from(p.wB)).toEqual([0, 2, 1]);
    expect(p.totalB).toBe(3);
    expect(p.boundsA).toEqual([0, 0, 2, 4]);
    expect(p.boundsAll).toEqual([0, 0, 2, 4]);
  });

  it("leaves out non-finite points from both types and every bound, and zeroes bad weights", () => {
    const p = packTcmPoints([0, NaN, 5, 1], [0, 1, Infinity, 1], [1, 1, 1, Number.NaN], [-1, 1, 1, 1]);
    expect(Array.from(p.wA)).toEqual([1, 0, 0, 0]);
    expect(Array.from(p.wB)).toEqual([0, 0, 0, 1]);
    expect(p.totalB).toBe(1);
    expect(p.boundsA).toEqual([0, 0, 0, 0]);
    expect(p.boundsAll).toEqual([0, 0, 1, 1]);
  });

  it("defaults each weighting to every point, and never packs a zero-length buffer", () => {
    const p = packTcmPoints([3], [4]);
    expect(Array.from(p.wA)).toEqual([1]);
    expect(Array.from(p.wB)).toEqual([1]);
    const empty = packTcmPoints([], []);
    expect(empty.xy.length).toBe(2);
    expect(empty.wA.length).toBe(1);
    expect(empty.boundsA).toBeNull();
  });
});

describe("densityOver", () => {
  it("is total weight per unit area, and 0 without an area", () => {
    expect(densityOver(10, [0, 0, 2, 5])).toBe(1);
    expect(densityOver(10, [0, 0, 0, 5])).toBe(0);
    expect(densityOver(10, null)).toBe(0);
  });
});

describe("planMarks", () => {
  const plan = { bbox: [10, 10, 20, 20] as [number, number, number, number], width: 100, height: 100, cell: 0.1 };

  it("covers the Γ grid padded by the Γ support, clipped to the A cells, plus a cell each side, on a lattice", () => {
    const m = planMarks({ plan, boundsA: [0, 0, 100, 100], support: 2, radius: 4, cellsPerRadius: 8, maxCells: 4e6, maxSide: 8192 })!;
    expect(m.cell).toBe(0.1); // the Γ cell is already finer than r/8
    expect(m.bbox[0]).toBeCloseTo(7.9, 9);
    expect(m.bbox[1]).toBeCloseTo(7.9, 9);
    expect(m.bbox[2]).toBeGreaterThanOrEqual(22.1);
    expect(m.bbox[3]).toBeGreaterThanOrEqual(22.1);
    const clipped = planMarks({ plan, boundsA: [15, 0, 30, 18], support: 2, radius: 4, cellsPerRadius: 8, maxCells: 4e6, maxSide: 8192 })!;
    expect(clipped.bbox[0]).toBeCloseTo(14.9, 9);
    expect(clipped.bbox[3]).toBeGreaterThanOrEqual(18.1);
    expect(clipped.bbox[3]).toBeLessThan(18.3);
  });

  it("refines to r / cellsPerRadius, and coarsens to fit the caps", () => {
    const fine = planMarks({ plan, boundsA: [0, 0, 100, 100], support: 2, radius: 0.4, cellsPerRadius: 8, maxCells: 4e6, maxSide: 8192 })!;
    expect(fine.cell).toBeCloseTo(0.05, 9);
    const capped = planMarks({
      plan,
      boundsA: [0, 0, 100, 100],
      support: 2,
      radius: 0.4,
      cellsPerRadius: 8,
      maxCells: 10_000,
      maxSide: 64,
    })!;
    expect(capped.width * capped.height).toBeLessThanOrEqual(10_000);
    expect(Math.max(capped.width, capped.height)).toBeLessThanOrEqual(64);
  });

  it("is null when no A cell reaches the Γ grid", () => {
    expect(
      planMarks({ plan, boundsA: [50, 50, 60, 60], support: 2, radius: 4, cellsPerRadius: 8, maxCells: 4e6, maxSide: 8192 }),
    ).toBeNull();
  });

  it("gives a single A cell a cell to sit in", () => {
    const one = planMarks({ plan, boundsA: [15, 15, 15, 15], support: 2, radius: 4, cellsPerRadius: 8, maxCells: 4e6, maxSide: 8192 })!;
    expect(one.width).toBeGreaterThanOrEqual(1);
    // At least a cell of margin on each side of it, for the bilinear fetch.
    expect(one.bbox[0]).toBeLessThanOrEqual(15 - one.cell);
    expect(one.bbox[2]).toBeGreaterThanOrEqual(15 + one.cell);
  });
});
