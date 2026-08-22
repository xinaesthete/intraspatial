import { describe, expect, it } from "vitest";
import { buildBucketGrid } from "../../../spatial/bucketGrid";
import type { ResolvedPlacement } from "../handle";
import { Graph, registerBuiltinOps } from "../index";
import { getOp } from "../registry";

// Build-time contract of `gridIndexOp`: shapes, placement and param validation. No device — the
// kernel itself is covered by `gridIndex.gpu.test.ts`.

registerBuiltinOps();

const BOUNDS = { minX: 0, minY: 0, maxX: 100, maxY: 80 };
const op = () => getOp("gridIndex");

/** An anisotropic, translated placement: 2 world units per x step, 3 per y, origin at (10, 5). */
const placed: ResolvedPlacement = {
  system: "global",
  worldFromArray: {
    origin: [10, 5, 0],
    axes: [
      [2, 0, 0],
      [0, 3, 0],
      [0, 0, 1],
    ],
  },
};

describe("gridIndexOp — shapes", () => {
  it("sizes start as cells + 1 and items as n, matching buildBucketGrid", () => {
    const cpu = buildBucketGrid([1, 2, 3], [4, 5, 6], 10, [BOUNDS.minX, BOUNDS.minY, BOUNDS.maxX, BOUNDS.maxY]);
    const shapes = op().inferShapes([{ kind: "points", n: 3 }], { cell: 10, ...BOUNDS });
    expect(shapes).toEqual([
      { kind: "points", n: cpu.cols * cpu.rows + 1 },
      { kind: "points", n: 3 },
      { kind: "opaque", name: "gridLattice" },
    ]);
  });

  it("keeps items bindable for an empty cloud", () => {
    expect(op().inferShapes([{ kind: "points", n: 0 }], { cell: 10, ...BOUNDS })[1]).toEqual({ kind: "points", n: 1 });
  });

  it("rejects a non-points input and bad params", () => {
    expect(() => op().inferShapes([{ kind: "grid", width: 4, height: 4 }], { cell: 1, ...BOUNDS })).toThrow(/points cloud/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 0, ...BOUNDS })).toThrow(/cell must be > 0/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 1, ...BOUNDS, maxX: -1 })).toThrow(/max bound below min/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 1, ...BOUNDS, minY: Number.NaN })).toThrow(/finite/);
  });
});

describe("gridIndexOp — placement", () => {
  it("scales the points' axes by the cell size and moves the origin to the lattice corner", () => {
    const [start, items, lattice] = op().inferPlacement!([placed], { cell: 10, ...BOUNDS, minX: 4, minY: 6 });
    // Index lists have no position.
    expect([start, items]).toEqual([undefined, undefined]);
    expect(lattice).toEqual({
      system: "global",
      // origin + minX·axes[0] + minY·axes[1] = (10 + 4·2, 5 + 6·3, 0); axes ×10, z untouched.
      worldFromArray: {
        origin: [18, 23, 0],
        axes: [
          [20, 0, 0],
          [0, 30, 0],
          [0, 0, 1],
        ],
      },
    });
  });

  it("maps cell (cx, cy) to the world position of that cell's corner", () => {
    const cell = 10;
    const lattice = op().inferPlacement!([placed], { cell, ...BOUNDS })[2]!;
    const at = (cx: number, cy: number) => {
      const { origin, axes } = lattice.worldFromArray;
      return [origin[0] + cx * axes[0][0] + cy * axes[1][0], origin[1] + cx * axes[0][1] + cy * axes[1][1]];
    };
    // Cell (2, 1) starts at point (20, 10) in the cloud's own coordinates, which the placement
    // puts at (10 + 20·2, 5 + 10·3) = (50, 35).
    expect(at(2, 1)).toEqual([50, 35]);
    expect(at(0, 0)).toEqual([10, 5]);
  });

  it("leaves everything array-space when the points are unplaced", () => {
    expect(op().inferPlacement!([undefined], { cell: 10, ...BOUNDS })).toEqual([undefined, undefined, undefined]);
  });

  it("stamps the lattice placement onto the built field", () => {
    const g = new Graph();
    const pts = g.source(
      { shape: { kind: "points", n: 2 }, dtype: "f32", data: new Float32Array([1, 2, 3, 4]), placement: placed },
      "points",
    );
    const [start, , lattice] = g.op("gridIndex", { points: pts }, { cell: 10, ...BOUNDS });
    expect(start!.placement).toBeUndefined();
    expect(lattice!.placement?.worldFromArray.axes[0]).toEqual([20, 0, 0]);
  });
});
