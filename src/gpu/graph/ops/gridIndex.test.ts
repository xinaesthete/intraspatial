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

/** The bundle shape the op declares, or a failure if it declared something else. */
function bundleParts(params: Record<string, unknown>, n = 3) {
  const s = op().inferShapes([{ kind: "points", n }], params)[0]!;
  if (s.kind !== "bundle") throw new Error(`expected a bundle, got ${s.kind}`);
  return s;
}

describe("gridIndexOp — shapes", () => {
  it("declares ONE bundle whose parts size to the lattice and the cloud", () => {
    const cpu = buildBucketGrid([1, 2, 3], [4, 5, 6], 10, [BOUNDS.minX, BOUNDS.minY, BOUNDS.maxX, BOUNDS.maxY]);
    expect(bundleParts({ cell: 10, ...BOUNDS })).toEqual({
      kind: "bundle",
      name: "gridIndex",
      parts: {
        start: { kind: "points", n: cpu.cols * cpu.rows + 1 },
        items: { kind: "points", n: 3 },
        // The lattice part IS the cell grid, which is what makes a consumer's extent inferable.
        lattice: { kind: "grid", width: cpu.cols, height: cpu.rows },
      },
    });
  });

  it("keeps items bindable for an empty cloud", () => {
    expect(bundleParts({ cell: 10, ...BOUNDS }, 0).parts.items).toEqual({ kind: "points", n: 1 });
  });

  it("rejects a non-points input and bad params", () => {
    expect(() => op().inferShapes([{ kind: "grid", width: 4, height: 4 }], { cell: 1, ...BOUNDS })).toThrow(/points cloud/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 0, ...BOUNDS })).toThrow(/cell must be > 0/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 1, ...BOUNDS, maxX: -1 })).toThrow(/max bound below min/);
    expect(() => op().inferShapes([{ kind: "points", n: 1 }], { cell: 1, ...BOUNDS, minY: Number.NaN })).toThrow(/finite/);
  });
});

describe("gridIndexOp — placement", () => {
  // The bundle itself is unplaced; the placement lives on its `lattice` part, which `execute` and
  // `cpuGolden` stamp — so the algebra is checked by running the golden.
  const latticePlacementOf = (params: Record<string, unknown>) =>
    op().cpuGolden!(
      [{ shape: { kind: "points", n: 2 }, dtype: "f32", data: new Float32Array([1, 2, 3, 4]), placement: placed }],
      params,
    )[0]!.parts?.lattice?.placement;

  it("scales the points' axes by the cell size and moves the origin to the lattice corner", () => {
    expect(op().inferPlacement!([placed], { cell: 10, ...BOUNDS })).toEqual([undefined]);
    expect(latticePlacementOf({ cell: 10, ...BOUNDS, minX: 4, minY: 6 })).toEqual({
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
    const lattice = latticePlacementOf({ cell, ...BOUNDS })!;
    const at = (cx: number, cy: number) => {
      const { origin, axes } = lattice.worldFromArray;
      return [origin[0] + cx * axes[0][0] + cy * axes[1][0], origin[1] + cx * axes[0][1] + cy * axes[1][1]];
    };
    // Cell (2, 1) starts at point (20, 10) in the cloud's own coordinates, which the placement
    // puts at (10 + 20·2, 5 + 10·3) = (50, 35).
    expect(at(2, 1)).toEqual([50, 35]);
    expect(at(0, 0)).toEqual([10, 5]);
  });

  it("leaves the lattice array-space when the points are unplaced", () => {
    const v = op().cpuGolden!([{ shape: { kind: "points", n: 1 }, dtype: "f32", data: new Float32Array([1, 2]) }], {
      cell: 10,
      ...BOUNDS,
    })[0]!;
    expect(op().inferPlacement!([undefined], { cell: 10, ...BOUNDS })).toEqual([undefined]);
    expect(v.parts?.lattice?.placement).toBeUndefined();
  });

  it("gives the graph one bundle output, unplaced", () => {
    const g = new Graph();
    const pts = g.source(
      { shape: { kind: "points", n: 2 }, dtype: "f32", data: new Float32Array([1, 2, 3, 4]), placement: placed },
      "points",
    );
    const outs = g.op("gridIndex", { points: pts }, { cell: 10, ...BOUNDS });
    expect(outs.length).toBe(1);
    expect(outs[0]!.shape.kind).toBe("bundle");
    expect(outs[0]!.placement).toBeUndefined();
  });
});

describe("gridIndex parts — extract and combine", () => {
  const bundleOf = () =>
    op().cpuGolden!([{ shape: { kind: "points", n: 2 }, dtype: "f32", data: new Float32Array([1, 2, 3, 4]), placement: placed }], {
      cell: 10,
      ...BOUNDS,
    })[0]!;

  it("hands back the very same part value, not a copy", () => {
    const bundle = bundleOf();
    for (const part of ["start", "items", "lattice"] as const) {
      const extract = getOp(`gridIndex.${part}`);
      expect(extract.inferShapes([bundle.shape], {})).toEqual([bundle.parts![part]!.shape]);
      // Identity, not equality: extraction BORROWS (ADR-0023), so a copy here would be the bug.
      expect(extract.cpuGolden!([bundle], {})[0]).toBe(bundle.parts![part]);
    }
  });

  it("rejects a bundle of the wrong type", () => {
    const wrong = { kind: "bundle", name: "knn", parts: {} } as const;
    expect(() => getOp("gridIndex.start").inferShapes([wrong], {})).toThrow(/expected a "gridIndex" bundle/);
    expect(() => getOp("gridIndex.start").inferShapes([{ kind: "points", n: 3 }], {})).toThrow(/expected a "gridIndex" bundle/);
  });

  it("round-trips through combine", () => {
    const bundle = bundleOf();
    const parts = ["start", "items", "lattice"].map((p) => bundle.parts![p]!);
    const combined = getOp("gridIndex.bundle").cpuGolden!(parts, {})[0]!;
    expect(combined.shape).toEqual(bundle.shape);
    for (const p of ["start", "items", "lattice"]) expect(combined.parts![p]).toBe(bundle.parts![p]);
  });
});
