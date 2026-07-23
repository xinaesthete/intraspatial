// Slice 2 of the stream-A placement plan (ADR-0018): binary-op placement agreement, the
// placement `splatDensity` constructs from bbox + resolution, and `hashSource` folding placement
// into source identity. All three are build-time / pure-function checks, so they live on the CPU
// path (the splat's *value* correctness under placement is in `splatPlacement.gpu.test.ts`).
import { beforeAll, describe, expect, it } from "vitest";
import type { Affine3 } from "../../coords";
import type { FieldValue, ResolvedPlacement } from "./handle";
import { Graph, registerElementOps } from "./index";
import { hashSource } from "./memo";

beforeAll(async () => {
  await registerElementOps(); // addFields / dotFields — the fieldArithmetic binary ops
});

const affine = (origin: [number, number, number], sx: number, sy: number): Affine3 => ({
  origin,
  axes: [
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1],
  ],
});

const placementIn = (system: string, wfa: Affine3): ResolvedPlacement => ({ system, worldFromArray: wfa });

/** A 4×4 grid source carrying whatever facets are passed. */
function gridSource(g: Graph, extra: Partial<FieldValue> = {}) {
  const value: FieldValue = { shape: { kind: "grid", width: 4, height: 4 }, dtype: "f32", data: new Float32Array(16).fill(1), ...extra };
  return g.source(value);
}

/** A points source carrying whatever facets are passed. */
function pointsSource(g: Graph, xs: number[], ys: number[], extra: Partial<FieldValue> = {}) {
  const data = new Float32Array(xs.length * 2);
  for (let i = 0; i < xs.length; i++) {
    data[2 * i] = xs[i]!;
    data[2 * i + 1] = ys[i]!;
  }
  const value: FieldValue = { shape: { kind: "points", n: xs.length }, dtype: "f32", data, ...extra };
  return g.source(value, "points");
}

describe("binary-op placement agreement (ADR-0018 decision 3 — reject add across systems)", () => {
  it("addGrids: two sources in DIFFERENT systems throw at graph build", () => {
    const g = new Graph();
    const a = gridSource(g, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const b = gridSource(g, { placement: placementIn("tissue", affine([0, 0, 0], 1, 1)) });
    expect(() => g.op("addGrids", { a, b })).toThrow(/different coordinate systems/i);
  });

  it("addGrids: same system builds fine (even with different matrices — two pyramid levels)", () => {
    const g = new Graph();
    const a = gridSource(g, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const b = gridSource(g, { placement: placementIn("global", affine([10, 20, 0], 0.5, 0.5)) });
    const [out] = g.op("addGrids", { a, b });
    // Passes through the first input's placement (a pointwise sum does not move the grid).
    expect(out!.placement).toEqual(a.placement);
  });

  it("addGrids: both array-space (absent) builds fine — today's behaviour", () => {
    const g = new Graph();
    const [out] = g.op("addGrids", { a: gridSource(g), b: gridSource(g) });
    expect(out!.placement).toBeUndefined();
  });

  it("addGrids: exactly one placed ⇒ throws (placed + array-space can't combine)", () => {
    const g = new Graph();
    const a = gridSource(g, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const b = gridSource(g); // unplaced
    expect(() => g.op("addGrids", { a, b })).toThrow(/cannot combine/i);
  });

  it("addFields (fieldArithmetic): different systems throw, same system builds", () => {
    const g = new Graph();
    const a = gridSource(g, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const b = gridSource(g, { placement: placementIn("tissue", affine([0, 0, 0], 1, 1)) });
    expect(() => g.op("addFields", { a, b })).toThrow(/different coordinate systems/i);

    const g2 = new Graph();
    const c = gridSource(g2, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const d = gridSource(g2, { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const [out] = g2.op("addFields", { a: c, b: d });
    expect(out!.placement).toEqual(c.placement);
  });
});

describe("splatDensity constructs its output grid's placement (ADR-0018)", () => {
  // bbox 24 wide over a 24-cell grid ⇒ cellSize 1; origin at bbox.min. Clean numbers to assert.
  const bbox = [10, 20, 34, 44];
  const expectedWfa = affine([10, 20, 0], (34 - 10) / 24, (44 - 20) / 24); // = diag(1,1), origin (10,20,0)

  it("placed points ⇒ output grid carries the points' system + a bbox/resolution worldFromArray", () => {
    const g = new Graph();
    const pts = pointsSource(g, [12, 30], [22, 40], { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const [out] = g.op("splatDensity", { points: pts }, { width: 24, height: 24, bbox });
    expect(out!.placement?.system).toBe("global");
    expect(out!.placement?.worldFromArray).toEqual(expectedWfa);
    // Numeric spot-check of a few known cells: world = origin + (i,j)·cellSize.
    const wfa = out!.placement!.worldFromArray;
    const cell = (i: number, j: number): [number, number, number] => [
      wfa.origin[0] + i * wfa.axes[0][0] + j * wfa.axes[1][0],
      wfa.origin[1] + i * wfa.axes[0][1] + j * wfa.axes[1][1],
      wfa.origin[2] + i * wfa.axes[0][2] + j * wfa.axes[1][2],
    ];
    expect(cell(0, 0)).toEqual([10, 20, 0]); // bbox.min
    expect(cell(24, 24)).toEqual([34, 44, 0]); // bbox.max
    expect(cell(5, 3)).toEqual([15, 23, 0]);
  });

  it("unplaced points ⇒ output placement absent (array space in ⇒ array space out, no fabricated identity)", () => {
    const g = new Graph();
    const pts = pointsSource(g, [12, 30], [22, 40]); // no placement
    const [out] = g.op("splatDensity", { points: pts }, { width: 24, height: 24, bbox });
    expect(out!.placement).toBeUndefined();
  });

  it("placed points but no explicit bbox ⇒ output placement absent (region unknown at build time)", () => {
    const g = new Graph();
    const pts = pointsSource(g, [12, 30], [22, 40], { placement: placementIn("global", affine([0, 0, 0], 1, 1)) });
    const [out] = g.op("splatDensity", { points: pts }, { width: 24, height: 24 }); // no bbox
    expect(out!.placement).toBeUndefined();
  });
});

describe("hashSource folds placement into source identity (ADR-0018)", () => {
  const bytes = () => new Float32Array([1, 2, 3, 4]);
  const shape = { kind: "grid", width: 2, height: 2 } as const;

  it("identical bytes + DIFFERENT placement ⇒ different key (no false memo hit)", () => {
    const a: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([0, 0, 0], 1, 1)) };
    const b: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([10, 20, 0], 0.5, 0.5)) };
    expect(hashSource(a)).not.toBe(hashSource(b));
  });

  it("identical bytes + DIFFERENT system ⇒ different key", () => {
    const a: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([0, 0, 0], 1, 1)) };
    const b: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("tissue", affine([0, 0, 0], 1, 1)) };
    expect(hashSource(a)).not.toBe(hashSource(b));
  });

  it("identical bytes + SAME placement ⇒ same key (still a hit)", () => {
    const a: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([1, 2, 3], 0.5, 0.5)) };
    const b: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([1, 2, 3], 0.5, 0.5)) };
    expect(hashSource(a)).toBe(hashSource(b));
  });

  it("absent placement keys exactly as it did before folding placement in (additive)", () => {
    // A placed source and an unplaced one with the same bytes are different fields.
    const unplaced: FieldValue = { shape, dtype: "f32", data: bytes() };
    const placed: FieldValue = { shape, dtype: "f32", data: bytes(), placement: placementIn("global", affine([0, 0, 0], 1, 1)) };
    expect(hashSource(unplaced)).not.toBe(hashSource(placed));
    // Two unplaced sources with identical bytes still collide (memo hit preserved).
    const unplaced2: FieldValue = { shape, dtype: "f32", data: bytes() };
    expect(hashSource(unplaced)).toBe(hashSource(unplaced2));
  });
});
