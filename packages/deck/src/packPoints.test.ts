import { describe, expect, it } from "vitest";
import { packPoints } from "./packPoints";

describe("packPoints", () => {
  it("packs [x, y, w] with weight 1 by default", () => {
    const { xyw, n, bounds } = packPoints([1, 2], [3, 4]);
    expect(n).toBe(2);
    expect([...xyw]).toEqual([1, 3, 1, 2, 4, 1]);
    expect(bounds).toEqual([1, 3, 2, 4]);
  });

  it("treats a non-finite weight as 0 — one NaN would poison its whole kernel footprint", () => {
    const { xyw } = packPoints([1, 2, 3], [1, 2, 3], [0.5, Number.NaN, Number.POSITIVE_INFINITY]);
    expect([xyw[2], xyw[5], xyw[8]]).toEqual([0.5, 0, 0]);
  });

  it("leaves out points with non-finite coordinates, and keeps them out of the bounds", () => {
    const { xyw, bounds } = packPoints([0, Number.POSITIVE_INFINITY, 10], [0, 5, Number.NaN]);
    expect(xyw[5]).toBe(0);
    expect(xyw[8]).toBe(0);
    expect([xyw[3], xyw[4], xyw[6], xyw[7]].every(Number.isFinite)).toBe(true);
    expect(bounds).toEqual([0, 0, 0, 0]);
  });

  it("bounds only the points that contribute", () => {
    const { bounds } = packPoints([0, 100, 5], [0, 100, 6], [0, 0, 1]);
    expect(bounds).toEqual([5, 6, 5, 6]);
  });

  it("has no bounds when nothing contributes", () => {
    expect(packPoints([1, 2], [1, 2], [0, Number.NaN]).bounds).toBeNull();
    expect(packPoints([], []).bounds).toBeNull();
  });
});
