import { describe, expect, it } from "vitest";
import { add, addScaled, clampLength, cross, dot, length, normalize, readVec3, scale, sub, writeVec3, ZERO3 } from "./vec3";

describe("vec3", () => {
  it("does the basic algebra", () => {
    expect(add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
    expect(sub([4, 5, 6], [1, 2, 3])).toEqual([3, 3, 3]);
    expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]);
    expect(addScaled([1, 1, 1], [2, 0, 0], 3)).toEqual([7, 1, 1]);
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(length([3, 4, 0])).toBe(5);
  });

  it("normalizes, with a safe fallback for the zero vector", () => {
    const n = normalize([3, 4, 0]);
    expect(length(n)).toBeCloseTo(1, 12);
    expect(normalize(ZERO3)).toEqual(ZERO3); // no NaN
    expect(normalize(ZERO3, 1e-9, [0, 1, 0])).toEqual([0, 1, 0]);
  });

  it("clamps length", () => {
    expect(length(clampLength([10, 0, 0], 2))).toBeCloseTo(2, 12);
    expect(clampLength([1, 0, 0], 2)).toEqual([1, 0, 0]); // shorter untouched
  });

  it("reads/writes flat buffers with bounds checks (no silent undefined)", () => {
    const buf = new Float32Array(9);
    writeVec3(buf, 1, [7, 8, 9]);
    expect(Array.from(readVec3(buf, 1))).toEqual([7, 8, 9]);
    expect(() => readVec3(buf, 3)).toThrow(/out of range/);
    expect(() => writeVec3(buf, 3, [1, 2, 3])).toThrow(/out of range/);
  });
});
