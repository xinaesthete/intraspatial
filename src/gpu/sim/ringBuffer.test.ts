import { describe, expect, it } from "vitest";
import { FieldRing, RingBuffer } from "./ringBuffer";
import type { FieldValue } from "../graph/handle";

describe("RingBuffer", () => {
  it("stores up to capacity and indexes frames newest→oldest", () => {
    const r = new RingBuffer(2, 3);
    r.push([1, 1]);
    r.push([2, 2]);
    r.push([3, 3]);
    r.push([4, 4]); // evicts [1,1]
    expect(r.frames).toBe(3);
    expect(Array.from(r.frame(0))).toEqual([4, 4]); // newest
    expect(Array.from(r.frame(2))).toEqual([2, 2]); // oldest still held
    expect(() => r.frame(3)).toThrow(/out of range/);
  });

  it("rejects a short push and resets", () => {
    const r = new RingBuffer(3, 2);
    expect(() => r.push([1, 2])).toThrow(/frameLength/);
    r.push([1, 2, 3]);
    r.reset();
    expect(r.frames).toBe(0);
  });
});

describe("FieldRing — history of any field", () => {
  it("round-trips a grid field's frames (the point: it works on fields, not just points)", () => {
    const shape = { kind: "grid", width: 2, height: 2 } as const;
    const fr = new FieldRing(shape, 4); // scalar 2×2 grid, capacity 4
    const g = (v: number): FieldValue => ({ shape, dtype: "f32", data: new Float32Array([v, v + 1, v + 2, v + 3]) });
    fr.push(g(0));
    fr.push(g(10));
    fr.push(g(20));
    expect(fr.frames).toBe(3);
    const newest = fr.frame(0);
    expect(newest.shape).toEqual(shape);
    expect(Array.from(newest.data ?? [])).toEqual([20, 21, 22, 23]);
    expect(Array.from(fr.frame(2).data ?? [])).toEqual([0, 1, 2, 3]); // oldest
  });

  it("carries element lanes (a vec3 points field)", () => {
    const s = { kind: "points", n: 2 } as const;
    const fr = new FieldRing(s, 3, { kind: "vec", n: 3 });
    fr.push({ shape: s, dtype: "f32", element: { kind: "vec", n: 3 }, data: new Float32Array([0, 1, 2, 3, 4, 5]) });
    expect(fr.ring.frameLength).toBe(2 * 3); // n · lanes
    expect(Array.from(fr.frame(0).data ?? [])).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
