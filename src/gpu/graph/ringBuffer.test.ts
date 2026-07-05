import { describe, expect, it } from "vitest";
import type { FieldValue } from "./handle";
import { FieldRing, RingBuffer } from "./ringBuffer";

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

  it("interpolates continuously through the buffer (sample)", () => {
    const r = new RingBuffer(1, 4);
    r.push([0]);
    r.push([10]);
    r.push([20]); // frame(0)=20 (newest), frame(1)=10, frame(2)=0 (oldest)
    expect(r.sample(0)[0]).toBeCloseTo(20, 6); // newest
    expect(r.sample(2)[0]).toBeCloseTo(0, 6); // oldest
    expect(r.sample(0.5)[0]).toBeCloseTo(15, 6); // halfway between 20 and 10
    expect(r.sample(1.25)[0]).toBeCloseTo(7.5, 6); // 1/4 from 10 toward 0
    expect(r.sample(5)[0]).toBeCloseTo(0, 6); // clamps to oldest
    expect(() => new RingBuffer(1, 2).sample(0)).toThrow(/empty/);
  });

  it("interpolates cubically through the buffer (sampleCubic)", () => {
    const r = new RingBuffer(1, 6);
    // a linear ramp: Catmull-Rom reproduces linear data EXACTLY, so cubic == linear here
    for (const v of [0, 10, 20, 30, 40]) r.push([v]); // frame(0)=40 .. frame(4)=0
    expect(r.sampleCubic(0)[0]).toBeCloseTo(40, 6); // passes through knots
    expect(r.sampleCubic(2)[0]).toBeCloseTo(20, 6);
    expect(r.sampleCubic(1.5)[0]).toBeCloseTo(25, 6); // interior linear ⇒ exact midpoint
    expect(r.sampleCubic(2.25)[0]).toBeCloseTo(17.5, 6);
    expect(r.sampleCubic(9)[0]).toBeCloseTo(0, 6); // clamps to oldest

    // a non-linear sequence: cubic passes through the knots but curves between them
    const c = new RingBuffer(1, 6);
    for (const v of [0, 0, 1, 0, 0]) c.push([v]); // an impulse
    expect(c.sampleCubic(2)[0]).toBeCloseTo(1, 6); // knot exact (frame 2 back = the 1)
    expect(c.sampleCubic(1)[0]).toBeCloseTo(0, 6);
    // overshoot between knots is the Catmull-Rom signature (linear would stay within [0,1])
    const mid = c.sampleCubic(1.5)[0];
    expect(mid).toBeGreaterThan(0.5);
    expect(mid).toBeLessThan(0.75);
    expect(() => new RingBuffer(1, 2).sampleCubic(0)).toThrow(/empty/);
  });

  it("reports its resident byteLength", () => {
    const r = new RingBuffer(6, 10); // 60 floats × 4 bytes
    expect(r.byteLength).toBe(60 * 4);
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

  it("samples an interpolated grid field and reports byteLength", () => {
    const shape = { kind: "grid", width: 2, height: 1 } as const;
    const fr = new FieldRing(shape, 4);
    fr.push({ shape, dtype: "f32", data: new Float32Array([0, 0]) });
    fr.push({ shape, dtype: "f32", data: new Float32Array([4, 8]) });
    const mid = fr.sample(0.5); // halfway between newest [4,8] and prev [0,0]
    expect(Array.from(mid.data ?? [])).toEqual([2, 4]);
    expect(mid.shape).toEqual(shape);
    expect(fr.byteLength).toBe(2 * 4 * 4); // 2 cells · 4 frames · 4 bytes
  });
});
