import { describe, expect, it } from "vitest";
import type { ElementType, FieldValue } from "./handle";
import { FieldOnePole, OnePole } from "./onePole";

const at = (a: ArrayLike<number> | undefined, i: number): number => a?.[i] ?? Number.NaN;

describe("OnePole", () => {
  it("snaps to the first target by default", () => {
    const f = new OnePole(1, { tau: 5 });
    expect(at(f.push([3]), 0)).toBeCloseTo(3, 12); // first push jumps, no ramp from zero
  });

  it("ramps from zero with the exact one-pole step when snapFirst is off", () => {
    const f = new OnePole(1, { tau: 1, snapFirst: false });
    const a = 1 - Math.exp(-1); // α for dt=1, τ=1 ≈ 0.632
    expect(at(f.push([1]), 0)).toBeCloseTo(a, 6); // 0 → α (f32 state, so ~ppm)
    expect(at(f.push([1]), 0)).toBeCloseTo(1 - Math.exp(-2), 6); // cumulative: 1 − e^-2
    expect(at(f.push([1]), 0)).toBeCloseTo(1 - Math.exp(-3), 6);
  });

  it("converges to a held target", () => {
    const f = new OnePole(1, { tau: 3, snapFirst: false });
    for (let i = 0; i < 200; i++) f.push([2]);
    expect(at(f.value(), 0)).toBeCloseTo(2, 6);
  });

  it("larger τ eases more slowly", () => {
    const fast = new OnePole(1, { tau: 1, snapFirst: false });
    const slow = new OnePole(1, { tau: 10, snapFirst: false });
    fast.push([1]);
    slow.push([1]);
    expect(at(fast.value(), 0)).toBeGreaterThan(at(slow.value(), 0)); // fast nearer the target after one step
  });

  it("composes over dt (one big step == several small steps)", () => {
    const big = new OnePole(1, { tau: 4, snapFirst: false });
    const small = new OnePole(1, { tau: 4, snapFirst: false });
    big.push([1], 3);
    small.push([1], 1);
    small.push([1], 1);
    small.push([1], 1);
    expect(at(big.value(), 0)).toBeCloseTo(at(small.value(), 0), 12); // 1 − e^-(3/4) either way
  });

  it("starts primed from an initial state", () => {
    const f = new OnePole(1, { tau: 1, initial: [0.5] });
    const a = 1 - Math.exp(-1);
    expect(at(f.push([1]), 0)).toBeCloseTo(0.5 + a * 0.5, 6); // filters from 0.5, no snap
  });

  it("smooths each lane independently", () => {
    const f = new OnePole(3, { tau: 2, snapFirst: false });
    const out = f.push([1, -1, 4]);
    const a = 1 - Math.exp(-0.5);
    expect(at(out, 0)).toBeCloseTo(a * 1, 6);
    expect(at(out, 1)).toBeCloseTo(a * -1, 6);
    expect(at(out, 2)).toBeCloseTo(a * 4, 6);
  });

  it("reset jumps to a value or re-arms the snap", () => {
    const f = new OnePole(1, { tau: 1 });
    f.push([1]);
    f.reset([9]);
    expect(at(f.value(), 0)).toBe(9);
    f.reset(); // re-arm: next push snaps again
    expect(at(f.push([2]), 0)).toBe(2);
  });

  it("validates construction and push", () => {
    expect(() => new OnePole(0, { tau: 1 })).toThrow(/frameLength/);
    expect(() => new OnePole(1, { tau: 0 })).toThrow(/tau/);
    expect(() => new OnePole(2, { tau: 1 }).push([1])).toThrow(/length/);
  });

  it("reports its resident bytes", () => {
    expect(new OnePole(4, { tau: 1 }).byteLength).toBe(16); // 4 × f32
  });
});

describe("FieldOnePole", () => {
  const VEC2: ElementType = { kind: "vec", n: 2 };
  const points2 = (data: number[]): FieldValue => ({
    shape: { kind: "points", n: data.length / 2 },
    dtype: "f32",
    element: VEC2,
    data: Float32Array.from(data),
  });

  it("eases a point cloud toward a target field", () => {
    const f = new FieldOnePole({ kind: "points", n: 2 }, { tau: 2, snapFirst: false }, VEC2);
    const target = points2([1, 0, 1, 0]); // 2 points × vec2 = 4 lanes
    const out = f.push(target);
    const a = 1 - Math.exp(-0.5);
    expect(at(out.data, 0)).toBeCloseTo(a, 6); // first point's x eased from 0
    expect(out.shape.kind).toBe("points");
    for (let i = 0; i < 200; i++) f.push(target);
    expect(at(f.value().data, 0)).toBeCloseTo(1, 6); // converges
  });
});
