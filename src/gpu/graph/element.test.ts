/** biome-ignore-all lint/style/noNonNullAssertion: test noise */
// Element algebra (ADR-0004): the math kernels directly, plus the ops through the
// executor in CPU mode (no GPU — fast `*.test.ts`, runs under vitest.config.ts).
import { beforeAll, describe, expect, it } from "vitest";
import type { ElementType, GpuField } from "./index";
import { Graph, pull, registerElementOps } from "./index";

beforeAll(async () => {
  await registerElementOps();
});

import { addFields, conjugate, crossFields, dotFields, magnitude, mulFields, normalize, packComplex } from "./elementMath";

const COMPLEX: ElementType = { kind: "complex" };
const QUAT: ElementType = { kind: "quaternion" };
const VEC3: ElementType = { kind: "vec", n: 3 };

const near = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) < eps;

describe("elementMath", () => {
  it("complex multiply matches (a+bi)(c+di)", () => {
    // (1+2i)(3+4i) = 3 +4i +6i +8i² = -5 +10i ; (0+1i)(0+1i) = -1
    const a = Float32Array.of(1, 2, 0, 1);
    const b = Float32Array.of(3, 4, 0, 1);
    const out = mulFields(COMPLEX, a, b);
    expect([...out]).toEqual([-5, 10, -1, 0]);
  });

  it("complex conjugate negates the imaginary lane; |z| is the modulus", () => {
    const z = Float32Array.of(3, -4);
    expect([...conjugate(COMPLEX, z)]).toEqual([3, 4]);
    expect([...magnitude(COMPLEX, z)]).toEqual([5]); // hypot(3,4)
  });

  it("quaternion product is the Hamilton product (i·j = k)", () => {
    // i = (0,1,0,0), j = (0,0,1,0) → k = (0,0,0,1)
    const i = Float32Array.of(0, 1, 0, 0);
    const j = Float32Array.of(0, 0, 1, 0);
    expect([...mulFields(QUAT, i, j)]).toEqual([0, 0, 0, 1]);
    // q·q* = |q|²  (real), here |q|² = 1+4+9+16 = 30
    const q = Float32Array.of(1, 2, 3, 4);
    const r = mulFields(QUAT, q, conjugate(QUAT, q));
    expect(near(r[0]!, 30)).toBe(true);
    expect(near(r[1]!, 0) && near(r[2]!, 0) && near(r[3]!, 0)).toBe(true);
  });

  it("vec dot, cross, normalize", () => {
    const a = Float32Array.of(1, 0, 0, 1, 2, 2); // two vec3 samples
    const b = Float32Array.of(0, 1, 0, 3, 0, 0);
    expect([...dotFields(VEC3, a, b)]).toEqual([0, 3]); // ⟂, then 1·3
    expect([...crossFields(VEC3, Float32Array.of(1, 0, 0), Float32Array.of(0, 1, 0))]).toEqual([0, 0, 1]);
    const n = normalize(VEC3, Float32Array.of(0, 3, 4));
    expect(near(n[1]!, 0.6) && near(n[2]!, 0.8)).toBe(true);
  });

  it("add is lane-wise for any element; mul rejects vec", () => {
    expect([...addFields(Float32Array.of(1, 2), Float32Array.of(3, 4))]).toEqual([4, 6]);
    expect(() => mulFields(VEC3, Float32Array.of(1, 2, 3), Float32Array.of(1, 2, 3))).toThrow(/no algebra product/);
  });
});

// A vec3 grid source built by hand (the builder's `grid` makes scalar fields).
function vec3Grid(g: Graph, samples: number[][]): GpuField {
  const data = new Float32Array(samples.length * 3);
  samples.forEach((s, i) => {
    data.set(s, i * 3);
  });
  return g.source({ shape: { kind: "grid", width: samples.length, height: 1 }, dtype: "f32", element: VEC3, data });
}

describe("element ops through the executor (CPU mode)", () => {
  it("complex(re, im) · complex(re, im) then magnitude, end to end", async () => {
    const g = new Graph();
    const re = g.grid(Float32Array.of(1, 0), 2, 1);
    const im = g.grid(Float32Array.of(2, 1), 2, 1);
    const z = g.op1("complex", { re, im }); // [1+2i, 0+1i]
    expect(z.element).toEqual(COMPLEX);
    const z2 = g.op1("mulFields", { a: z, b: z }); // [(1+2i)², (0+1i)²] = [-3+4i, -1+0i]
    const mag = g.op1("magnitude", { in: z2 }); // [|−3+4i|, |−1|] = [5, 1]
    expect(mag.element).toEqual({ kind: "scalar" });

    const out = await pull(g, mag, { mode: "cpu" });
    expect([...out.data!]).toEqual([5, 1]);
    // and the intermediate complex value packs interleaved as expected
    const zv = await pull(g, z, { mode: "cpu" });
    expect([...zv.data!]).toEqual([...packComplex(Float32Array.of(1, 0), Float32Array.of(2, 1))]);
  });

  it("vec3 dot through the graph yields a scalar field", async () => {
    const g = new Graph();
    const a = vec3Grid(g, [
      [1, 0, 0],
      [1, 2, 2],
    ]);
    const b = vec3Grid(g, [
      [0, 1, 0],
      [3, 0, 0],
    ]);
    const d = g.op1("dotFields", { a, b });
    expect(d.element).toEqual({ kind: "scalar" });
    const out = await pull(g, d, { mode: "cpu" });
    expect([...out.data!]).toEqual([0, 3]);
  });
});

describe("build-time element type-checking (inferElements)", () => {
  it("rejects a complex accessor on a scalar field", () => {
    const g = new Graph();
    const s = g.grid(Float32Array.of(1, 2, 3, 4), 2, 2);
    expect(() => g.op1("realPart", { in: s })).toThrow(/requires complex/);
  });

  it("rejects mulFields on vec (no algebra product) and element mismatch on add", () => {
    const g = new Graph();
    const v = vec3Grid(g, [[1, 2, 3]]);
    expect(() => g.op1("mulFields", { a: v, b: v })).toThrow(/no algebra product/);

    const re = g.grid(Float32Array.of(1), 1, 1);
    const im = g.grid(Float32Array.of(1), 1, 1);
    const z = g.op1("complex", { re, im });
    const s = g.grid(Float32Array.of(1), 1, 1);
    expect(() => g.op1("addFields", { a: z, b: s })).toThrow(/element mismatch/);
  });
});
