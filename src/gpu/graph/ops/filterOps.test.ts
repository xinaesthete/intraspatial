// The filter ops, and — more importantly — the algebra they implement.
//
// Most of what is worth testing here is not "does maskRange select the right rows" but
// "are these the right operators": the De Morgan and idempotence properties that made
// ADR-0005's original `AND = a·b` with `OR = max` wrong, and that only bite once a filter
// list becomes a filter graph.
import { describe, expect, it } from "vitest";
import type { FieldValue } from "../handle";
import type { OpType, Params } from "../op";
import { invertFieldOp, maxFieldsOp, minFieldsOp } from "./fieldArithmetic";
import { maskCountOp, maskEqualsOp, maskedDensityOp, maskRangeOp } from "./filterOps";

/** A per-row column / mask: `points{n}` carrying exactly n values. */
const col = (...v: number[]): FieldValue => ({ shape: { kind: "points", n: v.length }, dtype: "f32", data: Float32Array.from(v) });

/** An (x,y) cloud: `points{n}` carrying 2n values. Same shape, different lane count. */
const cloud = (xy: number[]): FieldValue => ({ shape: { kind: "points", n: xy.length / 2 }, dtype: "f32", data: Float32Array.from(xy) });

const run = (op: OpType, inputs: FieldValue[], params: Params = {}): FieldValue[] => op.cpuGolden!(inputs, params);
const out = (op: OpType, inputs: FieldValue[], params: Params = {}): number[] => Array.from(run(op, inputs, params)[0]!.data!);

describe("maskRange", () => {
  it("is a boxcar when softness is 0", () => {
    const v = col(-1, 0, 5, 10, 11);
    expect(out(maskRangeOp, [v], { lo: 0, hi: 10, softness: 0 })).toEqual([0, 1, 1, 1, 0]);
  });

  it("ramps at each edge when softness is set, reaching exactly 0 and 1", () => {
    // Ramp half-width 1 at lo=0 means: 0 at x=-1, 1/2 at x=0, 1 at x=+1.
    const v = col(-2, -1, 0, 1, 5);
    const m = out(maskRangeOp, [v], { lo: 0, hi: 10, softness: 1 });
    expect(m[0]).toBe(0); // fully outside, exactly — not a logistic tail
    expect(m[1]).toBe(0);
    expect(m[2]).toBeCloseTo(0.5, 6);
    expect(m[3]).toBe(1); // fully inside, exactly
    expect(m[4]).toBe(1);
  });

  it("is monotone across the rising edge", () => {
    const xs = Array.from({ length: 21 }, (_, i) => -2 + i * 0.2);
    const m = out(maskRangeOp, [col(...xs)], { lo: 0, hi: 100, softness: 1 });
    let drops = 0;
    for (let i = 1; i < m.length; i++) if (m[i]! < m[i - 1]! - 1e-7) drops++;
    expect(drops).toBe(0);
  });

  it("refuses an (x,y) cloud where a column belongs", () => {
    // Same declared shape as a 2-row column, twice the data. Without the length check this
    // would silently read coordinates as values.
    expect(() => run(maskRangeOp, [cloud([0, 0, 1, 1])], { lo: 0, hi: 1, softness: 0 })).toThrow(/point cloud, not a column/);
  });
});

describe("maskEquals", () => {
  it("selects one category code", () => {
    expect(out(maskEqualsOp, [col(0, 1, 2, 1)], { code: 1, tolerance: 0.5 })).toEqual([0, 1, 0, 1]);
  });

  it("tolerates a code that arrived through f32", () => {
    // 3 stored as 2.9999998 is the realistic case; an === would select nothing and the
    // failure would look like "that category is empty".
    expect(out(maskEqualsOp, [col(2.9999998, 4)], { code: 3, tolerance: 0.5 })).toEqual([1, 0]);
  });
});

// The reason the operator set was corrected. These properties are what a filter GRAPH needs
// and a filter LIST never did — MDV's flat conjunction cannot contain a diamond, so it never
// had to ask whether AND was idempotent.
describe("the fuzzy-set operators", () => {
  const a = col(0.8, 0.3, 1, 0);
  const b = col(0.6, 0.9, 0, 0);

  const AND = (x: FieldValue, y: FieldValue) => run(minFieldsOp, [x, y])[0]!;
  const OR = (x: FieldValue, y: FieldValue) => run(maxFieldsOp, [x, y])[0]!;
  const NOT = (x: FieldValue) => run(invertFieldOp, [x])[0]!;
  const vals = (f: FieldValue) => Array.from(f.data!);

  it("AND is idempotent, which the product t-norm is not", () => {
    // The diamond case: one gate reaching a join by two paths must not be counted twice.
    expect(vals(AND(a, a))).toEqual(vals(a));
    // What the ADR's original operator would have given, for contrast — the mask silently dims.
    const product = Array.from(a.data!, (x) => x * x);
    expect(product[0]).toBeCloseTo(0.64, 6);
    expect(vals(a)[0]).toBeCloseTo(0.8, 6);
  });

  it("satisfies De Morgan, which the ADR's mixed pair did not", () => {
    // NOT(NOT a AND NOT b) === a OR b
    const lhs = vals(NOT(AND(NOT(a), NOT(b))));
    const rhs = vals(OR(a, b));
    for (let i = 0; i < lhs.length; i++) expect(lhs[i]).toBeCloseTo(rhs[i]!, 6);
    // The mixed pair disagrees at exactly the values quoted in the design note.
    expect(1 - (1 - 0.8) * (1 - 0.6)).toBeCloseTo(0.92, 6);
    expect(Math.max(0.8, 0.6)).toBeCloseTo(0.8, 6);
  });

  it("NOT is involutive", () => {
    expect(vals(NOT(NOT(a)))).toEqual(vals(a));
  });

  it("agrees with every other t-norm on hard 0/1 masks", () => {
    // Why the defect survived review: invisible until masks are soft, and soft membership is
    // the entire reason the byte was widened to a weight.
    const h1 = col(1, 0, 1, 0);
    const h2 = col(1, 1, 0, 0);
    const product = Array.from(h1.data!, (x, i) => x * h2.data![i]!);
    expect(vals(AND(h1, h2))).toEqual(product);
  });

  it("expresses set difference as min(a, 1 - b)", () => {
    // "cells in gate A but not gate B" — a diff, not a re-brush.
    expect(vals(AND(col(1, 1, 0), NOT(col(1, 0, 0))))).toEqual([0, 1, 0]);
  });
});

describe("maskCount", () => {
  it("counts rows kept and total weight, which diverge once the mask is soft", () => {
    const [count, weight] = run(maskCountOp, [col(1, 0.75, 0.25, 0)]);
    expect(count!.data![0]).toBe(2); // > 1/2
    expect(weight!.data![0]).toBeCloseTo(2, 6);
    // A hard mask makes them agree, which is what makes the pair a useful readout: they
    // separate exactly when the selection has stopped being crisp.
    const [c2, w2] = run(maskCountOp, [col(1, 1, 0, 0)]);
    expect(c2!.data![0]).toBe(2);
    expect(w2!.data![0]).toBe(2);
  });
});

describe("maskedDensity", () => {
  const pts = cloud([0, 0, 10, 0, 0, 10, 10, 10]);
  const P = { width: 16, height: 16, sigma: 2, radiusSigma: 4 };
  const total = (f: FieldValue) => Array.from(f.data!).reduce((s, x) => s + x, 0);

  it("an all-zero mask leaves an empty field", () => {
    expect(total(run(maskedDensityOp, [pts, col(0, 0, 0, 0)], P)[0]!)).toBe(0);
  });

  it("mass scales with the selection", () => {
    const all = total(run(maskedDensityOp, [pts, col(1, 1, 1, 1)], P)[0]!);
    const half = total(run(maskedDensityOp, [pts, col(1, 1, 0, 0)], P)[0]!);
    expect(all).toBeGreaterThan(0);
    // Four symmetric corners in a symmetric bbox, so two of them carry half the mass.
    expect(half / all).toBeCloseTo(0.5, 2);
  });

  it("a soft weight contributes a partial kernel, not a whole one or none", () => {
    const hard = total(run(maskedDensityOp, [pts, col(1, 0, 0, 0)], P)[0]!);
    const soft = total(run(maskedDensityOp, [pts, col(0.25, 0, 0, 0)], P)[0]!);
    expect(soft / hard).toBeCloseTo(0.25, 4);
  });

  it("refuses a mask that does not belong to the cloud", () => {
    expect(() => run(maskedDensityOp, [pts, col(1, 1)], P)).toThrow(/belongs to the rows of one table/);
  });
});
