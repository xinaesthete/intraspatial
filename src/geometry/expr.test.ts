import { describe, expect, it } from "vitest";
import type { ParamSpec } from "../gpu/graph/op";
import {
  add,
  collectConsts,
  collectSpecs,
  constant,
  evalExpr,
  linear,
  mul,
  ramp,
  S,
  sub,
  THETA,
  toExpr,
  wgslExpr,
  wgslExprUniform,
  wgslFloat,
} from "./expr";

describe("Param-expression evaluation", () => {
  it("evaluates the free variables and constants", () => {
    expect(evalExpr(constant(3), 0.5, 1)).toBe(3);
    expect(evalExpr(S, 0.5, 1)).toBe(0.5);
    expect(evalExpr(THETA, 0.5, 1)).toBe(1);
  });

  it("evaluates arithmetic", () => {
    const e = add(mul(S, 2), sub(THETA, 1));
    expect(evalExpr(e, 0.5, 3)).toBe(0.5 * 2 + (3 - 1));
  });

  it("ramp is a 0 → peak progression along s", () => {
    const e = ramp(360);
    expect(evalExpr(e, 0, 0)).toBe(0);
    expect(evalExpr(e, 0.25, 0)).toBe(90);
    expect(evalExpr(e, 1, 0)).toBe(360);
  });

  it("linear interpolates a → b along s", () => {
    const e = linear(10, 20);
    expect(evalExpr(e, 0, 0)).toBe(10);
    expect(evalExpr(e, 0.5, 0)).toBe(15);
    expect(evalExpr(e, 1, 0)).toBe(20);
  });

  it("toExpr coerces a bare number to a constant", () => {
    expect(evalExpr(toExpr(7), 0, 0)).toBe(7);
    expect(evalExpr(toExpr(S), 0.3, 0)).toBe(0.3);
  });
});

describe("WGSL codegen mirrors the tree", () => {
  it("formats floats with a decimal point", () => {
    expect(wgslFloat(2)).toBe("2.0");
    expect(wgslFloat(-1.5)).toBe("-1.5");
    expect(wgslFloat(0.5)).toBe("0.5");
  });

  it("lowers the free variables to the shader locals s / th", () => {
    expect(wgslExpr(S)).toBe("s");
    expect(wgslExpr(THETA)).toBe("th");
  });

  it("lowers ramp and linear structurally", () => {
    expect(wgslExpr(ramp(360))).toBe("(360.0 * s)");
    expect(wgslExpr(linear(0, 360))).toBe("(0.0 + (360.0 * s))");
  });
});

describe("uniform codegen (structure/value split)", () => {
  it("reads constants from P[] in left-to-right order; collectConsts supplies the values", () => {
    const e = add(mul(constant(2), S), sub(THETA, constant(5)));
    const ctx = { next: 0 };
    expect(wgslExprUniform(e, ctx)).toBe("((P[0u] * s) + (th - P[1u]))");
    expect(ctx.next).toBe(2);
    expect(collectConsts(e, [])).toEqual([2, 5]);
  });

  it("same structure, different values → identical WGSL", () => {
    const shape = (peak: number) => wgslExprUniform(ramp(peak), { next: 0 });
    expect(shape(360)).toBe(shape(720));
    expect(shape(360)).toBe("(P[0u] * s)");
  });
});

describe("breeding genes", () => {
  it("collects the ParamSpecs its literals carry, in order", () => {
    const rSpec: ParamSpec = { name: "radius", type: "number", default: 3, min: 0, max: 10 };
    const tSpec: ParamSpec = { name: "twist", type: "number", default: 360, min: 0, max: 720 };
    const e = add(ramp(3, rSpec), constant(360, tSpec));
    expect(collectSpecs(e).map((s) => s.name)).toEqual(["radius", "twist"]);
  });

  it("a plain literal carries no gene", () => {
    expect(collectSpecs(constant(1))).toEqual([]);
  });
});
