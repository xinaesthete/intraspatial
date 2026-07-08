// Param-expression IR (ADR-0010, ADR-0007) — the typed, lazy expression a geometry-op
// parameter is. A Param-expression is a pure function of the free-variable environment
// `{s, θ}` (the Sweep and Spoke coordinates); it never does I/O and never awaits. A bare
// scalar is a **constant**; any progression along the sweep is a **named expression**
// (`ramp`, `linear`). The `s`/`θ`-dependence is therefore always explicit — there is no
// hidden op-baked ramp.
//
// One IR, lowered two ways from the *same* tree so they agree by construction (the repo's
// CPU-golden == GPU-kernel discipline, ADR-0003):
//   • `evalExpr(e, s, θ)`  — the CPU evaluator (the golden).
//   • `wgslExpr(e)`        — a WGSL source fragment over the shader locals `s`, `th`.
//
// Numeric literals may carry a `ParamSpec` (a *gene*, in the Todd–Latham FormGrow lineage):
// the breeding metadata `[name, default, min, max, step]` the Mutator (`src/evo`) enumerates.
// `collectSpecs` gathers them into the Geometry's breeding surface.

import type { ParamSpec } from "../gpu/graph/op";

/** A Param-expression node over the free variables `{s, θ}`. A small typed DAG (ADR-0007);
 *  grows additively as ops need more (the `{globals, sim-channels, data}` environment is
 *  designed-for, not built). */
export type Expr =
  | { kind: "const"; value: number; spec?: ParamSpec }
  | { kind: "s" }
  | { kind: "theta" }
  | { kind: "add"; a: Expr; b: Expr }
  | { kind: "sub"; a: Expr; b: Expr }
  | { kind: "mul"; a: Expr; b: Expr };

/** Anything a geometry-op will accept where a Param-expression is wanted: a bare number
 *  (a constant) or an already-built `Expr`. */
export type ExprLike = number | Expr;

/** The Sweep coordinate `s ∈ [0, 1]` (base → tip) as an expression. */
export const S: Expr = { kind: "s" };
/** The Spoke coordinate `θ ∈ [0, 2π)` (around the profile) as an expression. */
export const THETA: Expr = { kind: "theta" };

/** A constant Param-expression. `spec` attaches breeding metadata (a gene) to the literal. */
export function constant(value: number, spec?: ParamSpec): Expr {
  return spec ? { kind: "const", value, spec } : { kind: "const", value };
}

/** Coerce an {@link ExprLike} to an `Expr` (a bare number becomes a constant). */
export function toExpr(x: ExprLike): Expr {
  return typeof x === "number" ? constant(x) : x;
}

export function add(a: ExprLike, b: ExprLike): Expr {
  return { kind: "add", a: toExpr(a), b: toExpr(b) };
}
export function sub(a: ExprLike, b: ExprLike): Expr {
  return { kind: "sub", a: toExpr(a), b: toExpr(b) };
}
export function mul(a: ExprLike, b: ExprLike): Expr {
  return { kind: "mul", a: toExpr(a), b: toExpr(b) };
}

/** A linear progression `0 → peak` along the sweep — `peak · s`. Where "twist/taper harder
 *  toward the tip" lives, as a named expression rather than baked into an op. `spec` breeds
 *  the peak. */
export function ramp(peak: number, spec?: ParamSpec): Expr {
  return mul(constant(peak, spec), S);
}

/** A linear progression `a → b` along the sweep — `a + (b − a) · s`. */
export function linear(a: number, b: number): Expr {
  return add(constant(a), mul(constant(b - a), S));
}

/** Evaluate a Param-expression at `(s, θ)` on the CPU — the golden reference. */
export function evalExpr(e: Expr, s: number, theta: number): number {
  switch (e.kind) {
    case "const":
      return e.value;
    case "s":
      return s;
    case "theta":
      return theta;
    case "add":
      return evalExpr(e.a, s, theta) + evalExpr(e.b, s, theta);
    case "sub":
      return evalExpr(e.a, s, theta) - evalExpr(e.b, s, theta);
    case "mul":
      return evalExpr(e.a, s, theta) * evalExpr(e.b, s, theta);
  }
}

/** Format a JS number as a WGSL f32 literal: always with a decimal point (or an exponent),
 *  so `2` becomes `2.0` and never a WGSL integer. */
export function wgslFloat(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`wgslFloat: non-finite literal ${x}`);
  const str = x.toString();
  return /[.eE]/.test(str) ? str : `${str}.0`;
}

/** Lower a Param-expression to a WGSL source fragment over the shader locals `s` and `th`.
 *  The mirror of {@link evalExpr}: the same tree, so the CPU and GPU values agree. */
export function wgslExpr(e: Expr): string {
  switch (e.kind) {
    case "const":
      return wgslFloat(e.value);
    case "s":
      return "s";
    case "theta":
      return "th";
    case "add":
      return `(${wgslExpr(e.a)} + ${wgslExpr(e.b)})`;
    case "sub":
      return `(${wgslExpr(e.a)} - ${wgslExpr(e.b)})`;
    case "mul":
      return `(${wgslExpr(e.a)} * ${wgslExpr(e.b)})`;
  }
}

/** Gather the `ParamSpec` genes carried by an expression's literals, in traversal order —
 *  the breeding surface the Mutator (`src/evo`) enumerates. */
export function collectSpecs(e: Expr, out: ParamSpec[] = []): ParamSpec[] {
  switch (e.kind) {
    case "const":
      if (e.spec) out.push(e.spec);
      break;
    case "s":
    case "theta":
      break;
    case "add":
    case "sub":
    case "mul":
      collectSpecs(e.a, out);
      collectSpecs(e.b, out);
      break;
  }
  return out;
}
