// Composable procedural geometry-ops (ADR-0010) — the horn-grammar-first, Todd–Latham-lineage
// Swept catalogue, re-derived as a lazy typed expression-IR that lowers to a CPU golden and a
// codegen'd TGSL/WGSL per-vertex kernel. See `CONTEXT.md` for the ubiquitous language.

export type { Angle, AngleUnit } from "./angle";
export { deg, isAngle, rad, turns, unitToRadians } from "./angle";
export type { AngleLike } from "./angle-like";
export type { Expr, ExprLike } from "./expr";
export { add, collectSpecs, constant, evalExpr, linear, mul, ramp, S, sub, THETA, toExpr, wgslExpr } from "./expr";
export type { Mat4 } from "./placement";
export { applyNormal, applyPoint } from "./placement";
export type { BranchOptions, StackOptions } from "./structured";
export { branchPlacements, Structured, stackPlacements } from "./structured";
export type { Catalogue, HornConfig, Mesh, TessellateOptions, ToMeshOptions, Transform } from "./swept";
export { catalogue, FRAME_H, gridIndices, gridSampleAngles, horn, Swept, sweptShaderWgsl } from "./swept";

/** A Geometry value — a Swept leaf or a Structured assembly of instances (ADR-0010). */
export type Geometry = import("./swept").Swept | import("./structured").Structured;
