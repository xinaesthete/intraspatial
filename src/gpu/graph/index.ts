// Public entry point for the in-GPU operation graph runtime. Importing this
// registers the built-in ops and re-exports the core API: build a `Graph`, wire
// ops into it, and `pull` a sink to execute the minimal required subgraph.
//
// See docs/gpu-resource-sync.md for the model this implements.
export type { Dtype, ElementType, FieldValue, GpuField, NodeId, Shape, ShapeKind } from "./handle";
export { elementLabel, elementLanes, elementOf, elementsEqual, numCells, SCALAR, shapesEqual } from "./handle";
export type { ExecCtx, OpType, Params, ParamSpec, ParamType, PortSpec } from "./op";
export { allFinite, defaultParams, param } from "./op";
export type { EdgeRef, FeedbackHandle, GraphNode } from "./graph";
export { Graph } from "./graph";
export { getOp, hasOp, listOps, registerOp } from "./registry";
export type { GpuBackend, Root } from "./backend";
export { nodeBackend } from "./backend.node";
export type { AdvanceOptions, PullOptions, SimState } from "./executor";
export { advance, createSimState, pull, pullData } from "./executor";
export type { GraphMemo } from "./memo";
export { createMemo } from "./memo";
export { registerBuiltinOps, registerElementOps, registerWaveletOps } from "./ops/index";

// Side-effect import: ensure built-in ops are registered when this entry is used.
import "./ops/index";
