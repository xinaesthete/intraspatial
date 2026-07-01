// Public entry point for the in-GPU operation graph runtime. Importing this
// registers the built-in ops and re-exports the core API: build a `Graph`, wire
// ops into it, and `pull` a sink to execute the minimal required subgraph.
//
// See docs/gpu-resource-sync.md for the model this implements.
export type { Basis, Dtype, ElementType, FieldValue, GpuField, NodeId, Shape, ShapeKind } from "./handle";
export { basisLabel, basisOf, elementLabel, elementLanes, elementOf, elementsEqual, numCells, SCALAR, SPATIAL, shapesEqual } from "./handle";
export type { ExecCtx, OpHelp, OpType, Params, ParamSpec, ParamType, PortSpec } from "./op";
export { allFinite, defaultParams, param } from "./op";
export type { DelayHandle, EdgeRef, FeedbackHandle, GraphNode } from "./graph";
export { Graph } from "./graph";
export { getOp, hasOp, listOps, registerOp } from "./registry";
export type { GpuBackend, Root } from "./backend";
export { nodeBackend } from "./backend.node";
export type { AdvanceOptions, PullOptions, SimState } from "./executor";
export { advance, createSimState, pull, pullData, simStateBytes } from "./executor";
export { FieldRing, RingBuffer } from "./ringBuffer";
export type { MemoryReporting } from "./memory";
export { dtypeBytes, fieldBytes, fieldValueBytes, formatBytes, memoryBytes } from "./memory";
export type { GraphMemo } from "./memo";
export { createMemo } from "./memo";
export { registerBuiltinOps, registerElementOps, registerWaveletOps } from "./ops/index";

// Side-effect import: ensure built-in ops are registered when this entry is used.
import "./ops/index";
