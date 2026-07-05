// Public entry point for the in-GPU operation graph runtime. Importing this
// registers the built-in ops and re-exports the core API: build a `Graph`, wire
// ops into it, and `pull` a sink to execute the minimal required subgraph.
//
// See docs/gpu-resource-sync.md for the model this implements.

export type { GpuBackend, Root } from "./backend";
export { nodeBackend } from "./backend.node";
export type { AdvanceOptions, PullOptions, SimState } from "./executor";
export { advance, createSimState, pull, pullData, simStateBytes } from "./executor";
export type { DelayHandle, EdgeRef, FeedbackHandle, GraphNode } from "./graph";
export { Graph } from "./graph";
export type { Basis, Dtype, ElementType, FieldValue, GpuField, NodeId, Shape, ShapeKind } from "./handle";
export {
  basisLabel,
  basisOf,
  elementLabel,
  elementLanes,
  elementOf,
  elementsEqual,
  numCells,
  SCALAR,
  SPATIAL,
  shapesEqual,
} from "./handle";
export type { GraphMemo } from "./memo";
export { createMemo } from "./memo";
export type { MemoryReporting } from "./memory";
export { dtypeBytes, fieldBytes, fieldValueBytes, formatBytes, memoryBytes } from "./memory";
export { FieldOnePole, OnePole, type OnePoleOptions } from "./onePole";
export type { ExecCtx, OpHelp, OpType, ParamSpec, Params, ParamType, PortSpec } from "./op";
export { allFinite, defaultParams, param } from "./op";
export { registerBuiltinOps, registerElementOps, registerWaveletOps } from "./ops/index";
export { getOp, hasOp, listOps, registerOp } from "./registry";
export { FieldRing, RingBuffer } from "./ringBuffer";

// Side-effect import: ensure built-in ops are registered when this entry is used.
import "./ops/index";
