// View-driven multiscale datasource — the pure core (Milestone 0 of ADR-0008).
// Self-contained: no dependency on the dancer/evo/sim code, so it can graduate.

// Re-export the shared memory-reporting utilities (the datasource already builds on the
// op-graph's field model) so consumers report resident memory through one interface.
export { fieldBytes, formatBytes, type MemoryReporting, memoryBytes } from "../gpu/graph/memory";
export * from "./imageFacets";
export * from "./math";
export * from "./multiscale";
export * from "./select";
export * from "./syntheticLoader";
export * from "./tileCache";
export * from "./tileToField";
export * from "./types";
