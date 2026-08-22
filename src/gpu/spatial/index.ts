// GPU spatial-statistics catalogue — neighbour distances, KDE splat, Getis-Ord, ANNI, cKNN, fuzzy adjacency, separable convolution, …
// Barrel for the `intraspatial/gpu/spatial` subpath export; deep imports (`intraspatial/gpu/spatial/<file>`) also resolve.
export * from "./anni";
export * from "./cknn";
export * from "./convolveSeparable";
export * from "./crossPcf";
export * from "./emptySpace";
// Both fuzzy-adjacency modules declare a `FuzzyAdjacency` result interface; the fixed-kernel
// one wins the bare name, the adaptive one is reachable via its deep import.
export type { FuzzyAdjacency } from "./fuzzyAdjacency";
export * from "./fuzzyAdjacency";
export * from "./fuzzyAdjacencyAdaptive";
export * from "./getisOrd";
export * from "./gramEnvelope";
export * from "./gramMatrix";
export * from "./gramModes";
export * from "./gramTerrain";
export * from "./imageOverlayWgsl";
export * from "./kernelWgsl";
export * from "./knn";
export * from "./knnDescentGpu";
export * from "./kthNeighborDistance";
export * from "./markerWgsl";
export * from "./nnDistance";
export * from "./paintField";
export * from "./pcaGpu";
export * from "./quadratCorrelationGpu";
export * from "./similarityWgsl";
export * from "./splatDensity";
export * from "./tcm";
export * from "./tcmRender";
export * from "./umapLayoutGpu";
