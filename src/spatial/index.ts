// CPU-side spatial statistics & layout (PCF, quadrats, KNN-descent, UMAP, persistence, …) — the goldens the GPU ops are checked against.
// Barrel for the `intraspatial/spatial` subpath export; deep imports (`intraspatial/spatial/<file>`) also resolve.
export * from "./bucketGrid";
export * from "./cellCsv";
export * from "./contactNetwork";
export * from "./edgeCorrection";
export * from "./eigenSym";
export * from "./envelope";
export * from "./gram";
export * from "./kernelAnalysis";
// `mulberry32` is declared in both kernelAnalysis and umapLayout (identical PRNG); export one.
export { mulberry32 } from "./kernelAnalysis";
export * from "./kernelSpectrum";
export * from "./kernels";
export * from "./knnDescent";
export * from "./ngffTransform";
export * from "./pca";
export * from "./pcf";
export * from "./pcfBootstrap";
export * from "./pcfEnvelope";
export * from "./permute";
export * from "./persistence";
export * from "./pointPatterns";
export * from "./quadratCorrelation";
export * from "./scalarField";
export * from "./sublevelsetPersistence";
export * from "./syntheticManifolds";
export * from "./tcm";
export * from "./tcmKernel";
export * from "./umap";
export * from "./umapGraph";
export * from "./umapLayout";
