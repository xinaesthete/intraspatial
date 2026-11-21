// Simulation catalogue — reaction–diffusion, force/body/spline dynamics, the GPU dancer. (HsPf lives under ./hspf/.)
// Barrel for the `intraspatial/gpu/sim` subpath export; deep imports (`intraspatial/gpu/sim/<file>`) also resolve.
export * from "./body";
export * from "./dancerGpu";
export * from "./figures";
export * from "./forces";
export * from "./jet";
export * from "./quat";
export * from "./reactionDiffusion";
export * from "./splines";
// The CSynth-derived spring model. A namespace, because its names (`spawn`, `soft`, `CELL`) are
// short and would collide in this barrel.
export * as springs from "./springs/index";
export * from "./vec3";
