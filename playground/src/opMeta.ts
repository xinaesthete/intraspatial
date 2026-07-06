// Presentation metadata for the palette: which category each node falls under, and
// rich help (prose + display math) for its tooltip. Kept here (UI-side) so the runtime
// op definitions stay lean; an op may still carry its own `category`/`help` on OpType,
// which takes precedence (see specs.ts). Math strings are LaTeX, rendered with KaTeX.
import type { OpHelp } from "../../src/gpu/graph";

/** Category per node name (ops + sources). Drives palette grouping + the `/` palette. */
export const OP_CATEGORY: Record<string, string> = {
  // sources
  ringPoints: "Sources",
  blobPoints: "Sources",
  grayScottSeed: "Sources",
  grayScottSeedComplex: "Sources",
  noiseGrid: "Sources",
  waveGrid: "Sources",
  // image / grid
  splatDensity: "Image & grid",
  convolveSeparable: "Image & grid",
  getisOrd: "Image & grid",
  threshold: "Image & grid",
  addGrids: "Image & grid",
  // spatial / topology
  kthNeighborDistance: "Spatial & TDA",
  fuzzyAdjacency: "Spatial & TDA",
  fuzzyAdjacencyAdaptive: "Spatial & TDA",
  membershipToDistance: "Spatial & TDA",
  vietorisRipsPersistence: "Spatial & TDA",
  // simulation
  reactionDiffusionStep: "Simulation",
  reactionDiffusionComplex: "Simulation",
  feedback: "Simulation",
  // element algebra
  complex: "Complex",
  realPart: "Complex",
  imagPart: "Complex",
  conjugate: "Complex",
  magnitude: "Complex",
  addFields: "Arithmetic",
  subFields: "Arithmetic",
  mulFields: "Arithmetic",
  scaleField: "Arithmetic",
  dotFields: "Linear algebra",
  crossFields: "Linear algebra",
  normalizeField: "Linear algebra",
  // wavelet
  fdwt: "Wavelet",
  idwt: "Wavelet",
  thresholdDetail: "Wavelet",
};

/** Order categories appear in the palette (anything else sorts after, alphabetically). */
export const CATEGORY_ORDER: string[] = [
  "Sources",
  "Image & grid",
  "Arithmetic",
  "Complex",
  "Linear algebra",
  "Wavelet",
  "Spatial & TDA",
  "Simulation",
  "Other",
];

/** Rich help for the showcase ops (math rendered as KaTeX). */
export const OP_HELP: Record<string, OpHelp> = {
  complex: {
    detail:
      "Pack two real fields into one complex field — the honest representation of a 2-component signal (e.g. reaction–diffusion U,V). Wire U→re and V→im.",
    math: "z = \\mathrm{re} + i\\,\\mathrm{im}",
  },
  grayScottSeedComplex: {
    detail:
      "Reaction–diffusion seed as one complex field (re = U background ≈ 1, im = V ≈ 0 with a seeded square). Feed straight into Reaction–diffusion (complex).",
    math: "z = U + i\\,V",
  },
  conjugate: {
    detail: "Negate the imaginary part (complex) or vector part (quaternion).",
    math: "\\bar{z} = \\mathrm{re} - i\\,\\mathrm{im}",
  },
  magnitude: { detail: "Per-sample magnitude of any element → a real field.", math: "|z| = \\sqrt{\\mathrm{re}^2 + \\mathrm{im}^2}" },
  realPart: { math: "\\Re(z) = \\mathrm{re}" },
  imagPart: { math: "\\Im(z) = \\mathrm{im}" },
  addFields: { math: "(a + b)_i = a_i + b_i" },
  subFields: { math: "(a - b)_i = a_i - b_i" },
  scaleField: { detail: "Multiply every lane by a real scalar.", math: "a \\mapsto s\\,a" },
  mulFields: {
    detail: "The element's algebra product: ordinary × for scalar, complex multiply for complex, Hamilton product for quaternion.",
    math: "z\\,w = (ac - bd) + (ad + bc)\\,i",
  },
  dotFields: { detail: "Pointwise vector dot product → a real field.", math: "a \\cdot b = \\sum_i a_i b_i" },
  crossFields: { detail: "Pointwise vec3 cross product.", math: "a \\times b" },
  normalizeField: { detail: "Scale each sample to unit length.", math: "\\hat{a} = \\frac{a}{\\lVert a \\rVert}" },
  fdwt: {
    detail: "Separable 2D wavelet transform → a packed Mallat coefficient pyramid (the editable wavelet-domain representation).",
    math: "x \\;\\xrightarrow{\\;W\\;}\\; \\{\\,c_{LL},\\,c_{LH},\\,c_{HL},\\,c_{HH}\\,\\}",
  },
  idwt: {
    detail: "Inverse wavelet transform: resynthesise the spatial field from coefficients.",
    math: "c \\;\\xrightarrow{\\;W^{-1}\\;}\\; x",
  },
  thresholdDetail: {
    detail: "Wavelet shrinkage: pull detail coefficients toward zero (denoise / compress), leaving the LL approximation.",
    math: "\\eta_t(x) = \\operatorname{sgn}(x)\\,\\max(|x| - t,\\, 0)",
  },
  threshold: { detail: "Hard step, or a soft logistic ramp (a fuzzy threshold).", math: "\\sigma_k(x) = \\frac{1}{1 + e^{-(x - t)\\,k}}" },
  getisOrd: {
    detail: "Local hotspot z-score over a box neighbourhood.",
    math: "G^{*}_i = \\frac{\\sum_j w_{ij} x_j - \\bar{x}\\sum_j w_{ij}}{s\\,\\sqrt{\\dots}}",
  },
  reactionDiffusionComplex: {
    detail: "Gray–Scott carried as one complex field, re = U, im = V.",
    math: "\\partial_t U = D_u\\nabla^2 U - UV^2 + F(1-U)",
  },
  gradient: {
    detail: "Central-difference gradient → a vec2 field.",
    math: "\\nabla f = \\left(\\tfrac{\\partial f}{\\partial x},\\ \\tfrac{\\partial f}{\\partial y}\\right)",
  },
  gradientMagnitude: { detail: "Edge strength.", math: "\\lVert \\nabla f \\rVert = \\sqrt{f_x^2 + f_y^2}" },
  laplacian: { detail: "Sum of second derivatives (a blob / edge detector).", math: "\\nabla^2 f = f_{xx} + f_{yy}" },
  divergence: {
    detail: "Net outflow of a vector field → a scalar field.",
    math: "\\nabla \\cdot v = \\tfrac{\\partial v_x}{\\partial x} + \\tfrac{\\partial v_y}{\\partial y}",
  },
  structureOrientation: {
    detail: "Dominant local orientation: the principal eigenvector of the smoothed structure tensor, scaled by coherence.",
    math: "J = \\overline{\\nabla f\\,\\nabla f^{\\top}},\\quad \\theta = \\tfrac{1}{2}\\operatorname{atan2}(2J_{xy},\\,J_{xx}-J_{yy})",
  },
};
