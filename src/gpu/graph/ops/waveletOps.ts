// Wavelet-domain ops (ADR-0006): move a grid between the spatial basis and the
// wavelet (Mallat coefficient) basis, and operate on detail coefficients in between.
// The classic chain `fdwt → thresholdDetail → idwt` is a wavelet denoise, the
// project's headline "prove the pair as a tool" exercise — here as composable graph
// nodes. CPU Tier-1 (the reference math is the same one the docs demo uses); a GPU
// path can wrap the existing FDWT/IDWT kernels later.
//
// These are an OPT-IN op pack (`registerWaveletOps()`), NOT eagerly registered — see
// the Dawn-on-Node teardown lesson in ADR-0004 (eager module-graph growth on the base
// registry tipped unrelated render-fork GPU tests over).
import type { Basis, FieldValue, Shape } from "../handle";
import { SPATIAL, basisOf } from "../handle";
import type { OpType, Params } from "../op";
import { dwtBands, fdwt2d, idwt2d, isDetailBand } from "../dwt";
import type { Kernel } from "../dwt";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("wavelet op: input must be a grid");
  return { width: s.width, height: s.height };
}

const kernelOf = (params: Params): Kernel => (params.kernel === "9/7" ? "9/7" : "5/3");
const levelsOf = (params: Params): number => Math.max(1, (params.levels as number) | 0);

/** The wavelet decomposition contract carried by the input (ADR-0006): kernel + levels
 *  come from the field's `basis`, not from a per-op param. Throws if the input is not a
 *  wavelet field (apply Forward DWT first). */
function requireWavelet(b: Basis, op: string): { wavelet: Kernel; levels: number } {
  if (b.kind !== "wavelet") throw new Error(`${op}: input must be a wavelet field — apply Forward DWT first`);
  return { wavelet: b.wavelet, levels: b.levels };
}
const inputWavelet = (v: FieldValue, op: string) => requireWavelet(basisOf(v), op);

const KERNEL_PARAM = {
  name: "kernel",
  type: "enum" as const,
  default: "5/3",
  options: ["5/3", "9/7"],
  describe: "5/3 reversible (lossless) or 9/7 irreversible (lossy)",
};
const LEVELS_PARAM = { name: "levels", type: "int" as const, default: 3, min: 1, max: 8, describe: "decomposition levels" };

/** Forward DWT: spatial grid → packed Mallat coefficient grid (same shape). */
export const fdwtOp: OpType = {
  name: "fdwt",
  label: "Forward DWT",
  describe: "Separable 2D wavelet transform → packed coefficient grid (the wavelet-domain representation).",
  inputs: [{ name: "in", kind: "grid" }],
  outputs: [{ name: "coeffs", kind: "grid", dtype: "f32" }],
  params: [KERNEL_PARAM, LEVELS_PARAM],
  inferShapes: (inputs) => {
    gridShape(inputs[0]!);
    return [inputs[0]!];
  },
  // The output is a wavelet field tagging its own kernel + levels — the contract
  // idwt/thresholdDetail read instead of re-declaring (ADR-0006).
  inferBasis: (_inputs, params) => [{ kind: "wavelet", wavelet: kernelOf(params), levels: levelsOf(params) }],
  async execute(_ctx, inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const data = fdwt2d(inputs[0]!.data!, width, height, kernelOf(params), levelsOf(params));
    return [{ shape: inputs[0]!.shape, dtype: "f32", data }];
  },
  cpuGolden(inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    return [{ shape: inputs[0]!.shape, dtype: "f32", data: fdwt2d(inputs[0]!.data!, width, height, kernelOf(params), levelsOf(params)) }];
  },
};

/** Inverse DWT: a wavelet field → resynthesised spatial grid. Kernel + levels come
 *  from the input's basis (no params) — you can only idwt something that was fdwt'd. */
function idwtFV(input: FieldValue): FieldValue {
  const { width, height } = gridShape(input.shape);
  const { wavelet, levels } = inputWavelet(input, "idwt");
  return { shape: input.shape, dtype: "f32", data: idwt2d(input.data!, width, height, wavelet, levels) };
}

export const idwtOp: OpType = {
  name: "idwt",
  label: "Inverse DWT",
  describe: "Inverse 2D wavelet transform → spatial grid. Kernel + levels are read from the input wavelet field.",
  inputs: [{ name: "coeffs", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [],
  inferShapes: (inputs) => {
    gridShape(inputs[0]!);
    return [inputs[0]!];
  },
  // Requires a wavelet input; resynthesises back to the spatial basis.
  inferBasis: (inputs) => {
    requireWavelet(inputs[0]!, "idwt");
    return [SPATIAL];
  },
  execute: async (_ctx, inputs) => [idwtFV(inputs[0]!)],
  cpuGolden: (inputs) => [idwtFV(inputs[0]!)],
};

/** Shrink detail coefficients toward zero, leaving the LL approximation untouched —
 *  the denoising/compression operator in the wavelet basis. Soft = sign(x)·max(|x|−t,0)
 *  (the standard wavelet-shrinkage estimator); hard = zero below t. */
function thresholdDetail(coeffs: ArrayLike<number>, width: number, height: number, levels: number, t: number, soft: boolean): Float32Array {
  const out = Float32Array.from(coeffs);
  for (const band of dwtBands(width, height, levels)) {
    if (!isDetailBand(band)) continue;
    for (let y = band.y; y < band.y + band.h; y++) {
      for (let x = band.x; x < band.x + band.w; x++) {
        const i = y * width + x;
        const v = out[i]!;
        if (soft) out[i] = Math.sign(v) * Math.max(Math.abs(v) - t, 0);
        else if (Math.abs(v) < t) out[i] = 0;
      }
    }
  }
  return out;
}

/** Threshold detail subbands of a wavelet field (wavelet shrinkage). The level count
 *  comes from the input's basis, not a param — it always matches the producing fdwt. */
export const thresholdDetailOp: OpType = {
  name: "thresholdDetail",
  label: "Threshold detail",
  describe: "Wavelet-shrinkage: soft/hard threshold the detail subbands, leave the LL approximation.",
  inputs: [{ name: "coeffs", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "thresh", type: "number", default: 4, min: 0, max: 32, step: 0.25, describe: "shrinkage threshold" },
    { name: "soft", type: "bool", default: true, describe: "soft shrinkage instead of hard zeroing" },
  ],
  inferShapes: (inputs) => {
    gridShape(inputs[0]!);
    return [inputs[0]!];
  },
  // Operates inside the wavelet basis (input stays wavelet on output).
  inferBasis: (inputs) => {
    requireWavelet(inputs[0]!, "thresholdDetail");
    return [inputs[0]!];
  },
  execute: async (_ctx, inputs, params) => [thresholdDetailFV(inputs[0]!, params)],
  cpuGolden: (inputs, params) => [thresholdDetailFV(inputs[0]!, params)],
};

function thresholdDetailFV(input: FieldValue, params: Params): FieldValue {
  const { width, height } = gridShape(input.shape);
  const { levels } = inputWavelet(input, "thresholdDetail");
  const data = thresholdDetail(input.data!, width, height, levels, params.thresh as number, params.soft as boolean);
  return { shape: input.shape, dtype: "f32", data };
}
