// Tier-2 graph node wrapping `convolveSeparableResident` (grid -> grid windowing).
//
// One of the two ADR-0017 pilot conversions. `execute` now takes a GPU-resident input and leases
// a resident output, so a convolve sandwiched between other resident ops moves no bytes across
// the bus at all. `cpuGolden` is unchanged and remains the host-side reference oracle.
import { boxKernel, convolveSeparableResident, gaussianKernel } from "../../spatial/convolveSeparable";
import type { Shape } from "../handle";
import type { OpType, Params } from "../op";

function gridShape(s: Shape): { width: number; height: number } {
  if (s.kind !== "grid") throw new Error("convolveSeparable: input must be a grid");
  return { width: s.width, height: s.height };
}

function kernelFor(params: Params): Float32Array {
  const kind = params.kernel as string;
  const radius = params.radius as number;
  return kind === "gaussian" ? gaussianKernel((params.sigma as number) || radius / 3, radius) : boxKernel(radius);
}

export const convolveSeparableOp: OpType = {
  name: "convolveSeparable",
  label: "Separable convolution",
  describe: "Window a grid with a 1D box or Gaussian kernel on both axes.",
  inputs: [{ name: "grid", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "kernel", type: "enum", default: "box", options: ["box", "gaussian"] },
    { name: "radius", type: "int", default: 2, min: 1, max: 64 },
    { name: "sigma", type: "number", default: 0, min: 0, max: 32, step: 0.1, describe: "gaussian σ (0 = radius/3)" },
  ],
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const { width, height } = gridShape(inField.shape);
    const src = inField.buffer;
    if (!src) throw new Error("convolveSeparable: resident op received a non-resident input");
    // Lease the output rather than reusing module scratch: the executor owns this buffer's
    // lifetime and returns it to the pool once the last consumer has read it.
    const dst = await ctx.backend.lease(width * height * 4);
    await convolveSeparableResident(src.buffer, dst.buffer, width, height, kernelFor(params));
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const data = inputs[0]!.data!;
    const k = kernelFor(params);
    const r = (k.length - 1) / 2;
    const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
    const tmp = new Float32Array(width * height);
    const out = new Float32Array(width * height);
    // horizontal then vertical (matches the GPU two-pass)
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        let acc = 0;
        for (let t = 0; t < k.length; t++) acc += data[row * width + clamp(col + t - r, width - 1)]! * k[t]!;
        tmp[row * width + col] = acc;
      }
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        let acc = 0;
        for (let t = 0; t < k.length; t++) acc += tmp[clamp(row + t - r, height - 1) * width + col]! * k[t]!;
        out[row * width + col] = acc;
      }
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data: out }];
  },
};
