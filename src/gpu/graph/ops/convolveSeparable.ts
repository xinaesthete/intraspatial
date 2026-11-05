// Tier-2 graph node wrapping `convolveSeparableResident` (grid -> grid windowing).
//
// One of the two ADR-0017 pilot conversions. `execute` now takes a GPU-resident input and leases
// a resident output, so a convolve sandwiched between other resident ops moves no bytes across
// the bus at all. `cpuGolden` is unchanged and remains the host-side reference oracle.
import { boxKernel, convolveSeparableResident, gaussianKernel } from "#src/gpu/spatial/convolveSeparable";
import { channelAxis, type Shape } from "../handle";
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
  describe: "Window a grid with a 1D box or Gaussian kernel on both axes (each channel plane on its own).",
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
    const kernel = kernelFor(params);
    const device = await ctx.backend.getDevice();
    const planes = channelAxis(inField)?.length ?? 1;
    const plane = width * height * 4;
    // Lease the output rather than reusing module scratch: the executor owns this buffer's
    // lifetime and returns it to the pool once the last consumer has read it.
    const dst = await ctx.backend.lease(planes * plane);
    if (planes === 1) {
      await convolveSeparableResident(src.buffer, dst.buffer, width, height, kernel, device);
      return [{ shape: { kind: "grid", width, height }, dtype: "f32", buffer: dst }];
    }
    // A channel axis: each plane on its own, through two plane-sized scratch leases (a binding's
    // offset must be 256-aligned, which a plane's start need not be; a copy's need only be 4).
    const a = await ctx.backend.lease(plane);
    const b = await ctx.backend.lease(plane);
    for (let k = 0; k < planes; k++) {
      const into = device.createCommandEncoder({ label: "convolveSeparable:plane-in" });
      into.copyBufferToBuffer(src.buffer, k * plane, a.buffer, 0, plane);
      device.queue.submit([into.finish()]);
      await convolveSeparableResident(a.buffer, b.buffer, width, height, kernel, device);
      const out = device.createCommandEncoder({ label: "convolveSeparable:plane-out" });
      out.copyBufferToBuffer(b.buffer, 0, dst.buffer, k * plane, plane);
      device.queue.submit([out.finish()]);
    }
    ctx.backend.release(a);
    ctx.backend.release(b);
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const { width, height } = gridShape(inputs[0]!.shape);
    const all = inputs[0]!.data!;
    const k = kernelFor(params);
    const r = (k.length - 1) / 2;
    const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);
    const n = width * height;
    const planes = channelAxis(inputs[0]!)?.length ?? 1;
    const tmp = new Float32Array(n);
    const result = new Float32Array(planes * n);
    for (let p = 0; p < planes; p++) {
      const data = all.subarray(p * n, (p + 1) * n);
      const out = result.subarray(p * n, (p + 1) * n);
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
    }
    return [{ shape: { kind: "grid", width, height }, dtype: "f32", data: result }];
  },
};
