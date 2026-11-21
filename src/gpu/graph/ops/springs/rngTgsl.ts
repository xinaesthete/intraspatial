// The spring model's counter-based random numbers as TGSL functions: the same hash as
// `src/gpu/sim/springs/rng.ts`, so a kernel and its CPU golden draw the same stream.
import tgpu from "typegpu";
import * as d from "typegpu/data";

/** PCG hash of a u32. */
export const pcgGpu = tgpu.fn(
  [d.u32],
  d.u32,
)((v) => {
  "use gpu";
  const state = v * d.u32(747796405) + d.u32(2891336453);
  const word = ((state >> ((state >> d.u32(28)) + d.u32(4))) ^ state) * d.u32(277803737);
  return (word >> d.u32(22)) ^ word;
});

/** Uniform in [0, 1) from three u32 keys. */
export const rand01Gpu = tgpu.fn(
  [d.u32, d.u32, d.u32],
  d.f32,
)((a, b, c) => {
  "use gpu";
  return d.f32(pcgGpu(a ^ pcgGpu(b ^ pcgGpu(c)))) / 4294967296;
});

/** The per-step key every draw of one tick shares: `seed ^ step · φ`. */
export const stepKeyGpu = tgpu.fn(
  [d.u32, d.u32],
  d.u32,
)((seed, step) => {
  "use gpu";
  return seed ^ (step * d.u32(0x9e3779b1));
});

/** Hash of an unordered particle pair and a step: the same value whichever end asks, which is
 *  what lets both ends of a bond reach the same random decision. `pairHash` in rng.ts. */
export const pairHashGpu = tgpu.fn(
  [d.u32, d.u32, d.u32, d.u32],
  d.u32,
)((i, j, step, salt) => {
  "use gpu";
  let lo = i;
  let hi = j;
  if (j < i) {
    lo = j;
    hi = i;
  }
  return pcgGpu(lo ^ pcgGpu((hi * d.u32(0x9e3779b1)) ^ salt ^ pcgGpu(step)));
});
