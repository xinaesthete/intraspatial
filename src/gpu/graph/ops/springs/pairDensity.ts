// `pairDensity`: how crowded each particle is. Two numbers per particle, from one pass over the
// others within range:
//
//   density      Σ_j 1 / len³
//   directional  Σ_j max(0, d̂·v) / len²      crowding in the direction the particle is moving
//
// Density gates division and CSynth's density push-apart; the directional one feeds damping type
// 4, which brakes a particle hard as it runs into others. A dead particle has neither and
// contributes to neither.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { OpType, ParamSpec, Params } from "../../op";
import { rawBindGroup } from "../../residentBind";
import {
  cutoff,
  DEAD_SPECIES,
  hostF32,
  NO_CUTOFF,
  numbers,
  PAIR_RANGE,
  requireScalar,
  requireVec,
  residentBuffer,
  sameParticles,
  VEC2,
} from "./common";

const WG = 64;
const DensityParams = d.struct({ n: d.u32, rate: d.f32, pairRange: d.f32, densityRange: d.f32 });

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: DensityParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    prev: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    density: { storage: (n: number) => d.arrayOf(d.vec2f, n), access: "mutable" },
  })
  .$name("pairDensity");

const densityFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      let density = d.f32(0);
      let directional = d.f32(0);
      if (layout.$.species[i]! !== 255) {
        const me = layout.$.pos[i]!.xyz;
        const v = std.div(std.sub(me, layout.$.prev[i]!.xyz), layout.$.params.rate);
        for (let j = d.u32(0); j < layout.$.params.n; j++) {
          if (j !== i && layout.$.species[j]! !== 255) {
            const dd = std.sub(layout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            if (len > 0 && len <= layout.$.params.pairRange && len < layout.$.params.densityRange) {
              density += 1 / (len * len * len);
              directional += std.max(0, std.dot(dd, v) / len) / (len * len);
            }
          }
        }
      }
      layout.$.density[i] = d.vec2f(density, directional);
    }
  })
  .$name("pairDensity");

const makeParams = (root: Root) => root.createBuffer(DensityParams).$usage("uniform").$name("pairDensity:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([densityFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "pairDensity", "pairDensity"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const PARAMS: ParamSpec[] = [
  PAIR_RANGE,
  { name: "densityRange", type: "number", default: NO_CUTOFF, min: 0, describe: "Only neighbours nearer than this count towards density." },
  {
    name: "springrate",
    type: "number",
    default: 1,
    min: 1e-6,
    max: 4,
    step: 0.01,
    describe: "Step length, to turn the two positions into a velocity.",
  },
];
const num = numbers(PARAMS, "pairDensity");

function densityCpu(pos: ArrayLike<number>, prev: ArrayLike<number>, species: ArrayLike<number>, n: number, p: Params): Float32Array {
  const out = new Float32Array(n * 2);
  const rate = num(p, "springrate");
  const pairRange = cutoff(num(p, "pairRange"));
  const densityRange = cutoff(num(p, "densityRange"));
  for (let i = 0; i < n; i++) {
    if (species[i] === DEAD_SPECIES) continue;
    const vx = (pos[i * 4]! - prev[i * 4]!) / rate;
    const vy = (pos[i * 4 + 1]! - prev[i * 4 + 1]!) / rate;
    const vz = (pos[i * 4 + 2]! - prev[i * 4 + 2]!) / rate;
    let density = 0;
    let directional = 0;
    for (let j = 0; j < n; j++) {
      if (j === i || species[j] === DEAD_SPECIES) continue;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0 || len > pairRange || !(len < densityRange)) continue;
      density += 1 / (len * len * len);
      directional += Math.max(0, (dx * vx + dy * vy + dz * vz) / len) / (len * len);
    }
    out[i * 2] = density;
    out[i * 2 + 1] = directional;
  }
  return out;
}

export const pairDensityOp: OpType = {
  name: "pairDensity",
  label: "Density",
  describe: "How crowded each particle is, and how crowded in the direction it is moving.",
  category: "Springs",
  help: {
    math: "\\rho_i=\\sum_j \\ell_{ij}^{-3},\\qquad \\rho^{\\rightarrow}_i=\\sum_j \\max(0,\\hat d_{ij}\\cdot v_i)\\,\\ell_{ij}^{-2}",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "prev", kind: "points", dtype: "f32" },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  outputs: [{ name: "density", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    return [{ kind: "points", n: sameParticles(inputs, "pairDensity") }];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "pairDensity.pos");
    requireVec(inputs[1], 4, "pairDensity.prev");
    requireScalar(inputs[2], "pairDensity.species");
    return [VEC2];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "pairDensity",
    );
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(n * 8);
    pipe.params.write({
      n,
      rate: num(params, "springrate"),
      pairRange: cutoff(num(params, "pairRange")),
      densityRange: cutoff(num(params, "densityRange")),
    });
    const bind = rawBindGroup(
      device,
      root,
      layout,
      [
        pipe.params,
        residentBuffer(inputs[0], "pairDensity.pos"),
        residentBuffer(inputs[1], "pairDensity.prev"),
        residentBuffer(inputs[2], "pairDensity.species"),
        out.buffer,
      ],
      "pairDensity",
    );
    submitDispatch(device, "pairDensity", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC2, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "pairDensity",
    );
    const data = densityCpu(
      hostF32(inputs[0], "pairDensity.pos"),
      hostF32(inputs[1], "pairDensity.prev"),
      hostF32(inputs[2], "pairDensity.species"),
      n,
      params,
    );
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC2, data }];
  },
};
