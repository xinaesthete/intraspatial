// The two random terms of the spring model as graph ops. Both draw from a hash of (particle,
// step, salt), so a run is reproducible and the kernel matches its CPU golden draw for draw.
//
//   noiseForce     every particle, with probability `noiseprob` each step: a kick of up to
//                  ±noiseforce/2 on each axis. CSynth's noise.
//   motilityForce  every live particle, every step: a kick of up to ±motility on each axis,
//                  with the strength set per species. How cells wander.
//
// In 2D the z axis is left alone.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { rand01, SALT } from "#src/gpu/sim/springs/rng";
import type { ParamSpec, Params } from "../../op";
import { DEAD_SPECIES, numbers } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";
import { rand01Gpu, stepKeyGpu } from "./rngTgsl";

const SEED: ParamSpec = { name: "seed", type: "int", default: 1, min: 0, describe: "Seed of the random stream." };
const DIMS: ParamSpec = { name: "dims", type: "int", default: 3, min: 2, max: 3, describe: "2 leaves the z axis alone." };
const stepKey = (seed: number, step: number) => (seed ^ Math.imul(step, 0x9e3779b1)) >>> 0;

// ── noise ──────────────────────────────────────────────────────────────────────────────────────

const NoiseParams = d.struct({ n: d.u32, seed: d.u32, dims: d.u32, force: d.f32, prob: d.f32 });
const noiseLayout = tgpu
  .bindGroupLayout({
    params: { uniform: NoiseParams },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    step: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("noiseForce");

const noiseFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [KERNEL_WORKGROUP] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < noiseLayout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const key = stepKeyGpu(noiseLayout.$.params.seed, noiseLayout.$.step[0]!);
      if (noiseLayout.$.species[i]! !== 255 && rand01Gpu(i, key, d.u32(0x51ed270b)) < noiseLayout.$.params.prob) {
        let z = d.f32(0);
        if (noiseLayout.$.params.dims === 3) {
          z = rand01Gpu(i, key, d.u32(0xc2b2ae35)) - 0.5;
        }
        f = d.vec3f(rand01Gpu(i, key, d.u32(0x1b873593)) - 0.5, rand01Gpu(i, key, d.u32(0x85ebca6b)) - 0.5, z);
        f = d.vec3f(f.x * noiseLayout.$.params.force, f.y * noiseLayout.$.params.force, f.z * noiseLayout.$.params.force);
      }
      noiseLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("noiseForce");

const NOISE_PARAMS: ParamSpec[] = [
  { name: "noiseforce", type: "number", default: 0, min: 0, max: 2, step: 0.001, describe: "Size of a kick." },
  { name: "noiseprob", type: "number", default: 1, min: 0, max: 1, step: 0.01, describe: "Chance a particle is kicked in a step." },
  SEED,
  DIMS,
];
const noiseNum = numbers(NOISE_PARAMS, "noiseForce");

export const noiseForceOp = kernelOp<d.Infer<typeof NoiseParams>>({
  name: "noiseForce",
  label: "Noise",
  describe: "Random kicks: each particle, with a set chance each step.",
  inputs: [
    { name: "species", kind: "points", dtype: "u32" },
    { name: "step", kind: "scalar", dtype: "u32" },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: NOISE_PARAMS,
  layout: noiseLayout,
  kernel: noiseFn,
  uniform: (root) => root.createBuffer(NoiseParams).$usage("uniform").$name("noiseForce:params"),
  uniforms: (n, p) => ({
    n,
    seed: noiseNum(p, "seed"),
    dims: noiseNum(p, "dims"),
    force: noiseNum(p, "noiseforce"),
    prob: noiseNum(p, "noiseprob"),
  }),
  cpu([species, step], n, p) {
    const out = new Float32Array(n * 4);
    const key = stepKey(noiseNum(p, "seed"), step![0]!);
    const force = noiseNum(p, "noiseforce");
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES || !(rand01(i, key, SALT.noiseGate) < noiseNum(p, "noiseprob"))) continue;
      out[i * 4] = (rand01(i, key, SALT.noiseX) - 0.5) * force;
      out[i * 4 + 1] = (rand01(i, key, SALT.noiseY) - 0.5) * force;
      if (noiseNum(p, "dims") === 3) out[i * 4 + 2] = (rand01(i, key, SALT.noiseZ) - 0.5) * force;
    }
    return out;
  },
});

// ── motility ───────────────────────────────────────────────────────────────────────────────────

const MotilityParams = d.struct({ n: d.u32, seed: d.u32, dims: d.u32, strength: d.vec4f });
const motilityLayout = tgpu
  .bindGroupLayout({
    params: { uniform: MotilityParams },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    step: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("motilityForce");

const motilityFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [KERNEL_WORKGROUP] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < motilityLayout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const si = motilityLayout.$.species[i]!;
      let mot = d.f32(0);
      if (si === 0) {
        mot = motilityLayout.$.params.strength.x;
      }
      if (si === 1) {
        mot = motilityLayout.$.params.strength.y;
      }
      if (si === 2) {
        mot = motilityLayout.$.params.strength.z;
      }
      if (si === 3) {
        mot = motilityLayout.$.params.strength.w;
      }
      if (mot !== 0) {
        const key = stepKeyGpu(motilityLayout.$.params.seed, motilityLayout.$.step[0]!);
        let z = d.f32(0);
        if (motilityLayout.$.params.dims === 3) {
          z = (rand01Gpu(i, key, d.u32(0xd3a2646c)) - 0.5) * 2 * mot;
        }
        f = d.vec3f((rand01Gpu(i, key, d.u32(0x27d4eb2f)) - 0.5) * 2 * mot, (rand01Gpu(i, key, d.u32(0x165667b1)) - 0.5) * 2 * mot, z);
      }
      motilityLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("motilityForce");

const MOTILITY_PARAMS: ParamSpec[] = [
  ...Array.from({ length: 4 }, (_, s) => ({
    name: `motility${s}`,
    type: "number" as const,
    default: 0,
    min: 0,
    max: 1,
    step: 0.001,
    describe: `Size of the random kick for species ${s}.`,
    tags: ["motility"],
  })),
  SEED,
  DIMS,
];
const motilityNum = numbers(MOTILITY_PARAMS, "motilityForce");
const strengths = (p: Params) => [0, 1, 2, 3].map((s) => motilityNum(p, `motility${s}`));

export const motilityForceOp = kernelOp<d.Infer<typeof MotilityParams>>({
  name: "motilityForce",
  label: "Motility",
  describe: "A random kick every step, with its size set per species: how cells wander.",
  inputs: [
    { name: "species", kind: "points", dtype: "u32" },
    { name: "step", kind: "scalar", dtype: "u32" },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: MOTILITY_PARAMS,
  layout: motilityLayout,
  kernel: motilityFn,
  uniform: (root) => root.createBuffer(MotilityParams).$usage("uniform").$name("motilityForce:params"),
  uniforms(n, p) {
    const m = strengths(p);
    return { n, seed: motilityNum(p, "seed"), dims: motilityNum(p, "dims"), strength: d.vec4f(m[0]!, m[1]!, m[2]!, m[3]!) };
  },
  cpu([species, step], n, p) {
    const out = new Float32Array(n * 4);
    const key = stepKey(motilityNum(p, "seed"), step![0]!);
    const m = strengths(p);
    for (let i = 0; i < n; i++) {
      const mot = m[species![i]!] ?? 0;
      if (mot === 0) continue;
      out[i * 4] = (rand01(i, key, SALT.motileX) - 0.5) * 2 * mot;
      out[i * 4 + 1] = (rand01(i, key, SALT.motileY) - 0.5) * 2 * mot;
      if (motilityNum(p, "dims") === 3) out[i * 4 + 2] = (rand01(i, key, SALT.motileZ) - 0.5) * 2 * mot;
    }
    return out;
  },
});
