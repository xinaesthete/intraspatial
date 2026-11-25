// `outward`: a push away from a particle's own tissue, as a graph op. Each particle sums the
// directions away from its tissue neighbours, weighted by closeness. The sum points out of the
// tissue. Along it the particle gets two pushes, each set per species:
//
//   protrusion · exposure     exposure = max(0, 1 − Σw / full): 1 on a tip, 0 when buried
//   thrust                    the same size wherever the particle is: what a leader cell uses
//
//   away = −Σ_j w_j d̂_j,   w_j = 1 − len / range
//
// A neighbour counts when it is the particle's own species or one in `tissueMask`.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { OpType, ParamSpec, Params } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { VEC4 } from "./bondTable";
import { cutoff, DEAD_SPECIES, hostF32, numbers, PAIR_RANGE, requireScalar, requireVec, residentBuffer, sameParticles } from "./common";

const WG = 64;
/** Species this op has per-species strengths for. */
const SPECIES = 4;

const OutwardParams = d.struct({
  n: d.u32,
  tissueMask: d.u32,
  range: d.f32,
  full: d.f32,
  pairRange: d.f32,
  protrusion: d.vec4f,
  thrust: d.vec4f,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: OutwardParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("outward");

const outwardFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const si = layout.$.species[i]!;
      let protrude = d.f32(0);
      let thrust = d.f32(0);
      if (si === 0) {
        protrude = layout.$.params.protrusion.x;
        thrust = layout.$.params.thrust.x;
      }
      if (si === 1) {
        protrude = layout.$.params.protrusion.y;
        thrust = layout.$.params.thrust.y;
      }
      if (si === 2) {
        protrude = layout.$.params.protrusion.z;
        thrust = layout.$.params.thrust.z;
      }
      if (si === 3) {
        protrude = layout.$.params.protrusion.w;
        thrust = layout.$.params.thrust.w;
      }
      if (protrude !== 0 || thrust !== 0) {
        const me = layout.$.pos[i]!.xyz;
        let away = d.vec3f(0, 0, 0);
        let weight = d.f32(0);
        for (let j = d.u32(0); j < layout.$.params.n; j++) {
          const sj = layout.$.species[j]!;
          if (j !== i && sj !== 255) {
            if (sj === si || ((layout.$.params.tissueMask >> sj) & 1) === 1) {
              const dd = std.sub(layout.$.pos[j]!.xyz, me);
              const len = std.length(dd);
              if (len > 0 && len <= layout.$.params.pairRange && len < layout.$.params.range) {
                const w = 1 - len / layout.$.params.range;
                weight += w;
                away = std.sub(away, std.mul(w / len, dd));
              }
            }
          }
        }
        const awayLen = std.length(away);
        if (awayLen > 1e-6) {
          const exposure = std.max(0, 1 - weight / layout.$.params.full);
          f = std.mul((protrude * exposure + thrust) / awayLen, away);
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("outward");

const makeParams = (root: Root) => root.createBuffer(OutwardParams).$usage("uniform").$name("outward:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([outwardFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "outward", "outward"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const perSpecies = (name: string, describe: string): ParamSpec[] =>
  Array.from({ length: SPECIES }, (_, s) => ({
    name: `${name}${s}`,
    type: "number" as const,
    default: 0,
    min: 0,
    max: 2,
    step: 0.01,
    describe: `${describe} for species ${s}.`,
    tags: [name],
  }));

const PARAMS: ParamSpec[] = [
  ...perSpecies("protrusion", "Outward push that fades as the particle is buried,"),
  ...perSpecies("thrust", "Outward push of fixed size,"),
  { name: "protrusionRange", type: "number", default: 1.5, min: 0.1, max: 10, step: 0.1, describe: "How far a neighbour is felt." },
  {
    name: "protrusionFull",
    type: "number",
    default: 3.8,
    min: 0.1,
    max: 40,
    step: 0.1,
    describe: "Summed neighbour weight at which a particle counts as buried.",
  },
  {
    name: "tissueMask",
    type: "int",
    default: 0,
    min: 0,
    max: 0xff,
    describe: "Bit per species that counts as tissue, besides the particle's own.",
  },
  PAIR_RANGE,
];
const num = numbers(PARAMS, "outward");
const four = (p: Params, name: string) => Array.from({ length: SPECIES }, (_, s) => num(p, `${name}${s}`));

function outwardCpu(pos: ArrayLike<number>, species: ArrayLike<number>, n: number, p: Params): Float32Array {
  const out = new Float32Array(n * 4);
  const protrusion = four(p, "protrusion");
  const thrusts = four(p, "thrust");
  const range = num(p, "protrusionRange");
  const full = num(p, "protrusionFull");
  const mask = num(p, "tissueMask");
  const pairRange = cutoff(num(p, "pairRange"));
  for (let i = 0; i < n; i++) {
    const si = species[i]!;
    if (si === DEAD_SPECIES) continue;
    const protrude = protrusion[si] ?? 0;
    const thrust = thrusts[si] ?? 0;
    if (protrude === 0 && thrust === 0) continue;
    let ax = 0;
    let ay = 0;
    let az = 0;
    let weight = 0;
    for (let j = 0; j < n; j++) {
      const sj = species[j]!;
      if (j === i || sj === DEAD_SPECIES) continue;
      if (sj !== si && ((mask >> sj) & 1) !== 1) continue;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0 || len > pairRange || !(len < range)) continue;
      const w = 1 - len / range;
      weight += w;
      ax -= (dx * w) / len;
      ay -= (dy * w) / len;
      az -= (dz * w) / len;
    }
    const awayLen = Math.hypot(ax, ay, az);
    if (!(awayLen > 1e-6)) continue;
    const sc = (protrude * Math.max(0, 1 - weight / full) + thrust) / awayLen;
    out[i * 4] = ax * sc;
    out[i * 4 + 1] = ay * sc;
    out[i * 4 + 2] = az * sc;
  }
  return out;
}

export const outwardOp: OpType = {
  name: "outward",
  label: "Outward push",
  describe: "Pushes particles out of their own tissue: protrusion fades when buried, thrust does not.",
  category: "Springs",
  help: {
    detail:
      "Each particle looks at where its tissue neighbours are and is pushed the other way. On a tip, with " +
      "tissue on one side only, the directions add up. In the middle of the tissue they cancel. The " +
      "protrusion part also fades with how much tissue is near, reaching zero at `protrusionFull`, so a " +
      "tip with close neighbours pushes less than a bare one.",
    math: "F_i=\\bigl(p\\,\\max(0,1-\\tfrac{\\Sigma w}{w_{full}})+t\\bigr)\\,\\hat a_i,\\qquad a_i=-\\sum_j w_j\\hat d_{ij}",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    return [{ kind: "points", n: sameParticles(inputs, "outward") }];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "outward.pos");
    requireScalar(inputs[1], "outward.species");
    return [VEC4];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "outward",
    );
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(n * 16);
    const pr = four(params, "protrusion");
    const th = four(params, "thrust");
    pipe.params.write({
      n,
      tissueMask: num(params, "tissueMask"),
      range: num(params, "protrusionRange"),
      full: num(params, "protrusionFull"),
      pairRange: cutoff(num(params, "pairRange")),
      protrusion: d.vec4f(pr[0]!, pr[1]!, pr[2]!, pr[3]!),
      thrust: d.vec4f(th[0]!, th[1]!, th[2]!, th[3]!),
    });
    const bind = rawBindGroup(
      device,
      root,
      layout,
      [pipe.params, residentBuffer(inputs[0], "outward.pos"), residentBuffer(inputs[1], "outward.species"), out.buffer],
      "outward",
    );
    submitDispatch(device, "outward", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "outward",
    );
    const data = outwardCpu(hostF32(inputs[0], "outward.pos"), hostF32(inputs[1], "outward.species"), n, params);
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, data }];
  },
};
