// `pushApart`: CSynth's local push-apart as a graph op. Particles that come closer than a set
// distance push each other away; beyond it they ignore each other:
//
//   F_i = Σ_j  −f · (1 − smoothstep(0.5, 1, len / reach)) · d̂
//   reach = min(|i − j|, nonBackboneLen) · backboneScale + delta
//
// `reach` grows with separation along the chain, so chain neighbours may sit closer than
// strangers. Only species in `speciesMask` take part, on both sides of a pair.
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
const PushParams = d.struct({
  n: d.u32,
  speciesMask: d.u32,
  force: d.f32,
  nonBackboneLen: d.f32,
  backboneScale: d.f32,
  delta: d.f32,
  pairRange: d.f32,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: PushParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("pushApart");

const pushFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const si = layout.$.species[i]!;
      if (si !== 255 && ((layout.$.params.speciesMask >> si) & 1) === 1) {
        const me = layout.$.pos[i]!.xyz;
        for (let j = d.u32(0); j < layout.$.params.n; j++) {
          const sj = layout.$.species[j]!;
          if (j !== i && sj !== 255 && ((layout.$.params.speciesMask >> sj) & 1) === 1) {
            const dd = std.sub(layout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            if (len > 0 && len <= layout.$.params.pairRange) {
              const along = std.abs(d.f32(i) - d.f32(j));
              const reach = std.min(along, layout.$.params.nonBackboneLen) * layout.$.params.backboneScale + layout.$.params.delta;
              const t = std.clamp((len / reach - 0.5) / 0.5, 0, 1);
              const push = -layout.$.params.force * (1 - t * t * (3 - 2 * t));
              f = std.add(f, std.mul(push / len, dd));
            }
          }
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("pushApart");

const makeParams = (root: Root) => root.createBuffer(PushParams).$usage("uniform").$name("pushApart:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([pushFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "pushApart", "pushApart"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const PARAMS: ParamSpec[] = [
  { name: "pushapartlocalforce", type: "number", default: 0.2, min: 0, max: 2, step: 0.01, describe: "Strength of the push." },
  {
    name: "nonBackboneLen",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    step: 0.1,
    describe: "Closest approach for strangers, in units of the backbone spacing.",
  },
  { name: "backboneScale", type: "number", default: 1, min: 0, max: 10, step: 0.01, describe: "Spacing of chain neighbours." },
  { name: "pushapartdelta", type: "number", default: 0, min: -2, max: 2, step: 0.01, describe: "Added to the reach of every pair." },
  { name: "speciesMask", type: "int", default: 0xff, min: 0, max: 0xff, describe: "Bit per species that takes part." },
  PAIR_RANGE,
];
const num = numbers(PARAMS, "pushApart");

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function pushCpu(pos: ArrayLike<number>, species: ArrayLike<number>, n: number, p: Params): Float32Array {
  const out = new Float32Array(n * 4);
  const force = num(p, "pushapartlocalforce");
  const nonBackboneLen = num(p, "nonBackboneLen");
  const backboneScale = num(p, "backboneScale");
  const delta = num(p, "pushapartdelta");
  const mask = num(p, "speciesMask");
  const pairRange = cutoff(num(p, "pairRange"));
  const takesPart = (s: number) => s !== DEAD_SPECIES && ((mask >> s) & 1) === 1;
  for (let i = 0; i < n; i++) {
    if (!takesPart(species[i]!)) continue;
    let fx = 0;
    let fy = 0;
    let fz = 0;
    for (let j = 0; j < n; j++) {
      if (j === i || !takesPart(species[j]!)) continue;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0 || len > pairRange) continue;
      const reach = Math.min(Math.abs(i - j), nonBackboneLen) * backboneScale + delta;
      const sc = (-force * (1 - smoothstep(0.5, 1, len / reach))) / len;
      fx += dx * sc;
      fy += dy * sc;
      fz += dz * sc;
    }
    out[i * 4] = fx;
    out[i * 4 + 1] = fy;
    out[i * 4 + 2] = fz;
  }
  return out;
}

export const pushApartOp: OpType = {
  name: "pushApart",
  label: "Push apart",
  describe: "Particles closer than a set distance push each other away.",
  category: "Springs",
  help: { math: "F_i=-f\\sum_j\\bigl(1-\\operatorname{smoothstep}(0.5,1,\\ell_{ij}/r_{ij})\\bigr)\\,\\hat d_{ij}" },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    return [{ kind: "points", n: sameParticles(inputs, "pushApart") }];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "pushApart.pos");
    requireScalar(inputs[1], "pushApart.species");
    return [VEC4];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "pushApart",
    );
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(n * 16);
    pipe.params.write({
      n,
      speciesMask: num(params, "speciesMask"),
      force: num(params, "pushapartlocalforce"),
      nonBackboneLen: num(params, "nonBackboneLen"),
      backboneScale: num(params, "backboneScale"),
      delta: num(params, "pushapartdelta"),
      pairRange: cutoff(num(params, "pairRange")),
    });
    const bind = rawBindGroup(
      device,
      root,
      layout,
      [pipe.params, residentBuffer(inputs[0], "pushApart.pos"), residentBuffer(inputs[1], "pushApart.species"), out.buffer],
      "pushApart",
    );
    submitDispatch(device, "pushApart", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "pushApart",
    );
    const data = pushCpu(hostF32(inputs[0], "pushApart.pos"), hostF32(inputs[1], "pushApart.species"), n, params);
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, data }];
  },
};
