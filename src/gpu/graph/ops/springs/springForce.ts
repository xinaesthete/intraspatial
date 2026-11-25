// `springForce`: the bonded term of the spring model as a graph op (docs/spring-model-as-ops.md).
// Each particle gathers over its own row of the bond table and outputs the force on it:
//
//   F_i = Σ_j  min(s / len, 0.5 / rate) · (x_j − x_i)
//   s   = (len − rest) · strength · m[kind] · min(1000, (len / powBaseDist)^pow)
//
// `rest`, `strength` and `pow` come from the bond; `m[kind]` is a live multiplier per bond kind
// (CSynth's `roleforces`), so one control scales all bonds of a kind without rewriting the table.
// The cap limits how far a bond pulls each end in one step to half the gap, because the other
// end moves too. The ends can still meet; it bounds the move, not the rest length.
//
// One term, one kernel. The maths is the bonded loop of `stepSprings`, which stays the reference
// the summed ops must equal.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { FieldValue, Shape } from "../../handle";
import type { OpType, ParamSpec, Params } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { BOND_KINDS, BOND_TABLE_PORT, NO_LINK, slotsPerParticle, VEC4 } from "./bondTable";

const WG = 64;

const ForceParams = d.struct({
  n: d.u32,
  K: d.u32,
  springlen: d.f32,
  springlenfix: d.f32,
  springpow: d.f32,
  powBaseDist: d.f32,
  backboneStrength: d.f32,
  /** `0.5 / springrate`: the most a bond may move its end in a step, per unit length. */
  cap: d.f32,
  /** Strength multiplier of each bond kind: kinds 0–3, then 4–7. */
  kindLo: d.vec4f,
  kindHi: d.vec4f,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: ForceParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    link: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    spring: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("springForce");

const forceFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      const me = layout.$.pos[i]!.xyz;
      let f = d.vec3f(0, 0, 0);
      for (let k = d.u32(0); k < layout.$.params.K; k++) {
        const o = i * layout.$.params.K + k;
        const link = layout.$.link[o]!;
        if (link !== 0xffffffff) {
          const j = link & 0xffffff;
          const kind = link >> 24;
          const bond = layout.$.spring[o]!;
          const dd = std.sub(layout.$.pos[j]!.xyz, me);
          const len = std.length(dd);
          if (len > 0) {
            let mult = layout.$.params.kindLo.x;
            if (kind === 1) {
              mult = layout.$.params.kindLo.y;
            }
            if (kind === 2) {
              mult = layout.$.params.kindLo.z;
            }
            if (kind === 3) {
              mult = layout.$.params.kindLo.w;
            }
            if (kind === 4) {
              mult = layout.$.params.kindHi.x;
            }
            if (kind === 5) {
              mult = layout.$.params.kindHi.y;
            }
            if (kind === 6) {
              mult = layout.$.params.kindHi.z;
            }
            if (kind === 7) {
              mult = layout.$.params.kindHi.w;
            }
            const rest = bond.x * layout.$.params.springlen + layout.$.params.springlenfix;
            const power = std.min(1000, std.pow(len / layout.$.params.powBaseDist, bond.z + layout.$.params.springpow));
            let s = (len - rest) * bond.y * mult * power;
            // Neighbours along the chain are the backbone.
            if (j === i + 1 || j + 1 === i) {
              s = s * layout.$.params.backboneStrength;
            }
            f = std.add(f, std.mul(std.min(s / len, layout.$.params.cap), dd));
          }
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("springForce");

interface Pipe {
  pipeline: GPUComputePipeline;
  params: ReturnType<typeof makeParams>;
}

function makeParams(root: Root) {
  return root.createBuffer(ForceParams).$usage("uniform").$name("springForce:params");
}

// One pipeline and uniform block per backend root (roots are device singletons).
const pipes = new WeakMap<object, Promise<Pipe>>();

function getPipe(device: GPUDevice, root: Root): Promise<Pipe> {
  let p = pipes.get(root as object);
  if (!p) {
    p = (async () => ({
      pipeline: typegpuComputePipeline(device, root, tgpu.resolveWithContext([forceFn], { names: "strict" }), "springForce", "springForce"),
      params: makeParams(root),
    }))();
    pipes.set(root as object, p);
  }
  return p;
}

const KIND_PARAMS: ParamSpec[] = Array.from({ length: BOND_KINDS }, (_, kind) => ({
  name: `kind${kind}`,
  type: "number" as const,
  default: 1,
  min: 0,
  max: 4,
  step: 0.01,
  describe: `Strength multiplier for bonds of kind ${kind}.`,
  tags: ["kind"],
}));

const PARAMS: ParamSpec[] = [
  {
    name: "springforce",
    type: "number",
    default: 1,
    min: 0,
    max: 4,
    step: 0.01,
    describe: "Strength of every bond, before its kind's multiplier.",
  },
  { name: "springlen", type: "number", default: 1, min: 0, max: 4, step: 0.01, describe: "Scales every rest length." },
  { name: "springlenfix", type: "number", default: 0, min: -2, max: 2, step: 0.01, describe: "Added to every rest length." },
  { name: "springpow", type: "number", default: 0, min: -4, max: 4, step: 0.1, describe: "Distance power added to each bond's own." },
  { name: "powBaseDist", type: "number", default: 1, min: 1e-6, max: 100, describe: "The distance at which the power factor is 1." },
  {
    name: "backboneStrength",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    step: 0.01,
    describe: "Extra factor on bonds between chain neighbours.",
  },
  { name: "springrate", type: "number", default: 1, min: 1e-6, max: 4, step: 0.01, describe: "Step length; sets the overshoot cap." },
  ...KIND_PARAMS,
];

const num = (p: Params, name: string): number => {
  const spec = PARAMS.find((s) => s.name === name);
  const v = p[name] ?? spec?.default;
  if (typeof v !== "number" || Number.isNaN(v)) throw new Error(`springForce: param "${name}" must be a number`);
  return v;
};

/** Effective strength multiplier of each kind: `springforce · kind[k]`. */
function kindMultipliers(p: Params): number[] {
  const base = num(p, "springforce");
  return Array.from({ length: BOND_KINDS }, (_, kind) => base * num(p, `kind${kind}`));
}

function pointsN(shape: Shape, who: string): number {
  if (shape.kind !== "points") throw new Error(`${who}: expected a points field`);
  return shape.n;
}

function forceCpu(pos: ArrayLike<number>, link: Uint32Array, spring: ArrayLike<number>, n: number, K: number, p: Params): Float32Array {
  const out = new Float32Array(n * 4);
  const mult = kindMultipliers(p);
  const springlen = num(p, "springlen");
  const springlenfix = num(p, "springlenfix");
  const springpow = num(p, "springpow");
  const powBaseDist = num(p, "powBaseDist");
  const backbone = num(p, "backboneStrength");
  const cap = 0.5 / num(p, "springrate");
  for (let i = 0; i < n; i++) {
    let fx = 0;
    let fy = 0;
    let fz = 0;
    for (let k = 0; k < K; k++) {
      const o = i * K + k;
      const bits = link[o]!;
      if (bits === NO_LINK) continue;
      const j = bits & 0xffffff;
      const kind = bits >>> 24;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0) continue;
      const rest = spring[o * 4]! * springlen + springlenfix;
      let s = (len - rest) * spring[o * 4 + 1]! * mult[kind]! * Math.min(1000, (len / powBaseDist) ** (spring[o * 4 + 2]! + springpow));
      if (Math.abs(i - j) === 1) s *= backbone;
      const kk = Math.min(s / len, cap);
      fx += dx * kk;
      fy += dy * kk;
      fz += dz * kk;
    }
    out[i * 4] = fx;
    out[i * 4 + 1] = fy;
    out[i * 4 + 2] = fz;
  }
  return out;
}

function part(table: FieldValue, name: "link" | "spring"): FieldValue {
  const v = table.parts?.[name];
  if (!v) throw new Error(`springForce: bond table has no "${name}" part`);
  return v;
}

export const springForceOp: OpType = {
  name: "springForce",
  label: "Spring force",
  describe: "Force on each particle from the bonds in its row of the bond table.",
  category: "Springs",
  help: {
    detail:
      "Every bond pulls its two ends towards its rest length. A particle reads only its own row of the " +
      "bond table, where each of its bonds is listed, so nothing is written to any other particle. Add " +
      "the result to other forces and pass the sum to `Integrate (springs)`.",
    math: "F_i=\\sum_j \\min\\!\\left(\\tfrac{s_{ij}}{\\ell},\\tfrac{0.5}{\\text{rate}}\\right)(x_j-x_i),\\quad s_{ij}=(\\ell-\\ell_0)\\,k\\,m_{\\text{kind}}\\,(\\ell/d_0)^{p}",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "bonds", kind: "bundle", bundle: BOND_TABLE_PORT },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "springForce.pos");
    slotsPerParticle(inputs[1]!, n, "springForce");
    return [{ kind: "points", n }];
  },
  inferElements(inputs) {
    const el = inputs[0];
    if (!el || el.kind !== "vec" || el.n !== 4) throw new Error("springForce.pos: expected a vec4 points field (xyz, density)");
    return [VEC4];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const pos = inputs[0]!;
    const table = inputs[1]!;
    const n = pointsN(pos.shape, "springForce.pos");
    const K = slotsPerParticle(table.shape, n, "springForce");
    const link = part(table, "link").buffer;
    const spring = part(table, "spring").buffer;
    if (!pos.buffer || !link || !spring) throw new Error("springForce: resident op received a non-resident input");
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = await getPipe(device, root);

    const out = await ctx.backend.lease(n * 16);
    const mult = kindMultipliers(params);
    pipe.params.write({
      n,
      K,
      springlen: num(params, "springlen"),
      springlenfix: num(params, "springlenfix"),
      springpow: num(params, "springpow"),
      powBaseDist: num(params, "powBaseDist"),
      backboneStrength: num(params, "backboneStrength"),
      cap: 0.5 / num(params, "springrate"),
      kindLo: d.vec4f(mult[0]!, mult[1]!, mult[2]!, mult[3]!),
      kindHi: d.vec4f(mult[4]!, mult[5]!, mult[6]!, mult[7]!),
    });
    // Order matches the layout: params, pos, link, spring, force.
    const bind = rawBindGroup(device, root, layout, [pipe.params, pos.buffer, link, spring, out.buffer], "springForce");
    submitDispatch(device, "springForce", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const pos = inputs[0]!;
    const table = inputs[1]!;
    const n = pointsN(pos.shape, "springForce.pos");
    const K = slotsPerParticle(table.shape, n, "springForce");
    const link = part(table, "link").data;
    const spring = part(table, "spring").data;
    if (!pos.data || !(link instanceof Uint32Array) || !spring) throw new Error("springForce: cpu golden needs host data");
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, data: forceCpu(pos.data, link, spring, n, K, params) }];
  },
};
