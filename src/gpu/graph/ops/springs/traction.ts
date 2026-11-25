// `traction`: thrust as a pull on what a particle grips, rather than a force from nowhere.
//
// `thrust` is each particle's outward pull, as the `outward` op gives it with protrusion off. A
// particle grips the partners of its bonds whose kind is in `kinds` and that lie ahead of it,
// along its pull. It keeps its pull when it grips something, and each gripped partner is pulled
// back by a share of it, so the forces sum to zero:
//
//   F_i += T_i                               if Σ_k a_ik > 0
//   F_j −= T_i · a_ij / Σ_k a_ik             for each gripped partner j
//   a_ij = max(0, (x_j − x_i)·T̂_i / |x_j − x_i|)
//
// An elastic network then holds a pulling particle back with the strain it has put in, and it
// advances only where bonds give way. Each particle gathers over its own row of the bond table
// (the table lists a bond under both ends), so nothing is written to another particle.
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

const TractionParams = d.struct({ n: d.u32, K: d.u32, kinds: d.u32 });

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: TractionParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    thrust: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    link: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("traction");

const tractionFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = layout.$.params;
    if (i < P.n) {
      const me = layout.$.pos[i]!.xyz;
      let f = d.vec3f(0, 0, 0);
      // Its own pull, if it grips anything ahead.
      const own = layout.$.thrust[i]!.xyz;
      const ownLen = std.length(own);
      if (ownLen > 0) {
        const u = std.mul(1 / ownLen, own);
        let total = d.f32(0);
        for (let k = d.u32(0); k < P.K; k++) {
          const link = layout.$.link[i * P.K + k]!;
          if (link !== 0xffffffff && ((P.kinds >> (link >> 24)) & 1) === 1) {
            const dd = std.sub(layout.$.pos[link & 0xffffff]!.xyz, me);
            const len = std.length(dd);
            if (len > 0) {
              total += std.max(0, std.dot(dd, u) / len);
            }
          }
        }
        if (total > 0) {
          f = std.add(f, own);
        }
      }
      // A share of the pull of each partner that grips this particle.
      for (let k = d.u32(0); k < P.K; k++) {
        const link = layout.$.link[i * P.K + k]!;
        if (link !== 0xffffffff && ((P.kinds >> (link >> 24)) & 1) === 1) {
          const j = link & 0xffffff;
          const pull = layout.$.thrust[j]!.xyz;
          const pullLen = std.length(pull);
          if (pullLen > 0) {
            const u = std.mul(1 / pullLen, pull);
            const at = layout.$.pos[j]!.xyz;
            const toMe = std.sub(me, at);
            const myLen = std.length(toMe);
            let mine = d.f32(0);
            if (myLen > 0) {
              mine = std.max(0, std.dot(toMe, u) / myLen);
            }
            if (mine > 0) {
              let total = d.f32(0);
              for (let m = d.u32(0); m < P.K; m++) {
                const other = layout.$.link[j * P.K + m]!;
                if (other !== 0xffffffff && ((P.kinds >> (other >> 24)) & 1) === 1) {
                  const dd = std.sub(layout.$.pos[other & 0xffffff]!.xyz, at);
                  const len = std.length(dd);
                  if (len > 0) {
                    total += std.max(0, std.dot(dd, u) / len);
                  }
                }
              }
              f = std.sub(f, std.mul((pullLen * mine) / total, u));
            }
          }
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("traction");

const makeParams = (root: Root) => root.createBuffer(TractionParams).$usage("uniform").$name("traction:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([tractionFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "traction", "traction"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const PARAMS: ParamSpec[] = [
  {
    name: "kinds",
    type: "int",
    default: 0,
    min: 0,
    max: (1 << BOND_KINDS) - 1,
    describe: "Bit per bond kind a pulling particle grips through.",
  },
];

function kinds(p: Params): number {
  const v = p.kinds ?? 0;
  if (typeof v !== "number" || !Number.isInteger(v) || v < 0 || v >= 1 << BOND_KINDS) {
    throw new Error(`traction: kinds must be a bitmask of the ${BOND_KINDS} bond kinds, got ${JSON.stringify(v)}`);
  }
  return v;
}

function pointsN(shape: Shape, who: string): number {
  if (shape.kind !== "points") throw new Error(`${who}: expected a points field`);
  return shape.n;
}

/** The same sums on the host. */
function tractionCpu(
  pos: ArrayLike<number>,
  thrust: ArrayLike<number>,
  link: Uint32Array,
  n: number,
  K: number,
  mask: number,
): Float32Array {
  const out = new Float64Array(n * 4);
  const grips = (bits: number) => bits !== NO_LINK && ((mask >> (bits >>> 24)) & 1) === 1;
  /** Particle `i`'s forward cosine to each gripped partner, by slot; 0 for the rest. */
  const ahead = (i: number, ux: number, uy: number, uz: number): number[] =>
    Array.from({ length: K }, (_, k) => {
      const bits = link[i * K + k]!;
      if (!grips(bits)) return 0;
      const j = bits & 0xffffff;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      return len > 0 ? Math.max(0, (dx * ux + dy * uy + dz * uz) / len) : 0;
    });
  for (let i = 0; i < n; i++) {
    const tx = thrust[i * 4]!;
    const ty = thrust[i * 4 + 1]!;
    const tz = thrust[i * 4 + 2]!;
    const pull = Math.hypot(tx, ty, tz);
    if (!(pull > 0)) continue;
    const [ux, uy, uz] = [tx / pull, ty / pull, tz / pull];
    const a = ahead(i, ux, uy, uz);
    const total = a.reduce((s, v) => s + v, 0);
    if (!(total > 0)) continue;
    out[i * 4]! += tx;
    out[i * 4 + 1]! += ty;
    out[i * 4 + 2]! += tz;
    a.forEach((v, k) => {
      if (v === 0) return;
      const j = link[i * K + k]! & 0xffffff;
      const share = (pull * v) / total;
      out[j * 4]! -= ux * share;
      out[j * 4 + 1]! -= uy * share;
      out[j * 4 + 2]! -= uz * share;
    });
  }
  return Float32Array.from(out);
}

function linkPart(table: FieldValue): FieldValue {
  const v = table.parts?.link;
  if (!v) throw new Error(`traction: bond table has no "link" part`);
  return v;
}

export const tractionOp: OpType = {
  name: "traction",
  label: "Traction",
  describe: "Makes a pull act on what the particle grips: it pulls itself forward and its grip back.",
  category: "Springs",
  help: {
    detail:
      "Takes each particle's pull (from `Outward push` with protrusion off) and turns it into traction. A " +
      "particle grips the partners of its bonds of the chosen kinds that lie ahead of it. It keeps its pull " +
      "only if it grips something, and those partners are pulled back by the same total force, shared by " +
      "how far ahead each lies. The forces sum to zero, so a pull strains what it grips.",
    math: "F_i \\mathrel{+}= T_i,\\quad F_j \\mathrel{-}= T_i\\,\\frac{a_{ij}}{\\sum_k a_{ik}},\\quad a_{ij}=\\max\\!\\left(0,\\hat{T}_i\\cdot\\frac{x_j-x_i}{\\lVert x_j-x_i\\rVert}\\right)",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "thrust", kind: "points", dtype: "f32" },
    { name: "bonds", kind: "bundle", bundle: BOND_TABLE_PORT },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "traction.pos");
    if (pointsN(inputs[1]!, "traction.thrust") !== n) throw new Error("traction: thrust is not for the same particles as pos");
    slotsPerParticle(inputs[2]!, n, "traction");
    return [{ kind: "points", n }];
  },
  inferElements(inputs) {
    inputs.slice(0, 2).forEach((el, k) => {
      if (!el || el.kind !== "vec" || el.n !== 4) throw new Error(`traction.${k === 0 ? "pos" : "thrust"}: expected a vec4 points field`);
    });
    return [VEC4];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const [pos, thrust, table] = inputs as [FieldValue, FieldValue, FieldValue];
    const n = pointsN(pos.shape, "traction.pos");
    const K = slotsPerParticle(table.shape, n, "traction");
    const mask = kinds(params);
    const link = linkPart(table).buffer;
    if (!pos.buffer || !thrust.buffer || !link) throw new Error("traction: resident op received a non-resident input");
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(Math.max(n, 1) * 16);
    try {
      pipe.params.write({ n, K, kinds: mask });
      const bind = rawBindGroup(device, root, layout, [pipe.params, pos.buffer, thrust.buffer, link, out.buffer], "traction");
      submitDispatch(device, "traction", pipe.pipeline, bind, Math.max(1, Math.ceil(n / WG)));
    } catch (e) {
      ctx.backend.release(out);
      throw e;
    }
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const [pos, thrust, table] = inputs as [FieldValue, FieldValue, FieldValue];
    const n = pointsN(pos.shape, "traction.pos");
    const K = slotsPerParticle(table.shape, n, "traction");
    const link = linkPart(table).data;
    if (!pos.data || !thrust.data || !(link instanceof Uint32Array)) throw new Error("traction: cpu golden needs host data");
    return [
      { shape: { kind: "points", n }, dtype: "f32", element: VEC4, data: tractionCpu(pos.data, thrust.data, link, n, K, kinds(params)) },
    ];
  },
};
