// `adhesion`: short-range contact between particles, by species, as a graph op. Two particles
// that overlap push apart in proportion to the overlap. Between touching and a little further
// they attract, strongest midway and zero at both ends, so the force is continuous:
//
//   len < σ                 F = −repulsion · (σ − len)
//   σ ≤ len < σ(1 + range)  F = adhesion · 4u(1 − u),   u = (len − σ) / (σ · range)
//
// σ is the sum of the two radii. `repulsion` and `adhesion` are species × species tables, given
// as two small matrix inputs so that a model sets one number per pair of species.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { Shape } from "../../handle";
import type { OpType, ParamSpec, Params } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { VEC4 } from "./bondTable";
import { cutoff, DEAD_SPECIES, hostF32, numbers, PAIR_RANGE, requireScalar, requireVec, residentBuffer, sameParticles } from "./common";

const WG = 64;
const AdhesionParams = d.struct({ n: d.u32, speciesCount: d.u32, range: d.f32, pairRange: d.f32 });

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: AdhesionParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    radius: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    repulsion: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    adhesion: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("adhesion");

const adhesionFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const si = layout.$.species[i]!;
      if (si !== 255) {
        const me = layout.$.pos[i]!.xyz;
        const ri = layout.$.radius[i]!;
        for (let j = d.u32(0); j < layout.$.params.n; j++) {
          const sj = layout.$.species[j]!;
          // A species the tables do not cover takes no part, rather than reading another pair's cell.
          if (j !== i && si < layout.$.params.speciesCount && sj < layout.$.params.speciesCount) {
            const dd = std.sub(layout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            if (len > 0 && len <= layout.$.params.pairRange) {
              const pair = si * layout.$.params.speciesCount + sj;
              const sigma = ri + layout.$.radius[j]!;
              let along = d.f32(0);
              if (len < sigma) {
                along = -layout.$.repulsion[pair]! * (sigma - len);
              } else {
                const reach = sigma * layout.$.params.range;
                const stick = layout.$.adhesion[pair]!;
                if (stick !== 0 && len < sigma + reach) {
                  const u = (len - sigma) / reach;
                  along = stick * 4 * u * (1 - u);
                }
              }
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("adhesion");

const makeParams = (root: Root) => root.createBuffer(AdhesionParams).$usage("uniform").$name("adhesion:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([adhesionFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "adhesion", "adhesion"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const PARAMS: ParamSpec[] = [
  {
    name: "adhesionRange",
    type: "number",
    default: 0.5,
    min: 0.01,
    max: 2,
    step: 0.01,
    describe: "How far past touching the attraction reaches, as a fraction of the touching distance.",
  },
  PAIR_RANGE,
];
const num = numbers(PARAMS, "adhesion");

/** Species per side of a species × species table. */
function speciesCount(table: Shape, who: string): number {
  if (table.kind !== "matrix" || table.rows !== table.cols) throw new Error(`${who}: expected a square species × species matrix`);
  return table.rows;
}

/** The scalar force along the line to the other particle; positive pulls together. */
export function contactForce(len: number, sigma: number, repulsion: number, adhesion: number, range: number): number {
  if (len < sigma) return -repulsion * (sigma - len);
  const reach = sigma * range;
  if (adhesion === 0 || len >= sigma + reach) return 0;
  const u = (len - sigma) / reach;
  return adhesion * 4 * u * (1 - u);
}

function adhesionCpu(
  pos: ArrayLike<number>,
  species: ArrayLike<number>,
  radius: ArrayLike<number>,
  repulsion: ArrayLike<number>,
  adhesion: ArrayLike<number>,
  n: number,
  count: number,
  p: Params,
): Float32Array {
  const out = new Float32Array(n * 4);
  const range = num(p, "adhesionRange");
  const pairRange = cutoff(num(p, "pairRange"));
  for (let i = 0; i < n; i++) {
    const si = species[i]!;
    // A species the tables do not cover takes no part, as in the kernel.
    if (si === DEAD_SPECIES || si >= count) continue;
    let fx = 0;
    let fy = 0;
    let fz = 0;
    for (let j = 0; j < n; j++) {
      const sj = species[j]!;
      if (j === i || sj === DEAD_SPECIES || sj >= count) continue;
      const dx = pos[j * 4]! - pos[i * 4]!;
      const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
      const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0 || len > pairRange) continue;
      const pair = si * count + sj;
      const sc = contactForce(len, radius[i]! + radius[j]!, repulsion[pair]!, adhesion[pair]!, range) / len;
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

export const adhesionOp: OpType = {
  name: "adhesion",
  label: "Adhesion",
  describe: "Overlapping particles repel; particles just apart attract. Strengths are set per pair of species.",
  category: "Springs",
  help: {
    math: "F=\\begin{cases}-k_r(\\sigma-\\ell)&\\ell<\\sigma\\\\ 4k_a\\,u(1-u)&\\sigma\\le\\ell<\\sigma(1+r)\\end{cases}\\quad u=\\frac{\\ell-\\sigma}{\\sigma r}",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "radius", kind: "points", dtype: "f32" },
    { name: "repulsion", kind: "matrix", dtype: "f32" },
    { name: "adhesion", kind: "matrix", dtype: "f32" },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: PARAMS,
  inferShapes(inputs) {
    const n = sameParticles(inputs.slice(0, 3), "adhesion");
    if (speciesCount(inputs[3]!, "adhesion.repulsion") !== speciesCount(inputs[4]!, "adhesion.adhesion")) {
      throw new Error("adhesion: the repulsion and adhesion tables are different sizes");
    }
    return [{ kind: "points", n }];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "adhesion.pos");
    requireScalar(inputs[1], "adhesion.species");
    requireScalar(inputs[2], "adhesion.radius");
    requireScalar(inputs[3], "adhesion.repulsion");
    requireScalar(inputs[4], "adhesion.adhesion");
    return [VEC4];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = sameParticles(
      inputs.slice(0, 3).map((v) => v.shape),
      "adhesion",
    );
    const count = speciesCount(inputs[3]!.shape, "adhesion.repulsion");
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(n * 16);
    pipe.params.write({ n, speciesCount: count, range: num(params, "adhesionRange"), pairRange: cutoff(num(params, "pairRange")) });
    const bind = rawBindGroup(
      device,
      root,
      layout,
      [
        pipe.params,
        residentBuffer(inputs[0], "adhesion.pos"),
        residentBuffer(inputs[1], "adhesion.species"),
        residentBuffer(inputs[2], "adhesion.radius"),
        residentBuffer(inputs[3], "adhesion.repulsion"),
        residentBuffer(inputs[4], "adhesion.adhesion"),
        out.buffer,
      ],
      "adhesion",
    );
    submitDispatch(device, "adhesion", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs, params) {
    const n = sameParticles(
      inputs.slice(0, 3).map((v) => v.shape),
      "adhesion",
    );
    const count = speciesCount(inputs[3]!.shape, "adhesion.repulsion");
    const data = adhesionCpu(
      hostF32(inputs[0], "adhesion.pos"),
      hostF32(inputs[1], "adhesion.species"),
      hostF32(inputs[2], "adhesion.radius"),
      hostF32(inputs[3], "adhesion.repulsion"),
      hostF32(inputs[4], "adhesion.adhesion"),
      n,
      count,
      params,
    );
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, data }];
  },
};
