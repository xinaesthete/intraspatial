// Bond turnover as graph ops: `bondBreak` and `bondForm`. Each takes the bond table and returns a
// new one, so the table is state on a feedback edge and never leaves the device. This is where
// viscoelasticity comes from: a network whose bonds never break is elastic, and one whose bonds
// turn over relaxes its stress (docs/bond-primitive.md).
//
// Both keep the table's invariant without one row writing another: a particle decides about its
// own row only, from values both ends of a bond hold identically.
//
//   bondBreak  a bond goes when either end is dead, when its strain passes a threshold, or at
//              random with a set lifetime (a hash of the unordered pair, so both ends agree)
//   bondForm   each particle with a free slot proposes to its nearest eligible neighbour; a bond
//              forms only where two particles propose to each other
//
// The maths is `breakBonds` and `formBonds` in src/gpu/sim/springs/turnover.ts.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { pairHash, probThreshold, SALT } from "#src/gpu/sim/springs/rng";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { FieldValue, ResidentBuffer, Shape } from "../../handle";
import type { ExecCtx, OpType, ParamSpec, Params } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { BOND_TABLE_PORT, bondTableValue, NO_LINK, slotsPerParticle, VEC4 } from "./bondTable";
import { cutoff, DEAD_SPECIES, NO_CUTOFF, numbers, pointsN } from "./common";
import { pairHashGpu } from "./rngTgsl";

const WG = 64;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);
const f32s = (n: number) => d.arrayOf(d.f32, n);

type Resolvable = Parameters<typeof tgpu.resolveWithContext>[0][number];
/** One pipeline per backend root for a kernel. */
function pipelineOf(name: string, kernel: Resolvable): (device: GPUDevice, root: Root) => GPUComputePipeline {
  const cache = new WeakMap<object, GPUComputePipeline>();
  return (device, root) => {
    let p = cache.get(root as object);
    if (!p) {
      p = typegpuComputePipeline(device, root, tgpu.resolveWithContext([kernel], { names: "strict" }), name, name);
      cache.set(root as object, p);
    }
    return p;
  };
}

interface TableBuffers {
  readonly n: number;
  readonly K: number;
  readonly link: ResidentBuffer;
  readonly spring: ResidentBuffer;
}

function residentTable(table: FieldValue, n: number, who: string): TableBuffers {
  const K = slotsPerParticle(table.shape, n, who);
  const link = table.parts?.link?.buffer;
  const spring = table.parts?.spring?.buffer;
  if (!link || !spring) throw new Error(`${who}: resident op received a bond table that is not on the device`);
  return { n, K, link, spring };
}

function hostTable(table: FieldValue, n: number, who: string): { K: number; link: Uint32Array; spring: Float32Array } {
  const K = slotsPerParticle(table.shape, n, who);
  const link = table.parts?.link?.data;
  const spring = table.parts?.spring?.data;
  if (!(link instanceof Uint32Array) || !(spring instanceof Float32Array))
    throw new Error(`${who}: cpu golden needs the bond table on the host`);
  return { K, link, spring };
}

/** A new resident bond table over two fresh leases. */
async function leaseTable(ctx: ExecCtx, slots: number) {
  const link = await ctx.backend.lease(slots * 4);
  let spring: ResidentBuffer;
  try {
    spring = await ctx.backend.lease(slots * 16);
  } catch (e) {
    ctx.backend.release(link);
    throw e;
  }
  const shape: Shape = { kind: "points", n: slots };
  const value = bondTableValue(new Uint32Array(0), new Float32Array(0));
  return {
    link,
    spring,
    value: {
      ...value,
      shape: { ...value.shape, parts: { link: shape, spring: shape } } as Shape,
      parts: {
        link: { shape, dtype: "u32" as const, buffer: link },
        spring: { shape, dtype: "f32" as const, element: VEC4, buffer: spring },
      },
    } satisfies FieldValue,
  };
}

const TABLE_INPUTS = [
  { name: "pos", kind: "points" as const, dtype: "f32" as const },
  { name: "species", kind: "points" as const, dtype: "u32" as const },
  { name: "bonds", kind: "bundle" as const, bundle: BOND_TABLE_PORT },
];
const STEP_INPUT = { name: "step", kind: "scalar" as const, dtype: "u32" as const };
const TABLE_OUTPUT = [{ name: "bonds", kind: "bundle" as const, bundle: BOND_TABLE_PORT }];
const SEED: ParamSpec = { name: "seed", type: "int", default: 1, min: 0, describe: "Seed of the random stream." };
const LENGTHS: ParamSpec[] = [
  { name: "springlen", type: "number", default: 1, min: 0, max: 4, step: 0.01, describe: "Scales every rest length." },
  { name: "springlenfix", type: "number", default: 0, min: -2, max: 2, step: 0.01, describe: "Added to every rest length." },
];
const host = (v: FieldValue | undefined, who: string) => {
  if (!v?.data) throw new Error(`${who}: cpu golden needs host data`);
  return v.data;
};
const device = (v: FieldValue | undefined, who: string) => {
  if (!v?.buffer) throw new Error(`${who}: resident op received a non-resident input`);
  return v.buffer;
};

// ── bondBreak ──────────────────────────────────────────────────────────────────────────────────

const BreakParams = d.struct({
  n: d.u32,
  K: d.u32,
  kinds: d.u32,
  tensionOnly: d.u32,
  /** A bond breaks at random when its pair's hash is below this. 0: never. */
  tauThreshold: d.u32,
  salt: d.u32,
  maxStrain: d.f32,
  springlen: d.f32,
  springlenfix: d.f32,
});
const breakLayout = tgpu
  .bindGroupLayout({
    params: { uniform: BreakParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    link: { storage: u32s, access: "readonly" },
    spring: { storage: vec4s, access: "readonly" },
    step: { storage: u32s, access: "readonly" },
    linkOut: { storage: u32s, access: "mutable" },
    springOut: { storage: vec4s, access: "mutable" },
  })
  .$name("bondBreak");

const breakFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = breakLayout.$.params;
    if (i < P.n) {
      const me = breakLayout.$.pos[i]!.xyz;
      const dead = breakLayout.$.species[i]! === 255;
      for (let k = d.u32(0); k < P.K; k++) {
        const o = i * P.K + k;
        const link = breakLayout.$.link[o]!;
        const bond = breakLayout.$.spring[o]!;
        let kept = link;
        if (link !== 0xffffffff) {
          const j = link & 0xffffff;
          const kind = link >> 24;
          let gone = dead || breakLayout.$.species[j]! === 255;
          if (!gone && ((P.kinds >> kind) & 1) === 1) {
            const len = std.length(std.sub(breakLayout.$.pos[j]!.xyz, me));
            let strain = len / (bond.x * P.springlen + P.springlenfix) - 1;
            if (P.tensionOnly === 0) {
              strain = std.abs(strain);
            }
            if (strain > P.maxStrain) {
              gone = true;
            }
            if (pairHashGpu(i, j, breakLayout.$.step[0]!, P.salt) < P.tauThreshold) {
              gone = true;
            }
          }
          if (gone) {
            kept = 0xffffffff;
          }
        }
        breakLayout.$.linkOut[o] = kept;
        breakLayout.$.springOut[o] = d.vec4f(bond.x, bond.y, bond.z, bond.w);
      }
    }
  })
  .$name("bondBreak");

const BREAK_PARAMS: ParamSpec[] = [
  { name: "kinds", type: "int", default: 0xff, min: 0, max: 0xff, describe: "Bit per bond kind this rule may break." },
  {
    name: "maxStrain",
    type: "number",
    default: NO_CUTOFF,
    min: 0,
    describe: "A bond breaks when its strain passes this. Leave very large for never.",
  },
  {
    name: "tensionOnly",
    type: "bool",
    default: false,
    describe: "Only stretched bonds break by strain, so a squeezed network keeps its bonds.",
  },
  {
    name: "tau",
    type: "number",
    default: NO_CUTOFF,
    min: 1,
    describe: "Mean lifetime of a bond in steps, breaking at random. Leave very large for never.",
  },
  ...LENGTHS,
  SEED,
];
const breakNum = numbers(BREAK_PARAMS, "bondBreak");
const breakPipe = pipelineOf("bondBreak", breakFn);
const breakUniforms = new WeakMap<object, ReturnType<typeof makeBreakUniform>>();
const makeBreakUniform = (root: Root) => root.createBuffer(BreakParams).$usage("uniform").$name("bondBreak:params");

/** The hash threshold of a random lifetime: 0 when the lifetime is "never". */
const tauThreshold = (tau: number) => (tau >= NO_CUTOFF || !Number.isFinite(tau) ? 0 : probThreshold(1 / tau));

export const bondBreakOp: OpType = {
  name: "bondBreak",
  label: "Break bonds",
  describe: "Removes bonds that are overstrained, have reached the end of a random lifetime, or lead to a dead particle.",
  category: "Springs",
  help: {
    detail:
      "Each bond is listed under both of its ends. Each end decides alone whether the bond goes, from " +
      "things both ends see the same: the bond's length, and a random number worked out from the pair. " +
      "So they always agree, and the table stays consistent.",
  },
  inputs: [...TABLE_INPUTS, STEP_INPUT],
  outputs: TABLE_OUTPUT,
  params: BREAK_PARAMS,
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "bondBreak.pos");
    if (pointsN(inputs[1]!, "bondBreak.species") !== n)
      throw new Error("bondBreak: pos and species are for different numbers of particles");
    slotsPerParticle(inputs[2]!, n, "bondBreak");
    return [inputs[2]!];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = pointsN(inputs[0]!.shape, "bondBreak.pos");
    const table = residentTable(inputs[2]!, n, "bondBreak");
    const dev = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    let uniform = breakUniforms.get(root as object);
    if (!uniform) {
      uniform = makeBreakUniform(root);
      breakUniforms.set(root as object, uniform);
    }
    // Read the settings and inputs before leasing: a bad one must not leave the outputs leased.
    const values = {
      n,
      K: table.K,
      kinds: breakNum(params, "kinds"),
      tensionOnly: params.tensionOnly ? 1 : 0,
      tauThreshold: tauThreshold(breakNum(params, "tau")),
      salt: (breakNum(params, "seed") ^ SALT.bondBreak) >>> 0,
      maxStrain: cutoff(breakNum(params, "maxStrain")),
      springlen: breakNum(params, "springlen"),
      springlenfix: breakNum(params, "springlenfix"),
    };
    const pos = device(inputs[0], "bondBreak.pos");
    const species = device(inputs[1], "bondBreak.species");
    const step = device(inputs[3], "bondBreak.step");
    const out = await leaseTable(ctx, n * table.K);
    try {
      uniform.write(values);
      const bind = rawBindGroup(
        dev,
        root,
        breakLayout,
        [uniform, pos, species, table.link, table.spring, step, out.link.buffer, out.spring.buffer],
        "bondBreak",
      );
      submitDispatch(dev, "bondBreak", breakPipe(dev, root), bind, Math.ceil(n / WG));
    } catch (e) {
      ctx.backend.release(out.link);
      ctx.backend.release(out.spring);
      throw e;
    }
    return [out.value];
  },
  cpuGolden(inputs, params) {
    const n = pointsN(inputs[0]!.shape, "bondBreak.pos");
    const pos = host(inputs[0], "bondBreak.pos");
    const species = host(inputs[1], "bondBreak.species");
    const { K, link, spring } = hostTable(inputs[2]!, n, "bondBreak");
    const step = host(inputs[3], "bondBreak.step")[0]!;
    const out = link.slice();
    const kinds = breakNum(params, "kinds");
    const maxStrain = cutoff(breakNum(params, "maxStrain"));
    const threshold = tauThreshold(breakNum(params, "tau"));
    const salt = (breakNum(params, "seed") ^ SALT.bondBreak) >>> 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) {
        const o = i * K + k;
        const bits = link[o]!;
        if (bits === NO_LINK) continue;
        const j = bits & 0xffffff;
        let gone = species[i] === DEAD_SPECIES || species[j] === DEAD_SPECIES;
        if (!gone && ((kinds >> (bits >>> 24)) & 1) === 1) {
          const len = Math.hypot(pos[j * 4]! - pos[i * 4]!, pos[j * 4 + 1]! - pos[i * 4 + 1]!, pos[j * 4 + 2]! - pos[i * 4 + 2]!);
          const strain = len / (spring[o * 4]! * breakNum(params, "springlen") + breakNum(params, "springlenfix")) - 1;
          if ((params.tensionOnly ? strain : Math.abs(strain)) > maxStrain) gone = true;
          if (pairHash(i, j, step, salt) < threshold) gone = true;
        }
        if (gone) out[o] = NO_LINK;
      }
    }
    return [bondTableValue(out, spring.slice())];
  },
};

// ── bondForm ───────────────────────────────────────────────────────────────────────────────────

const FormParams = d.struct({
  n: d.u32,
  K: d.u32,
  speciesCount: d.u32,
  /** A pair may bond this step when its hash is below this. */
  probThreshold: d.u32,
  salt: d.u32,
  /** 1: a new bond's rest length is its current length. 0: `rest`. */
  restCurrent: d.u32,
  range: d.f32,
  rest: d.f32,
  springlen: d.f32,
  springlenfix: d.f32,
  /** Strength of a new bond of each kind: kinds 0–3, then 4–7. */
  strengthLo: d.vec4f,
  strengthHi: d.vec4f,
});
const proposeLayout = tgpu
  .bindGroupLayout({
    params: { uniform: FormParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    link: { storage: u32s, access: "readonly" },
    kindOf: { storage: f32s, access: "readonly" },
    step: { storage: u32s, access: "readonly" },
    proposal: { storage: u32s, access: "mutable" },
  })
  .$name("bondFormPropose");

const proposeFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = proposeLayout.$.params;
    if (i < P.n) {
      let best = d.u32(0xffffffff);
      let bestLen = d.f32(3e38);
      const si = proposeLayout.$.species[i]!;
      let room = false;
      for (let k = d.u32(0); k < P.K; k++) {
        if (proposeLayout.$.link[i * P.K + k]! === 0xffffffff) {
          room = true;
        }
      }
      if (si !== 255 && room) {
        const me = proposeLayout.$.pos[i]!.xyz;
        for (let j = d.u32(0); j < P.n; j++) {
          const sj = proposeLayout.$.species[j]!;
          if (j !== i && sj !== 255) {
            const len = std.length(std.sub(proposeLayout.$.pos[j]!.xyz, me));
            if (proposeLayout.$.kindOf[si * P.speciesCount + sj]! >= 0 && len > 0 && len <= P.range && len < bestLen) {
              // Not already bonded to it, and it has a slot to take the bond in.
              let bonded = false;
              let theirRoom = false;
              for (let k = d.u32(0); k < P.K; k++) {
                const mine = proposeLayout.$.link[i * P.K + k]!;
                if (mine !== 0xffffffff && (mine & 0xffffff) === j) {
                  bonded = true;
                }
                if (proposeLayout.$.link[j * P.K + k]! === 0xffffffff) {
                  theirRoom = true;
                }
              }
              if (!bonded && theirRoom && pairHashGpu(i, j, proposeLayout.$.step[0]!, P.salt) < P.probThreshold) {
                best = j;
                bestLen = len;
              }
            }
          }
        }
      }
      proposeLayout.$.proposal[i] = best;
    }
  })
  .$name("bondFormPropose");

const acceptLayout = tgpu
  .bindGroupLayout({
    params: { uniform: FormParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    link: { storage: u32s, access: "readonly" },
    spring: { storage: vec4s, access: "readonly" },
    kindOf: { storage: f32s, access: "readonly" },
    proposal: { storage: u32s, access: "readonly" },
    linkOut: { storage: u32s, access: "mutable" },
    springOut: { storage: vec4s, access: "mutable" },
  })
  .$name("bondFormAccept");

const acceptFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = acceptLayout.$.params;
    if (i < P.n) {
      const j = acceptLayout.$.proposal[i]!;
      let mutual = false;
      if (j !== 0xffffffff) {
        mutual = acceptLayout.$.proposal[j]! === i;
      }
      // The new bond goes in the first empty slot.
      let placed = false;
      for (let k = d.u32(0); k < P.K; k++) {
        const o = i * P.K + k;
        const link = acceptLayout.$.link[o]!;
        const bond = acceptLayout.$.spring[o]!;
        if (mutual && !placed && link === 0xffffffff) {
          placed = true;
          // Looked up lower species first, so both ends store the same kind whatever the matrix.
          const sa = std.min(acceptLayout.$.species[i]!, acceptLayout.$.species[j]!);
          const sb = std.max(acceptLayout.$.species[i]!, acceptLayout.$.species[j]!);
          const kind = d.u32(acceptLayout.$.kindOf[sa * P.speciesCount + sb]!);
          let strength = P.strengthLo.x;
          if (kind === 1) {
            strength = P.strengthLo.y;
          }
          if (kind === 2) {
            strength = P.strengthLo.z;
          }
          if (kind === 3) {
            strength = P.strengthLo.w;
          }
          if (kind === 4) {
            strength = P.strengthHi.x;
          }
          if (kind === 5) {
            strength = P.strengthHi.y;
          }
          if (kind === 6) {
            strength = P.strengthHi.z;
          }
          if (kind === 7) {
            strength = P.strengthHi.w;
          }
          let rest = P.rest;
          if (P.restCurrent === 1) {
            const len = std.length(std.sub(acceptLayout.$.pos[j]!.xyz, acceptLayout.$.pos[i]!.xyz));
            rest = (len - P.springlenfix) / P.springlen;
          }
          acceptLayout.$.linkOut[o] = j | (kind << 24);
          acceptLayout.$.springOut[o] = d.vec4f(rest, strength, 0, 0);
        } else {
          acceptLayout.$.linkOut[o] = link;
          acceptLayout.$.springOut[o] = d.vec4f(bond.x, bond.y, bond.z, bond.w);
        }
      }
    }
  })
  .$name("bondFormAccept");

const FORM_PARAMS: ParamSpec[] = [
  { name: "range", type: "number", default: 1.5, min: 0, max: 100, step: 0.01, describe: "Largest distance at which a bond forms." },
  {
    name: "prob",
    type: "number",
    default: 1,
    min: 0,
    max: 1,
    step: 0.0001,
    describe: "Chance per step that a given pair is allowed to bond.",
  },
  {
    name: "restCurrent",
    type: "bool",
    default: true,
    describe: "A new bond's rest length is its current length, so it forms without stress.",
  },
  {
    name: "rest",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    step: 0.01,
    describe: "Rest length of a new bond, when not its current length.",
  },
  ...Array.from({ length: 8 }, (_, kind) => ({
    name: `strength${kind}`,
    type: "number" as const,
    default: 1,
    min: 0,
    max: 10,
    step: 0.01,
    describe: `Strength of a new bond of kind ${kind}.`,
    tags: ["strength"],
  })),
  ...LENGTHS,
  SEED,
];
const formNum = numbers(FORM_PARAMS, "bondForm");
const proposePipe = pipelineOf("bondFormPropose", proposeFn);
const acceptPipe = pipelineOf("bondFormAccept", acceptFn);
const formUniforms = new WeakMap<object, ReturnType<typeof makeFormUniform>>();
const makeFormUniform = (root: Root) => root.createBuffer(FormParams).$usage("uniform").$name("bondForm:params");
const strengths = (p: Params) => Array.from({ length: 8 }, (_, kind) => formNum(p, `strength${kind}`));

function speciesSide(shape: Shape): number {
  if (shape.kind !== "matrix" || shape.rows !== shape.cols) throw new Error("bondForm.kindOf: expected a square species × species matrix");
  return shape.rows;
}

/** Whether a new bond takes its current length as rest. That divides by `springlen`, so a
 *  scale of zero cannot be asked for together with it. */
function restCurrent(params: Record<string, unknown>): boolean {
  if (params.restCurrent === false) return false;
  if (formNum(params, "springlen") === 0) throw new Error("bondForm: restCurrent needs a springlen that is not zero");
  return true;
}

export const bondFormOp: OpType = {
  name: "bondForm",
  label: "Form bonds",
  describe: "Forms a bond between two nearby particles when each is the other's nearest eligible neighbour.",
  category: "Springs",
  help: {
    detail:
      "Every particle with room for another bond picks its nearest neighbour it could bond to. A bond " +
      "forms only where two particles pick each other. A particle gains at most one bond a step. The " +
      "`kindOf` matrix says what kind of bond two species make, with −1 for none.",
  },
  inputs: [...TABLE_INPUTS, { name: "kindOf", kind: "matrix", dtype: "f32" }, STEP_INPUT],
  outputs: TABLE_OUTPUT,
  params: FORM_PARAMS,
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "bondForm.pos");
    if (pointsN(inputs[1]!, "bondForm.species") !== n) throw new Error("bondForm: pos and species are for different numbers of particles");
    slotsPerParticle(inputs[2]!, n, "bondForm");
    speciesSide(inputs[3]!);
    return [inputs[2]!];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = pointsN(inputs[0]!.shape, "bondForm.pos");
    const table = residentTable(inputs[2]!, n, "bondForm");
    const s = strengths(params);
    // Numbers and input buffers before any lease, so a bad call cannot keep a buffer. The uniform
    // itself is written only after the leases, with nothing awaited before the submits: two
    // formations on one device share the uniform, and a write before an await let the other
    // call's rest length win.
    const written = {
      n,
      K: table.K,
      speciesCount: speciesSide(inputs[3]!.shape),
      probThreshold: probThreshold(formNum(params, "prob")),
      salt: (formNum(params, "seed") ^ SALT.bondForm) >>> 0,
      restCurrent: restCurrent(params) ? 1 : 0,
      range: formNum(params, "range"),
      rest: formNum(params, "rest"),
      springlen: formNum(params, "springlen"),
      springlenfix: formNum(params, "springlenfix"),
      strengthLo: d.vec4f(s[0]!, s[1]!, s[2]!, s[3]!),
      strengthHi: d.vec4f(s[4]!, s[5]!, s[6]!, s[7]!),
    };
    const pos = device(inputs[0], "bondForm.pos");
    const species = device(inputs[1], "bondForm.species");
    const kindOf = device(inputs[3], "bondForm.kindOf");
    const step = device(inputs[4], "bondForm.step");
    const dev = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    let uniform = formUniforms.get(root);
    if (!uniform) {
      uniform = makeFormUniform(root);
      formUniforms.set(root, uniform);
    }
    // The proposals live only between the two passes.
    const proposal = await ctx.backend.lease(n * 4);
    try {
      const out = await leaseTable(ctx, n * table.K);
      try {
        uniform.write(written);
        const propose = rawBindGroup(
          dev,
          root,
          proposeLayout,
          [uniform, pos, species, table.link, kindOf, step, proposal.buffer],
          "bondForm:propose",
        );
        submitDispatch(dev, "bondForm:propose", proposePipe(dev, root), propose, Math.ceil(n / WG));
        const accept = rawBindGroup(
          dev,
          root,
          acceptLayout,
          [uniform, pos, species, table.link, table.spring, kindOf, proposal.buffer, out.link.buffer, out.spring.buffer],
          "bondForm:accept",
        );
        submitDispatch(dev, "bondForm:accept", acceptPipe(dev, root), accept, Math.ceil(n / WG));
        ctx.backend.release(proposal);
        return [out.value];
      } catch (e) {
        ctx.backend.release(out.link);
        ctx.backend.release(out.spring);
        throw e;
      }
    } catch (e) {
      ctx.backend.release(proposal);
      throw e;
    }
  },
  cpuGolden(inputs, params) {
    restCurrent(params); // refuse a zero scale here, as `execute` does, whether or not a bond forms
    const n = pointsN(inputs[0]!.shape, "bondForm.pos");
    const pos = host(inputs[0], "bondForm.pos");
    const species = host(inputs[1], "bondForm.species");
    const { K, link, spring } = hostTable(inputs[2]!, n, "bondForm");
    const kindOf = host(inputs[3], "bondForm.kindOf");
    const count = speciesSide(inputs[3]!.shape);
    const step = host(inputs[4], "bondForm.step")[0]!;
    const range = formNum(params, "range");
    const threshold = probThreshold(formNum(params, "prob"));
    const salt = (formNum(params, "seed") ^ SALT.bondForm) >>> 0;
    const dist = (a: number, b: number) =>
      Math.hypot(pos[b * 4]! - pos[a * 4]!, pos[b * 4 + 1]! - pos[a * 4 + 1]!, pos[b * 4 + 2]! - pos[a * 4 + 2]!);
    const freeSlot = (i: number) => {
      for (let k = 0; k < K; k++) if (link[i * K + k] === NO_LINK) return i * K + k;
      return -1;
    };
    const bonded = (i: number, j: number) => {
      for (let k = 0; k < K; k++) if (link[i * K + k] !== NO_LINK && (link[i * K + k]! & 0xffffff) === j) return true;
      return false;
    };
    const proposal = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      const si = species[i]!;
      if (si === DEAD_SPECIES || freeSlot(i) < 0) continue;
      let bestLen = Number.POSITIVE_INFINITY;
      for (let j = 0; j < n; j++) {
        const sj = species[j]!;
        if (j === i || sj === DEAD_SPECIES || !(kindOf[si * count + sj]! >= 0)) continue;
        // In single precision, as the kernel measures it, so that ties fall the same way.
        const len = Math.fround(dist(i, j));
        if (!(len > 0 && len <= Math.fround(range) && len < bestLen)) continue;
        if (bonded(i, j) || freeSlot(j) < 0 || !(pairHash(i, j, step, salt) < threshold)) continue;
        proposal[i] = j;
        bestLen = len;
      }
    }
    const linkOut = link.slice();
    const springOut = spring.slice();
    const s = strengths(params);
    for (let i = 0; i < n; i++) {
      const j = proposal[i]!;
      if (j < 0 || proposal[j] !== i) continue;
      const o = freeSlot(i);
      const kind = kindOf[Math.min(species[i]!, species[j]!) * count + Math.max(species[i]!, species[j]!)]!;
      linkOut[o] = (j | (kind << 24)) >>> 0;
      const rest = !restCurrent(params)
        ? formNum(params, "rest")
        : (dist(i, j) - formNum(params, "springlenfix")) / formNum(params, "springlen");
      springOut.set([rest, s[kind]!, 0, 0], o * 4);
    }
    return [bondTableValue(linkOut, springOut)];
  },
};
