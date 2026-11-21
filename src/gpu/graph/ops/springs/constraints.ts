// Constraints as graph ops. Each takes the positions the integrator produced and returns them
// corrected; chain them between `springIntegrate` and the feedback edge, in this order:
//
//   rodConstraint     holds a particle within a range of distances from one other particle
//   centreConstraint  shrinks everything towards a point by a factor each step
//   fixConstraint     pins flagged particles to given positions
//   pullFix           moves particles that have a pull target a set fraction of the way to it
//
// Dead particles and the spare lane of the positions are left alone.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { ParamSpec } from "../../op";
import { DEAD_SPECIES, numbers } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";

const WG = KERNEL_WORKGROUP;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);
const POS = { name: "pos", kind: "points", dtype: "f32", lanes: 4 } as const;
const SPECIES = { name: "species", kind: "points", dtype: "u32" } as const;
const OUT = { name: "pos", dtype: "f32", lanes: 4 } as const;

// ── rod ────────────────────────────────────────────────────────────────────────────────────────

const RodParams = d.struct({ n: d.u32, springlen: d.f32, springlenfix: d.f32 });
const rodLayout = tgpu
  .bindGroupLayout({
    params: { uniform: RodParams },
    pos: { storage: vec4s, access: "readonly" },
    before: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    rod: { storage: vec4s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("rodConstraint");

const rodFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < rodLayout.$.params.n) {
      const here = rodLayout.$.pos[i]!;
      let next = d.vec3f(here.x, here.y, here.z);
      const rod = rodLayout.$.rod[i]!;
      if (rodLayout.$.species[i]! !== 255 && rod.x >= 0 && rod.x < d.f32(rodLayout.$.params.n) && rod.w >= 0) {
        const other = rodLayout.$.before[d.u32(rod.x)]!.xyz;
        const dd = std.sub(other, here.xyz);
        const len = std.length(dd);
        if (len >= 0.001) {
          const lo = rod.y * rodLayout.$.params.springlen + rodLayout.$.params.springlenfix;
          const hi = rod.z * rodLayout.$.params.springlen + rodLayout.$.params.springlenfix;
          const want = std.min(hi, std.max(lo, len)) * rod.w + len * (1 - rod.w);
          next = std.sub(other, std.mul(want / len, dd));
        }
      }
      rodLayout.$.out[i] = d.vec4f(next, here.w);
    }
  })
  .$name("rodConstraint");

const ROD_PARAMS: ParamSpec[] = [
  { name: "springlen", type: "number", default: 1, min: 0, max: 4, step: 0.01, describe: "Scales the rod's two lengths." },
  { name: "springlenfix", type: "number", default: 0, min: -2, max: 2, step: 0.01, describe: "Added to the rod's two lengths." },
];
const rodNum = numbers(ROD_PARAMS, "rodConstraint");

export const rodConstraintOp = kernelOp<d.Infer<typeof RodParams>>({
  name: "rodConstraint",
  label: "Rod",
  describe: "Holds each particle within a range of distances from one other particle.",
  inputs: [
    POS,
    { name: "before", kind: "points", dtype: "f32", lanes: 4 },
    SPECIES,
    { name: "rod", kind: "points", dtype: "f32", lanes: 4 },
  ],
  output: OUT,
  params: ROD_PARAMS,
  layout: rodLayout,
  kernel: rodFn,
  uniform: (root) => root.createBuffer(RodParams).$usage("uniform").$name("rodConstraint:params"),
  uniforms: (n, p) => ({ n, springlen: rodNum(p, "springlen"), springlenfix: rodNum(p, "springlenfix") }),
  cpu([pos, before, species, rod], n, p) {
    const out = Float32Array.from(pos!);
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const j = rod![o]!;
      const damp = rod![o + 3]!;
      if (species![i] === DEAD_SPECIES || !(j >= 0 && j < n && damp >= 0)) continue;
      const dx = before![j * 4]! - pos![o]!;
      const dy = before![j * 4 + 1]! - pos![o + 1]!;
      const dz = before![j * 4 + 2]! - pos![o + 2]!;
      const len = Math.hypot(dx, dy, dz);
      if (len < 0.001) continue;
      const lo = rod![o + 1]! * rodNum(p, "springlen") + rodNum(p, "springlenfix");
      const hi = rod![o + 2]! * rodNum(p, "springlen") + rodNum(p, "springlenfix");
      const sc = (Math.min(hi, Math.max(lo, len)) * damp + len * (1 - damp)) / len;
      out[o] = before![j * 4]! - dx * sc;
      out[o + 1] = before![j * 4 + 1]! - dy * sc;
      out[o + 2] = before![j * 4 + 2]! - dz * sc;
    }
    return out;
  },
});

// ── centre ─────────────────────────────────────────────────────────────────────────────────────

const CentreParams = d.struct({ n: d.u32, damp: d.f32, centre: d.vec3f });
const centreLayout = tgpu
  .bindGroupLayout({
    params: { uniform: CentreParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("centreConstraint");

const centreFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < centreLayout.$.params.n) {
      const here = centreLayout.$.pos[i]!;
      let next = d.vec3f(here.x, here.y, here.z);
      if (centreLayout.$.species[i]! !== 255) {
        const c = centreLayout.$.params.centre;
        next = std.add(c, std.mul(centreLayout.$.params.damp, std.sub(here.xyz, c)));
      }
      centreLayout.$.out[i] = d.vec4f(next, here.w);
    }
  })
  .$name("centreConstraint");

const CENTRE_PARAMS: ParamSpec[] = [
  {
    name: "springCentreDamp",
    type: "number",
    default: 1,
    min: 0,
    max: 1,
    step: 0.001,
    describe: "Fraction of the distance from the centre kept each step.",
  },
  { name: "centreX", type: "number", default: 0, describe: "Centre, x." },
  { name: "centreY", type: "number", default: 0, describe: "Centre, y." },
  { name: "centreZ", type: "number", default: 0, describe: "Centre, z." },
];
const centreNum = numbers(CENTRE_PARAMS, "centreConstraint");

export const centreConstraintOp = kernelOp<d.Infer<typeof CentreParams>>({
  name: "centreConstraint",
  label: "Centre",
  describe: "Shrinks everything towards a point by a factor each step.",
  inputs: [POS, SPECIES],
  output: OUT,
  params: CENTRE_PARAMS,
  layout: centreLayout,
  kernel: centreFn,
  uniform: (root) => root.createBuffer(CentreParams).$usage("uniform").$name("centreConstraint:params"),
  uniforms: (n, p) => ({
    n,
    damp: centreNum(p, "springCentreDamp"),
    centre: d.vec3f(centreNum(p, "centreX"), centreNum(p, "centreY"), centreNum(p, "centreZ")),
  }),
  cpu([pos, species], n, p) {
    const out = Float32Array.from(pos!);
    const c = [centreNum(p, "centreX"), centreNum(p, "centreY"), centreNum(p, "centreZ")];
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES) continue;
      for (let k = 0; k < 3; k++) out[i * 4 + k] = c[k]! + (pos![i * 4 + k]! - c[k]!) * centreNum(p, "springCentreDamp");
    }
    return out;
  },
});

// ── fix ────────────────────────────────────────────────────────────────────────────────────────

const FixParams = d.struct({ n: d.u32 });
const fixLayout = tgpu
  .bindGroupLayout({
    params: { uniform: FixParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    fix: { storage: vec4s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("fixConstraint");

const fixFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < fixLayout.$.params.n) {
      const here = fixLayout.$.pos[i]!;
      const fix = fixLayout.$.fix[i]!;
      let next = d.vec3f(here.x, here.y, here.z);
      if (fixLayout.$.species[i]! !== 255 && fix.x === 1) {
        next = d.vec3f(fix.y, fix.z, fix.w);
      }
      fixLayout.$.out[i] = d.vec4f(next, here.w);
    }
  })
  .$name("fixConstraint");

export const fixConstraintOp = kernelOp<d.Infer<typeof FixParams>>({
  name: "fixConstraint",
  label: "Fix",
  describe: "Pins flagged particles to given positions.",
  inputs: [POS, SPECIES, { name: "fix", kind: "points", dtype: "f32", lanes: 4 }],
  output: OUT,
  params: [],
  layout: fixLayout,
  kernel: fixFn,
  uniform: (root) => root.createBuffer(FixParams).$usage("uniform").$name("fixConstraint:params"),
  uniforms: (n) => ({ n }),
  cpu([pos, species, fix], n) {
    const out = Float32Array.from(pos!);
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES || fix![i * 4] !== 1) continue;
      for (let k = 0; k < 3; k++) out[i * 4 + k] = fix![i * 4 + 1 + k]!;
    }
    return out;
  },
});

// ── pull as a fix ──────────────────────────────────────────────────────────────────────────────

const PullFixParams = d.struct({ n: d.u32, damp: d.f32 });
const pullFixLayout = tgpu
  .bindGroupLayout({
    params: { uniform: PullFixParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    pull: { storage: vec4s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("pullFix");

const pullFixFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < pullFixLayout.$.params.n) {
      const here = pullFixLayout.$.pos[i]!;
      const target = pullFixLayout.$.pull[i]!;
      let next = d.vec3f(here.x, here.y, here.z);
      if (pullFixLayout.$.species[i]! !== 255 && target.w !== 0) {
        next = std.add(std.mul(pullFixLayout.$.params.damp, target.xyz), std.mul(1 - pullFixLayout.$.params.damp, here.xyz));
      }
      pullFixLayout.$.out[i] = d.vec4f(next, here.w);
    }
  })
  .$name("pullFix");

const PULL_FIX_PARAMS: ParamSpec[] = [
  {
    name: "pullfixdamp",
    type: "number",
    default: 0,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Fraction of the way to its target a particle is moved each step.",
  },
];
const pullFixNum = numbers(PULL_FIX_PARAMS, "pullFix");

export const pullFixOp = kernelOp<d.Infer<typeof PullFixParams>>({
  name: "pullFix",
  label: "Pull (as a fix)",
  describe: "Moves particles that have a pull target a set fraction of the way to it.",
  inputs: [POS, SPECIES, { name: "pull", kind: "points", dtype: "f32", lanes: 4 }],
  output: OUT,
  params: PULL_FIX_PARAMS,
  layout: pullFixLayout,
  kernel: pullFixFn,
  uniform: (root) => root.createBuffer(PullFixParams).$usage("uniform").$name("pullFix:params"),
  uniforms: (n, p) => ({ n, damp: pullFixNum(p, "pullfixdamp") }),
  cpu([pos, species, pull], n, p) {
    const out = Float32Array.from(pos!);
    const damp = pullFixNum(p, "pullfixdamp");
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES || pull![i * 4 + 3] === 0) continue;
      for (let k = 0; k < 3; k++) out[i * 4 + k] = damp * pull![i * 4 + k]! + (1 - damp) * pos![i * 4 + k]!;
    }
    return out;
  },
});
