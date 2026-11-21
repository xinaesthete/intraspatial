// Terms that act on one particle at a time, as graph ops:
//
//   mobility    scales a force by a factor per species: how easily that kind of particle moves.
//               It goes on the summed force, before the random terms are added.
//   viscosity   drag against the particle's own velocity:  F = −visc · v · |v|^(viscpow − 1)
//   container   a soft spherical wall about the origin:    F = −force · (r − R) · r̂  for r > R
//   pullForce   a spring to a point of the particle's own: F = min(k·|w|, 1/rate) · |d|^pow · d
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { ParamSpec, Params } from "../../op";
import { DEAD_SPECIES, numbers } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";

const WG = KERNEL_WORKGROUP;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);

// ── mobility ───────────────────────────────────────────────────────────────────────────────────

const MobilityParams = d.struct({ n: d.u32, mobility: d.vec4f });
const mobilityLayout = tgpu
  .bindGroupLayout({
    params: { uniform: MobilityParams },
    force: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    scaled: { storage: vec4s, access: "mutable" },
  })
  .$name("mobility");

const mobilityFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < mobilityLayout.$.params.n) {
      const si = mobilityLayout.$.species[i]!;
      let mob = d.f32(0);
      if (si === 0) {
        mob = mobilityLayout.$.params.mobility.x;
      }
      if (si === 1) {
        mob = mobilityLayout.$.params.mobility.y;
      }
      if (si === 2) {
        mob = mobilityLayout.$.params.mobility.z;
      }
      if (si === 3) {
        mob = mobilityLayout.$.params.mobility.w;
      }
      mobilityLayout.$.scaled[i] = std.mul(mob, mobilityLayout.$.force[i]!);
    }
  })
  .$name("mobility");

const MOBILITY_PARAMS: ParamSpec[] = Array.from({ length: 4 }, (_, s) => ({
  name: `mobility${s}`,
  type: "number" as const,
  default: 1,
  min: 0,
  max: 2,
  step: 0.01,
  describe: `How much of the force on species ${s} is felt: 1 is all of it, 0 none. It does not stop motion already under way.`,
  tags: ["mobility"],
}));
const mobilityNum = numbers(MOBILITY_PARAMS, "mobility");
const mobilities = (p: Params) => [0, 1, 2, 3].map((s) => mobilityNum(p, `mobility${s}`));

export const mobilityOp = kernelOp<d.Infer<typeof MobilityParams>>({
  name: "mobility",
  label: "Mobility",
  describe: "Scales a force by a factor per species.",
  inputs: [
    { name: "force", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: MOBILITY_PARAMS,
  layout: mobilityLayout,
  kernel: mobilityFn,
  uniform: (root) => root.createBuffer(MobilityParams).$usage("uniform").$name("mobility:params"),
  uniforms(n, p) {
    const m = mobilities(p);
    return { n, mobility: d.vec4f(m[0]!, m[1]!, m[2]!, m[3]!) };
  },
  cpu([force, species], n, p) {
    const out = new Float32Array(n * 4);
    const m = mobilities(p);
    for (let i = 0; i < n; i++) {
      const mob = m[species![i]!] ?? 0;
      for (let c = 0; c < 4; c++) out[i * 4 + c] = force![i * 4 + c]! * mob;
    }
    return out;
  },
});

// ── viscosity ──────────────────────────────────────────────────────────────────────────────────

const ViscosityParams = d.struct({ n: d.u32, visc: d.f32, viscpow: d.f32, rate: d.f32 });
const viscosityLayout = tgpu
  .bindGroupLayout({
    params: { uniform: ViscosityParams },
    pos: { storage: vec4s, access: "readonly" },
    prev: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("viscosity");

const viscosityFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < viscosityLayout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      if (viscosityLayout.$.species[i]! !== 255) {
        const v = std.div(std.sub(viscosityLayout.$.pos[i]!.xyz, viscosityLayout.$.prev[i]!.xyz), viscosityLayout.$.params.rate);
        const ll = std.dot(v, v);
        if (ll > 0) {
          f = std.mul(-viscosityLayout.$.params.visc * std.pow(ll, viscosityLayout.$.params.viscpow * 0.5 - 0.5), v);
        }
      }
      viscosityLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("viscosity");

const VISCOSITY_PARAMS: ParamSpec[] = [
  { name: "visc", type: "number", default: 0, min: 0, max: 2, step: 0.001, describe: "Strength of the drag." },
  { name: "viscpow", type: "number", default: 1, min: 0, max: 4, step: 0.1, describe: "Power of speed the drag grows with: 1 is linear." },
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
const viscosityNum = numbers(VISCOSITY_PARAMS, "viscosity");

export const viscosityOp = kernelOp<d.Infer<typeof ViscosityParams>>({
  name: "viscosity",
  label: "Viscosity",
  describe: "Drag against each particle's own velocity.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "prev", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: VISCOSITY_PARAMS,
  layout: viscosityLayout,
  kernel: viscosityFn,
  uniform: (root) => root.createBuffer(ViscosityParams).$usage("uniform").$name("viscosity:params"),
  uniforms: (n, p) => ({ n, visc: viscosityNum(p, "visc"), viscpow: viscosityNum(p, "viscpow"), rate: viscosityNum(p, "springrate") }),
  cpu([pos, prev, species], n, p) {
    const out = new Float32Array(n * 4);
    const rate = viscosityNum(p, "springrate");
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES) continue;
      const vx = (pos![i * 4]! - prev![i * 4]!) / rate;
      const vy = (pos![i * 4 + 1]! - prev![i * 4 + 1]!) / rate;
      const vz = (pos![i * 4 + 2]! - prev![i * 4 + 2]!) / rate;
      const ll = vx * vx + vy * vy + vz * vz;
      if (!(ll > 0)) continue;
      const sc = -viscosityNum(p, "visc") * ll ** (viscosityNum(p, "viscpow") * 0.5 - 0.5);
      out[i * 4] = vx * sc;
      out[i * 4 + 1] = vy * sc;
      out[i * 4 + 2] = vz * sc;
    }
    return out;
  },
});

// ── container ──────────────────────────────────────────────────────────────────────────────────

const ContainerParams = d.struct({ n: d.u32, force: d.f32, radius: d.f32 });
const containerLayout = tgpu
  .bindGroupLayout({
    params: { uniform: ContainerParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("container");

const containerFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < containerLayout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      if (containerLayout.$.species[i]! !== 255) {
        const here = containerLayout.$.pos[i]!.xyz;
        const r = std.length(here);
        const over = r - containerLayout.$.params.radius;
        if (over > 0) {
          f = std.mul((-containerLayout.$.params.force * over) / r, here);
        }
      }
      containerLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("container");

const CONTAINER_PARAMS: ParamSpec[] = [
  { name: "modelSphereForce", type: "number", default: 0, min: 0, max: 2, step: 0.01, describe: "Stiffness of the wall." },
  {
    name: "modelSphereRadius",
    type: "number",
    default: 1,
    min: 0,
    max: 1000,
    step: 0.1,
    describe: "Radius of the wall, about the origin.",
  },
];
const containerNum = numbers(CONTAINER_PARAMS, "container");

export const containerOp = kernelOp<d.Infer<typeof ContainerParams>>({
  name: "container",
  label: "Container",
  describe: "A soft spherical wall: particles outside it are pushed back in.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: CONTAINER_PARAMS,
  layout: containerLayout,
  kernel: containerFn,
  uniform: (root) => root.createBuffer(ContainerParams).$usage("uniform").$name("container:params"),
  uniforms: (n, p) => ({ n, force: containerNum(p, "modelSphereForce"), radius: containerNum(p, "modelSphereRadius") }),
  cpu([pos, species], n, p) {
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      if (species![i] === DEAD_SPECIES) continue;
      const x = pos![i * 4]!;
      const y = pos![i * 4 + 1]!;
      const z = pos![i * 4 + 2]!;
      const r = Math.hypot(x, y, z);
      const over = r - containerNum(p, "modelSphereRadius");
      if (!(over > 0)) continue;
      const sc = (-containerNum(p, "modelSphereForce") * over) / r;
      out[i * 4] = x * sc;
      out[i * 4 + 1] = y * sc;
      out[i * 4 + 2] = z * sc;
    }
    return out;
  },
});

// ── pull ───────────────────────────────────────────────────────────────────────────────────────

const PullParams = d.struct({ n: d.u32, force: d.f32, power: d.f32, cap: d.f32 });
const pullLayout = tgpu
  .bindGroupLayout({
    params: { uniform: PullParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    pull: { storage: vec4s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("pullForce");

const pullFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < pullLayout.$.params.n) {
      let f = d.vec3f(0, 0, 0);
      const target = pullLayout.$.pull[i]!;
      if (pullLayout.$.species[i]! !== 255 && target.w !== 0) {
        const dd = std.sub(target.xyz, pullLayout.$.pos[i]!.xyz);
        const kk = std.min(pullLayout.$.params.force * std.abs(target.w), pullLayout.$.params.cap);
        const sc = kk * std.pow(std.length(dd), pullLayout.$.params.power);
        if (std.abs(sc) < 3e38) {
          f = std.mul(sc, dd);
        }
      }
      pullLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("pullForce");

const PULL_PARAMS: ParamSpec[] = [
  {
    name: "pullspringforce",
    type: "number",
    default: 0,
    min: 0,
    max: 2,
    step: 0.001,
    describe: "Strength of the pull, times each particle's own weight.",
  },
  { name: "pullspringpow", type: "number", default: 0, min: -2, max: 2, step: 0.1, describe: "Power of distance the pull grows with." },
  { name: "springrate", type: "number", default: 1, min: 1e-6, max: 4, step: 0.01, describe: "Step length; sets the cap on the pull." },
];
const pullNum = numbers(PULL_PARAMS, "pullForce");

export const pullForceOp = kernelOp<d.Infer<typeof PullParams>>({
  name: "pullForce",
  label: "Pull",
  describe: "A spring from each particle to a point of its own.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "pull", kind: "points", dtype: "f32", lanes: 4 },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: PULL_PARAMS,
  layout: pullLayout,
  kernel: pullFn,
  uniform: (root) => root.createBuffer(PullParams).$usage("uniform").$name("pullForce:params"),
  uniforms: (n, p) => ({ n, force: pullNum(p, "pullspringforce"), power: pullNum(p, "pullspringpow"), cap: 1 / pullNum(p, "springrate") }),
  cpu([pos, species, pull], n, p) {
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const w = pull![i * 4 + 3]!;
      if (species![i] === DEAD_SPECIES || w === 0) continue;
      const dx = pull![i * 4]! - pos![i * 4]!;
      const dy = pull![i * 4 + 1]! - pos![i * 4 + 1]!;
      const dz = pull![i * 4 + 2]! - pos![i * 4 + 2]!;
      const kk = Math.min(pullNum(p, "pullspringforce") * Math.abs(w), 1 / pullNum(p, "springrate"));
      const sc = kk * Math.hypot(dx, dy, dz) ** pullNum(p, "pullspringpow");
      if (!Number.isFinite(sc)) continue;
      out[i * 4] = dx * sc;
      out[i * 4 + 1] = dy * sc;
      out[i * 4 + 2] = dz * sc;
    }
    return out;
  },
});
