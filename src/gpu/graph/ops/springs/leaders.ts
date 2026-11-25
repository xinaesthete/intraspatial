// Tip-cell selection as graph ops, on the device (`selectLeaders` in
// src/gpu/sim/springs/growth.ts). Exposed surface cells turn into leaders at random, but never
// close to an existing leader, and a leader that ends up buried turns back.
//
//   leaderStatus  what each particle is this round: a leader that stays, a leader that reverts
//                 (it has a full shell of tissue neighbours), or a cell that has drawn the
//                 chance to become one and is exposed enough
//   leaderApply   the new species. A candidate becomes a leader unless a staying leader, or a
//                 candidate with a lower index, is within `spacing`
//
// The second rule is what makes it parallel: no candidate needs to know what another decided,
// only who else was a candidate. It runs every `every` steps and leaves species alone otherwise.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { applyLeaderStatuses, LEADER_STATUS, leaderStatuses } from "#src/gpu/sim/springs/growth";
import { rateThreshold, SALT } from "#src/gpu/sim/springs/rng";
import type { ParamSpec } from "../../op";
import { numbers } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";
import { pcgGpu, stepKeyGpu } from "./rngTgsl";

const WG = KERNEL_WORKGROUP;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);

const StatusParams = d.struct({
  n: d.u32,
  cell: d.u32,
  leader: d.u32,
  every: d.u32,
  threshold: d.u32,
  seed: d.u32,
  maxNeighbours: d.f32,
  revertNeighbours: d.f32,
});
const statusLayout = tgpu
  .bindGroupLayout({
    params: { uniform: StatusParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    step: { storage: u32s, access: "readonly" },
    status: { storage: u32s, access: "mutable" },
  })
  .$name("leaderStatus");

const statusFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = statusLayout.$.params;
    if (i < P.n) {
      const si = statusLayout.$.species[i]!;
      const step = statusLayout.$.step[0]!;
      const due = step % P.every === 0;
      let status = d.u32(0);
      if (si === P.leader) {
        status = 2;
      }
      let look = due && si === P.leader;
      if (due && si === P.cell) {
        const key = stepKeyGpu(P.seed, step);
        look = pcgGpu(i ^ pcgGpu(key ^ pcgGpu(d.u32(0x68e31da4)))) < P.threshold;
      }
      if (look) {
        const me = statusLayout.$.pos[i]!.xyz;
        let near = d.f32(0);
        for (let j = d.u32(0); j < P.n; j++) {
          const sj = statusLayout.$.species[j]!;
          if (j !== i && (sj === P.cell || sj === P.leader) && std.length(std.sub(statusLayout.$.pos[j]!.xyz, me)) < 1.3) {
            near += 1;
          }
        }
        if (si === P.leader) {
          if (near >= P.revertNeighbours) {
            status = 3;
          }
        } else if (near >= 1 && near <= P.maxNeighbours) {
          status = 1;
        }
      }
      statusLayout.$.status[i] = status;
    }
  })
  .$name("leaderStatus");

const STATUS_PARAMS: ParamSpec[] = [
  { name: "from", type: "int", default: 0, min: 0, max: 254, describe: "Species that can become a leader." },
  { name: "to", type: "int", default: 2, min: 0, max: 254, describe: "Species a leader is." },
  { name: "tau", type: "number", default: 60, min: 1, describe: "Mean rounds before an exposed cell becomes a leader." },
  {
    name: "maxNeighbours",
    type: "number",
    default: 4,
    min: 0,
    max: 20,
    describe: "A cell with more tissue neighbours than this is not exposed enough.",
  },
  {
    name: "revertNeighbours",
    type: "number",
    default: 6,
    min: 0,
    max: 20,
    describe: "A leader with this many tissue neighbours has been overgrown and turns back.",
  },
  { name: "every", type: "int", default: 1, min: 1, max: 1000, describe: "Run once every this many steps." },
  { name: "seed", type: "int", default: 1, min: 0, describe: "Seed of the random stream." },
];
const statusNum = numbers(STATUS_PARAMS, "leaderStatus");
if (SALT.leader !== 0x68e31da4) throw new Error("leaderStatus: the kernel's salt must be SALT.leader");

export const leaderStatusOp = kernelOp<d.Infer<typeof StatusParams>>({
  name: "leaderStatus",
  label: "Leaders: status",
  describe: "Finds the leaders that stay, the leaders that have been overgrown, and the exposed cells that could become leaders.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "step", kind: "scalar", dtype: "u32" },
  ],
  output: { name: "status", dtype: "u32" },
  params: STATUS_PARAMS,
  layout: statusLayout,
  kernel: statusFn,
  uniform: (root) => root.createBuffer(StatusParams).$usage("uniform").$name("leaderStatus:params"),
  uniforms: (n, p) => ({
    n,
    cell: statusNum(p, "from"),
    leader: statusNum(p, "to"),
    every: Math.max(1, statusNum(p, "every")),
    threshold: rateThreshold(statusNum(p, "tau")),
    seed: statusNum(p, "seed"),
    maxNeighbours: statusNum(p, "maxNeighbours"),
    revertNeighbours: statusNum(p, "revertNeighbours"),
  }),
  cpu([pos, species, step], n, p) {
    const to = statusNum(p, "to");
    if (step![0]! % Math.max(1, statusNum(p, "every")) !== 0)
      return Uint32Array.from({ length: n }, (_, i) => (species![i] === to ? LEADER_STATUS.stays : 0));
    return leaderStatuses(pos!, species!, n, {
      from: statusNum(p, "from"),
      to,
      tau: statusNum(p, "tau"),
      maxNeighbours: statusNum(p, "maxNeighbours"),
      revertNeighbours: statusNum(p, "revertNeighbours"),
      seed: statusNum(p, "seed"),
      step: step![0]!,
    });
  },
});

// ── leaderApply ────────────────────────────────────────────────────────────────────────────────

const ApplyParams = d.struct({ n: d.u32, cell: d.u32, leader: d.u32, spacing: d.f32 });
const applyLayout = tgpu
  .bindGroupLayout({
    params: { uniform: ApplyParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    status: { storage: u32s, access: "readonly" },
    out: { storage: u32s, access: "mutable" },
  })
  .$name("leaderApply");

const applyFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = applyLayout.$.params;
    if (i < P.n) {
      let species = applyLayout.$.species[i]!;
      const status = applyLayout.$.status[i]!;
      if (status === 3) {
        species = P.cell;
      }
      if (status === 1) {
        const me = applyLayout.$.pos[i]!.xyz;
        let blocked = false;
        for (let j = d.u32(0); j < P.n; j++) {
          const other = applyLayout.$.status[j]!;
          if (j !== i && (other === 2 || (other === 1 && j < i)) && std.length(std.sub(applyLayout.$.pos[j]!.xyz, me)) < P.spacing) {
            blocked = true;
          }
        }
        if (!blocked) {
          species = P.leader;
        }
      }
      applyLayout.$.out[i] = species;
    }
  })
  .$name("leaderApply");

const APPLY_PARAMS: ParamSpec[] = [
  { name: "from", type: "int", default: 0, min: 0, max: 254, describe: "Species a reverting leader goes back to." },
  { name: "to", type: "int", default: 2, min: 0, max: 254, describe: "Species a leader is." },
  {
    name: "spacing",
    type: "number",
    default: 4,
    min: 0,
    max: 100,
    step: 0.1,
    describe: "No new leader within this distance of another: how far apart branches start.",
  },
];
const applyNum = numbers(APPLY_PARAMS, "leaderApply");

export const leaderApplyOp = kernelOp<d.Infer<typeof ApplyParams>>({
  name: "leaderApply",
  label: "Leaders: apply",
  describe:
    "Turns overgrown leaders back into cells, and candidates into leaders unless another leader or an earlier candidate is too near.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "status", kind: "points", dtype: "u32" },
  ],
  output: { name: "species", dtype: "u32" },
  params: APPLY_PARAMS,
  layout: applyLayout,
  kernel: applyFn,
  uniform: (root) => root.createBuffer(ApplyParams).$usage("uniform").$name("leaderApply:params"),
  uniforms: (n, p) => ({ n, cell: applyNum(p, "from"), leader: applyNum(p, "to"), spacing: applyNum(p, "spacing") }),
  cpu([pos, species, status], n, p) {
    const out = Uint32Array.from(species!);
    applyLeaderStatuses(pos!, out, status!, n, { from: applyNum(p, "from"), to: applyNum(p, "to"), spacing: applyNum(p, "spacing") });
    return out;
  },
});
