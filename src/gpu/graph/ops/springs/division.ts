// Cell division as graph ops, all on the device. A division brings a dead slot to life beside
// its parent, so nothing is renumbered and bonds stay valid. It is split into steps so that each
// can be replaced, and so that a model can carry per-particle values these ops know nothing about:
//
//   divisionDraw       which particles want to divide this step: the model's own rule (a species,
//                      not too crowded, a random draw). Anything that outputs a 0/1 field can
//                      stand in for it, such as an agent's own program.
//   divisionPlan       who actually divides, and into which slot. The k-th particle that wants
//                      to is matched with the k-th free slot, by two prefix sums; if there are
//                      more of one than the other, the extras do nothing.
//   divideMove         new positions: parent and daughter half a step apart along a random axis.
//   restAfterDivision  both start at rest: their previous position is their new one.
//   inheritField       any per-particle field, with each daughter taking its parent's value.
//                      One of these per field the particles carry: species, radius, anything.
//
// The plan is a u32 per particle: `PLAN_NONE`, `PLAN_DIVIDES` for a parent, or for a daughter the
// index of its parent. The maths is `divideCells` in src/gpu/sim/springs/growth.ts.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { hash3, rand01, rateThreshold, SALT } from "#src/gpu/sim/springs/rng";
import { encodeScan, getScanCtx } from "#src/gpu/scan/prefixSum";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { FieldValue, ResidentBuffer } from "../../handle";
import type { OpType, ParamSpec } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { DEAD_SPECIES, NO_CUTOFF, numbers, pointsN, requireScalar } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";
import { pcgGpu, rand01Gpu, stepKeyGpu } from "./rngTgsl";

const WG = KERNEL_WORKGROUP;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);

/** A particle that neither divides nor is born this step. */
export const PLAN_NONE = 0xffffffff;
/** A parent that divides this step. A daughter's plan is its parent's index instead. */
export const PLAN_DIVIDES = 0xfffffffe;

const stepKey = (seed: number, step: number) => (seed ^ Math.imul(step, 0x9e3779b1)) >>> 0;
const SEED: ParamSpec = { name: "seed", type: "int", default: 1, min: 0, describe: "Seed of the random stream." };

// ── divisionDraw ───────────────────────────────────────────────────────────────────────────────

const DrawParams = d.struct({ n: d.u32, species: d.u32, threshold: d.u32, seed: d.u32, maxDensity: d.f32 });
const drawLayout = tgpu
  .bindGroupLayout({
    params: { uniform: DrawParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    step: { storage: u32s, access: "readonly" },
    wants: { storage: u32s, access: "mutable" },
  })
  .$name("divisionDraw");

const drawFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = drawLayout.$.params;
    if (i < P.n) {
      let wants = d.u32(0);
      if (drawLayout.$.species[i]! === P.species && !(drawLayout.$.pos[i]!.w > P.maxDensity)) {
        const key = stepKeyGpu(P.seed, drawLayout.$.step[0]!);
        if (pcgGpu(i ^ pcgGpu(key ^ pcgGpu(d.u32(0xb55a4f09)))) < P.threshold) {
          wants = 1;
        }
      }
      drawLayout.$.wants[i] = wants;
    }
  })
  .$name("divisionDraw");

const DRAW_PARAMS: ParamSpec[] = [
  { name: "species", type: "int", default: 0, min: 0, max: 254, describe: "The species that divides." },
  { name: "tau", type: "number", default: 200, min: 1, describe: "Mean steps between divisions of an uncrowded particle." },
  {
    name: "maxDensity",
    type: "number",
    default: NO_CUTOFF,
    min: 0,
    describe: "A particle more crowded than this does not divide: growth stops under pressure.",
  },
  SEED,
];
const drawNum = numbers(DRAW_PARAMS, "divisionDraw");
if (SALT.divide !== 0xb55a4f09) throw new Error("divisionDraw: the kernel's salt must be SALT.divide");

export const divisionDrawOp = kernelOp<d.Infer<typeof DrawParams>>({
  name: "divisionDraw",
  label: "Division: who wants to",
  describe: "Marks the particles that would divide this step: one species, not too crowded, at random with a mean interval.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "step", kind: "scalar", dtype: "u32" },
  ],
  output: { name: "wants", dtype: "u32" },
  params: DRAW_PARAMS,
  layout: drawLayout,
  kernel: drawFn,
  uniform: (root) => root.createBuffer(DrawParams).$usage("uniform").$name("divisionDraw:params"),
  uniforms: (n, p) => ({
    n,
    species: drawNum(p, "species"),
    threshold: rateThreshold(drawNum(p, "tau")),
    seed: drawNum(p, "seed"),
    maxDensity: Math.min(NO_CUTOFF, drawNum(p, "maxDensity")),
  }),
  cpu([pos, species, step], n, p) {
    const out = new Uint32Array(n);
    const key = stepKey(drawNum(p, "seed"), step![0]!);
    const threshold = rateThreshold(drawNum(p, "tau"));
    for (let i = 0; i < n; i++) {
      if (species![i] !== drawNum(p, "species") || pos![i * 4 + 3]! > drawNum(p, "maxDensity")) continue;
      if (hash3(i, key, SALT.divide) < threshold) out[i] = 1;
    }
    return out;
  },
});

// ── divisionPlan ───────────────────────────────────────────────────────────────────────────────

const PlanParams = d.struct({ n: d.u32 });
const flagsLayout = tgpu
  .bindGroupLayout({
    params: { uniform: PlanParams },
    wants: { storage: u32s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    divider: { storage: u32s, access: "mutable" },
    free: { storage: u32s, access: "mutable" },
  })
  .$name("divisionFlags");
/** Clean 0/1 flags to scan: a live particle that wants to divide, and a free slot. */
const flagsFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < flagsLayout.$.params.n) {
      const dead = flagsLayout.$.species[i]! === 255;
      let divider = d.u32(0);
      let free = d.u32(0);
      if (dead) {
        free = 1;
      } else if (flagsLayout.$.wants[i]! !== 0) {
        divider = 1;
      }
      flagsLayout.$.divider[i] = divider;
      flagsLayout.$.free[i] = free;
    }
  })
  .$name("divisionFlags");

const scatterLayout = tgpu
  .bindGroupLayout({
    params: { uniform: PlanParams },
    divider: { storage: u32s, access: "readonly" },
    rank: { storage: u32s, access: "readonly" },
    parentOf: { storage: u32s, access: "mutable" },
  })
  .$name("divisionScatter");
/** `parentOf[k]` = the k-th particle that wants to divide. Ranks are distinct, so no two threads
 *  write one place. */
const scatterFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < scatterLayout.$.params.n && scatterLayout.$.divider[i]! === 1) {
      scatterLayout.$.parentOf[scatterLayout.$.rank[i]!] = i;
    }
  })
  .$name("divisionScatter");

const matchLayout = tgpu
  .bindGroupLayout({
    params: { uniform: PlanParams },
    divider: { storage: u32s, access: "readonly" },
    free: { storage: u32s, access: "readonly" },
    dividerRank: { storage: u32s, access: "readonly" },
    freeRank: { storage: u32s, access: "readonly" },
    dividers: { storage: u32s, access: "readonly" },
    frees: { storage: u32s, access: "readonly" },
    parentOf: { storage: u32s, access: "readonly" },
    plan: { storage: u32s, access: "mutable" },
  })
  .$name("divisionMatch");
const matchFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < matchLayout.$.params.n) {
      // As many divisions as there are both takers and room.
      const pairs = std.min(matchLayout.$.dividers[0]!, matchLayout.$.frees[0]!);
      let plan = d.u32(0xffffffff);
      if (matchLayout.$.divider[i]! === 1 && matchLayout.$.dividerRank[i]! < pairs) {
        plan = 0xfffffffe;
      }
      if (matchLayout.$.free[i]! === 1 && matchLayout.$.freeRank[i]! < pairs) {
        plan = matchLayout.$.parentOf[matchLayout.$.freeRank[i]!]!;
      }
      matchLayout.$.plan[i] = plan;
    }
  })
  .$name("divisionMatch");

type Resolvable = Parameters<typeof tgpu.resolveWithContext>[0][number];
interface PlanPipes {
  flags: GPUComputePipeline;
  scatter: GPUComputePipeline;
  match: GPUComputePipeline;
  uniform: ReturnType<typeof makePlanUniform>;
}
const makePlanUniform = (root: Root) => root.createBuffer(PlanParams).$usage("uniform").$name("divisionPlan:params");
const planPipes = new WeakMap<object, PlanPipes>();
function getPlanPipes(device: GPUDevice, root: Root): PlanPipes {
  let p = planPipes.get(root as object);
  if (!p) {
    const make = (name: string, fn: Resolvable) =>
      typegpuComputePipeline(device, root, tgpu.resolveWithContext([fn], { names: "strict" }), name, name);
    p = {
      flags: make("divisionFlags", flagsFn),
      scatter: make("divisionScatter", scatterFn),
      match: make("divisionMatch", matchFn),
      uniform: makePlanUniform(root),
    };
    planPipes.set(root as object, p);
  }
  return p;
}

/** The plan on the host: the k-th divider with the k-th free slot, in index order. */
export function planDivisions(wants: ArrayLike<number>, species: ArrayLike<number>, n: number): Uint32Array {
  const plan = new Uint32Array(n).fill(PLAN_NONE);
  const dividers: number[] = [];
  const frees: number[] = [];
  for (let i = 0; i < n; i++) {
    if (species[i] === DEAD_SPECIES) frees.push(i);
    else if (wants[i] !== 0) dividers.push(i);
  }
  const pairs = Math.min(dividers.length, frees.length);
  for (let k = 0; k < pairs; k++) {
    plan[dividers[k]!] = PLAN_DIVIDES;
    plan[frees[k]!] = dividers[k]!;
  }
  return plan;
}

export const divisionPlanOp: OpType = {
  name: "divisionPlan",
  label: "Division: plan",
  describe: "Matches each particle that wants to divide with a free slot for its daughter.",
  category: "Springs",
  help: {
    detail:
      "Count along the particles: the first that wants to divide gets the first free slot, the second " +
      "gets the second, and so on. The counting is a running total (a prefix sum), which the device " +
      "can do in parallel. When slots run out, the rest wait.",
  },
  inputs: [
    { name: "wants", kind: "points", dtype: "u32" },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  outputs: [{ name: "plan", kind: "points", dtype: "u32" }],
  params: [],
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "divisionPlan.wants");
    if (pointsN(inputs[1]!, "divisionPlan.species") !== n)
      throw new Error("divisionPlan: wants and species are for different numbers of particles");
    return [{ kind: "points", n }];
  },
  inferElements(inputs) {
    requireScalar(inputs[0], "divisionPlan.wants");
    requireScalar(inputs[1], "divisionPlan.species");
    return [{ kind: "scalar" }];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs) {
    const n = pointsN(inputs[0]!.shape, "divisionPlan.wants");
    const wants = inputs[0]!.buffer;
    const species = inputs[1]!.buffer;
    if (!wants || !species) throw new Error("divisionPlan: resident op received a non-resident input");
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipes = getPlanPipes(device, root);
    const scan = await getScanCtx(device);
    const bytes = n * 4;
    // Leased one by one; if one fails, or the encoding below throws, the ones in hand go back.
    const leased: ResidentBuffer[] = [];
    const lease = async () => {
      const b = await ctx.backend.lease(bytes);
      leased.push(b);
      return b;
    };
    try {
      const divider = await lease();
      const free = await lease();
      const parentOf = await lease();
      const plan = await lease();
      pipes.uniform.write({ n });

      // One command buffer: flags, two scans, scatter, match.
      const enc = device.createCommandEncoder({ label: "divisionPlan" });
      const run = (label: string, pipeline: GPUComputePipeline, bind: GPUBindGroup) => {
        const pass = enc.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(n / WG));
        pass.end();
      };
      run(
        "divisionPlan:flags",
        pipes.flags,
        rawBindGroup(device, root, flagsLayout, [pipes.uniform, wants, species, divider.buffer, free.buffer], "divisionPlan:flags"),
      );
      const dividers = encodeScan(scan, "u32", divider.buffer, n, enc, undefined, "springs:dividers");
      const frees = encodeScan(scan, "u32", free.buffer, n, enc, undefined, "springs:frees");
      run(
        "divisionPlan:scatter",
        pipes.scatter,
        rawBindGroup(device, root, scatterLayout, [pipes.uniform, divider.buffer, dividers.dst, parentOf.buffer], "divisionPlan:scatter"),
      );
      run(
        "divisionPlan:match",
        pipes.match,
        rawBindGroup(
          device,
          root,
          matchLayout,
          [
            pipes.uniform,
            divider.buffer,
            free.buffer,
            dividers.dst,
            frees.dst,
            dividers.totalBuf,
            frees.totalBuf,
            parentOf.buffer,
            plan.buffer,
          ],
          "divisionPlan:match",
        ),
      );
      device.queue.submit([enc.finish()]);
      for (const scratch of [divider, free, parentOf]) ctx.backend.release(scratch);
      leased.length = 0;
      const out: FieldValue[] = [{ shape: { kind: "points", n }, dtype: "u32", buffer: plan }];
      return out;
    } catch (e) {
      for (const b of leased) ctx.backend.release(b);
      throw e;
    }
  },
  cpuGolden(inputs) {
    const n = pointsN(inputs[0]!.shape, "divisionPlan.wants");
    const wants = inputs[0]!.data;
    const species = inputs[1]!.data;
    if (!wants || !species) throw new Error("divisionPlan: cpu golden needs host data");
    return [{ shape: { kind: "points", n }, dtype: "u32", data: planDivisions(wants, species, n) }];
  },
};

// ── divideMove ─────────────────────────────────────────────────────────────────────────────────

const MoveParams = d.struct({ n: d.u32, seed: d.u32, dims: d.u32, half: d.f32 });
const moveLayout = tgpu
  .bindGroupLayout({
    params: { uniform: MoveParams },
    pos: { storage: vec4s, access: "readonly" },
    plan: { storage: u32s, access: "readonly" },
    step: { storage: u32s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("divideMove");

/** Half the offset between the two daughters of parent `p`: a random direction, times `half`. */
const splitGpu = tgpu.fn(
  [d.u32, d.u32, d.u32, d.f32],
  d.vec3f,
)((p, key, dims, half) => {
  "use gpu";
  const angle = rand01Gpu(p, key, d.u32(0x2545f491)) * 6.283185307179586;
  let cz = d.f32(0);
  if (dims === 3) {
    cz = rand01Gpu(p, key, d.u32(0x9e3779b9)) * 2 - 1;
  }
  const flat = std.sqrt(1 - cz * cz);
  return d.vec3f(std.cos(angle) * flat * half, std.sin(angle) * flat * half, cz * half);
});

const moveFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = moveLayout.$.params;
    if (i < P.n) {
      const here = moveLayout.$.pos[i]!;
      const plan = moveLayout.$.plan[i]!;
      const key = stepKeyGpu(P.seed, moveLayout.$.step[0]!);
      let next = d.vec4f(here.x, here.y, here.z, here.w);
      if (plan === 0xfffffffe) {
        next = d.vec4f(std.sub(here.xyz, splitGpu(i, key, P.dims, P.half)), here.w);
      } else if (plan !== 0xffffffff) {
        const parent = moveLayout.$.pos[plan]!;
        next = d.vec4f(std.add(parent.xyz, splitGpu(plan, key, P.dims, P.half)), parent.w);
      }
      moveLayout.$.out[i] = d.vec4f(next.x, next.y, next.z, next.w);
    }
  })
  .$name("divideMove");

const MOVE_PARAMS: ParamSpec[] = [
  { name: "offset", type: "number", default: 0.5, min: 0, max: 4, step: 0.01, describe: "Distance between the two daughters." },
  { name: "dims", type: "int", default: 3, min: 2, max: 3, describe: "2 keeps the split in the plane." },
  SEED,
];
const moveNum = numbers(MOVE_PARAMS, "divideMove");
if (SALT.divideDir !== 0x2545f491 || SALT.divideZ !== 0x9e3779b9)
  throw new Error("divideMove: the kernel's salts must be SALT.divideDir and SALT.divideZ");

export const divideMoveOp = kernelOp<d.Infer<typeof MoveParams>>({
  name: "divideMove",
  label: "Division: positions",
  describe: "Places each dividing particle and its daughter half a step either side of where the parent was.",
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "plan", kind: "points", dtype: "u32" },
    { name: "step", kind: "scalar", dtype: "u32" },
  ],
  output: { name: "pos", dtype: "f32", lanes: 4 },
  params: MOVE_PARAMS,
  layout: moveLayout,
  kernel: moveFn,
  uniform: (root) => root.createBuffer(MoveParams).$usage("uniform").$name("divideMove:params"),
  uniforms: (n, p) => ({ n, seed: moveNum(p, "seed"), dims: moveNum(p, "dims"), half: moveNum(p, "offset") / 2 }),
  cpu([pos, plan, step], n, p) {
    const out = Float32Array.from(pos!);
    const key = stepKey(moveNum(p, "seed"), step![0]!);
    const half = moveNum(p, "offset") / 2;
    const split = (parent: number) => {
      const angle = rand01(parent, key, SALT.divideDir) * 2 * Math.PI;
      const cz = moveNum(p, "dims") === 3 ? rand01(parent, key, SALT.divideZ) * 2 - 1 : 0;
      const flat = Math.sqrt(1 - cz * cz);
      return [Math.cos(angle) * flat * half, Math.sin(angle) * flat * half, cz * half];
    };
    for (let i = 0; i < n; i++) {
      const role = plan![i]!;
      if (role === PLAN_NONE) continue;
      const parent = role === PLAN_DIVIDES ? i : role;
      const sign = role === PLAN_DIVIDES ? -1 : 1;
      const dd = split(parent);
      for (let k = 0; k < 3; k++) out[i * 4 + k] = pos![parent * 4 + k]! + sign * dd[k]!;
      out[i * 4 + 3] = pos![parent * 4 + 3]!;
    }
    return out;
  },
});

// ── restAfterDivision ──────────────────────────────────────────────────────────────────────────

const RestParams = d.struct({ n: d.u32 });
const restLayout = tgpu
  .bindGroupLayout({
    params: { uniform: RestParams },
    prev: { storage: vec4s, access: "readonly" },
    pos: { storage: vec4s, access: "readonly" },
    plan: { storage: u32s, access: "readonly" },
    out: { storage: vec4s, access: "mutable" },
  })
  .$name("restAfterDivision");

const restFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < restLayout.$.params.n) {
      const was = restLayout.$.prev[i]!;
      const now = restLayout.$.pos[i]!;
      if (restLayout.$.plan[i]! !== 0xffffffff) {
        restLayout.$.out[i] = d.vec4f(now.x, now.y, now.z, now.w);
      } else {
        restLayout.$.out[i] = d.vec4f(was.x, was.y, was.z, was.w);
      }
    }
  })
  .$name("restAfterDivision");

export const restAfterDivisionOp = kernelOp<d.Infer<typeof RestParams>>({
  name: "restAfterDivision",
  label: "Division: at rest",
  describe: "Sets the previous position of every particle that divided or was born to its new one, so both start at rest.",
  inputs: [
    { name: "prev", kind: "points", dtype: "f32", lanes: 4 },
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "plan", kind: "points", dtype: "u32" },
  ],
  output: { name: "prev", dtype: "f32", lanes: 4 },
  params: [],
  layout: restLayout,
  kernel: restFn,
  uniform: (root) => root.createBuffer(RestParams).$usage("uniform").$name("restAfterDivision:params"),
  uniforms: (n) => ({ n }),
  cpu([prev, pos, plan], n) {
    const out = Float32Array.from(prev!);
    for (let i = 0; i < n; i++) if (plan![i] !== PLAN_NONE) for (let k = 0; k < 4; k++) out[i * 4 + k] = pos![i * 4 + k]!;
    return out;
  },
});

// ── inheritField ───────────────────────────────────────────────────────────────────────────────

const InheritParams = d.struct({ n: d.u32, lanes: d.u32 });
const inheritLayout = tgpu
  .bindGroupLayout({
    params: { uniform: InheritParams },
    // Bound as words whatever the field holds: a daughter takes its parent's bits.
    field: { storage: u32s, access: "readonly" },
    plan: { storage: u32s, access: "readonly" },
    out: { storage: u32s, access: "mutable" },
  })
  .$name("inheritField");

const inheritFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = inheritLayout.$.params;
    if (i < P.n) {
      let from = i;
      const plan = inheritLayout.$.plan[i]!;
      if (plan < 0xfffffffe) {
        from = plan;
      }
      for (let c = d.u32(0); c < P.lanes; c++) {
        inheritLayout.$.out[i * P.lanes + c] = inheritLayout.$.field[from * P.lanes + c]!;
      }
    }
  })
  .$name("inheritField");

const makeInheritUniform = (root: Root) => root.createBuffer(InheritParams).$usage("uniform").$name("inheritField:params");
const inheritPipes = new WeakMap<object, { pipeline: GPUComputePipeline; uniform: ReturnType<typeof makeInheritUniform> }>();

/** Words per particle of a field value. */
function lanesOf(v: FieldValue): number {
  const el = v.element;
  return !el || el.kind === "scalar" ? 1 : el.kind === "vec" ? el.n : el.kind === "complex" ? 2 : 4;
}

export const inheritFieldOp: OpType = {
  name: "inheritField",
  label: "Division: inherit",
  describe: "Any per-particle field, with each newborn taking its parent's value.",
  category: "Springs",
  help: {
    detail:
      "Division does not know what the particles carry. Whatever they carry (species, radius, an " +
      "age, a genome), pass each field through one of these and the daughters inherit it.",
  },
  inputs: [
    { name: "field", kind: "points" },
    { name: "plan", kind: "points", dtype: "u32" },
  ],
  outputs: [{ name: "field", kind: "points" }],
  params: [],
  inferShapes(inputs) {
    const n = pointsN(inputs[0]!, "inheritField.field");
    if (pointsN(inputs[1]!, "inheritField.plan") !== n)
      throw new Error("inheritField: field and plan are for different numbers of particles");
    return [inputs[0]!];
  },
  inferElements(inputs) {
    return [inputs[0] ?? { kind: "scalar" }];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs) {
    const field = inputs[0]!;
    const n = pointsN(field.shape, "inheritField.field");
    const plan = inputs[1]!.buffer;
    if (!field.buffer || !plan) throw new Error("inheritField: resident op received a non-resident input");
    const lanes = lanesOf(field);
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    let pipe = inheritPipes.get(root as object);
    if (!pipe) {
      const resolved = tgpu.resolveWithContext([inheritFn], { names: "strict" });
      pipe = {
        pipeline: typegpuComputePipeline(device, root, resolved, "inheritField", "inheritField"),
        uniform: makeInheritUniform(root),
      };
      inheritPipes.set(root as object, pipe);
    }
    const out = await ctx.backend.lease(n * lanes * 4);
    pipe.uniform.write({ n, lanes });
    const bind = rawBindGroup(device, root, inheritLayout, [pipe.uniform, field.buffer, plan, out.buffer], "inheritField");
    const enc = device.createCommandEncoder({ label: "inheritField" });
    const pass = enc.beginComputePass({ label: "inheritField" });
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    return [{ ...field, data: undefined, buffer: out }];
  },
  cpuGolden(inputs) {
    const field = inputs[0]!;
    const n = pointsN(field.shape, "inheritField.field");
    const plan = inputs[1]!.data;
    if (!field.data || !plan) throw new Error("inheritField: cpu golden needs host data");
    const lanes = lanesOf(field);
    const out = field.data.slice();
    for (let i = 0; i < n; i++) {
      const role = plan[i]!;
      if (role >= PLAN_DIVIDES) continue;
      for (let c = 0; c < lanes; c++) out[i * lanes + c] = field.data[role * lanes + c]!;
    }
    return [{ ...field, buffer: undefined, data: out }];
  },
};
