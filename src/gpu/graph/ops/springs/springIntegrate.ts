// `springIntegrate`: CSynth's integrator as a graph op (docs/spring-model-as-ops.md). State is two
// position fields, `pos` and `prev`; velocity is their difference. A force is a velocity change
// per step, as in CSynth, so its parameter values carry over:
//
//   v      = (pos − prev) / rate
//   v'     = v · use + F − gravity·ŷ          `use` is the damping: how much old velocity is kept
//   v'     = v' · soft(|v'| / maxvel)          a soft speed cap
//   pos'   = pos + v' · rate                   halved when v'·F < 0 (CSynth's fix for bistable pairs)
//   prev'  = pos
//
// `density` supplies the two numbers `pairDensity` computes: the first is stored in the spare lane
// of the new positions (CSynth keeps it there for the next step), the second strengthens damping
// type 4. A dead particle (`species` = 255) does not move.
//
// It closes the feedback loop: wire `pos` and `prev` back to the state they came from. The maths
// is the tail of `stepSprings`, which stays the reference.
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
import { DEAD_SPECIES, hostF32, numbers, requireScalar, requireVec, residentBuffer, sameParticles } from "./common";

const WG = 64;

const IntegrateParams = d.struct({
  n: d.u32,
  damptype: d.u32,
  damp: d.f32,
  rate: d.f32,
  maxvel: d.f32,
  gravity: d.f32,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: IntegrateParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    prev: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    density: { storage: (n: number) => d.arrayOf(d.vec2f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    posOut: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
    prevOut: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("springIntegrate");

const integrateFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      const here = layout.$.pos[i]!;
      const crowd = layout.$.density[i]!;
      // An unused slot stays where it is.
      let next = d.vec3f(here.x, here.y, here.z);
      let spare = here.w;
      if (layout.$.species[i]! !== 255) {
        const f = layout.$.force[i]!.xyz;
        const rate = layout.$.params.rate;
        const damp = layout.$.params.damp;
        let v = std.div(std.sub(here.xyz, layout.$.prev[i]!.xyz), rate);
        const speed = std.length(v);

        // The proportion of the old velocity that is kept.
        let use = damp;
        if (layout.$.params.damptype === 2) {
          const sq = speed * speed;
          use = std.max(0, 1 - damp * sq * sq);
        }
        if (layout.$.params.damptype === 3) {
          // (1 + damp·s³)^(-1/3), the same law as (s⁻³ + damp)^(-1/3) / s. A resting particle
          // has s = 0, and s⁻³ overflows f32, which used to throw the force away.
          const s3 = speed;
          use = std.pow(1 + damp * s3 * s3 * s3, -1 / 3);
        }
        if (layout.$.params.damptype === 4) {
          // Speed is exaggerated by the directional density, to damp collisions hard.
          // damp·(1 + (s·damp)³)^(-1/3), the same law as ((s·damp)⁻³ + 1)^(-1/3) / s.
          // damp = 0, or a resting particle, used to divide by zero or overflow f32.
          const s4 = speed * (1 + crowd.y);
          const sd = s4 * damp;
          use = std.min(1, std.max(0, damp * std.pow(1 + sd * sd * sd, -1 / 3)));
        }
        if (damp < 0) {
          use = -damp;
        }

        v = std.add(std.mul(use, v), f);
        v = d.vec3f(v.x, v.y - layout.$.params.gravity, v.z);

        // Soft speed cap: untouched below 0.7 of the limit, then bent towards it.
        const rr = std.length(v) / layout.$.params.maxvel;
        if (rr > 0.7) {
          v = std.mul((1 - 0.09 / (rr - 0.4)) / rr, v);
        }
        const total = v.x + v.y + v.z;
        if (!(std.abs(total) < 3e38)) {
          v = d.vec3f(0, 0, 0);
        }

        next = std.add(here.xyz, std.mul(rate, v));
        // A particle moving against its force takes half the step.
        if (std.dot(v, f) < 0) {
          next = std.mul(0.5, std.add(here.xyz, next));
        }
        spare = crowd.x;
      }
      layout.$.posOut[i] = d.vec4f(next, spare);
      layout.$.prevOut[i] = d.vec4f(here.xyz, here.w);
    }
  })
  .$name("springIntegrate");

const makeParams = (root: Root) => root.createBuffer(IntegrateParams).$usage("uniform").$name("springIntegrate:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([integrateFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "springIntegrate", "springIntegrate"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

const PARAMS: ParamSpec[] = [
  {
    name: "damp",
    type: "number",
    default: 0.9,
    min: -1,
    max: 1,
    step: 0.01,
    describe: "Damping; its meaning depends on the damping type.",
  },
  {
    name: "damptype",
    type: "int",
    default: 1,
    min: 1,
    max: 4,
    describe: "1: keep a fixed fraction of velocity. 2 to 4: CSynth's speed-dependent laws, which damp fast particles harder.",
  },
  {
    name: "springrate",
    type: "number",
    default: 1,
    min: 1e-6,
    max: 4,
    step: 0.01,
    describe: "Step length: positions move by velocity times this.",
  },
  { name: "springmaxvel", type: "number", default: 0.25, min: 1e-6, max: 10, step: 0.01, describe: "Soft cap on speed." },
  { name: "gravity", type: "number", default: 0, min: -1, max: 1, step: 0.001, describe: "Constant pull along −y." },
];
const num = numbers(PARAMS, "springIntegrate");

function soft(rr: number): number {
  return rr <= 0.7 ? 1 : (1 - 0.09 / (rr - 0.4)) / rr;
}

function integrateCpu(
  pos: ArrayLike<number>,
  prev: ArrayLike<number>,
  force: ArrayLike<number>,
  density: ArrayLike<number>,
  species: ArrayLike<number>,
  n: number,
  p: Params,
) {
  const posOut = new Float32Array(n * 4);
  const prevOut = new Float32Array(n * 4);
  const damp = num(p, "damp");
  const damptype = num(p, "damptype");
  const rate = num(p, "springrate");
  const maxvel = num(p, "springmaxvel");
  const gravity = num(p, "gravity");
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    for (let c = 0; c < 4; c++) prevOut[o + c] = pos[o + c]!;
    if (species[i] === DEAD_SPECIES) {
      for (let c = 0; c < 4; c++) posOut[o + c] = pos[o + c]!;
      continue;
    }
    const fx = force[o]!;
    const fy = force[o + 1]!;
    const fz = force[o + 2]!;
    let vx = (pos[o]! - prev[o]!) / rate;
    let vy = (pos[o + 1]! - prev[o + 1]!) / rate;
    let vz = (pos[o + 2]! - prev[o + 2]!) / rate;
    const speed = Math.hypot(vx, vy, vz);
    let use = damp;
    if (damptype === 2) use = Math.max(0, 1 - damp * speed ** 4);
    else if (damptype === 3) {
      const v = speed;
      use = (1 + damp * v ** 3) ** (-1 / 3);
    } else if (damptype === 4) {
      const v = speed * (1 + density[i * 2 + 1]!);
      const sd = v * damp;
      use = Math.min(1, Math.max(0, damp * (1 + sd ** 3) ** (-1 / 3)));
    }
    if (damp < 0) use = -damp;
    vx = vx * use + fx;
    vy = vy * use + fy - gravity;
    vz = vz * use + fz;
    const clip = soft(Math.hypot(vx, vy, vz) / maxvel);
    vx *= clip;
    vy *= clip;
    vz *= clip;
    if (!Number.isFinite(vx + vy + vz)) {
      vx = 0;
      vy = 0;
      vz = 0;
    }
    let nx = pos[o]! + vx * rate;
    let ny = pos[o + 1]! + vy * rate;
    let nz = pos[o + 2]! + vz * rate;
    if (vx * fx + vy * fy + vz * fz < 0) {
      nx = (pos[o]! + nx) * 0.5;
      ny = (pos[o + 1]! + ny) * 0.5;
      nz = (pos[o + 2]! + nz) * 0.5;
    }
    posOut[o] = nx;
    posOut[o + 1] = ny;
    posOut[o + 2] = nz;
    posOut[o + 3] = density[i * 2]!;
  }
  return { posOut, prevOut };
}

export const springIntegrateOp: OpType = {
  name: "springIntegrate",
  label: "Integrate (springs)",
  describe: "Advance positions by one step from the summed force, with CSynth's damping and speed cap.",
  category: "Springs",
  help: {
    detail:
      "The state is two sets of positions: now and one step ago. Their difference is the velocity. A " +
      "force is a change of velocity per step. Feed `pos` and `prev` back to where they came from to " +
      "close the loop.",
    math: "v'=u\\,v+F,\\quad x'=x+v'\\,\\text{rate}",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "prev", kind: "points", dtype: "f32" },
    { name: "force", kind: "points", dtype: "f32" },
    { name: "density", kind: "points", dtype: "f32" },
    { name: "species", kind: "points", dtype: "u32" },
  ],
  outputs: [
    { name: "pos", kind: "points", dtype: "f32" },
    { name: "prev", kind: "points", dtype: "f32" },
  ],
  params: PARAMS,
  inferShapes(inputs) {
    const n = sameParticles(inputs, "springIntegrate");
    return [
      { kind: "points", n },
      { kind: "points", n },
    ];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "springIntegrate.pos");
    requireVec(inputs[1], 4, "springIntegrate.prev");
    requireVec(inputs[2], 4, "springIntegrate.force");
    requireVec(inputs[3], 2, "springIntegrate.density");
    requireScalar(inputs[4], "springIntegrate.species");
    return [VEC4, VEC4];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "springIntegrate",
    );
    // Validate before either lease: a bad param or a host input used to keep the output
    // buffers, and the pool never destroys them.
    const uniform = {
      n,
      damptype: num(params, "damptype"),
      damp: num(params, "damp"),
      rate: num(params, "springrate"),
      maxvel: num(params, "springmaxvel"),
      gravity: num(params, "gravity"),
    };
    const buffers = [
      residentBuffer(inputs[0], "springIntegrate.pos"),
      residentBuffer(inputs[1], "springIntegrate.prev"),
      residentBuffer(inputs[2], "springIntegrate.force"),
      residentBuffer(inputs[3], "springIntegrate.density"),
      residentBuffer(inputs[4], "springIntegrate.species"),
    ];
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);

    // One lease per output port; both are written by the one dispatch.
    const posOut = await ctx.backend.lease(n * 16);
    try {
      const prevOut = await ctx.backend.lease(n * 16);
      try {
        pipe.params.write(uniform);
        const bind = rawBindGroup(device, root, layout, [pipe.params, ...buffers, posOut.buffer, prevOut.buffer], "springIntegrate");
        submitDispatch(device, "springIntegrate", pipe.pipeline, bind, Math.ceil(n / WG));
        const shape: Shape = { kind: "points", n };
        return [
          { shape, dtype: "f32", element: VEC4, buffer: posOut },
          { shape, dtype: "f32", element: VEC4, buffer: prevOut },
        ];
      } catch (e) {
        ctx.backend.release(prevOut);
        throw e;
      }
    } catch (e) {
      ctx.backend.release(posOut);
      throw e;
    }
  },
  cpuGolden(inputs, params) {
    const n = sameParticles(
      inputs.map((v) => v.shape),
      "springIntegrate",
    );
    const { posOut, prevOut } = integrateCpu(
      hostF32(inputs[0], "springIntegrate.pos"),
      hostF32(inputs[1], "springIntegrate.prev"),
      hostF32(inputs[2], "springIntegrate.force"),
      hostF32(inputs[3], "springIntegrate.density"),
      hostF32(inputs[4], "springIntegrate.species"),
      n,
      params,
    );
    const shape: Shape = { kind: "points", n };
    return [
      { shape, dtype: "f32", element: VEC4, data: posOut },
      { shape, dtype: "f32", element: VEC4, data: prevOut },
    ];
  },
};
