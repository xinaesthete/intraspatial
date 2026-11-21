// `addForces`: the sum of two force fields, on the device. `addFields` does the same on the host;
// a simulation needs it resident, or every tick would bring both forces back to add them.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { OpType } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { VEC4 } from "./bondTable";
import { hostF32, requireVec, residentBuffer, sameParticles } from "./common";

const WG = 64;
const AddParams = d.struct({ n: d.u32 });

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: AddParams },
    a: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    b: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    sum: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("addForces");

const addFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      layout.$.sum[i] = std.add(layout.$.a[i]!, layout.$.b[i]!);
    }
  })
  .$name("addForces");

const makeParams = (root: Root) => root.createBuffer(AddParams).$usage("uniform").$name("addForces:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; params: ReturnType<typeof makeParams> }>();

function getPipe(device: GPUDevice, root: Root) {
  let p = pipes.get(root as object);
  if (!p) {
    const resolved = tgpu.resolveWithContext([addFn], { names: "strict" });
    p = { pipeline: typegpuComputePipeline(device, root, resolved, "addForces", "addForces"), params: makeParams(root) };
    pipes.set(root as object, p);
  }
  return p;
}

export const addForcesOp: OpType = {
  name: "addForces",
  label: "Add forces",
  describe: "The sum of two force fields, on the device.",
  category: "Springs",
  inputs: [
    { name: "a", kind: "points", dtype: "f32" },
    { name: "b", kind: "points", dtype: "f32" },
  ],
  outputs: [{ name: "force", kind: "points", dtype: "f32" }],
  params: [],
  inferShapes(inputs) {
    return [{ kind: "points", n: sameParticles(inputs, "addForces") }];
  },
  inferElements(inputs) {
    requireVec(inputs[0], 4, "addForces.a");
    requireVec(inputs[1], 4, "addForces.b");
    return [VEC4];
  },
  resident: true,
  async execute(ctx, inputs) {
    const n = sameParticles([inputs[0]!.shape, inputs[1]!.shape], "addForces");
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = getPipe(device, root);
    const out = await ctx.backend.lease(n * 16);
    pipe.params.write({ n });
    const bind = rawBindGroup(
      device,
      root,
      layout,
      [pipe.params, residentBuffer(inputs[0], "addForces.a"), residentBuffer(inputs[1], "addForces.b"), out.buffer],
      "addForces",
    );
    submitDispatch(device, "addForces", pipe.pipeline, bind, Math.ceil(n / WG));
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, buffer: out }];
  },
  cpuGolden(inputs) {
    const n = sameParticles([inputs[0]!.shape, inputs[1]!.shape], "addForces");
    const a = hostF32(inputs[0], "addForces.a");
    const b = hostF32(inputs[1], "addForces.b");
    const data = new Float32Array(n * 4);
    for (let k = 0; k < data.length; k++) data[k] = a[k]! + b[k]!;
    return [{ shape: { kind: "points", n }, dtype: "f32", element: VEC4, data }];
  },
};
