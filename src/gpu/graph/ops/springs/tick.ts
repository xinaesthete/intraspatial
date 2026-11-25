// `tick`: the step counter as a graph value. Random terms draw from a hash of (particle, step,
// salt), so the step has to be a value in the graph: a scalar on a feedback edge that this op
// adds one to.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";

const TickParams = d.struct({ n: d.u32 });
const layout = tgpu
  .bindGroupLayout({
    params: { uniform: TickParams },
    step: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    next: { storage: (n: number) => d.arrayOf(d.u32, n), access: "mutable" },
  })
  .$name("tick");

const tickFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [KERNEL_WORKGROUP] })((input) => {
    "use gpu";
    if (input.gid.x < layout.$.params.n) {
      layout.$.next[input.gid.x] = layout.$.step[input.gid.x]! + 1;
    }
  })
  .$name("tick");

export const tickOp = kernelOp<d.Infer<typeof TickParams>>({
  name: "tick",
  label: "Tick",
  describe: "Adds one to the step counter.",
  inputs: [{ name: "step", kind: "scalar", dtype: "u32" }],
  output: { name: "step", dtype: "u32" },
  params: [],
  layout,
  kernel: tickFn,
  uniform: (root) => root.createBuffer(TickParams).$usage("uniform").$name("tick:params"),
  uniforms: (n) => ({ n }),
  cpu: ([step]) => Uint32Array.of((step![0]! + 1) >>> 0),
});
