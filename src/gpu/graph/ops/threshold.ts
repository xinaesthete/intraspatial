// Native (non-wrapper) graph op: pointwise threshold of a grid. The first op that
// runs a real compute pass *through the backend* rather than delegating to a legacy
// `*Gpu` function — it proves the backend seam end to end. Authored as a `"use gpu"`
// kernel with an explicit layout-bound pipeline (portable across the Node/Dawn and
// browser backends; layout-bound is required for Dawn-on-Node teardown stability,
// ADR-0003) and read back via TypeGPU `.read()`.
//
// Hard mode: out = (x >= t) ? 1 : 0. Soft mode: a logistic ramp
// 1/(1+exp(-(x-t)*softness)) — a smooth ("fuzzy") threshold, the windowing move
// applied to thresholding.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { Root } from "../backend";
import type { OpType, Params } from "../op";
import { rawBindGroup } from "../residentBind";

const WG = 64;

const ThreshParams = d.struct({ n: d.u32, thresh: d.f32, soft: d.u32, softness: d.f32 });

const layout = tgpu.bindGroupLayout({
  params: { uniform: ThreshParams },
  src: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  dst: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const threshFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      const x = layout.$.src[i]!;
      let out = d.f32(0);
      if (layout.$.params.soft === 1) {
        out = 1 / (1 + std.exp(-(x - layout.$.params.thresh) * layout.$.params.softness));
      } else {
        if (x >= layout.$.params.thresh) {
          out = d.f32(1);
        }
      }
      layout.$.dst[i] = out;
    }
  })
  .$name("threshold");

interface Pipe {
  pipeline: GPUComputePipeline;
  /** Only the uniform block is module-scoped now. Since ADR-0017 the src/dst storage buffers
   *  come from the executor's lease, so this op no longer keeps a private grid-sized scratch
   *  pool at all — the value it is handed *is* the buffer it reads. */
  params: ReturnType<typeof makeParams>;
}

function makeParams(root: Root) {
  return root.createBuffer(ThreshParams).$usage("uniform");
}

// One pipeline + uniform block per backend root (roots are device singletons).
const pipes = new WeakMap<object, Promise<Pipe>>();

function getPipe(device: GPUDevice, root: Root): Promise<Pipe> {
  let p = pipes.get(root as object);
  if (!p) {
    p = (async () => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([threshFn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "threshold" } });
      return { pipeline, params: makeParams(root) };
    })();
    pipes.set(root as object, p);
  }
  return p;
}

function softFlag(params: Params): number {
  return params.soft ? 1 : 0;
}
function softness(params: Params): number {
  return (params.softness as number) || 10;
}

function applyCpu(data: ArrayLike<number>, params: Params): Float32Array {
  const t = params.thresh as number;
  const soft = softFlag(params);
  const k = softness(params);
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = soft === 1 ? 1 / (1 + Math.exp(-(data[i]! - t) * k)) : data[i]! >= t ? 1 : 0;
  }
  return out;
}

export const thresholdOp: OpType = {
  name: "threshold",
  label: "Threshold",
  describe: "Pointwise hard step or soft (logistic) threshold of a grid.",
  inputs: [{ name: "in", kind: "grid" }],
  outputs: [{ name: "out", kind: "grid", dtype: "f32" }],
  params: [
    { name: "thresh", type: "number", default: 0.5, min: -10, max: 10, step: 0.01 },
    { name: "soft", type: "bool", default: false, describe: "logistic ramp instead of a hard step" },
    { name: "softness", type: "number", default: 10, min: 0.1, max: 100, step: 0.1 },
  ],
  inferShapes(inputs) {
    return [inputs[0]!];
  },
  resident: true,
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const src = inField.buffer;
    if (!src) throw new Error("threshold: resident op received a non-resident input");
    const n = src.byteLength / 4;
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = await getPipe(device, root);

    // Pointwise, but still not in place: invariant 1 requires read-modify-write over a shared
    // field to use two physical buffers, and the input buffer may be aliased by another reader.
    const dst = await ctx.backend.lease(src.byteLength);
    pipe.params.write({ n, thresh: params.thresh as number, soft: softFlag(params), softness: softness(params) });

    // Raw bind group over the pooled buffers — see residentBind.ts for why these are not
    // wrapped as TypeGPU buffers. Order matches the `layout` declaration: params, src, dst.
    const bind = rawBindGroup(device, root, layout, [pipe.params, src.buffer, dst.buffer]);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);

    // No readback — the value stays resident and the next op binds this buffer directly.
    return [{ shape: inField.shape, dtype: "f32", buffer: dst }];
  },
  cpuGolden(inputs, params) {
    const inField = inputs[0]!;
    return [{ shape: inField.shape, dtype: "f32", data: applyCpu(inField.data!, params) }];
  },
};
