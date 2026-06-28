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

function makePool(root: Root, n: number) {
  const cap = Math.max(1, n);
  return {
    n,
    src: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    dst: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    params: root.createBuffer(ThreshParams).$usage("uniform"),
  };
}

interface Pipe {
  pipeline: GPUComputePipeline;
  pool: ReturnType<typeof makePool>;
}

// One pipeline + pool per backend root (roots are device singletons).
const pipes = new WeakMap<object, Promise<Pipe>>();

function getPipe(device: GPUDevice, root: Root): Promise<Pipe> {
  let p = pipes.get(root as object);
  if (!p) {
    p = (async () => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([threshFn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "threshold" } });
      return { pipeline, pool: makePool(root, 1) };
    })();
    pipes.set(root as object, p);
  }
  return p;
}

function ensurePool(root: Root, pipe: Pipe, n: number) {
  if (pipe.pool.n >= n) return pipe.pool;
  pipe.pool = makePool(root, Math.max(n, pipe.pool.n * 2));
  return pipe.pool;
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
  async execute(ctx, inputs, params) {
    const inField = inputs[0]!;
    const host = inField.data!;
    const n = host.length;
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    const pipe = await getPipe(device, root);
    const pool = ensurePool(root, pipe, n);

    device.queue.writeBuffer(root.unwrap(pool.src), 0, Float32Array.from(host) as BufferSource);
    pool.params.write({ n, thresh: params.thresh as number, soft: softFlag(params), softness: softness(params) });

    const bind = root.unwrap(root.createBindGroup(layout, { params: pool.params, src: pool.src, dst: pool.dst }));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);

    const got = (await pool.dst.read()) as ArrayLike<number>;
    const data = Float32Array.from({ length: n }, (_, i) => got[i]!);
    return [{ shape: inField.shape, dtype: "f32", data }];
  },
  cpuGolden(inputs, params) {
    const inField = inputs[0]!;
    return [{ shape: inField.shape, dtype: "f32", data: applyCpu(inField.data!, params) }];
  },
};
