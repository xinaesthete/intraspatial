// GPU complex multiply — the first element-algebra op backed by a real compute pass
// (ADR-0004). Authored as a `"use gpu"` (TGSL) kernel with an explicit layout-bound
// pipeline built once (ADR-0003: required for Dawn-on-Node teardown stability), read
// back via TypeGPU `.read()`. A complex field packs interleaved [re0,im0,re1,im1,...],
// so sample `i` lives at lanes 2i, 2i+1; the kernel is the lane-for-lane image of
// `mulFields({kind:"complex"}, …)` in elementMath.ts.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { writeView } from "../../device";
import type { Root } from "../backend";

const WG = 64;

const CMulParams = d.struct({ n: d.u32 }); // n = sample count (array length = 2n)

const layout = tgpu.bindGroupLayout({
  params: { uniform: CMulParams },
  a: { storage: (k: number) => d.arrayOf(d.f32, k), access: "readonly" },
  b: { storage: (k: number) => d.arrayOf(d.f32, k), access: "readonly" },
  dst: { storage: (k: number) => d.arrayOf(d.f32, k), access: "mutable" },
});

const cmulFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    if (i < layout.$.params.n) {
      const re = i + i; // 2i — multiplication-free to dodge TGSL's float-division/typing edges
      const im = re + d.u32(1);
      const ar = layout.$.a[re]!;
      const ai = layout.$.a[im]!;
      const br = layout.$.b[re]!;
      const bi = layout.$.b[im]!;
      layout.$.dst[re] = ar * br - ai * bi;
      layout.$.dst[im] = ar * bi + ai * br;
    }
  })
  .$name("complexMul");

function makePool(root: Root, len: number) {
  const cap = Math.max(2, len);
  return {
    len,
    a: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    b: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    dst: root.createBuffer(d.arrayOf(d.f32, cap)).$usage("storage"),
    params: root.createBuffer(CMulParams).$usage("uniform"),
  };
}

interface Pipe {
  pipeline: GPUComputePipeline;
  pool: ReturnType<typeof makePool>;
}

const pipes = new WeakMap<object, Promise<Pipe>>();

function getPipe(device: GPUDevice, root: Root): Promise<Pipe> {
  let p = pipes.get(root as object);
  if (!p) {
    p = (async () => {
      const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([cmulFn], { names: "strict" });
      const module = device.createShaderModule({ code });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
      const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "complexMul" } });
      return { pipeline, pool: makePool(root, 2) };
    })();
    pipes.set(root as object, p);
  }
  return p;
}

function ensurePool(root: Root, pipe: Pipe, len: number) {
  if (pipe.pool.len >= len) return pipe.pool;
  pipe.pool = makePool(root, Math.max(len, pipe.pool.len * 2));
  return pipe.pool;
}

/** Pointwise complex product of two interleaved [re,im,...] arrays (length 2n). */
export async function complexMulGpu(device: GPUDevice, root: Root, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  if (a.length !== b.length) throw new Error("complexMulGpu: length mismatch");
  const len = a.length;
  const n = len >> 1;
  const pipe = await getPipe(device, root);
  const pool = ensurePool(root, pipe, len);

  writeView(device.queue, root.unwrap(pool.a), a);
  writeView(device.queue, root.unwrap(pool.b), b);
  pool.params.write({ n });

  const bind = root.unwrap(root.createBindGroup(layout, { params: pool.params, a: pool.a, b: pool.b, dst: pool.dst }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe.pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await pool.dst.read()) as ArrayLike<number>;
  return Float32Array.from({ length: len }, (_, i) => got[i]!);
}
