// GPU lowering of a Swept geometry — one compute invocation per grid vertex (ADR-0003). The
// shader body is **codegen'd from the geometry IR** (`Swept.toWgsl`), so it is the same
// closed-form `eval(s, θ)` as the CPU golden `Swept.tessellate`, just run on the device; the
// parity test (`swept.gpu.test.ts`) pins them together.
//
// Param **values** flow through a storage buffer `P` (laid out by `Swept.paramVector`), so the
// emitted WGSL depends only on the horn's *structure*. Two same-structure horns therefore share
// one pipeline (cached by code) — you render a whole breeding grid of varied, smoothly-morphing
// horns by uploading a new `paramVector` per instance, with no per-frame recompile.
//
// `createSweptGpu` returns the geometry **resident on the GPU** (`positions`/`normals` are
// STORAGE|VERTEX buffers) so a WebGPU render pass can bind them directly — no CPU round-trip.
// `sweptMeshGpu` reads them back to typed arrays (the parity test + interop escape hatch).
//
// We hand-author the WGSL (a *dynamic* per-structure shader can't be a statically-authored
// `"use gpu"` kernel) but bind and read it through TypeGPU's Dawn-stable path — the TypeGPU/WGSL
// backend, never TSL.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { Root } from "../gpu/graph/backend";
import type { Mesh, Swept } from "./swept";
import { gridIndices } from "./sweptGrid";

const WG = 64;

const Params = d.struct({ slices: d.u32, stacks: d.u32, vertexCount: d.u32, _pad: d.u32 });

// Binding order fixes the WGSL `@binding` indices in `sweptShaderWgsl`: params=0, pos=1, nor=2,
// P (the param values)=3.
const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  outPos: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  outNor: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  pbuf: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
});

// One compute pipeline per (device, shader-structure). Same-structure horns hit the cache.
const pipes = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
function getPipe(device: GPUDevice, root: Root, code: string): GPUComputePipeline {
  let byCode = pipes.get(device);
  if (!byCode) {
    byCode = new Map();
    pipes.set(device, byCode);
  }
  let pipe = byCode.get(code);
  if (!pipe) {
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [root.unwrap(layout)] });
    pipe = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "main" } });
    byCode.set(code, pipe);
  }
  return pipe;
}

/** A Swept geometry resident on the GPU: `positions`/`normals` are `STORAGE|VERTEX|COPY_SRC`
 *  buffers (flat f32, 3 lanes/vertex) a render pass binds directly. `update` re-runs the kernel
 *  with a new param vector (same structure) into the **same** buffers — cheap, no recompile. */
export interface SweptGpuHandle {
  positions: GPUBuffer;
  normals: GPUBuffer;
  vertexCount: number;
  slices: number;
  stacks: number;
  /** Re-evaluate with a new `Swept.paramVector()` (must match this geometry's structure). */
  update(params: Float32Array): void;
}

/** Build GPU-resident geometry for a Swept horn, evaluated by the codegen'd kernel. The initial
 *  dispatch uses `g.paramVector()`; call `update` to re-evaluate a same-structure genotype. */
export function createSweptGpu(device: GPUDevice, root: Root, g: Swept, slices: number, stacks: number): SweptGpuHandle {
  const cols = slices + 1;
  const rows = stacks + 1;
  const vertexCount = cols * rows;
  const lanes = vertexCount * 3;
  const paramCount = Math.max(1, g.paramVector().length);

  const pipeline = getPipe(device, root, g.toWgsl());

  const dims = root.createBuffer(Params).$usage("uniform");
  dims.write({ slices, stacks, vertexCount, _pad: 0 });
  const pbuf = root.createBuffer(d.arrayOf(d.f32, paramCount)).$usage("storage");

  // Raw vertex/storage buffers so a render pass can bind them; wrapped for the compute bind group.
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC;
  const posRaw = device.createBuffer({ size: lanes * 4, usage });
  const norRaw = device.createBuffer({ size: lanes * 4, usage });
  const outPos = root.createBuffer(d.arrayOf(d.f32, lanes), posRaw).$usage("storage");
  const outNor = root.createBuffer(d.arrayOf(d.f32, lanes), norRaw).$usage("storage");

  const bind = root.unwrap(root.createBindGroup(layout, { params: dims, outPos, outNor, pbuf }));

  function update(params: Float32Array): void {
    pbuf.write(params.length === paramCount ? params : Float32Array.from({ length: paramCount }, (_, i) => params[i] ?? 0));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(vertexCount / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  update(g.paramVector());

  return { positions: posRaw, normals: norRaw, vertexCount, slices, stacks, update };
}

/** Tessellate a Swept geometry on the GPU and read it back as a {@link Mesh} — the same form as
 *  the CPU path (positions/normals from the kernel; indices host-side, identical). Used by the
 *  parity test and as an interop escape hatch; rendering should prefer {@link createSweptGpu}. */
export async function sweptMeshGpu(device: GPUDevice, root: Root, g: Swept, slices: number, stacks: number): Promise<Mesh> {
  const handle = createSweptGpu(device, root, g, slices, stacks);
  const lanes = handle.vertexCount * 3;
  const posV = root.createBuffer(d.arrayOf(d.f32, lanes), handle.positions);
  const norV = root.createBuffer(d.arrayOf(d.f32, lanes), handle.normals);
  const gotPos = (await posV.read()) as ArrayLike<number>;
  const gotNor = (await norV.read()) as ArrayLike<number>;
  const positions = Float32Array.from({ length: lanes }, (_, i) => gotPos[i] ?? 0);
  const normals = Float32Array.from({ length: lanes }, (_, i) => gotNor[i] ?? 0);
  return { positions, normals, indices: gridIndices(slices, stacks), vertexCount: handle.vertexCount, slices, stacks };
}
