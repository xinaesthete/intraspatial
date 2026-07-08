// GPU lowering of a Swept geometry — one compute invocation per grid vertex (ADR-0003). The
// shader body is **codegen'd from the geometry IR** (`Swept.toWgsl`), so it is the same
// closed-form `eval(s, θ)` as the CPU golden `Swept.tessellate`, just run on the device; the
// parity test (`swept.gpu.test.ts`) pins them together.
//
// Isolated in its own module (imported dynamically by `toMesh`, and directly by the GPU test)
// so the device / TypeGPU machinery stays out of the CPU module graph — Dawn-on-Node teardown
// is sensitive to what pulls a GPU pipeline in (see `element.gpu.test.ts`). We hand-author the
// WGSL (a *dynamic* per-geometry shader can't be a statically-authored `"use gpu"` kernel) but
// bind and read it back through TypeGPU's Dawn-stable path — the TypeGPU/WGSL backend, never TSL.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { Root } from "../gpu/graph/backend";
import { gridIndices, type Mesh, type Swept } from "./swept";

const WG = 64;

const Params = d.struct({ slices: d.u32, stacks: d.u32, vertexCount: d.u32, _pad: d.u32 });

// Binding order fixes the WGSL `@binding` indices in `sweptShaderWgsl`: params=0, pos=1, nor=2.
const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  outPos: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  outNor: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

/** Tessellate a Swept geometry on the GPU, returning the same {@link Mesh} form as the CPU
 *  path (positions/normals filled by the kernel; indices computed host-side, identical). */
export async function sweptMeshGpu(device: GPUDevice, root: Root, g: Swept, slices: number, stacks: number): Promise<Mesh> {
  const cols = slices + 1;
  const rows = stacks + 1;
  const vertexCount = cols * rows;
  const lanes = vertexCount * 3;

  const code = g.toWgsl();
  const module = device.createShaderModule({ code });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [root.unwrap(layout)] });
  const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "main" } });

  const params = root.createBuffer(Params).$usage("uniform");
  const outPos = root.createBuffer(d.arrayOf(d.f32, lanes)).$usage("storage");
  const outNor = root.createBuffer(d.arrayOf(d.f32, lanes)).$usage("storage");
  params.write({ slices, stacks, vertexCount, _pad: 0 });

  const bind = root.unwrap(root.createBindGroup(layout, { params, outPos, outNor }));
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(vertexCount / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const gotPos = (await outPos.read()) as ArrayLike<number>;
  const gotNor = (await outNor.read()) as ArrayLike<number>;
  const positions = Float32Array.from({ length: lanes }, (_, i) => gotPos[i] ?? 0);
  const normals = Float32Array.from({ length: lanes }, (_, i) => gotNor[i] ?? 0);
  return { positions, normals, indices: gridIndices(slices, stacks), vertexCount, slices, stacks };
}
