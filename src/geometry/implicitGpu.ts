// GPU lowering of an Implicit geometry's distance field — one compute invocation per sample point.
// The shader's `sdScene(p)` is **codegen'd from the SDF IR** (`Implicit.toWgsl`), so it is the same
// field as the CPU golden `evalSdf`; the parity test (`implicit.gpu.test.ts`) pins them together.
//
// This is the field-evaluation building block, not a full mesh extraction: it samples `sdScene` at
// a caller-supplied set of points and reads the distances back. A raymarch render pass would bind
// the same `sdScene`; the GPU surface-nets extraction pass (classify → compact → connect) is the
// designed-for next step (ADR-0010; see `docs/explainers/surface-nets-dual-contouring.html`). We
// hand-author the WGSL wrapper — a dynamic per-structure shader can't be a static `"use gpu"`
// kernel — and bind it through TypeGPU's Dawn-stable path (ADR-0003), never TSL.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { Root } from "../gpu/graph/backend";
import type { Implicit } from "./implicit";

const WG = 64;

const Params = d.struct({ count: d.u32, _p0: d.u32, _p1: d.u32, _p2: d.u32 });

// Binding order fixes the WGSL `@binding` indices below: params=0, inPts=1, outDist=2, P=3.
const layout = tgpu.bindGroupLayout({
  params: { uniform: Params },
  inPts: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
  outDist: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
  pbuf: { storage: (n: number) => d.arrayOf(d.f32, n), access: "readonly" },
});

// P is declared here (before `sdScene` uses it) so the codegen'd body can read param values.
const BINDINGS = /* wgsl */ `
struct Params { count: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inPts: array<f32>;
@group(0) @binding(2) var<storage, read_write> outDist: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<f32>;
`;

const MAIN = /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let p = vec3<f32>(inPts[i * 3u], inPts[i * 3u + 1u], inPts[i * 3u + 2u]);
  outDist[i] = sdScene(p);
}
`;

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

/** Evaluate an Implicit geometry's signed-distance field at `points` (flat `[x,y,z, …]`) on the
 *  GPU, via the codegen'd `sdScene`. Returns one distance per point — the GPU image of `evalSdf`. */
export async function sampleSdfGpu(device: GPUDevice, root: Root, g: Implicit, points: Float32Array): Promise<Float32Array> {
  const count = Math.floor(points.length / 3);
  const paramVec = g.paramVector();
  const paramCount = Math.max(1, paramVec.length);

  const code = `${BINDINGS}\n${g.toWgsl()}\n${MAIN}`;
  const pipeline = getPipe(device, root, code);

  const dims = root.createBuffer(Params).$usage("uniform");
  dims.write({ count, _p0: 0, _p1: 0, _p2: 0 });
  const inPts = root.createBuffer(d.arrayOf(d.f32, Math.max(3, points.length))).$usage("storage");
  inPts.write(points.length >= 3 ? points : Float32Array.from({ length: 3 }, (_, i) => points[i] ?? 0));
  const outDist = root.createBuffer(d.arrayOf(d.f32, Math.max(1, count))).$usage("storage");
  const pbuf = root.createBuffer(d.arrayOf(d.f32, paramCount)).$usage("storage");
  pbuf.write(paramVec.length === paramCount ? paramVec : Float32Array.from({ length: paramCount }, (_, i) => paramVec[i] ?? 0));

  const bind = root.unwrap(root.createBindGroup(layout, { params: dims, inPts, outDist, pbuf }));

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(Math.max(1, count) / WG));
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = (await outDist.read()) as ArrayLike<number>;
  return Float32Array.from({ length: count }, (_, i) => got[i] ?? 0);
}
