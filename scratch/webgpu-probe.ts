// Probe: does the Dawn `webgpu` binding give a working WebGPU device under Bun?
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const gpu = create([]) as GPU;
const adapter = await gpu.requestAdapter();
if (!adapter) throw new Error("no adapter");
const device = await adapter.requestDevice();
console.log("got device:", !!device);

// Trivial compute: out[i] = in[i] * 2
const N = 8;
const input = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7]);

const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(inBuf, 0, input);
const outBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const readBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

const module = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read> inp: array<u32>;
    @group(0) @binding(1) var<storage, read_write> outp: array<u32>;
    @compute @workgroup_size(${N})
    fn main(@builtin(global_invocation_id) gid: vec3u) {
      outp[gid.x] = inp[gid.x] << 1u;
    }`,
});
const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
const bind = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
  ],
});
const enc = device.createCommandEncoder();
const pass = enc.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bind);
pass.dispatchWorkgroups(1);
pass.end();
enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, input.byteLength);
device.queue.submit([enc.finish()]);

await readBuf.mapAsync(GPUMapMode.READ);
const result = new Uint32Array(readBuf.getMappedRange().slice(0));
readBuf.unmap();
console.log("input: ", Array.from(input));
console.log("result:", Array.from(result));
const ok = result.every((v, i) => v === input[i] * 2);
console.log(ok ? "✅ WebGPU compute works under Bun" : "❌ wrong result");
process.exit(ok ? 0 : 1);
