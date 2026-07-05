import tgpu from "typegpu";
import * as d from "typegpu/data";
import { expect, test } from "vitest";
import { getDevice } from "../src/gpu/device";

test("typegpu buffer write/read roundtrip (Node + Dawn)", async () => {
  const device = await getDevice();
  const root = tgpu.initFromDevice({ device });
  const buf = root.createBuffer(d.arrayOf(d.u32, 4), [10, 20, 30, 40]).$usage("storage");
  const back = await buf.read();
  expect(Array.from(back)).toEqual([10, 20, 30, 40]);
});

test("raw-WGSL compute over a typegpu-managed buffer doubles values", async () => {
  const device = await getDevice();
  const root = tgpu.initFromDevice({ device });
  const N = 8;
  const io = root.createBuffer(d.arrayOf(d.u32, N), [0, 1, 2, 3, 4, 5, 6, 7]).$usage("storage");
  const gpuBuf = root.unwrap(io);

  const module = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;
      @compute @workgroup_size(${N})
      fn main(@builtin(global_invocation_id) gid: vec3u) {
        data[gid.x] = data[gid.x] << 1u;
      }`,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuBuf } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([enc.finish()]);

  const back = await io.read();
  expect(Array.from(back)).toEqual([0, 2, 4, 6, 8, 10, 12, 14]);
});
