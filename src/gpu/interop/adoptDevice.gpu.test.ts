import { describe, it, expect } from "vitest";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getDevice } from "../device";
import { adoptDevice } from "./adoptDevice";

// Prove the interop seam: a device someone ELSE created (here the Dawn device, standing in
// for three.js's `renderer.backend.device`) can be adopted, and our TypeGPU compute runs on
// it and reads back. Same code path the browser uses to ride three.js's device.

const WG = 64;

const layout = tgpu.bindGroupLayout({
  out: { storage: (n: number) => d.arrayOf(d.f32, n), access: "mutable" },
});

const fillFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })(({ gid }) => {
    "use gpu";
    const i = gid.x;
    if (i < d.u32(layout.$.out.length)) {
      layout.$.out[i] = d.f32(i) * 2;
    }
  })
  .$name("fillDouble");

describe("adoptDevice interop seam", () => {
  it("runs compute on an externally-owned device and reads it back", async () => {
    const device = await getDevice(); // pretend this came from a host renderer
    const backend = adoptDevice(device, "test-host");
    expect(backend.kind).toBe("test-host");

    const root = await backend.getRoot();
    expect(await backend.getDevice()).toBe(device);

    const n = 256;
    const out = root.createBuffer(d.arrayOf(d.f32, n)).$usage("storage");

    const { code, usedBindGroupLayouts } = tgpu.resolveWithContext([fillFn], { names: "strict" });
    const module = device.createShaderModule({ code });
    const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: usedBindGroupLayouts.map((l) => root.unwrap(l)) });
    const pipeline = device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint: "fillDouble" } });

    const group = root.createBindGroup(layout, { out: out });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, root.unwrap(group));
    pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);

    const got = await backend.readbackF32(root.unwrap(out), n);
    expect(got.length).toBe(n);
    for (let i = 0; i < n; i++) expect(got[i]).toBeCloseTo(i * 2, 5);
  });
});

// silence unused import if std tree-shakes; std is imported to match the kernel idiom used
// elsewhere and keep the transpiler's global set consistent.
void std;
