// What the deck particle layers decide per instance, run as a compute pass on a real spring model:
// each bond in the slot table is drawn exactly once, and the species read from the attribute buffer
// are the host's. This is the layers' own WGSL (`particleWgsl.ts`) on the buffers `SpringsGpu`
// exposes; only the projection and shading need a browser.
import { describe, expect, it } from "vite-plus/test";
import { BOND_WGSL, CATEGORY_WGSL } from "../packages/deck/src/particleWgsl";
import { compileShader, getDevice, readBackBytes } from "../src/gpu/device";
import {
  ATTR_FLOATS,
  ATTR_SPECIES,
  bondCount,
  createSpheroidGpuRunner,
  DEAD,
  liveCount,
  MATRIX,
  SLOT_WORDS,
  slotKind,
  spheroidScenario,
} from "../src/gpu/sim/springs/index";

const WGSL = /* wgsl */ `
${CATEGORY_WGSL}
${BOND_WGSL}
struct Params { n: u32, K: u32, slotWords: u32, attrStride: u32, speciesAt: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
@group(0) @binding(2) var<storage, read> attrs: array<u32>;
// Per slot: 0 not drawn, else 1 + bond kind. Per particle: its category.
@group(0) @binding(3) var<storage, read_write> drawn: array<u32>;
@group(0) @binding(4) var<storage, read_write> category: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x;
  if (slot < P.n * P.K) {
    let e = bond_ends(slots[slot * P.slotWords], slot, P.K);
    drawn[slot] = select(0u, 1u + e.kind, e.draw);
  }
  if (slot < P.n) {
    category[slot] = particle_category(attrs[slot * P.attrStride + P.speciesAt], true);
  }
}
`;

describe("deck particle layers on the spring model's buffers", () => {
  it("draws each bond once and reads the host's species, after stepping on the device", async () => {
    const sc = spheroidScenario({ cellRadius: 3, matrixRadius: 8, seed: 5 });
    const runner = await createSpheroidGpuRunner(sc, { hostCurrent: false, syncEvery: 10 });
    await runner.advance(20);
    // Two syncs of ten steps, none added at the end: the host is current as of step 20.
    expect(runner.synced).toBe(2);
    const { gpu } = runner;
    const { n, K } = gpu;

    const device = await getDevice();
    const module = await compileShader(device, WGSL, "test:particle-decode");
    const pipeline = device.createComputePipeline({
      label: "test:particle-decode",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const drawn = device.createBuffer({ label: "test:drawn", size: n * K * 4, usage });
    const category = device.createBuffer({ label: "test:category", size: n * 4, usage });
    const params = device.createBuffer({ label: "test:params", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([n, K, SLOT_WORDS, ATTR_FLOATS, ATTR_SPECIES, 0, 0, 0]));
    const bind = device.createBindGroup({
      label: "test:particle-decode",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: gpu.slots } },
        { binding: 2, resource: { buffer: gpu.attrs } },
        { binding: 3, resource: { buffer: drawn } },
        { binding: 4, resource: { buffer: category } },
      ],
    });
    const enc = device.createCommandEncoder({ label: "test:particle-decode" });
    const pass = enc.beginComputePass({ label: "test:particle-decode" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil((n * K) / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    const drawnHost = new Uint32Array(await readBackBytes(device, "test:drawn", drawn, 0, n * K * 4));
    const categoryHost = new Uint32Array(await readBackBytes(device, "test:category", category, 0, n * 4));

    // Bonds: as many drawn as the host's table holds, and kind for kind.
    const drawnByKind = [0, 0, 0, 0];
    let drawnTotal = 0;
    for (const v of drawnHost) {
      if (v === 0) continue;
      drawnTotal++;
      drawnByKind[Math.min(v - 1, 3)]!++;
    }
    const hostByKind = [0, 0, 0, 0];
    const bits = sc.topology.slotBits;
    for (let w = 0; w < bits.length; w += SLOT_WORDS) {
      if (bits[w] !== 0xffffffff) hostByKind[Math.min(slotKind(bits[w]!), 3)]!++;
    }
    expect(bondCount(sc.topology)).toBeGreaterThan(200);
    expect(drawnTotal).toBe(bondCount(sc.topology));
    // The host table holds each bond twice, once per end.
    expect(drawnByKind).toEqual(hostByKind.map((c) => c / 2));

    // Species: the same counts as the host state, dead slots included.
    const count = (value: number) => categoryHost.reduce((c, v) => c + (v === value ? 1 : 0), 0);
    expect(count(DEAD)).toBe(n - liveCount(sc.state));
    expect(count(MATRIX)).toBe(liveCount(sc.state, MATRIX));
    let mismatches = 0;
    for (let i = 0; i < n; i++) if (categoryHost[i] !== sc.state.species[i]) mismatches++;
    expect(mismatches).toBe(0);
  });
});
