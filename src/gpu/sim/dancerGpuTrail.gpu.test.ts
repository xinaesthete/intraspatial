import { describe, it, expect } from "vitest";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { getDevice } from "../device";
import { DancerGpuSim, type DancerGpuParams } from "./dancerGpu";

const P: DancerGpuParams = {
  constrain: 0.5, cohere: 0.4, cohereRadius: 3, separate: 0.6, separateRadius: 1.4, orbit: 0.35,
  swim: 0, vortex: 0, solenoid: 0, partner: 0.5, partnerOffset: 1, caller: 1, callerGain: 0.09,
  callerSpeed: 0.6, period: 480, callerSeed: 0, timeFactor: 0.2, jerkLimit: 0.05, linDamp: 0.96,
  angDamp: 0.9, speedLimit: 1.2, face: 0.5, maxRadius: 40,
};

describe("DancerGpuSim trail history", () => {
  it("seeds the ring, then appends the current position each step (touched vs untouched slots)", async () => {
    const device = await getDevice();
    const n = 8;
    const cap = 6;
    const sim = new DancerGpuSim(device, n, 3, P);
    sim.init();

    const root = tgpu.initFromDevice({ device });
    // vec3f-strided (matches the kernel's array<vec3f> — 16 bytes/sample, NOT packed f32)
    const hist = root.createBuffer(d.arrayOf(d.vec3f, n * cap)).$usage("storage");
    sim.setTrailTarget(root.unwrap(hist), cap); // seeds every slot to the current position

    const K = 3;
    for (let s = 0; s < K; s++) sim.step(); // writes slots 0..K-1 (head = frame % cap)

    const raw = (await root.createBuffer(d.arrayOf(d.vec3f, n * cap), root.unwrap(hist)).read()) as { x: number; y: number; z: number }[];
    const at = (i: number, slot: number, c: number): number => {
      const v = raw[i * cap + slot] ?? { x: 0, y: 0, z: 0 };
      return c === 0 ? v.x : c === 1 ? v.y : v.z;
    };

    let finite = true;
    let maxUntouchedSpread = 0; // untouched slots (K..cap-1) should all equal the seed position
    let minTouchedMove = Infinity; // touched slots should have moved off the seed
    for (let i = 0; i < n; i++) {
      const seedSlot = cap - 1; // never written by K<cap steps ⇒ still the seed position
      for (let c = 0; c < 3; c++) {
        for (let slot = K; slot < cap; slot++) {
          if (!Number.isFinite(at(i, slot, c))) finite = false;
          maxUntouchedSpread = Math.max(maxUntouchedSpread, Math.abs(at(i, slot, c) - at(i, seedSlot, c)));
        }
      }
      // the newest touched slot (K-1) should differ from the seed (the agent moved)
      let move = 0;
      for (let c = 0; c < 3; c++) move += Math.abs(at(i, K - 1, c) - at(i, seedSlot, c));
      minTouchedMove = Math.min(minTouchedMove, move);
    }
    expect(finite).toBe(true);
    expect(maxUntouchedSpread).toBeLessThan(1e-6); // untouched slots identical to the seed
    expect(minTouchedMove).toBeGreaterThan(1e-4); // every agent's newest sample moved
    expect(sim.trailCapacity()).toBe(cap);
    expect(sim.trailHead()).toBe((K - 1) % cap); // newest slot index after K steps
  });
});
