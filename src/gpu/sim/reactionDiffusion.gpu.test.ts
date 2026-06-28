import { describe, it, expect } from "vitest";
import { grayScottStepCpu, grayScottStepsGpu, seedGrayScott } from "./reactionDiffusion";
import type { GrayScottState } from "./reactionDiffusion";

const PARAMS = { du: 0.16, dv: 0.08, feed: 0.06, kill: 0.062, dt: 1 };

describe("Gray–Scott reaction–diffusion step", () => {
  it("matches the CPU golden for one step", async () => {
    const seed = seedGrayScott(24, 24, 0.05);
    const gpu = await grayScottStepsGpu(seed, 1, PARAMS);
    const cpu = grayScottStepCpu(seed, PARAMS);

    let maxU = 0, maxV = 0;
    for (let i = 0; i < gpu.u.length; i++) {
      maxU = Math.max(maxU, Math.abs(gpu.u[i]! - cpu.u[i]!));
      maxV = Math.max(maxV, Math.abs(gpu.v[i]! - cpu.v[i]!));
    }
    expect(maxU).toBeLessThan(1e-5);
    expect(maxV).toBeLessThan(1e-5);
  });

  it("advances stably and keeps a bounded, evolving pattern", async () => {
    const w = 24, h = 24;
    const seed = seedGrayScott(w, h, 0.05);

    const out: GrayScottState = await grayScottStepsGpu(seed, 40, PARAMS);
    let finite = true, minV = Infinity, maxV = -Infinity, changed = 0;
    for (let i = 0; i < out.v.length; i++) {
      if (!Number.isFinite(out.u[i]!) || !Number.isFinite(out.v[i]!)) finite = false;
      minV = Math.min(minV, out.v[i]!);
      maxV = Math.max(maxV, out.v[i]!);
      changed = Math.max(changed, Math.abs(out.v[i]! - seed.v[i]!));
    }
    expect(finite).toBe(true);
    expect(minV).toBeGreaterThan(-0.01); // bounded below
    expect(maxV).toBeLessThan(1.01); // bounded above
    expect(maxV).toBeGreaterThan(0.05); // a V pattern persists, not washed out
    expect(changed).toBeGreaterThan(1e-3); // the dynamics actually ran
  });
});
