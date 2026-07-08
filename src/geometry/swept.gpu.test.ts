// Golden parity: the codegen'd TGSL/WGSL per-vertex kernel must reproduce the CPU golden
// `(s, θ)` loop (ADR-0003 / ADR-0010). Because both lower the *same* Swept IR, they are the
// same closed-form function twice — this test pins them numerically. Kept tiny: Dawn-on-Node
// teardown segfaults past enough cumulative GPU work, so use a small grid (ADR-0002/0003).
import { describe, expect, it } from "vitest";
// Import the backend directly (not via the ops barrel) so the built-in op registry — and any
// `"use gpu"` kernel modules it pulls in — stays out of this Dawn test process (element.gpu note).
import { nodeBackend } from "../gpu/graph/backend.node";
import { linear, ramp } from "./expr";
import { horn } from "./swept";
import { sweptMeshGpu } from "./sweptGpu";

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return m;
};

// Compare normals by direction (1 − |cos|): the finite-difference framing subtracts near-equal
// positions, so f32 vs f64 cancellation shows up in the length more than the direction.
const maxNormalMisalign = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let worst = 0;
  for (let i = 0; i + 2 < a.length; i += 3) {
    const dot = (a[i] ?? 0) * (b[i] ?? 0) + (a[i + 1] ?? 0) * (b[i + 1] ?? 0) + (a[i + 2] ?? 0) * (b[i + 2] ?? 0);
    worst = Math.max(worst, 1 - Math.abs(dot));
  }
  return worst;
};

describe("Swept geometry: GPU kernel matches the CPU golden", () => {
  it("reproduces positions and normals for a bent, twisted, tapered horn", async () => {
    const g = horn({ radius: ramp(1.2), exponent: 1, length: 3 })
      .twist(ramp(360))
      .bend(ramp(50))
      .scale(linear(1, 0.3));

    const slices = 12;
    const stacks = 8;
    const cpu = g.tessellate({ slices, stacks });

    const device = await nodeBackend.getDevice();
    const root = await nodeBackend.getRoot();
    const gpu = await sweptMeshGpu(device, root, g, slices, stacks);

    expect(gpu.vertexCount).toBe(cpu.vertexCount);
    expect(maxAbsDiff(gpu.positions, cpu.positions)).toBeLessThan(1e-4);
    expect(maxNormalMisalign(gpu.normals, cpu.normals)).toBeLessThan(5e-3);
  });
});
