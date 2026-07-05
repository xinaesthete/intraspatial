// GPU-backed element algebra (ADR-0004): the complex-multiply compute kernel must
// match its CPU reference, and the complex reaction–diffusion node must run end to end
// on the real (Dawn) backend. Small scenarios — Dawn-on-Node teardown segfaults past
// "enough" cumulative GPU work, so keep grids tiny (ADR-0002/0003).
//
// `complexMulGpu` is imported DIRECTLY here, not via the op registry: pulling that
// `"use gpu"` module into the registry's module graph destabilised Dawn teardown in
// unrelated forks, so `mulFields` stays CPU Tier-1 and the kernel is validated in
// isolation in this dedicated GPU fork instead.
import { beforeAll, describe, expect, it } from "vitest";
import { seedGrayScott } from "../sim/reactionDiffusion";
import { mulFields, packComplex } from "./elementMath";
import type { ElementType } from "./index";
import { advance, createSimState, Graph, nodeBackend, registerElementOps } from "./index";
import { complexMulGpu } from "./ops/complexMulGpu";

const COMPLEX: ElementType = { kind: "complex" };

beforeAll(async () => {
  await registerElementOps();
});

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
};

describe("element algebra on the GPU", () => {
  it("complexMulGpu kernel matches the CPU mulFields reference", async () => {
    const n = 256; // 256 complex samples → 512 interleaved lanes
    const a = new Float32Array(2 * n);
    const b = new Float32Array(2 * n);
    for (let i = 0; i < 2 * n; i++) {
      a[i] = ((i * 2654435761) % 1000) / 1000 - 0.5;
      b[i] = ((i * 40503) % 1000) / 1000 - 0.5;
    }
    const device = await nodeBackend.getDevice();
    const root = await nodeBackend.getRoot();
    const gpu = await complexMulGpu(device, root, a, b);
    const cpu = mulFields(COMPLEX, a, b);
    expect(maxAbsDiff(gpu, cpu)).toBeLessThan(1e-5);
  });

  it("reaction–diffusion (complex) advances on the GPU, matching the CPU integrator", async () => {
    const w = 8,
      h = 8;
    const seed = seedGrayScott(w, h, 0.05);
    const make = () => {
      const g = new Graph();
      const z0 = g.source({
        shape: { kind: "grid", width: w, height: h },
        dtype: "f32",
        element: COMPLEX,
        data: packComplex(seed.u, seed.v),
      });
      const fb = g.feedback(z0, "Z");
      const zNext = g.op1("reactionDiffusionComplex", { state: fb.state }, { steps: 3 });
      fb.close(zNext);
      return { g, zNext };
    };
    const a = make(),
      b = make();
    const onGpu = await advance(a.g, a.zNext, { steps: 3, state: createSimState(), mode: "gpu" });
    const onCpu = await advance(b.g, b.zNext, { steps: 3, state: createSimState(), mode: "cpu" });
    expect(maxAbsDiff(onGpu.data!, onCpu.data!)).toBeLessThan(1e-3);
  });
});
