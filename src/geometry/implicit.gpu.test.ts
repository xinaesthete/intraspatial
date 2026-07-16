// Golden parity: the codegen'd WGSL `sdScene` must reproduce the CPU golden `evalSdf` (ADR-0003 /
// ADR-0010). Both lower the *same* SDF IR, so they are the same field twice — this pins them
// numerically over a grid of sample points, and doubles as the backstop that the `P`-buffer slot
// order matches `paramVector` (a mismatch smears distances). Kept small: Dawn-on-Node teardown
// segfaults past enough cumulative GPU work (ADR-0002/0003).
import { describe, expect, it } from "vitest";
import { nodeBackend } from "../gpu/graph/backend.node";
import type { Implicit } from "./implicit";
import { box, evalSdf, sphere } from "./implicit";
import { sampleSdfGpu } from "./implicitGpu";
import type { Vec3 } from "./superellipsoid";

// A small lattice of sample points across the domain.
function gridPoints(n: number, bound: number): Float32Array {
  const out: number[] = [];
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        out.push(-bound + (2 * bound * i) / (n - 1), -bound + (2 * bound * j) / (n - 1), -bound + (2 * bound * k) / (n - 1));
      }
  return Float32Array.from(out);
}

const maxAbsDiff = (pts: Float32Array, gpu: Float32Array, node: Parameters<typeof evalSdf>[0]) => {
  let m = 0;
  for (let i = 0; i < gpu.length; i++) {
    const p: Vec3 = [pts[i * 3] ?? 0, pts[i * 3 + 1] ?? 0, pts[i * 3 + 2] ?? 0];
    m = Math.max(m, Math.abs((gpu[i] ?? 0) - evalSdf(node, p)));
  }
  return m;
};

describe("Implicit geometry: GPU sdScene matches the CPU golden", () => {
  it("reproduces the field of a smooth-unioned, transformed CSG tree", async () => {
    const g = sphere(1)
      .translate(-0.4, 0, 0)
      .smoothUnion(box(0.6, 0.4, 0.5).translate(0.4, 0, 0), 0.3)
      .subtract(sphere(0.5).translate(0, 0.6, 0));

    const pts = gridPoints(6, 1.6); // 216 points — small enough for Dawn
    const device = await nodeBackend.getDevice();
    const root = await nodeBackend.getRoot();
    const gpu = await sampleSdfGpu(device, root, g, pts);

    expect(gpu.length).toBe(pts.length / 3);
    expect(maxAbsDiff(pts, gpu, g.node)).toBeLessThan(1e-4);
  });

  it("reproduces a noise-displaced field (bit-exact hash; f32/f64 interp only)", async () => {
    // The value-noise hash is pure u32 arithmetic, identical CPU/GPU, so the only divergence is the
    // f32-vs-f64 interpolation of four octaves — a looser but still tight bound than analytic fields.
    const g: Implicit = sphere(1).displace(0.35, 2.5);
    const pts = gridPoints(6, 1.4);
    const device = await nodeBackend.getDevice();
    const root = await nodeBackend.getRoot();
    const gpu = await sampleSdfGpu(device, root, g, pts);
    expect(maxAbsDiff(pts, gpu, g.node)).toBeLessThan(1e-3);
  });
});
