import { describe, it, expect } from "vitest";
import { Graph, pullData } from "./index";
import { pointHotspotsGpu } from "../spatial/getisOrd";

// Composition tests: the graph runtime reproduces the hand-written GPU pipeline.
// Kept in its own file (one fork) because the splat render path + compute roots do
// "enough work" that Dawn's process-exit teardown is fragile when many GPU
// scenarios share a process — the suite isolates per file for exactly this reason
// (see vitest.gpu.config.ts). Executor-only checks live in graph_exec.gpu.test.ts.

function fixture() {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < 40; i++) { xs.push(50 + (i % 7) * 0.6); ys.push(50 + Math.floor(i / 7) * 0.6); }
  for (let i = 0; i < 20; i++) { xs.push((i * 13) % 100); ys.push((i * 29) % 100); }
  return { xs, ys, bbox: [0, 0, 100, 100] as [number, number, number, number] };
}

describe("graph runtime — composition", () => {
  it("graph[splat -> getisOrd] matches pointHotspotsGpu (the hand-written composition)", async () => {
    const { xs, ys, bbox } = fixture();
    const opts = { width: 40, height: 40, sigma: 2, radius: 2 };

    const g = new Graph();
    const pts = g.points(xs, ys);
    const dens = g.op1("splatDensity", { points: pts }, { width: opts.width, height: opts.height, sigma: opts.sigma, bbox });
    const z = g.op1("getisOrd", { grid: dens }, { radius: opts.radius });
    const got = await pullData(g, z);

    const ref = await pointHotspotsGpu(xs, ys, { ...opts, bbox });

    expect(got.length).toBe(ref.z.length);
    let maxAbs = 0;
    for (let i = 0; i < got.length; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i]! - ref.z[i]!));
    expect(maxAbs).toBeLessThan(1e-4); // same underlying GPU path → essentially identical
  });
});
