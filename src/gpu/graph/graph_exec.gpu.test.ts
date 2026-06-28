import { describe, it, expect } from "vitest";
import { Graph, pull, pullData } from "./index";

// Executor-semantics tests: validate/fallback equivalence and fork-join dedup.
// Separate file (own fork) — see the note in graph.gpu.test.ts on Dawn teardown.

function fixture() {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < 40; i++) { xs.push(50 + (i % 7) * 0.6); ys.push(50 + Math.floor(i / 7) * 0.6); }
  for (let i = 0; i < 20; i++) { xs.push((i * 13) % 100); ys.push((i * 29) % 100); }
  return { xs, ys, bbox: [0, 0, 100, 100] as [number, number, number, number] };
}

describe("graph runtime — executor", () => {
  it("native threshold op matches its CPU golden (gpu vs cpu executor mode)", async () => {
    const w = 16, h = 16;
    const grid = Float32Array.from({ length: w * h }, (_, i) => ((i * 2654435761) % 1000) / 1000);
    const g = new Graph();
    const src = g.grid(grid, w, h);
    const soft = g.op1("threshold", { in: src }, { thresh: 0.4, soft: true, softness: 12 });

    const viaGpu = await pullData(g, soft, { mode: "gpu" });
    const viaCpu = await pullData(g, soft, { mode: "cpu" });
    let maxAbs = 0;
    for (let i = 0; i < viaGpu.length; i++) maxAbs = Math.max(maxAbs, Math.abs(viaGpu[i]! - viaCpu[i]!));
    expect(maxAbs).toBeLessThan(1e-5);
  });

  it("executes a shared producer exactly once (diamond dedup)", async () => {
    const { xs, ys, bbox } = fixture();
    const g = new Graph();
    const pts = g.points(xs, ys);
    const dens = g.op1("splatDensity", { points: pts }, { width: 24, height: 24, sigma: 2, bbox });
    // diamond: dens fans out to two consumers that join again at addGrids
    const a = g.op1("convolveSeparable", { grid: dens }, { kernel: "box", radius: 1 });
    const b = g.op1("convolveSeparable", { grid: dens }, { kernel: "gaussian", radius: 2 });
    const joined = g.op1("addGrids", { a, b }, { wa: 1, wb: -1 });

    const produced: string[] = [];
    await pull(g, joined, { onValue: (k) => produced.push(k) });
    const densKey = produced.filter((k) => k.startsWith(dens.producer + ":"));
    expect(densKey.length).toBe(1); // density computed once despite two downstream paths
  });
});
