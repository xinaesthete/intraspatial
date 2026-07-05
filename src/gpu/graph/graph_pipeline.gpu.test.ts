import { describe, expect, it } from "vitest";
import { Graph, pullData } from "./index";

// The headline vertical slice: a 3-node graph splat -> convolve -> threshold,
// executed end to end by the runtime. Own file (own fork) — see graph.gpu.test.ts.

describe("graph runtime — vertical slice", () => {
  it("runs splat -> convolve -> threshold and flags the dense cluster", async () => {
    const xs: number[] = [],
      ys: number[] = [];
    for (let i = 0; i < 40; i++) {
      xs.push(50 + (i % 7) * 0.6);
      ys.push(50 + Math.floor(i / 7) * 0.6);
    }
    for (let i = 0; i < 20; i++) {
      xs.push((i * 13) % 100);
      ys.push((i * 29) % 100);
    }
    const bbox: [number, number, number, number] = [0, 0, 100, 100];

    const w = 24,
      h = 24;
    const g = new Graph();
    const pts = g.points(xs, ys);
    const dens = g.op1("splatDensity", { points: pts }, { width: w, height: h, sigma: 2, bbox });
    const smooth = g.op1("convolveSeparable", { grid: dens }, { kernel: "box", radius: 1 });
    const mask = g.op1("threshold", { in: smooth }, { thresh: 0.5, soft: false });
    const out = await pullData(g, mask);

    let ones = 0;
    for (const v of out) {
      expect(v === 0 || v === 1).toBe(true);
      if (v === 1) ones++;
    }
    expect(ones).toBeGreaterThan(0);
    const col = Math.round(((52 - 0) / 100) * w - 0.5);
    const row = Math.round(((100 - 52) / 100) * h - 0.5);
    expect(out[row * w + col]).toBe(1);
  });
});
