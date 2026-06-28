import { describe, it, expect } from "vitest";
import { Graph, advance, createSimState, pull } from "./index";
import { grayScottStepsGpu, seedGrayScott } from "../sim/reactionDiffusion";

describe("feedback (unit-delay) semantics", () => {
  // An accumulator: state_{t+1} = state_t + 1. CPU-only (addGrids), so no GPU work.
  function accumulator() {
    const g = new Graph();
    const zero = g.grid(new Float32Array(4).fill(0), 2, 2);
    const one = g.grid(new Float32Array(4).fill(1), 2, 2);
    const fb = g.feedback(zero, "acc");
    const sum = g.op1("addGrids", { a: fb.state, b: one }, { wa: 1, wb: 1 });
    fb.close(sum);
    return { g, sum };
  }

  it("pull runs one tick from the seed (init + 1)", async () => {
    const { g, sum } = accumulator();
    const v = await pull(g, sum);
    expect(Array.from(v.data!)).toEqual([1, 1, 1, 1]);
  });

  it("advance steps the loop, persisting state across ticks", async () => {
    const { g, sum } = accumulator();
    const state = createSimState();
    const v = await advance(g, sum, { steps: 5, state });
    expect(Array.from(v.data!)).toEqual([5, 5, 5, 5]); // 0→1→2→3→4→5

    // continuing reuses the stored state
    const v2 = await advance(g, sum, { steps: 2, state });
    expect(Array.from(v2.data!)).toEqual([7, 7, 7, 7]);

    // reset re-seeds from init
    const v3 = await advance(g, sum, { steps: 1, state, reset: true });
    expect(Array.from(v3.data!)).toEqual([1, 1, 1, 1]);
  });

  it("rejects a cycle that does not pass through a feedback node", () => {
    const g = new Graph();
    const a = g.grid(new Float32Array(4).fill(1), 2, 2);
    const b = g.op1("addGrids", { a, b: a }, {});
    // hand-craft an illegal self-cycle: b reads itself
    g.getNode(b.producer).inputs.a = { node: b.producer, port: "out" };
    return expect(pull(g, b)).rejects.toThrow(/cycle/);
  });
});

describe("reaction-diffusion as a feedback loop (GPU)", () => {
  it("advancing the loop matches a direct multi-step run", async () => {
    const w = 24, h = 24, perTick = 5, ticks = 4;
    const seed = seedGrayScott(w, h, 0.05);

    const g = new Graph();
    const u0 = g.grid(seed.u, w, h);
    const v0 = g.grid(seed.v, w, h);
    const fbU = g.feedback(u0, "U");
    const fbV = g.feedback(v0, "V");
    const step = g.op("reactionDiffusionStep", { u: fbU.state, v: fbV.state }, { steps: perTick });
    const [uNext, vNext] = step;
    fbU.close(uNext!);
    fbV.close(vNext!);

    const state = createSimState();
    const out = await advance(g, vNext!, { steps: ticks, state }); // V at the final tick

    const ref = await grayScottStepsGpu(seed, perTick * ticks); // same total Euler steps
    let maxAbs = 0;
    for (let i = 0; i < out.data!.length; i++) maxAbs = Math.max(maxAbs, Math.abs(out.data![i]! - ref.v[i]!));
    expect(maxAbs).toBeLessThan(1e-4);
  });
});
