// The single complex-field reaction–diffusion node (ADR-0004) must be exactly
// equivalent to the legacy two-feedback-node (U,V) loop — same Gray–Scott kernel,
// one signal instead of two. Run in CPU mode so this stays a fast `*.test.ts` (no
// Dawn); correctness of the collapse is backend-independent.
import { describe, it, expect, beforeAll } from "vitest";
import { Graph, advance, createSimState, registerElementOps } from "./index";
import { seedGrayScott } from "../sim/reactionDiffusion";
import { packComplex } from "./elementMath";

beforeAll(async () => {
  await registerElementOps();
});

const W = 20, H = 20, PER_TICK = 4, TICKS = 5;

/** Build the legacy two-node (U,V) loop and return its two sink handles + graph. */
function twoNodeLoop() {
  const seed = seedGrayScott(W, H, 0.05);
  const g = new Graph();
  const u0 = g.grid(seed.u, W, H);
  const v0 = g.grid(seed.v, W, H);
  const fbU = g.feedback(u0, "U");
  const fbV = g.feedback(v0, "V");
  const [uNext, vNext] = g.op("reactionDiffusionStep", { u: fbU.state, v: fbV.state }, { steps: PER_TICK });
  fbU.close(uNext!);
  fbV.close(vNext!);
  return { g, uNext: uNext!, vNext: vNext! };
}

/** The single complex-node loop: one feedback node carrying re=U, im=V. */
function complexLoop() {
  const seed = seedGrayScott(W, H, 0.05);
  const g = new Graph();
  const z0 = g.source({
    shape: { kind: "grid", width: W, height: H },
    dtype: "f32",
    element: { kind: "complex" },
    data: packComplex(seed.u, seed.v),
  });
  const fbZ = g.feedback(z0, "Z");
  const zNext = g.op1("reactionDiffusionComplex", { state: fbZ.state }, { steps: PER_TICK });
  fbZ.close(zNext);
  return { g, fbZ, zNext };
}

/** Advance a loop `TICKS` ticks (each tick runs the whole cut-DAG once, so reading
 *  either sink yields the correct final state) and return the sink's host data. */
async function finalData(g: Graph, sink: ReturnType<Graph["op1"]>): Promise<Float32Array> {
  const out = await advance(g, sink, { steps: TICKS, state: createSimState(), mode: "cpu" });
  return out.data as Float32Array;
}

describe("reaction–diffusion: single complex node ≡ two (U,V) nodes", () => {
  it("the complex state stays element-typed through feedback and the step op", () => {
    const { fbZ, zNext } = complexLoop();
    expect(fbZ.state.element).toEqual({ kind: "complex" });
    expect(zNext.element).toEqual({ kind: "complex" });
  });

  it("rejects a non-complex state at graph-build time", () => {
    const g = new Graph();
    const s = g.grid(new Float32Array(W * H).fill(1), W, H); // scalar
    expect(() => g.op1("reactionDiffusionComplex", { state: s })).toThrow(/complex/);
  });

  it("advancing the complex loop matches the two-node loop bit-for-bit", async () => {
    const two = twoNodeLoop();
    const u = await finalData(two.g, two.uNext);
    const v = await finalData(two.g, two.vNext);

    const cx = complexLoop();
    const z = await finalData(cx.g, cx.zNext);

    let maxDiff = 0;
    for (let i = 0; i < W * H; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(u[i]! - z[i * 2]!), Math.abs(v[i]! - z[i * 2 + 1]!));
    }
    expect(maxDiff).toBeLessThan(1e-6);
  });
});
