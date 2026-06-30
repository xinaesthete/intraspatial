// End-to-end runtime test for the dance field: build the exact graph the playground
// example wires (seed → feedback → danceField → back), then `advance` the feedback loop
// and confirm the swarm evolves and the state persists across ticks. This exercises the
// real executor path — feedback (z⁻¹) state carry, vec3 element inference in graph.op,
// and the 2N→N shape inference — without a GPU device (mode: "cpu" runs cpuGolden).
import { describe, expect, it } from "vitest";
import { advance, createSimState, getOp, Graph, hasOp } from "../index";
import { packSwarm, seedSwarm } from "../../sim/danceField";

function buildDanceGraph(n: number) {
  const g = new Graph();
  const seed = g.source(
    { shape: { kind: "points", n: n * 2 }, dtype: "f32", element: { kind: "vec", n: 3 }, data: packSwarm(seedSwarm(n, 1)) },
    "danceSwarmSeed",
  );
  const fb = g.feedback(seed, "swarm");
  const [state, swarm] = g.op("danceField", { state: fb.state }, { steps: 2 });
  fb.close(state!);
  return { g, swarm: swarm! };
}

describe("danceField op wiring", () => {
  it("registers in the builtin op set", () => {
    expect(hasOp("danceField")).toBe(true);
    expect(getOp("danceField").outputs.map((o) => o.name)).toEqual(["state", "swarm"]);
  });

  it("infers the swarm output shape as N points from a 2N-row state input", () => {
    const op = getOp("danceField");
    const [state, swarm] = op.inferShapes([{ kind: "points", n: 80 }], {});
    expect(state).toEqual({ kind: "points", n: 80 });
    expect(swarm).toEqual({ kind: "points", n: 40 });
  });

  it("rejects a non-vec3 state element at build time", () => {
    const g = new Graph();
    const bad = g.source({ shape: { kind: "points", n: 20 }, dtype: "f32", data: new Float32Array(40) }, "bad");
    expect(() => g.op("danceField", { state: bad })).toThrow(/vec3/);
  });

  it("advances the feedback loop, evolving the swarm while carrying state across ticks", async () => {
    const n = 64;
    const { g, swarm } = buildDanceGraph(n);
    const store = createSimState();

    const f0 = await advance(g, swarm, { steps: 1, state: store, mode: "cpu" });
    const after = await advance(g, swarm, { steps: 40, state: store, mode: "cpu" });

    expect(after.shape).toEqual({ kind: "points", n }); // the swarm port is N positions
    expect(after.data!.length).toBe(n * 2); // [x,y] pairs for the scatter
    for (let i = 0; i < after.data!.length; i++) expect(Number.isFinite(after.data![i]!)).toBe(true);

    // The loop actually moved the swarm between tick 1 and tick 41.
    let moved = 0;
    for (let i = 0; i < f0.data!.length; i++) if (Math.abs(after.data![i]! - f0.data![i]!) > 1e-3) moved++;
    expect(moved).toBeGreaterThan(n); // most coordinates changed
  });
});
