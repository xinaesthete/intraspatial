// End-to-end runtime test for the composed Ceilidh dancer: build the same force-graph the
// playground example wires (swarm body → feedback → bodyTap → forces → Σ addFields →
// integrate → back), then `advance` the loop and confirm the swarm evolves and stays
// bounded. Exercises the real executor path — multi-input ops, vec3 element inference, the
// 5N-block body field, and the feedback state-carry — without a GPU device (mode "cpu").
import { beforeAll, describe, expect, it } from "vitest";
import { advance, createSimState, getOp, Graph, hasOp, registerElementOps, type GpuField } from "../index";
import { registerForceOps } from "./danceForces";
import { BODY_BLOCK_COUNT, seedSwarmBody } from "../../sim/body";

beforeAll(async () => {
  await registerElementOps(); // addFields
  registerForceOps();
});

function buildCeilidh(n: number) {
  const g = new Graph();
  const body0 = g.source(
    { shape: { kind: "points", n: n * BODY_BLOCK_COUNT }, dtype: "f32", element: { kind: "vec", n: 3 }, data: seedSwarmBody(n, 1) },
    "swarmSeed",
  );
  const fb = g.feedback(body0, "body");
  const taps = g.op("bodyTap", { body: fb.state });
  const pos = taps[0];
  const vel = taps[1];
  if (!pos || !vel) throw new Error("bodyTap did not yield pos/vel");
  const fC = g.op1("constrain", { pos });
  const fH = g.op1("cohere", { pos });
  const fS = g.op1("separate", { pos });
  const fO = g.op1("orbit", { pos, vel });
  const s1 = g.op1("addFields", { a: fC, b: fH });
  const s2 = g.op1("addFields", { a: s1, b: fS });
  const total = g.op1("addFields", { a: s2, b: fO });
  const integ = g.op("integrate", { body: fb.state, force: total });
  const bodyNext = integ[0];
  const swarm = integ[1];
  if (!bodyNext || !swarm) throw new Error("integrate did not yield body/swarm");
  fb.close(bodyNext);
  return { g, swarm };
}

describe("dance force building blocks", () => {
  it("register and expose the expected ports", () => {
    registerForceOps();
    expect(hasOp("constrain")).toBe(true);
    expect(getOp("bodyTap").outputs.map((o) => o.name)).toEqual(["pos", "vel"]);
    expect(getOp("integrate").outputs.map((o) => o.name)).toEqual(["body", "swarm"]);
  });

  it("bodyTap infers N agents from a 5N-row body field", () => {
    const [pos, vel] = getOp("bodyTap").inferShapes([{ kind: "points", n: 100 }], {});
    expect(pos).toEqual({ kind: "points", n: 20 });
    expect(vel).toEqual({ kind: "points", n: 20 });
  });

  it("rejects a non-vec3 body at build time", () => {
    const g = new Graph();
    const bad = g.source({ shape: { kind: "points", n: 25 }, dtype: "f32", data: new Float32Array(50) }, "bad");
    expect(() => g.op("bodyTap", { body: bad as GpuField })).toThrow(/vec3/);
  });

  it("advances the composed loop, evolving the swarm and staying bounded", async () => {
    const n = 48;
    const { g, swarm } = buildCeilidh(n);
    const store = createSimState();

    const f0 = await advance(g, swarm, { steps: 1, state: store, mode: "cpu" });
    const after = await advance(g, swarm, { steps: 60, state: store, mode: "cpu" });

    expect(after.shape).toEqual({ kind: "points", n }); // swarm = N xy pairs
    const data = after.data;
    if (!data) throw new Error("no swarm data");
    expect(data.length).toBe(n * 2);
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      expect(v !== undefined && Number.isFinite(v)).toBe(true);
      expect(Math.abs(v ?? 0)).toBeLessThanOrEqual(41); // maxRadius backstop
    }

    // motion happened between tick 1 and tick 61
    const d0 = f0.data;
    if (!d0) throw new Error("no first-frame data");
    let moved = 0;
    for (let i = 0; i < data.length; i++) if (Math.abs((data[i] ?? 0) - (d0[i] ?? 0)) > 1e-3) moved++;
    expect(moved).toBeGreaterThan(n);
  });

  it("drives figures via a graph-native clock and the caller (couples scramble)", async () => {
    const n = 32;
    const g = new Graph();
    const body0 = g.source(
      { shape: { kind: "points", n: n * BODY_BLOCK_COUNT }, dtype: "f32", element: { kind: "vec", n: 3 }, data: seedSwarmBody(n, 2) },
      "swarmSeed",
    );
    const fb = g.feedback(body0, "body");
    const taps = g.op("bodyTap", { body: fb.state });
    const pos = taps[0];
    const vel = taps[1];
    if (!pos || !vel) throw new Error("no taps");
    // clock loop
    const t0 = g.source({ shape: { kind: "scalar" }, dtype: "f32", data: new Float32Array([0]) }, "clockStart");
    const cfb = g.feedback(t0, "clock");
    const t = g.op1("clock", { prev: cfb.state }, { rate: 1 });
    cfb.close(t);
    // caller with a short period so a figure boundary is crossed within the run
    const force = g.op1("caller", { pos, vel, frame: t }, { period: 20, tightness: 1.2 });
    const integ = g.op("integrate", { body: fb.state, force });
    const bodyNext = integ[0];
    const swarm = integ[1];
    if (!bodyNext || !swarm) throw new Error("no integrate outputs");
    fb.close(bodyNext);

    const store = createSimState();
    // The clock advances each tick; pull it out to confirm.
    const clockAfter = await advance(g, t, { steps: 50, state: store, mode: "cpu" });
    expect(clockAfter.data?.[0]).toBeCloseTo(50, 6);

    const s = await advance(g, swarm, { steps: 1, state: store, mode: "cpu" });
    expect(s.data?.length).toBe(n * 2);
    expect(Array.from(s.data ?? []).every((v) => Number.isFinite(v))).toBe(true);
  });
});
