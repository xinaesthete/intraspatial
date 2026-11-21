import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import {
  addSpring,
  chainScenario,
  createState,
  createTopology,
  runSprings,
  type SpringParams,
  type SpringState,
  spawn,
  springGraph,
  springParams,
} from "#src/gpu/sim/springs/index";
import { advance, createSimState, pullData } from "../../index";

/** Bonded springs and the integrator only: every other term of the fused step switched off. */
function bondedOnly(over: Partial<SpringParams> = {}): SpringParams {
  return springParams({
    damp: 0.8,
    damptype: 1,
    springforce: 0.3,
    springmaxvel: 0.5,
    pushapartlocalforce: 0,
    backboneStrength: 1.5,
    noiseforce: 0,
    ...over,
  });
}

/** Largest absolute difference between two arrays, or Infinity when either holds a non-finite. */
function worst(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let err = 0;
  for (let k = 0; k < a.length; k++) {
    const e = Math.abs(a[k]! - b[k]!);
    if (!Number.isFinite(e)) return Number.POSITIVE_INFINITY;
    if (e > err) err = e;
  }
  return err;
}

/** A tangled chain with a few long-range bonds of other kinds, so every branch of the force runs. */
function tangle(n: number): { s: SpringState; t: ReturnType<typeof createTopology> } {
  const sc = chainScenario({ n, seed: 4 });
  const t = sc.topology;
  addSpring(t, 2, 20, 1.5, 0.7, 0, 1);
  addSpring(t, 5, 31, 0.8, 1.2, 1, 2);
  addSpring(t, 9, 40, 2.0, 0.4, -0.5, 3);
  return { s: sc.state, t };
}

describe("spring model as graph ops", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("springForce's kernel matches its CPU golden", async () => {
    const { s, t } = tangle(48);
    const { graph, force } = springGraph(s, t, bondedOnly({ roleforces: [1, 0.5, 2, 1.5, 1, 1, 1, 1] }));
    const gpu = await pullData(graph, force);
    const cpu = await pullData(graph, force, { mode: "cpu" });
    expect(gpu.length).toBe(48 * 4);
    expect(worst(gpu, cpu)).toBeLessThan(2e-5);
    expect(Math.max(...Array.from(cpu, Math.abs))).toBeGreaterThan(1e-3);
  });

  it("the graph on the CPU equals the fused step, term for term", async () => {
    for (const damptype of [1, 2, 3, 4] as const) {
      const { s, t } = tangle(48);
      const p = bondedOnly({ damptype, damp: damptype === 1 ? 0.8 : 0.6 });
      const { graph, pos } = springGraph(s, t, p);
      const out = await advance(graph, pos, { steps: 25, state: createSimState(), mode: "cpu" });
      runSprings(s, t, p, 25);
      // Not exact: the fused step keeps the force in double precision between its two halves.
      expect(worst(out.data!, s.pos), `damptype ${damptype}`).toBeLessThan(2e-5);
    }
  });

  it("the graph on the device equals the fused step, with one download for the whole run", async () => {
    const { s, t } = tangle(48);
    const p = bondedOnly();
    const { graph, pos } = springGraph(s, t, p);
    const downloads: string[] = [];
    const state = createSimState();
    const note = (k: string, dir: string) => {
      if (dir === "download") downloads.push(k);
    };
    const out = await advance(graph, pos, { steps: 25, state, materialise: "last", onBridge: note });
    runSprings(s, t, p, 25);
    expect(worst(out.data!, s.pos)).toBeLessThan(2e-4);
    expect(downloads.length).toBe(1);

    // And none at all for a view that binds the buffer: the state's own, valid until the next tick.
    downloads.length = 0;
    const resident = await advance(graph, pos, { steps: 10, state, materialise: "none", onBridge: note });
    expect(downloads).toEqual([]);
    expect(resident.buffer?.byteLength).toBe(48 * 16);
    const after = await advance(graph, pos, { steps: 1, state, materialise: "last" });
    runSprings(s, t, p, 11);
    expect(worst(after.data!, s.pos)).toBeLessThan(4e-4);
  });

  it("holds a two-particle bond at its rest length", async () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0, 0);
    spawn(s, 1, 3, 0, 0, 0);
    const t = createTopology(2, 4);
    addSpring(t, 0, 1, 1, 1, 0, 0);
    const { graph, pos } = springGraph(s, t, bondedOnly({ damp: 0.5, springforce: 0.2, backboneStrength: 1 }));
    const out = await advance(graph, pos, { steps: 200, state: createSimState() });
    const d = out.data!;
    expect(Math.hypot(d[4]! - d[0]!, d[5]! - d[1]!, d[6]! - d[2]!)).toBeCloseTo(1, 3);
  });
});
