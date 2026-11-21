import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import {
  addSpring,
  chainScenario,
  createState,
  createTopology,
  DEAD,
  runSprings,
  type SpringParams,
  type SpringState,
  spawn,
  speciesTable,
  springGraph,
  springParams,
  termsWithoutOps,
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

  /** A loose ball of three species with a few unused slots and some bonds: every pair term has work. */
  function tissue(n: number) {
    const s = createState(n);
    let seed = 12345;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    for (let i = 0; i < n; i++) {
      if (i % 11 === 5) continue; // left dead
      const species = i % 7 === 0 ? 2 : i % 3 === 0 ? 1 : 0;
      spawn(s, i, (rnd() - 0.5) * 5, (rnd() - 0.5) * 5, (rnd() - 0.5) * 5, species);
      s.radius[i] = 0.4 + 0.2 * rnd();
    }
    // A little initial velocity, so the directional density is not zero.
    for (let k = 0; k < n * 4; k++) if (k % 4 !== 3) s.prev[k] = s.pos[k]! - (rnd() - 0.5) * 0.05;
    const t = createTopology(n, 6);
    for (let i = 0; i + 2 < n; i += 3) if (s.species[i] !== DEAD && s.species[i + 2] !== DEAD) addSpring(t, i, i + 2, 1.2, 0.5, 0, 1);
    return { s, t };
  }

  function pairParams(over: Partial<SpringParams> = {}): SpringParams {
    return bondedOnly({
      damptype: 4,
      damp: 0.7,
      pairRange: 3,
      densityRange: 1.5,
      pushapartlocalforce: 0.05,
      pairSpeciesMask: 0b011,
      repulsion: speciesTable([
        [0, 0, 0.3],
        [0, 1, 0.2],
        [1, 1, 0.1],
        [0, 2, 0.3],
        [2, 2, 0.3],
      ]),
      adhesion: speciesTable([
        [0, 0, 0.04],
        [0, 2, 0.2],
      ]),
      adhesionRange: 0.4,
      protrusion: [0.05, 0, 0, 0],
      thrust: [0, 0, 0.12, 0],
      protrusionRange: 2,
      protrusionFull: 6,
      tissueMask: 0b101,
      ...over,
    });
  }

  it("each pair op's kernel matches its CPU golden", async () => {
    const { s, t } = tissue(66);
    const { graph, terms, density } = springGraph(s, t, pairParams());
    const worstBy: Record<string, number> = {};
    const largest: Record<string, number> = {};
    for (const [name, field] of [...Object.entries(terms), ["pairDensity", density] as const]) {
      const gpu = await pullData(graph, field);
      const cpu = await pullData(graph, field, { mode: "cpu" });
      worstBy[name] = worst(gpu, cpu);
      largest[name] = Math.max(...Array.from(cpu, Math.abs));
    }
    expect(Object.keys(worstBy).sort()).toEqual(["adhesion", "outward", "pairDensity", "pushApart", "springForce"]);
    // Every term does something on this state, and no kernel strays from its golden.
    expect(Math.min(...Object.values(largest))).toBeGreaterThan(1e-3);
    expect(Math.max(...Object.values(worstBy)), JSON.stringify(worstBy)).toBeLessThan(1e-4);
  });

  /** Positions compared lane by lane: the largest xyz difference, and the largest relative
   *  difference in the density lane (densities run to hundreds, so absolute error says little). */
  function againstFused(got: ArrayLike<number>, want: ArrayLike<number>): { xyz: number; density: number } {
    let xyz = 0;
    let density = 0;
    for (let k = 0; k < want.length; k++) {
      const e = Math.abs(got[k]! - want[k]!);
      if (!Number.isFinite(e)) return { xyz: Number.POSITIVE_INFINITY, density: Number.POSITIVE_INFINITY };
      if (k % 4 === 3) density = Math.max(density, e / Math.max(1, Math.abs(want[k]!)));
      else xyz = Math.max(xyz, e);
    }
    return { xyz, density };
  }

  it("the graph with pair terms equals the fused step, on the CPU and on the device", async () => {
    // Eight steps: this state is a crowded tangle, and rounding differences grow by about half
    // again each step, so a long run would be testing chaos, not the ops.
    const steps = 8;
    const reference = tissue(66);
    const p = pairParams();
    const cpuRun = tissue(66);
    const cpuGraph = springGraph(cpuRun.s, cpuRun.t, p);
    const onCpu = await advance(cpuGraph.graph, cpuGraph.pos, { steps, state: createSimState(), mode: "cpu" });
    const gpuRun = tissue(66);
    const gpuGraph = springGraph(gpuRun.s, gpuRun.t, p);
    const downloads: string[] = [];
    const onGpu = await advance(gpuGraph.graph, gpuGraph.pos, {
      steps,
      state: createSimState(),
      materialise: "last",
      onBridge: (k, dir) => {
        if (dir === "download") downloads.push(k);
      },
    });
    runSprings(reference.s, reference.t, p, steps);
    const cpu = againstFused(onCpu.data!, reference.s.pos);
    const gpu = againstFused(onGpu.data!, reference.s.pos);
    expect(cpu.xyz, JSON.stringify(cpu)).toBeLessThan(2e-5);
    expect(cpu.density, JSON.stringify(cpu)).toBeLessThan(1e-3);
    expect(gpu.xyz, JSON.stringify(gpu)).toBeLessThan(5e-4);
    expect(gpu.density, JSON.stringify(gpu)).toBeLessThan(1e-2);
    expect(downloads.length).toBe(1);
    // Dead slots did not move, and the live ones did.
    expect(onGpu.data![5 * 4]).toBe(reference.s.pos[5 * 4]);
    expect(againstFused(onGpu.data!, tissue(66).s.pos).xyz).toBeGreaterThan(1e-2);
    // The density lane holds real densities, not the zeros it started with.
    expect(Math.max(...Array.from({ length: 66 }, (_, i) => onGpu.data![i * 4 + 3]!))).toBeGreaterThan(0.1);
  });

  it("refuses params that need a term with no op yet", () => {
    const { s, t } = tissue(22);
    expect(() => springGraph(s, t, pairParams({ noiseforce: 0.1, visc: 0.2 }))).toThrow(/noise.*|viscosity/);
    expect(termsWithoutOps(pairParams({ pushapartforce: 1 }), t)).toEqual(["global push-apart (pushapartforce)"]);
  });
});
