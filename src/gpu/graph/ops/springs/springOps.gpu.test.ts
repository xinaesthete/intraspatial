import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import {
  addRod,
  addSpring,
  CELL,
  chainScenario,
  chromatinScenario,
  createState,
  createTopology,
  DEAD,
  LEADER,
  readSpringState,
  runSprings,
  setFix,
  setPull,
  setTorsion,
  type SpringParams,
  type SpringState,
  spawn,
  speciesTable,
  spheroidScenario,
  springGraph,
  springParams,
  topologyDefect,
} from "#src/gpu/sim/springs/index";
import { nodeBackend } from "../../backend.node";
import { advance, createSimState, type GpuField, pullData } from "../../index";
import { springIntegrateOp } from "./springIntegrate";
import { tickOp } from "./tick";

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

  it("keeps a stationary particle's force under damping types 3 and 4", async () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0, 0);
    spawn(s, 1, 2, 0, 0, 0);
    s.prev.set(s.pos);
    const t = createTopology(2, 4);
    addSpring(t, 0, 1, 1, 1, 0, 0);
    for (const spec of [
      { damptype: 3 as const, damp: 0.6 },
      { damptype: 4 as const, damp: 0 },
    ]) {
      const { graph, pos } = springGraph(s, t, bondedOnly({ ...spec, springforce: 0.5, backboneStrength: 1 }));
      const gpu = await pullData(graph, pos);
      const cpu = await pullData(graph, pos, { mode: "cpu" });
      expect(worst(gpu, cpu), `damptype ${spec.damptype}`).toBeLessThan(2e-5);
      expect(Math.abs(gpu[0]! - s.pos[0]!), `damptype ${spec.damptype} moved`).toBeGreaterThan(1e-3);
    }
  });

  it("does not keep an output lease when a spring op rejects before dispatch", async () => {
    const live = () => nodeBackend.poolStats().live;
    const before = live();
    const shape = { kind: "points" as const, n: 1 };
    const ctx = { backend: nodeBackend };
    const inputs = [
      { shape, dtype: "f32" as const },
      { shape, dtype: "f32" as const },
      { shape, dtype: "f32" as const },
      { shape, dtype: "f32" as const },
      { shape, dtype: "u32" as const },
    ];
    await expect(springIntegrateOp.execute(ctx, inputs, { damp: Number.NaN })).rejects.toThrow(/damp/);
    await expect(springIntegrateOp.execute(ctx, inputs, { damp: 0.9 })).rejects.toThrow(/non-resident/);
    await expect(tickOp.execute(ctx, [{ shape: { kind: "scalar" }, dtype: "u32" }], {})).rejects.toThrow(/non-resident/);
    expect(live()).toBe(before);
  });

  it("returns the first integrate lease when the second lease fails", async () => {
    const before = nodeBackend.poolStats().live;
    const buf = await nodeBackend.lease(64);
    let calls = 0;
    const backend = {
      ...nodeBackend,
      lease(byteLength: number, usage?: number) {
        calls += 1;
        if (calls === 2) return Promise.reject(new Error("second lease failed"));
        return nodeBackend.lease(byteLength, usage);
      },
    };
    const n = 1;
    const input = (dtype: "f32" | "u32") => ({ shape: { kind: "points" as const, n }, dtype, buffer: buf });
    try {
      await expect(
        springIntegrateOp.execute({ backend }, [input("f32"), input("f32"), input("f32"), input("f32"), input("u32")], { damp: 0.9 }),
      ).rejects.toThrow(/second lease failed/);
      expect(nodeBackend.poolStats().live).toBe(before + 1);
    } finally {
      nodeBackend.release(buf);
    }
    expect(nodeBackend.poolStats().live).toBe(before);
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

  /** The tissue with every per-particle extra: rods, pinned and pulled particles, torsion springs. */
  function rigged(n: number) {
    const { s, t } = tissue(n);
    const live = (i: number) => s.species[i] !== DEAD;
    for (let i = 3; i + 9 < n; i += 9) if (live(i) && live(i + 4)) addRod(t, i, i + 4, 0.8, 1.6, 0.7);
    for (let i = 8; i < n; i += 17) if (live(i)) setFix(t, i, s.pos[i * 4]! + 0.1, s.pos[i * 4 + 1]!, s.pos[i * 4 + 2]! - 0.1);
    for (let i = 2; i < n; i += 13) if (live(i)) setPull(t, i, 0.5, -0.5, 0.25, 0.8);
    for (let i = 0; i + 3 < n; i += 5) setTorsion(t, i, 0.9, 0.6);
    return { s, t };
  }

  function everything(over: Partial<SpringParams> = {}): SpringParams {
    return pairParams({
      visc: 0.05,
      viscpow: 1.5,
      modelSphereForce: 0.3,
      modelSphereRadius: 2,
      mobility: [1, 0.4, 0.8, 1],
      noiseforce: 0.02,
      noiseprob: 0.6,
      motility: [0.03, 0, 0.05, 0],
      springCentreDamp: 0.98,
      springCentre: [0.2, -0.1, 0.3],
      pullspringforce: 0.05,
      pullspringpow: 0.5,
      pullfixdamp: 0.1,
      torsionspringforce: 0.5,
      backbonetorsionspringforce: 0.2,
      seed: 7,
      ...over,
    });
  }

  it("every remaining term's kernel matches its CPU golden", async () => {
    const { s, t } = rigged(66);
    const { graph, terms, force, pos } = springGraph(s, t, everything());
    const fields: Record<string, GpuField> = { ...terms, "force (after mobility and random terms)": force, "pos (after constraints)": pos };
    const worstBy: Record<string, number> = {};
    const largest: Record<string, number> = {};
    for (const [name, field] of Object.entries(fields)) {
      const gpu = await pullData(graph, field);
      const cpu = await pullData(graph, field, { mode: "cpu" });
      worstBy[name] = worst(gpu, cpu);
      largest[name] = Math.max(...Array.from(cpu, Math.abs));
    }
    expect(Object.keys(terms).sort()).toEqual(
      [
        "adhesion",
        "container",
        "motilityForce",
        "noiseForce",
        "outward",
        "pullForce",
        "pushApart",
        "springForce",
        "torsionForce",
        "viscosity",
      ].sort(),
    );
    expect(Math.min(...Object.values(largest)), JSON.stringify(largest)).toBeGreaterThan(1e-4);
    expect(Math.max(...Object.values(worstBy)), JSON.stringify(worstBy)).toBeLessThan(2e-4);
  });

  it("the random terms draw the same numbers on the device as on the CPU, and new ones each step", async () => {
    const { s, t } = rigged(66);
    const { graph, terms } = springGraph(s, t, everything());
    const state = createSimState();
    const first = await advance(graph, terms.noiseForce!, { steps: 1, state });
    const second = await advance(graph, terms.noiseForce!, { steps: 1, state });
    const cpuState = createSimState();
    const firstCpu = await advance(graph, terms.noiseForce!, { steps: 1, state: cpuState, mode: "cpu" });
    const secondCpu = await advance(graph, terms.noiseForce!, { steps: 1, state: cpuState, mode: "cpu" });
    expect(worst(first.data!, firstCpu.data!)).toBeLessThan(1e-6);
    expect(worst(second.data!, secondCpu.data!)).toBeLessThan(1e-6);
    expect(worst(first.data!, second.data!)).toBeGreaterThan(1e-3);
  });

  it("the graph with every term but CSynth's own pair forces equals the fused step", async () => {
    const steps = 6;
    const p = everything();
    const reference = rigged(66);
    const cpuRun = rigged(66);
    const cpuGraph = springGraph(cpuRun.s, cpuRun.t, p);
    const onCpu = await advance(cpuGraph.graph, cpuGraph.pos, { steps, state: createSimState(), mode: "cpu" });
    const gpuRun = rigged(66);
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
    expect(gpu.xyz, JSON.stringify(gpu)).toBeLessThan(5e-4);
    expect(gpu.density, JSON.stringify(gpu)).toBeLessThan(1e-2);
    expect(downloads.length).toBe(1);
    // A pinned particle is exactly where it was pinned.
    expect([onGpu.data![8 * 4], onGpu.data![8 * 4 + 1], onGpu.data![8 * 4 + 2]]).toEqual(Array.from(reference.s.pos.subarray(32, 35)));
  });

  /** A contact map and a distance map over the first `size` particles, with a few pairs marked
   *  missing (−10) and excluded (−2000) so the gate has every case to handle. */
  function maps(s: SpringState, size: number) {
    const contact = new Float32Array(size * size);
    const dist = new Float32Array(size * size);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const gap = Math.abs(i - j);
        let c = gap === 0 ? 0 : 1 / (1 + gap * 0.5);
        if ((i + j) % 9 === 4) c = -10;
        if ((i + j) % 17 === 6) c = -2000;
        contact[i * size + j] = c;
        dist[i * size + j] = gap === 0 ? 0 : 0.6 + 0.25 * gap;
      }
    }
    return { size, contact, dist };
  }

  function csynth(over: Partial<SpringParams> = {}): SpringParams {
    return everything({
      pairRange: undefined,
      densityRange: Number.POSITIVE_INFINITY,
      pairSpeciesMask: 0b111,
      powBaseDist: 2,
      springpow: 0.3,
      maxBackboneDist: 0.8,
      pushapartforce: 0.004,
      pushapartpow: -1.5,
      pushapartDensityFactor: 0.0002,
      pushapartDensityPow: -1,
      fractforce: 0.0005,
      fractpow: -1,
      backboneforce: 0.05,
      backboneScale: 0.9,
      xyzforce: 0.02,
      xyzpow: 0.5,
      xyzMaxDist: 4,
      contactforcesc: 0.03,
      contactthreshold: 0.05,
      backboneContactStrength: 1.4,
      wrongfade: 0.5,
      boidalignment: 0.02,
      boidcohesion: 0.01,
      boidseparation: 0.003,
      boidseparationpow: -1,
      boidrad: 1.2,
      m_force: 0.4,
      m_k: 1.5,
      m_alpha: 0.5,
      m_c: 0.8,
      representativeContact: 0.7,
      maxv: 0.9,
      patchval: 0.5,
      patchwidth: 3,
      ...over,
    });
  }

  const CSYNTH_TERMS = [
    "backboneForce",
    "boids",
    "contactForce",
    "distanceForce",
    "fractalForce",
    "lorentzianContact",
    "pushApartDensity",
    "pushApartGlobal",
  ];

  it("each of CSynth's pair ops matches its CPU golden, with a contact map that gates pairs", async () => {
    const { s, t } = rigged(66);
    // One step first on the host, so the density lane the density term reads is not zero.
    const warm = csynth();
    const m = maps(s, 50);
    runSprings(s, t, warm, 1, m);
    const { graph, terms } = springGraph(s, t, warm, m);
    const worstBy: Record<string, number> = {};
    const largest: Record<string, number> = {};
    for (const name of CSYNTH_TERMS) {
      const gpu = await pullData(graph, terms[name]!);
      const cpu = await pullData(graph, terms[name]!, { mode: "cpu" });
      worstBy[name] = worst(gpu, cpu) / Math.max(1, ...Array.from(cpu, Math.abs));
      largest[name] = Math.max(...Array.from(cpu, Math.abs));
    }
    expect(Math.min(...Object.values(largest)), JSON.stringify(largest)).toBeGreaterThan(1e-6);
    expect(Math.max(...Object.values(worstBy)), JSON.stringify(worstBy)).toBeLessThan(2e-4);
  });

  it("the graph with every term of the model equals the fused step", async () => {
    const steps = 5;
    const p = csynth();
    const reference = rigged(66);
    const m = maps(reference.s, 50);
    const cpuRun = rigged(66);
    const cpuGraph = springGraph(cpuRun.s, cpuRun.t, p, m);
    const onCpu = await advance(cpuGraph.graph, cpuGraph.pos, { steps, state: createSimState(), mode: "cpu" });
    const gpuRun = rigged(66);
    const gpuGraph = springGraph(gpuRun.s, gpuRun.t, p, m);
    const onGpu = await advance(gpuGraph.graph, gpuGraph.pos, { steps, state: createSimState(), materialise: "last" });
    runSprings(reference.s, reference.t, p, steps, m);
    expect(Object.keys(gpuGraph.terms).length).toBe(18);
    const cpu = againstFused(onCpu.data!, reference.s.pos);
    const gpu = againstFused(onGpu.data!, reference.s.pos);
    expect(cpu.xyz, JSON.stringify(cpu)).toBeLessThan(2e-5);
    expect(gpu.xyz, JSON.stringify(gpu)).toBeLessThan(5e-4);
  });

  it("the chromatin scenarios run as graphs and match the fused step", async () => {
    for (const mode of ["distances", "contacts"] as const) {
      const reference = chromatinScenario({ n: 40, mode, seed: 3 });
      const run = chromatinScenario({ n: 40, mode, seed: 3 });
      const g = springGraph(run.state, run.topology, run.params, run.matrices);
      const onGpu = await advance(g.graph, g.pos, { steps: 12, state: createSimState(), materialise: "last" });
      runSprings(reference.state, reference.topology, reference.params, 12, reference.matrices);
      const gpu = againstFused(onGpu.data!, reference.state.pos);
      expect(gpu.xyz, `${mode} ${JSON.stringify(gpu)} ${Object.keys(g.terms).join(",")}`).toBeLessThan(5e-4);
      expect(againstFused(onGpu.data!, chromatinScenario({ n: 40, mode, seed: 3 }).state.pos).xyz).toBeGreaterThan(1e-3);
    }
  });

  it("the spheroid's forces are all ops: its graph matches the fused step in 2D and 3D", async () => {
    for (const dims of [2, 3] as const) {
      const cfg = { dims, cellRadius: 2, matrixRadius: 4, leaderThrust: 0.4, protrusion: 0.05, seed: 5 };
      const reference = spheroidScenario(cfg);
      const run = spheroidScenario(cfg);
      // A leader, so thrust and leader adhesion have something to act on.
      for (const sc of [reference, run]) sc.state.species[sc.state.species.indexOf(CELL)] = LEADER;
      const g = springGraph(run.state, run.topology, run.params);
      const onGpu = await advance(g.graph, g.pos, { steps: 10, state: createSimState(), materialise: "last" });
      runSprings(reference.state, reference.topology, reference.params, 10);
      const gpu = againstFused(onGpu.data!, reference.state.pos);
      expect(gpu.xyz, `${dims}D ${JSON.stringify(gpu)} ${Object.keys(g.terms).join(",")}`).toBeLessThan(5e-4);
    }
  });

  it("the interpreted pair kernel matches its golden, and the separate pair ops it stands in for", async () => {
    const p = pairParams();
    const separate = tissue(66);
    const fusedRun = tissue(66);
    const a = springGraph(separate.s, separate.t, p);
    const b = springGraph(fusedRun.s, fusedRun.t, p, undefined, {}, { fusePairs: true });
    expect(Object.keys(b.terms).sort()).toEqual(["pairInterpreter", "springForce"]);
    expect(b.packed).toBe(true);
    // Its own golden.
    const gpu = await pullData(b.graph, b.terms.pairInterpreter!);
    const cpu = await pullData(b.graph, b.terms.pairInterpreter!, { mode: "cpu" });
    expect(worst(gpu, cpu)).toBeLessThan(1e-4);
    // The three ops it replaces, summed. One pass adds a pair's terms before scaling by the
    // direction, three passes scale each first, so the last bits differ.
    const parts = await Promise.all(["pushApart", "adhesion", "outward"].map((name) => pullData(a.graph, a.terms[name]!)));
    const summed = parts[0]!.map((v, k) => v + parts[1]![k]! + parts[2]![k]!);
    expect(worst(gpu, summed)).toBeLessThan(1e-4);
    expect(Math.max(...Array.from(summed, Math.abs))).toBeGreaterThan(1e-2);
    // And the density it computes on the way.
    expect(
      worst(await pullData(b.graph, b.density), await pullData(a.graph, a.density)) / Math.max(...(await pullData(a.graph, a.density))),
    ).toBeLessThan(1e-5);
  });

  it("a graph with its pair terms fused equals the fused step, and never leaves the device", async () => {
    const steps = 8;
    const p = pairParams();
    const reference = tissue(66);
    const run = tissue(66);
    const g = springGraph(run.s, run.t, p, undefined, {}, { fusePairs: true });
    const state = createSimState();
    await advance(g.graph, g.pos, { steps: 1, state, materialise: "none" });
    const bridges: string[] = [];
    const out = await advance(g.graph, g.pos, {
      steps: steps - 1,
      state,
      materialise: "last",
      onBridge: (k, dir) => bridges.push(`${dir} ${k}`),
    });
    runSprings(reference.s, reference.t, p, steps);
    const got = againstFused(out.data!, reference.s.pos);
    expect(got.xyz, JSON.stringify(got)).toBeLessThan(5e-4);
    expect(got.density, JSON.stringify(got)).toBeLessThan(1e-2);
    // One download, of the sink at the end; no uploads once the first tick has run.
    expect(bridges.length).toBe(1);
  });

  it("the spheroid's whole tick runs with its pair terms fused", async () => {
    const cfg = { dims: 3 as const, cellRadius: 2, matrixRadius: 4, leaderThrust: 0.4, protrusion: 0.05, divisionTime: 25, seed: 5 };
    const counts = async (fusePairs: boolean) => {
      const sc = spheroidScenario(cfg);
      const start = sc.state.species.filter((v) => v === CELL).length;
      const g = springGraph(sc.state, sc.topology, sc.params, undefined, sc.graphRules?.(), { fusePairs });
      const state = createSimState();
      await advance(g.graph, g.pos, { steps: 40, state, materialise: "none" });
      await readSpringState(state, sc.state, sc.topology);
      return {
        start,
        tissue: sc.state.species.filter((v) => v === CELL || v === LEADER).length,
        defect: topologyDefect(sc.topology),
        terms: Object.keys(g.terms),
      };
    };
    const separate = await counts(false);
    const fused = await counts(true);
    expect(fused.terms).toContain("pairInterpreter");
    expect(fused.terms).not.toContain("adhesion");
    expect(fused.defect).toBeUndefined();
    expect(fused.tissue).toBeGreaterThan(fused.start + 3);
    expect(Math.abs(fused.tissue - separate.tissue)).toBeLessThanOrEqual(Math.ceil(separate.tissue * 0.1));
  });
});
