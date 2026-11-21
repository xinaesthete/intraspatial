import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import {
  addSpring,
  bondCount,
  type BreakRule,
  breakBonds,
  CELL,
  chainScenario,
  chromatinScenario,
  createSpringSim,
  createState,
  createTopology,
  DEAD,
  distanceCorrelation,
  distanceMatrix,
  divideCells,
  type FormRule,
  formBonds,
  LEADER,
  MAX_SPECIES,
  readSpringState,
  relaxationScenario,
  selectLeaders,
  SPRING_STATE,
  spheroidGraph,
  spheroidScenario,
  springGraph,
  type SpringState,
  type SpringTopology,
  spawn,
  springParams,
  topologyDefect,
} from "#src/gpu/sim/springs/index";
import { nodeBackend } from "../../backend.node";
import {
  advance,
  createSimState,
  disposeSimState,
  type FieldValue,
  Graph,
  type GpuField,
  hasOp,
  payloadsNotHeld,
  pull,
  registerOp,
  simStateValue,
} from "../../index";
import { extractOp, packOp } from "../bundleOps";
import { bondTableOf, VEC4 } from "./bondTable";
import { pairInterpreterOp } from "./pairInterpreter";

/** A jittered 2D lattice of two species, bonded to near neighbours, with a few slots left dead. */
function sheet(cols: number, rows: number, K = 8): { s: SpringState; t: SpringTopology } {
  const n = cols * rows + 6;
  const s = createState(n);
  let seed = 99;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      spawn(s, i, c + (r & 1) * 0.5 + (rnd() - 0.5) * 0.3, r * 0.87 + (rnd() - 0.5) * 0.3, 0, (c + r) % 3 === 0 ? 0 : 1);
    }
  }
  const t = createTopology(n, K);
  for (let i = 0; i < cols * rows; i++) {
    for (const j of [i + 1, i + cols]) {
      if (j < cols * rows && (j !== i + 1 || (i + 1) % cols !== 0) && rnd() < 0.7) addSpring(t, i, j, 0.9 + rnd() * 0.2, 1, 0, 1);
    }
  }
  return { s, t };
}

function sources(g: Graph, s: SpringState, t: SpringTopology, step: number) {
  const points = { kind: "points", n: s.n } as const;
  return {
    pos: g.source({ shape: points, dtype: "f32", element: VEC4, data: s.pos.slice() }, "pos"),
    species: g.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species"),
    bonds: g.source(bondTableOf(t), "bonds"),
    step: g.source({ shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(step) }, "step"),
  };
}

const links = (v: FieldValue) => Array.from(v.parts!.link!.data!);
const springs = (v: FieldValue) => v.parts!.spring!.data!;

/** Both ways of running an op, and the first index where their link tables differ (-1: none). */
async function bothWays(g: Graph, field: GpuField) {
  const gpu = await pull(g, field);
  const cpu = await pull(g, field, { mode: "cpu" });
  const a = links(gpu);
  const b = links(cpu);
  // A table of the wrong length differs where the shorter one ends.
  const firstDifference = a.length === b.length ? a.findIndex((v, k) => v !== b[k]) : Math.min(a.length, b.length);
  return { gpu, cpu, firstDifference };
}

describe("spring model rules as graph ops", () => {
  beforeAll(async () => {
    await getDevice();
  });

  it("bondBreak breaks exactly the bonds the host rule breaks, on the device and in its golden", async () => {
    const rules: BreakRule[] = [
      { kinds: 0b10, maxStrain: 0.08 },
      { kinds: 0b10, maxStrain: 0.03, tensionOnly: true },
      { kinds: 0b10, tau: 4 },
      { kinds: 0b10, maxStrain: 0.1, tau: 9 },
    ];
    for (const rule of rules) {
      const { s, t } = sheet(9, 7);
      s.species[10] = DEAD; // its bonds must go whatever the rule says
      s.step = 37;
      const p = springParams({ seed: 5 });
      const g = new Graph();
      const out = g.op1("bondBreak", sources(g, s, t, s.step), {
        kinds: rule.kinds,
        maxStrain: rule.maxStrain ?? Number.POSITIVE_INFINITY,
        tensionOnly: rule.tensionOnly ?? false,
        tau: rule.tau ?? Number.POSITIVE_INFINITY,
        seed: p.seed,
      });
      const { gpu, cpu, firstDifference } = await bothWays(g, out);
      const before = links(bondTableOf(t));
      const broken = breakBonds(s, t, p, rule);
      expect(firstDifference, JSON.stringify(rule)).toBe(-1);
      expect(links(cpu)).toEqual(links(bondTableOf(t)));
      expect(broken, JSON.stringify(rule)).toBeGreaterThan(2);
      expect(links(gpu).filter((v, k) => v !== before[k]).length).toBe(broken * 2);
      expect(topologyDefect(t)).toBeUndefined();
    }
  });

  it("bondForm forms exactly the bonds the host rule forms, on the device and in its golden", async () => {
    for (const prob of [1, 0.4]) {
      const { s, t } = sheet(9, 7);
      s.step = 12;
      const p = springParams({ seed: 3, dims: 2 });
      const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => -1);
      kindOf[1 * MAX_SPECIES + 1] = 1;
      kindOf[0 * MAX_SPECIES + 1] = 2;
      kindOf[1 * MAX_SPECIES + 0] = 2;
      const rule: FormRule = {
        range: 1.25,
        kindOf,
        strength: [1, 0.5, 2, 1, 1, 1, 1, 1],
        rest: "current",
        prob,
        speciesCount: MAX_SPECIES,
      };
      const g = new Graph();
      const src = sources(g, s, t, s.step);
      const out = g.op1(
        "bondForm",
        {
          pos: src.pos,
          species: src.species,
          bonds: src.bonds,
          kindOf: g.source(
            { shape: { kind: "matrix", rows: MAX_SPECIES, cols: MAX_SPECIES }, dtype: "f32", data: Float32Array.from(kindOf) },
            "kindOf",
          ),
          step: src.step,
        },
        { range: rule.range, prob, seed: p.seed, strength1: 0.5, strength2: 2 },
      );
      const { gpu, cpu, firstDifference } = await bothWays(g, out);
      const formed = formBonds(s, t, p, rule);
      const host = bondTableOf(t);
      expect(firstDifference, `prob ${prob}`).toBe(-1);
      expect(links(cpu)).toEqual(links(host));
      expect(formed, `prob ${prob}`).toBeGreaterThan(3);
      // Rest lengths and strengths of the new bonds too.
      let worst = 0;
      const want = springs(host);
      const got = springs(gpu);
      for (let k = 0; k < want.length; k++) worst = Math.max(worst, Math.abs(got[k]! - want[k]!));
      expect(worst).toBeLessThan(1e-5);
      expect(topologyDefect(t)).toBeUndefined();
    }
  });

  /** A clump of cells with free slots scattered among them, and a density in the spare lane. */
  function clump(n: number, free: number): SpringState {
    const s = createState(n);
    let seed = 4242;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const every = Math.max(2, Math.floor(n / free));
    let left = free;
    for (let i = 0; i < n; i++) {
      if (i % every === 1 && left-- > 0) continue;
      spawn(s, i, (rnd() - 0.5) * 6, (rnd() - 0.5) * 6, (rnd() - 0.5) * 6, i % 5 === 0 ? 1 : 0, 0.4 + 0.2 * rnd());
      s.pos[i * 4 + 3] = rnd() * 10;
      s.prev.set([s.pos[i * 4]! - 0.01, s.pos[i * 4 + 1]!, s.pos[i * 4 + 2]!, s.pos[i * 4 + 3]!], i * 4);
    }
    return s;
  }

  const u32Of = (v: FieldValue) => Array.from(v.data!);
  const worstOf = (a: ArrayLike<number>, b: ArrayLike<number>) => {
    let w = a.length === b.length ? 0 : Number.POSITIVE_INFINITY;
    for (let k = 0; k < a.length; k++) w = Math.max(w, Math.abs(a[k]! - b[k]!));
    return w;
  };

  it("division on the device equals the host rule: who divides, where the daughters go, what they inherit", async () => {
    // Plenty of room; fewer free slots than takers; and no room at all.
    for (const [tau, free, dims] of [
      [3, 30, 3],
      [1, 7, 2],
      [1, 0, 3],
    ] as const) {
      const s = clump(120, free);
      s.step = 21;
      const p = springParams({ seed: 11, dims });
      const rule = { species: 0, tau, maxDensity: 6, offset: 0.5 };
      const g = new Graph();
      const points = { kind: "points", n: s.n } as const;
      const pos = g.source({ shape: points, dtype: "f32", element: VEC4, data: s.pos.slice() }, "pos");
      const prev = g.source({ shape: points, dtype: "f32", element: VEC4, data: s.prev.slice() }, "prev");
      const species = g.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species");
      const radius = g.source({ shape: points, dtype: "f32", data: s.radius.slice() }, "radius");
      const step = g.source({ shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(s.step) }, "step");
      const wants = g.op1("divisionDraw", { pos, species, step }, { species: 0, tau, maxDensity: 6, seed: p.seed });
      const plan = g.op1("divisionPlan", { wants, species });
      const moved = g.op1("divideMove", { pos, plan, step }, { offset: 0.5, dims, seed: p.seed });
      const rested = g.op1("restAfterDivision", { prev, pos: moved, plan });
      const newSpecies = g.op1("inheritField", { field: species, plan });
      const newRadius = g.op1("inheritField", { field: radius, plan });

      const liveBefore = Array.from(s.species).filter((v) => v !== DEAD).length;
      const born = divideCells(s, p, rule);
      const label = `tau ${tau}, ${free} free, ${dims}D`;
      for (const mode of ["gpu", "cpu"] as const) {
        expect(u32Of(await pull(g, newSpecies, { mode })), `${label} ${mode} species`).toEqual(Array.from(s.species));
        expect(worstOf((await pull(g, moved, { mode })).data!, s.pos), `${label} ${mode} pos`).toBeLessThan(2e-5);
        expect(worstOf((await pull(g, rested, { mode })).data!, s.prev), `${label} ${mode} prev`).toBeLessThan(2e-5);
        expect(worstOf((await pull(g, newRadius, { mode })).data!, s.radius), `${label} ${mode} radius`).toBe(0);
      }
      expect(born, label).toBe(free === 0 ? 0 : Math.min(free, born));
      if (free > 0) expect(born, label).toBeGreaterThan(2);
      expect(Array.from(s.species).filter((v) => v !== DEAD).length).toBe(liveBefore + born);
    }
  });

  it("leader selection on the device equals the host rule, and waits for its round", async () => {
    for (const step of [40, 50, 60]) {
      const s = clump(140, 12);
      // Some existing leaders: a few exposed, a few buried.
      for (const i of [0, 30, 62, 95, 131]) if (s.species[i] !== DEAD) s.species[i] = 2;
      s.step = step;
      const p = springParams({ seed: 9, dims: 3 });
      const rule = { from: 0, to: 2, tau: 2, maxNeighbours: 5, spacing: 1.6, revertNeighbours: 3 };
      const g = new Graph();
      const points = { kind: "points", n: s.n } as const;
      const pos = g.source({ shape: points, dtype: "f32", element: VEC4, data: s.pos.slice() }, "pos");
      const species = g.source({ shape: points, dtype: "u32", data: Uint32Array.from(s.species) }, "species");
      const stepField = g.source({ shape: { kind: "scalar" }, dtype: "u32", data: Uint32Array.of(step) }, "step");
      const status = g.op1("leaderStatus", { pos, species, step: stepField }, { ...rule, every: 20, seed: p.seed });
      const out = g.op1("leaderApply", { pos, species, status }, rule);
      const before = Array.from(s.species);
      // Rounds are every 20 steps here: 40 and 60 are due, 50 is not.
      const changed = step % 20 === 0 ? selectLeaders(s, p, rule) : 0;
      expect(u32Of(await pull(g, out)), `step ${step} gpu`).toEqual(Array.from(s.species));
      expect(u32Of(await pull(g, out, { mode: "cpu" })), `step ${step} cpu`).toEqual(Array.from(s.species));
      expect(Array.from(s.species).filter((v, k) => v !== before[k]).length).toBe(changed);
      if (step % 20 === 0) expect(changed, `step ${step}`).toBeGreaterThan(2);
    }
  });

  const SMALL = {
    cellRadius: 2,
    matrixRadius: 5,
    yieldStrain: 0.12,
    bondLifetime: 150,
    reformProb: 0.05,
    divisionTime: 25,
    leaderThrust: 0.4,
    leaderTime: 30,
    ligand: 0.02,
    seed: 4,
  };
  const count = (s: SpringState, species: number) => Array.from(s.species).filter((v) => v === species).length;

  it("the spheroid's whole tick as a graph makes the host scenario's decisions exactly, in 2D and 3D", async () => {
    for (const dims of [2, 3] as const) {
      const cfg = { ...SMALL, dims, matrixRadius: dims === 2 ? 5 : 4 };
      const steps = 30;
      const host = spheroidScenario(cfg);
      const cellsAtStart = count(host.state, CELL);
      const bondsAtStart = bondCount(host.topology);
      host.advance(steps);

      const run = spheroidScenario(cfg);
      const g = spheroidGraph(run);
      const state = createSimState();
      await advance(g.graph, g.pos, { steps, state, mode: "cpu" });
      await readSpringState(state, run.state, run.topology);
      const label = `${dims}D`;
      expect(run.state.step, label).toBe(host.state.step);
      expect(Array.from(run.state.species), `${label} species`).toEqual(Array.from(host.state.species));
      expect(Array.from(bondTableOf(run.topology).parts!.link!.data!), `${label} bonds`).toEqual(
        Array.from(bondTableOf(host.topology).parts!.link!.data!),
      );
      expect(
        worstOf(
          run.state.pos.filter((_, k) => k % 4 !== 3),
          host.state.pos.filter((_, k) => k % 4 !== 3),
        ),
        `${label} positions`,
      ).toBeLessThan(1e-3);
      // And the run did something: cells divided, leaders appeared, bonds turned over.
      expect(count(host.state, CELL) + count(host.state, LEADER), label).toBeGreaterThan(cellsAtStart + 3);
      expect(host.totals.broken, label).toBeGreaterThan(3);
      expect(bondCount(host.topology), label).not.toBe(bondsAtStart);
      expect(topologyDefect(run.topology)).toBeUndefined();
    }
  });

  it("the same tick on the device never brings state to the host, and stays close to the host scenario", async () => {
    const cfg = { ...SMALL, dims: 2 as const };
    const steps = 60;
    const host = spheroidScenario(cfg);
    host.advance(steps);

    const run = spheroidScenario(cfg);
    const g = spheroidGraph(run);
    const state = createSimState();
    const downloads: string[] = [];
    await advance(g.graph, g.pos, {
      steps,
      state,
      materialise: "none",
      onBridge: (k, dir) => {
        if (dir === "download") downloads.push(k);
      },
    });
    expect(downloads).toEqual([]);
    await readSpringState(state, run.state, run.topology);
    expect(run.state.step).toBe(steps);
    expect(topologyDefect(run.topology)).toBeUndefined();
    // Single precision can tip a bond over its yield threshold a step early or late, after which
    // the two runs are different runs of the same model. So: the same amounts, not the same bits.
    const tissue = (s: SpringState) => count(s, CELL) + count(s, LEADER);
    expect(Math.abs(tissue(run.state) - tissue(host.state))).toBeLessThanOrEqual(Math.ceil(tissue(host.state) * 0.1));
    expect(Math.abs(bondCount(run.topology) - bondCount(host.topology))).toBeLessThanOrEqual(Math.ceil(bondCount(host.topology) * 0.05));
    expect(tissue(run.state)).toBeGreaterThan(tissue(spheroidScenario(cfg).state) + 5);
  });

  it("a spring sim runs a scenario on the device: one upload of each input, no downloads, buffers for a view", async () => {
    const sc = spheroidScenario({ ...SMALL, dims: 2 });
    const bridges: string[] = [];
    const sim = createSpringSim(sc, { onBridge: (k, dir) => bridges.push(`${dir} ${k}`) });
    expect(sim.buffers()).toBeUndefined();
    await sim.advance(1);
    const firstTick = bridges.length;
    expect(bridges.every((b) => b.startsWith("upload"))).toBe(true);
    await sim.advance(40);
    // Every input was uploaded for the first tick and has stayed on the device since.
    expect(bridges.length).toBe(firstTick);
    const buffers = sim.buffers();
    expect(buffers?.positions.size).toBe(sc.state.n * 16);
    expect(buffers?.link.size).toBe(sc.state.n * sc.topology.K * 4);
    // The host state is stale until asked for.
    expect(sc.state.step).toBe(0);
    const cellsBefore = count(sc.state, CELL);
    await sim.sync();
    expect(sc.state.step).toBe(41);
    expect(count(sc.state, CELL) + count(sc.state, LEADER)).toBeGreaterThan(cellsBefore);
    expect(topologyDefect(sc.topology)).toBeUndefined();
    expect(sim.builds).toBe(1);
    await sim.dispose();
  });

  it("a change of settings rebuilds the graph from the running state and carries on", async () => {
    const sc = spheroidScenario({ ...SMALL, dims: 2, leaderTime: 10 });
    const sim = createSpringSim(sc);
    await sim.advance(60);
    await sim.sync();
    const leaders = count(sc.state, LEADER);
    const tissue = count(sc.state, CELL) + leaders;
    expect(leaders).toBeGreaterThan(0);
    // Switch leaders off and division off: the leaders turn back, and the tissue stops growing.
    sc.configure({ leaderThrust: 0, divisionTime: Number.POSITIVE_INFINITY });
    await sim.advance(30);
    await sim.sync();
    expect(sim.builds).toBe(2);
    expect(sc.state.step).toBe(90);
    expect(count(sc.state, LEADER)).toBe(0);
    expect(count(sc.state, CELL)).toBe(tissue);
    // Nothing changed: no rebuild.
    await sim.advance(5);
    expect(sim.builds).toBe(2);
    await sim.dispose();
  });

  it("a spring sim runs the scenarios that have no rules, and the one with turnover only", async () => {
    const chromatin = chromatinScenario({ n: 60, mode: "distances", seed: 2 });
    const size = chromatin.matrices!.size;
    const fit = () => distanceCorrelation(distanceMatrix(chromatin.state, size), chromatin.matrices!.dist!, size);
    const before = fit();
    const chromatinSim = createSpringSim(chromatin);
    await chromatinSim.advance(300);
    await chromatinSim.sync();
    expect(chromatin.state.step).toBe(300);
    expect(fit()).toBeGreaterThan(Math.max(0.9, before + 0.2));
    await chromatinSim.dispose();

    const strip = relaxationScenario({ seed: 3, bondLifetime: 40 });
    const host = relaxationScenario({ seed: 3, bondLifetime: 40 });
    const stripSim = createSpringSim(strip);
    await stripSim.advance(80);
    await stripSim.sync();
    host.advance(80);
    expect(topologyDefect(strip.topology)).toBeUndefined();
    expect(host.totals.broken).toBeGreaterThan(20);
    // Bonds break at random with the same hash on both sides, so the tables should agree closely.
    expect(Math.abs(bondCount(strip.topology) - bondCount(host.topology))).toBeLessThanOrEqual(Math.ceil(bondCount(host.topology) * 0.03));
    await stripSim.dispose();
  });

  it("a packed graph gives the same bits as an unpacked one, from one state buffer and one inputs buffer", async () => {
    for (const dims of [2, 3] as const) {
      const cfg = { ...SMALL, dims, matrixRadius: dims === 2 ? 5 : 4 };
      const steps = 25;
      const run = async (packed: boolean) => {
        const sc = spheroidScenario(cfg);
        const g = springGraph(sc.state, sc.topology, sc.params, undefined, sc.graphRules?.(), { packed });
        const state = createSimState();
        const bridges: string[] = [];
        await advance(g.graph, g.pos, { steps: 1, state, materialise: "none" });
        await advance(g.graph, g.pos, { steps: steps - 1, state, materialise: "none", onBridge: (k, dir) => bridges.push(`${dir} ${k}`) });
        const held = { state: simStateValue(state, SPRING_STATE.packed), inputs: simStateValue(state, SPRING_STATE.inputs) };
        await readSpringState(state, sc.state, sc.topology);
        return { sc, bridges, held };
      };
      const plain = await run(false);
      const packed = await run(true);
      const label = `${dims}D`;
      // The same kernels on the same numbers: not close, identical.
      expect(Array.from(packed.sc.state.species), `${label} species`).toEqual(Array.from(plain.sc.state.species));
      expect(Array.from(bondTableOf(packed.sc.topology).parts!.link!.data!), `${label} bonds`).toEqual(
        Array.from(bondTableOf(plain.sc.topology).parts!.link!.data!),
      );
      expect(worstOf(packed.sc.state.pos, plain.sc.state.pos), `${label} positions`).toBe(0);
      expect(worstOf(packed.sc.state.radius, plain.sc.state.radius), `${label} radius`).toBe(0);
      expect(packed.sc.state.step).toBe(steps);
      expect(count(packed.sc.state, CELL) + count(packed.sc.state, LEADER), label).toBeGreaterThan(
        count(spheroidScenario(cfg).state, CELL),
      );
      // After the first tick nothing crosses to or from the host, packed or not.
      expect(packed.bridges, label).toEqual([]);
      expect(plain.bridges, label).toEqual([]);
      // Every part of the state is a range of one buffer, at an aligned offset; likewise the inputs.
      for (const bundle of [packed.held.state, packed.held.inputs]) {
        const parts = Object.values(bundle?.parts ?? {});
        expect(parts.length).toBeGreaterThan(1);
        expect(new Set(parts.map((v) => v.buffer?.buffer)).size).toBe(1);
        expect(parts.every((v) => v.buffer !== undefined && (v.buffer.offset ?? 0) % 256 === 0)).toBe(true);
        expect(new Set(parts.map((v) => v.buffer?.offset ?? 0)).size).toBe(parts.length);
      }
      expect(plain.held.state).toBeUndefined();
    }
  });

  it("a packed bundle crosses to an op that binds whole buffers as a copy, and back to the host part by part", async () => {
    // `addGrids` is a host op and `threshold` binds whole buffers: neither knows about ranges.
    const spec = {
      name: "testPacked",
      label: "Test packed",
      parts: [
        { name: "a", kind: "grid" as const },
        { name: "b", kind: "grid" as const },
      ],
    };
    for (const op of [packOp(spec), extractOp(spec, "a"), extractOp(spec, "b")]) if (!hasOp(op.name)) registerOp(op);
    const g = new Graph();
    const a = g.grid(Float32Array.of(0.1, 0.9, 0.4, 0.6), 2, 2);
    const b = g.grid(Float32Array.of(0.8, 0.2, 0.7, 0.3), 2, 2);
    const packed = g.op1("testPacked.pack", {
      a: g.op1("threshold", { in: a }, { thresh: 0.5 }),
      b: g.op1("threshold", { in: b }, { thresh: 0.5 }),
    });
    const second = g.op1("testPacked.b", { bundle: packed });
    // A slice into an op that takes whole buffers, then into a host op.
    const again = g.op1("threshold", { in: second }, { thresh: 0.5 });
    const sum = g.op1("addGrids", { a: again, b: g.op1("testPacked.a", { bundle: packed }) }, { wa: 1, wb: 10 });
    expect(Array.from((await pull(g, sum)).data!)).toEqual([1, 10, 1, 10]);
    const whole = await pull(g, packed);
    expect(Array.from(whole.parts!.a!.data!)).toEqual([0, 1, 0, 1]);
    expect(Array.from(whole.parts!.b!.data!)).toEqual([1, 0, 1, 0]);
  });

  it("a spring sim runs packed and fused, and hands out its state as ranges of one buffer", async () => {
    const plain = spheroidScenario({ ...SMALL, dims: 2 });
    const fusedScenario = spheroidScenario({ ...SMALL, dims: 2 });
    const a = createSpringSim(plain);
    const b = createSpringSim(fusedScenario, { fusePairs: true });
    await a.advance(40);
    await b.advance(40);
    const ranges = b.buffers()!;
    const all = [ranges.positions, ranges.previous, ranges.species, ranges.radius, ranges.link, ranges.spring];
    expect(new Set(all.map((r) => r.buffer)).size).toBe(1);
    expect(new Set(all.map((r) => r.offset)).size).toBe(all.length);
    expect(ranges.species.size).toBe(fusedScenario.state.n * 4);
    expect(new Set([a.buffers()!.positions.buffer, a.buffers()!.species.buffer]).size).toBe(2);
    await a.sync();
    await b.sync();
    expect(fusedScenario.state.step).toBe(40);
    expect(topologyDefect(fusedScenario.topology)).toBeUndefined();
    const tissue = (s: SpringState) => count(s, CELL) + count(s, LEADER);
    expect(Math.abs(tissue(fusedScenario.state) - tissue(plain.state))).toBeLessThanOrEqual(Math.ceil(tissue(plain.state) * 0.1));
    // A change of settings rebuilds a fused, packed graph too.
    fusedScenario.configure({ leaderThrust: 0 });
    await b.advance(10);
    await b.sync();
    expect(b.builds).toBe(2);
    expect(count(fusedScenario.state, LEADER)).toBe(0);
    await a.dispose();
    await b.dispose();
  });

  it("advances the step stored in a packed graph when nothing reads it", async () => {
    const sc = chainScenario({ n: 8, seed: 1 });
    sc.params = { ...sc.params, noiseforce: 0 };
    sc.state.step = 12;
    const g = springGraph(sc.state, sc.topology, sc.params, undefined, {}, { packed: true });
    const state = createSimState();
    const before = nodeBackend.poolStats().live;
    try {
      const sink = await advance(g.graph, g.pos, { steps: 2, state, materialise: "none" });
      // Packed, the state holds a copy; the position sink is ours to return.
      for (const p of payloadsNotHeld(state, sink)) if ("buffer" in p) nodeBackend.release(p);
      await readSpringState(state, sc.state, sc.topology);
      expect(sc.state.step).toBe(14);
    } finally {
      disposeSimState(state, nodeBackend);
    }
    expect(nodeBackend.poolStats().live).toBe(before);
  });

  it("a default chain hands out every range, keeps its run through a failed tick, and leaks nothing", async () => {
    const before = nodeBackend.poolStats().live;
    const sc = chainScenario({ n: 8, seed: 1 });
    const sim = createSpringSim(sc, { packed: true });
    await sim.advance(3);
    await sim.advance(2);
    const ranges = sim.buffers();
    expect(ranges?.radius.size).toBe(sc.state.n * 4);
    await expect(sim.advance(1.5)).rejects.toThrow(/whole number/);
    await expect(sim.advance(Number.POSITIVE_INFINITY)).rejects.toThrow(/whole number/);
    // A setting the tick rejects: the run so far survives, and carries on once it is restored.
    const good = sc.params;
    sc.params = { ...good, damp: Number.NaN };
    const failed = await sim.advance(1).then(
      () => false,
      () => true,
    );
    sc.params = good;
    await sim.advance(1);
    await sim.sync();
    expect([failed, sc.state.step, sim.builds]).toEqual([true, 6, 3]);
    const plain = createSpringSim(chainScenario({ n: 8, seed: 1 }));
    await plain.advance(1);
    expect(plain.buffers()?.radius.size).toBe(8 * 4);
    await plain.dispose();
    await sim.dispose();
    expect(nodeBackend.poolStats().live).toBe(before);
  });

  it("the pair interpreter refuses a program that is not plain u32 words, on both paths", async () => {
    const shape = { kind: "points" as const, n: 2 };
    const program = (extra: Partial<FieldValue>): FieldValue => ({
      shape: { kind: "bundle", parts: {} } as FieldValue["shape"],
      dtype: "f32",
      parts: { program: { shape, dtype: "u32", data: new Uint32Array(4), ...extra } },
    });
    const bad = [program({ axes: [{ name: "c", type: "channel", length: 2 }] }), program({ element: VEC4 }), program({ dtype: "f32" })];
    const state = program({});
    const refusals: boolean[] = [];
    for (const fixed of bad) {
      const gpu = await pairInterpreterOp.execute({ backend: nodeBackend }, [state, fixed], {}).then(
        () => "ran",
        (e: Error) => e.message,
      );
      let cpu = "ran";
      try {
        pairInterpreterOp.cpuGolden?.([state, fixed], {});
      } catch (e) {
        cpu = (e as Error).message;
      }
      refusals.push(/scalar u32 words/.test(gpu) && /scalar u32 words/.test(cpu));
    }
    expect(refusals).toEqual([true, true, true]);
  });

  it("returns what a tick leased when one of its ops rejects", async () => {
    const before = nodeBackend.poolStats().live;
    const sc = chainScenario({ n: 8, seed: 1 });
    const g = springGraph(sc.state, sc.topology, { ...sc.params, damp: Number.NaN });
    const state = createSimState();
    await expect(advance(g.graph, g.pos, { steps: 1, state, materialise: "none" })).rejects.toThrow();
    await expect(advance(g.graph, g.pos, { steps: 1.5, state })).rejects.toThrow(/whole number/);
    disposeSimState(state, nodeBackend);
    expect(nodeBackend.poolStats().live).toBe(before);
  });

  it("clears leaders at build when selection is off and no other rule is active", async () => {
    const sc = spheroidScenario({
      ...SMALL,
      dims: 2,
      leaderThrust: 0,
      divisionTime: Number.POSITIVE_INFINITY,
      yieldStrain: Number.POSITIVE_INFINITY,
      bondLifetime: Number.POSITIVE_INFINITY,
      ligand: 0,
    });
    const i = sc.state.species.indexOf(CELL);
    expect(i).toBeGreaterThanOrEqual(0);
    sc.state.species[i] = LEADER;
    const sim = createSpringSim(sc);
    await sim.advance(1);
    await sim.sync();
    expect(sc.state.species[i]).toBe(CELL);
    await sim.dispose();
  });

  it("refuses a sync that overlaps an advance", async () => {
    const sc = chainScenario({ n: 8, seed: 1 });
    const sim = createSpringSim(sc);
    const running = sim.advance(3);
    await expect(sim.sync()).rejects.toThrow(/still running/);
    await running;
    await sim.sync();
    expect(sc.state.step).toBe(3);
    await sim.dispose();
  });

  it("downloads state after a batch whose later tick fails", async () => {
    const make = () => chainScenario({ n: 8, seed: 1 });
    let leases = 0;
    const counting = {
      ...nodeBackend,
      lease(byteLength: number, usage?: number) {
        leases += 1;
        return nodeBackend.lease(byteLength, usage);
      },
    };
    const probe = createSpringSim(make(), { backend: counting });
    await probe.advance(1);
    const perTick = leases;
    await probe.dispose();
    expect(perTick).toBeGreaterThan(0);

    let seen = 0;
    const failing = {
      ...nodeBackend,
      lease(byteLength: number, usage?: number) {
        if (seen >= perTick) return Promise.reject(new Error("lease failed"));
        seen += 1;
        return nodeBackend.lease(byteLength, usage);
      },
    };
    const sc = make();
    const sim = createSpringSim(sc, { backend: failing });
    await expect(sim.advance(2)).rejects.toThrow(/lease failed/);
    await sim.sync();
    expect(sc.state.step).toBe(1);
    await sim.dispose();
  });

  it("releases device state after in-flight work settles and then refuses calls", async () => {
    const before = nodeBackend.poolStats().live;
    const sim = createSpringSim(chainScenario({ n: 8, seed: 2 }));
    const running = sim.advance(4);
    const released = sim.dispose();
    await expect(sim.sync()).rejects.toThrow(/disposed/);
    await running;
    await released;
    expect(nodeBackend.poolStats().live).toBe(before);
    await expect(sim.advance(1)).rejects.toThrow(/disposed/);
  });
});
