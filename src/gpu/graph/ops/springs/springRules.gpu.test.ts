import { beforeAll, describe, expect, it } from "vite-plus/test";
import { getDevice } from "#src/gpu/device";
import {
  addSpring,
  bondCount,
  type BreakRule,
  breakBonds,
  CELL,
  createState,
  createTopology,
  DEAD,
  divideCells,
  type FormRule,
  formBonds,
  LEADER,
  MAX_SPECIES,
  readSpringState,
  selectLeaders,
  spheroidGraph,
  spheroidScenario,
  type SpringState,
  type SpringTopology,
  spawn,
  springParams,
  topologyDefect,
} from "#src/gpu/sim/springs/index";
import { advance, createSimState, type FieldValue, Graph, type GpuField, pull } from "../../index";
import { bondTableOf, VEC4 } from "./bondTable";

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
  return { gpu, cpu, firstDifference: a.findIndex((v, k) => v !== b[k]) };
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
    for (let i = 0; i < n; i++) {
      if (i % Math.max(2, Math.floor(n / free)) === 1 && free-- > 0) continue;
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
});
