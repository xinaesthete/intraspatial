import { describe, expect, it } from "vite-plus/test";
import {
  addRod,
  addSpring,
  breakBonds,
  CELL,
  chromatinScenario,
  createSpheroidGpuRunner,
  createSpringsGpu,
  createState,
  createTopology,
  formBonds,
  liveCount,
  MAX_SPECIES,
  rand01,
  runSprings,
  type Scenario,
  setFix,
  setPull,
  setTorsion,
  type SpringMatrices,
  type SpringParams,
  type SpringState,
  type SpringTopology,
  speciesTable,
  spawn,
  spheroidScenario,
  springParams,
  topologyDefect,
} from "./index";

function cloneState(s: SpringState): SpringState {
  return {
    ...s,
    pos: s.pos.slice(),
    prev: s.prev.slice(),
    scratch: s.scratch.slice(),
    species: s.species.slice(),
    radius: s.radius.slice(),
  };
}

function cloneTopology(t: SpringTopology): SpringTopology {
  const slots = t.slots.slice();
  return { ...t, slots, slotBits: new Uint32Array(slots.buffer), rod: t.rod.slice(), fix: t.fix.slice(), pull: t.pull.slice() };
}

/** Largest position difference between two states, and the index it is at. */
function worst(a: SpringState, b: SpringState): { err: number; at: number } {
  let err = 0;
  let at = -1;
  for (let k = 0; k < a.n * 4; k++) {
    if (k % 4 === 3) continue;
    const e = Math.abs(a.pos[k]! - b.pos[k]!);
    // A NaN must fail every tolerance check, and must not be overwritten by a later finite one.
    if (!Number.isFinite(e)) return { err: Infinity, at: k };
    if (!(e <= err)) {
      err = e;
      at = k;
    }
  }
  return { err, at };
}

/** Run `steps` on both paths from the same start and return the worst position difference. */
async function stepBoth(
  s: SpringState,
  t: SpringTopology,
  p: SpringParams,
  steps: number,
  m?: SpringMatrices,
): Promise<{ err: number; at: number }> {
  const gs = cloneState(s);
  const gt = cloneTopology(t);
  const gpu = await createSpringsGpu(s.n, t.K);
  gpu.upload(gs, gt, m);
  gpu.step(p, steps);
  await gpu.download(gs, gt);
  runSprings(s, t, p, steps, m);
  return worst(s, gs);
}

describe("springs on the GPU", () => {
  it("matches the CPU step with every CSynth term switched on", async () => {
    const n = 40;
    const s = createState(n);
    for (let i = 0; i < n; i++) spawn(s, i, rand01(i, 3, 1) * 6, rand01(i, 3, 2) * 6, rand01(i, 3, 3) * 6);
    // A first step's worth of motion, so velocity-dependent terms have something to act on.
    for (let k = 0; k < n * 4; k++) s.prev[k] = s.pos[k]! - (rand01(k, 9, 4) - 0.5) * 0.02;
    const t = createTopology(n, 4);
    for (let i = 0; i + 1 < n; i++) addSpring(t, i, i + 1, 1, 0.5, 0, i % 3);
    addSpring(t, 0, 20, 2, 0.3, 1, 2);
    addRod(t, 5, 6, 0.8, 1.2);
    setFix(t, 0, 3, 3, 3);
    setPull(t, 39, 0, 0, 0, 2);
    setTorsion(t, 10, 1.1, 0.02);
    setTorsion(t, 11, -2, 0.01);
    const contact = new Float32Array(n * n);
    const dist = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = Math.abs(i - j);
        dist[i * n + j] = Math.sqrt(d) + 0.1;
        contact[i * n + j] = d === 0 ? 0 : 1 / d;
      }
    }
    // Missing data and a chain boundary, as CSynth marks them.
    contact[3 * n + 9] = contact[9 * n + 3] = -999;
    contact[19 * n + 20] = contact[20 * n + 19] = -9898;
    const p = springParams({
      damp: 0.9,
      damptype: 4,
      springforce: 0.05,
      springpow: 0.2,
      powBaseDist: 2,
      roleforces: [1, 2, 0.5, 1, 1, 1, 1, 1],
      roleforcesFix: [0, 0.01, 0, 0, 0, 0, 0, 0],
      backboneStrength: 2,
      pullspringforce: 0.01,
      pullspringpow: 0.5,
      pullfixdamp: 0.1,
      torsionspringforce: 0.5,
      backbonetorsionspringforce: 0.002,
      backbonetorsionspringangle: 0.7,
      torsionMaxLen: 6,
      backboneforce: 0.02,
      pushapartlocalforce: 0.05,
      pushapartforce: 0.004,
      pushapartpow: -1,
      pushapartDensityFactor: 0.0001,
      fractforce: 0.0005,
      xyzforce: 0.01,
      contactforcesc: 0.01,
      contactthreshold: 0.02,
      wrongfade: 1.5,
      m_force: 0.3,
      m_k: 2,
      m_c: 3,
      boidalignment: 0.01,
      boidcohesion: 0.01,
      boidseparation: 0.0005,
      boidrad: 2,
      visc: 0.05,
      gravity: 0.0005,
      modelSphereForce: 0.02,
      modelSphereRadius: 4,
      springCentre: [3, 3, 3],
      springCentreDamp: 0.999,
      noiseforce: 0.01,
      noiseprob: 0.5,
      motility: [0.005, 0, 0, 0],
      protrusion: [0.004, 0, 0, 0],
      thrust: [0.01, 0, 0, 0],
      protrusionRange: 2.5,
      densityRange: 3,
      repulsion: speciesTable([[0, 0, 0.1]]),
      adhesion: speciesTable([[0, 0, 0.02]]),
    });
    const { err, at } = await stepBoth(s, t, p, 6, { size: n, contact, dist });
    if (err > 2e-4) expect.fail(`positions differ by ${err} at flat index ${at}`);
  });

  it("matches the CPU step for each damping type", async () => {
    for (const damptype of [1, 2, 3] as const) {
      const s = createState(12);
      for (let i = 0; i < 12; i++) spawn(s, i, rand01(i, 5, 1) * 3, rand01(i, 5, 2) * 3, rand01(i, 5, 3) * 3);
      for (let k = 0; k < 48; k++) s.prev[k] = s.pos[k]! - (rand01(k, 6, 4) - 0.5) * 0.05;
      const t = createTopology(12, 2);
      for (let i = 0; i + 1 < 12; i++) addSpring(t, i, i + 1, 1, 0.2);
      const { err } = await stepBoth(s, t, springParams({ damptype, damp: damptype === 1 ? 0.9 : 5, springforce: 0.5 }), 5);
      if (err > 1e-4) expect.fail(`damptype ${damptype}: positions differ by ${err}`);
    }
  });

  it("matches the CPU on the chromatin and spheroid scenarios", async () => {
    const check = async (name: string, sc: Scenario, steps: number) => {
      const { err } = await stepBoth(sc.state, sc.topology, sc.params, steps, sc.matrices);
      if (err > 5e-4) expect.fail(`${name}: positions differ by ${err}`);
    };
    await check("chromatin distances", chromatinScenario({ n: 50, mode: "distances" }), 8);
    await check("chromatin contacts", chromatinScenario({ n: 50, mode: "contacts" }), 8);
    // Motility off: a random kick of ±0.02 that lands differently in f32 would swamp the check.
    await check("spheroid", spheroidScenario({ cellRadius: 2, matrixRadius: 6, maxCells: 40, motility: 0 }), 8);
    await check(
      "spheroid 3D with protrusion",
      spheroidScenario({ dims: 3, cellRadius: 2, matrixRadius: 4.5, maxCells: 60, motility: 0, protrusion: 0.02, protrusionRange: 2 }),
      6,
    );
  });

  it("runs a spheroid on the device, dividing on the host", async () => {
    const sc = spheroidScenario({ cellRadius: 2, matrixRadius: 7, maxCells: 200, divisionTime: 60, yieldStrain: 0.1 });
    const start = liveCount(sc.state, CELL);
    const runner = await createSpheroidGpuRunner(sc);
    await runner.advance(400);
    expect(sc.state.step).toBe(400);
    expect(liveCount(sc.state, CELL)).toBeGreaterThan(start * 1.5);
    expect(sc.totals.divisions).toBe(liveCount(sc.state, CELL) - start);
    expect(topologyDefect(sc.topology)).toBeUndefined();
  });

  it("downloads the state of the step it was called at, however far the model runs meanwhile", async () => {
    const make = async () => {
      const sc = chromatinScenario({ n: 40, mode: "distances", seed: 2 });
      const gpu = await createSpringsGpu(sc.state.n, sc.topology.K);
      gpu.upload(sc.state, sc.topology, sc.matrices);
      gpu.step(sc.params, 3);
      return { sc, gpu };
    };
    const quiet = await make();
    await quiet.gpu.download(quiet.sc.state, quiet.sc.topology);

    const busy = await make();
    const pending = busy.gpu.download(busy.sc.state, busy.sc.topology);
    await expect(busy.gpu.download(busy.sc.state, busy.sc.topology)).rejects.toThrow(/another download/);
    busy.gpu.step(busy.sc.params, 5);
    await pending;
    expect(busy.sc.state.step).toBe(3);
    expect(busy.gpu.stepIndex).toBe(8);
    expect(worst(busy.sc.state, quiet.sc.state).err).toBe(0);
  });

  it("breaks exactly the bonds the CPU breaks", async () => {
    const n = 100;
    const s = createState(n);
    let k = 0;
    for (let j = 0; j < 10; j++)
      for (let i = 0; i < 10; i++) spawn(s, k++, (i + (j & 1) * 0.5) * (1 + 0.04 * i), (j * Math.sqrt(3)) / 2, 0);
    const t = createTopology(n, 6);
    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        const d = Math.hypot(s.pos[b * 4]! - s.pos[a * 4]!, s.pos[b * 4 + 1]! - s.pos[a * 4 + 1]!);
        if (d < 1.45) addSpring(t, a, b, 1, 0.1, 0, a % 2);
      }
    }
    s.step = 17;
    const gt = cloneTopology(t);
    const p = springParams({ seed: 5 });
    const rule = { kinds: 0b10, maxStrain: 0.2, tau: 6, tensionOnly: true };
    const gpu = await createSpringsGpu(n, 6);
    gpu.upload(s, gt);
    gpu.breakBonds(p, rule);
    await gpu.download(cloneState(s), gt);
    const broken = breakBonds(s, t, p, rule);
    expect(broken).toBeGreaterThan(10);
    let mismatch = -1;
    for (let w = 0; w < t.slotBits.length; w += 4) if (t.slotBits[w] !== gt.slotBits[w]) mismatch = w;
    expect(mismatch).toBe(-1);
    expect(topologyDefect(gt)).toBeUndefined();
  });

  it("forms exactly the bonds the CPU forms", async () => {
    const n = 120;
    const s = createState(n);
    for (let i = 0; i < n; i++) spawn(s, i, rand01(i, 2, 1) * 9, rand01(i, 2, 2) * 9, 0, i % 2);
    const t = createTopology(n, 3);
    const gt = cloneTopology(t);
    const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => -1);
    kindOf[0] = 1;
    kindOf[1] = 2;
    kindOf[MAX_SPECIES] = 2;
    const rule = { range: 1.4, kindOf, strength: [0, 0.3, 0.7], rest: "current", prob: 0.8, speciesCount: MAX_SPECIES } as const;
    const p = springParams({ seed: 3, dims: 2 });
    const gpu = await createSpringsGpu(n, 3);
    gpu.upload(s, gt);
    let formed = 0;
    // Several rounds: later rounds meet rows that are part full.
    for (let round = 0; round < 4; round++) {
      formed += formBonds(s, t, p, rule);
      gpu.formBonds(p, rule);
    }
    await gpu.download(cloneState(s), gt);
    expect(formed).toBeGreaterThan(30);
    let mismatch = -1;
    let restErr = 0;
    for (let w = 0; w < t.slotBits.length; w += 4) {
      if (t.slotBits[w] !== gt.slotBits[w]) mismatch = w;
      else if (t.slotBits[w] !== 0xffffffff)
        restErr = Math.max(restErr, Math.abs(t.slots[w + 1]! - gt.slots[w + 1]!), Math.abs(t.slots[w + 2]! - gt.slots[w + 2]!));
    }
    if (mismatch >= 0) {
      const row = Math.floor(mismatch / 12);
      const show = (tt: SpringTopology) => Array.from(tt.slotBits.subarray(row * 12, row * 12 + 12)).filter((_, q) => q % 4 === 0);
      expect.fail(`row ${row}: cpu ${show(t).join(",")} gpu ${show(gt).join(",")}`);
    }
    expect(restErr).toBeLessThan(1e-5);
    expect(topologyDefect(gt)).toBeUndefined();
  });
});
