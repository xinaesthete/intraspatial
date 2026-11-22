import { describe, expect, it } from "vite-plus/test";
import {
  addRod,
  addSpring,
  adhesionForce,
  applyLines,
  bondCount,
  bondReadout,
  bondsOf,
  breakBonds,
  CELL,
  chainScenario,
  chromatinScenario,
  createState,
  createTopology,
  DEAD,
  DEFAULT_LINE_SPRINGS,
  dihedral,
  distanceCorrelation,
  distanceMatrix,
  divideCells,
  formBonds,
  getSpring,
  helixLine,
  killParticle,
  LEADER,
  lineCells,
  liveCount,
  MATRIX,
  MAX_SPECIES,
  pairRand01,
  phaseDiagram,
  projectedShape,
  rand01,
  relaxationScenario,
  removeSpring,
  runSprings,
  selectLeaders,
  type Scenario,
  setFix,
  setTorsion,
  shape3d,
  sheetLines,
  shollProfile,
  snapEnd,
  type SpringParams,
  type SpringState,
  speciesTable,
  spawn,
  spheroidDefaults,
  spheroidScenario,
  springParams,
  stepSprings,
  topologyDefect,
  torsionForces,
} from "./index";

/** Overdamped, no clipping in the range these tests use, no pair terms: springs only. */
function plain(over: Partial<SpringParams> = {}): SpringParams {
  return springParams({ damp: 0, damptype: 1, springforce: 1, springmaxvel: 10, pushapartlocalforce: 0, backboneStrength: 1, ...over });
}

function dist(s: SpringState, a: number, b: number): number {
  return Math.hypot(s.pos[b * 4]! - s.pos[a * 4]!, s.pos[b * 4 + 1]! - s.pos[a * 4 + 1]!, s.pos[b * 4 + 2]! - s.pos[a * 4 + 2]!);
}

/** A triangular lattice patch of live particles of one species; returns its size. */
function lattice(s: SpringState, cols: number, rows: number, species = 0): number {
  let k = 0;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) spawn(s, k++, i + (j & 1) * 0.5, (j * Math.sqrt(3)) / 2, 0, species);
  return k;
}

describe("random streams", () => {
  it("are deterministic, in range, and symmetric for a pair", () => {
    let lo = 1;
    let hi = 0;
    let sum = 0;
    for (let i = 0; i < 4000; i++) {
      const r = rand01(i, 7, 3);
      lo = Math.min(lo, r);
      hi = Math.max(hi, r);
      sum += r;
    }
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(hi).toBeLessThan(1);
    expect(sum / 4000).toBeCloseTo(0.5, 1);
    expect(rand01(5, 7, 3)).toBe(rand01(5, 7, 3));
    expect(pairRand01(3, 90, 12, 1)).toBe(pairRand01(90, 3, 12, 1));
    expect(pairRand01(3, 90, 12, 1)).not.toBe(pairRand01(3, 90, 13, 1));
  });
});

describe("spring topology", () => {
  it("stores a spring under both ends and removes it from both", () => {
    const t = createTopology(4, 2);
    expect(addSpring(t, 0, 1, 1.5, 2, 0, 3)).toBe(true);
    expect(getSpring(t, 1, 0)).toEqual({ partner: 0, kind: 3, rest: 1.5, strength: 2, pow: 0 });
    expect(bondCount(t)).toBe(1);
    expect(topologyDefect(t)).toBeUndefined();
    removeSpring(t, 1, 0);
    expect(getSpring(t, 0, 1)).toBeUndefined();
    expect(bondCount(t)).toBe(0);
  });

  it("refuses a spring when either row is full, and changes nothing", () => {
    const t = createTopology(4, 2);
    addSpring(t, 0, 1);
    addSpring(t, 0, 2);
    expect(addSpring(t, 3, 0)).toBe(false);
    expect(bondCount(t)).toBe(2);
    expect(topologyDefect(t)).toBeUndefined();
  });

  it("updates an existing spring in place", () => {
    const t = createTopology(2, 1);
    addSpring(t, 0, 1, 1);
    expect(addSpring(t, 0, 1, 2)).toBe(true);
    expect(getSpring(t, 1, 0)?.rest).toBe(2);
  });
});

describe("stepSprings: bonded terms", () => {
  it("relaxes a spring to its rest length and conserves the centre of mass", () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 3, 0, 0);
    const t = createTopology(2);
    addSpring(t, 0, 1, 1, 0.2);
    runSprings(s, t, plain(), 200);
    expect(dist(s, 0, 1)).toBeCloseTo(1, 4);
    expect((s.pos[0]! + s.pos[4]!) / 2).toBeCloseTo(1.5, 5);
  });

  it("scales strength by kind through roleforces", () => {
    const move = (kind: number) => {
      const s = createState(2);
      spawn(s, 0, 0, 0, 0);
      spawn(s, 1, 3, 0, 0);
      const t = createTopology(2);
      addSpring(t, 0, 1, 1, 0.01, 0, kind);
      const roleforces = [1, 4, 1, 1, 1, 1, 1, 1];
      stepSprings(s, t, plain({ roleforces }));
      return s.pos[0]!;
    };
    expect(move(1)).toBeCloseTo(4 * move(0), 6);
  });

  it("holds a rod's length and a fix's position", () => {
    const s = createState(3);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 3, 0, 0);
    spawn(s, 2, 0, 5, 0);
    const t = createTopology(3);
    setFix(t, 0, 0, 0, 0);
    addRod(t, 1, 0, 2);
    addSpring(t, 0, 2, 1, 0.3);
    runSprings(s, t, plain(), 100);
    expect(dist(s, 0, 1)).toBeCloseTo(2, 4);
    expect([s.pos[0], s.pos[1], s.pos[2]]).toEqual([0, 0, 0]);
    // The spring's free end did all the moving, because the other end is pinned.
    expect(dist(s, 0, 2)).toBeCloseTo(1, 3);
  });
});

describe("torsion springs", () => {
  it("apply no net force and no net torque", () => {
    const pos = [0.1, 1, 0.2, 0, 0, 0, 0, 1, 0.1, 0.4, 0, 2.1, 0.3, -0.5, 0.2, 0];
    // Twelve elements, as documented: three per particle.
    const out = new Float64Array(12);
    expect(torsionForces(pos, 0, 1, 2, 3, 1, 0.05, out)).toBe(true);
    const sum = [0, 0, 0];
    const torque = [0, 0, 0];
    for (let k = 0; k < 4; k++) {
      const [x, y, z] = [pos[k * 4]!, pos[k * 4 + 1]!, pos[k * 4 + 2]!];
      const [fx, fy, fz] = [out[k * 3]!, out[k * 3 + 1]!, out[k * 3 + 2]!];
      sum[0]! += fx;
      sum[1]! += fy;
      sum[2]! += fz;
      torque[0]! += y * fz - z * fy;
      torque[1]! += z * fx - x * fz;
      torque[2]! += x * fy - y * fx;
    }
    expect(Math.hypot(...sum)).toBeLessThan(1e-12);
    expect(Math.hypot(...torque)).toBeLessThan(1e-12);
    // The end forces are normal to their planes, so they do not stretch a-b or c-d.
    expect(Math.abs(out[0]! * 0.1 + out[1]! * 1 + out[2]! * 0.2)).toBeLessThan(1e-12);
  });

  it("apply nothing to a straight line of points", () => {
    const pos = [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
    expect(torsionForces(pos, 0, 1, 2, 3, 1, 1, new Float64Array(13))).toBe(false);
    expect(dihedral(pos, 0, 1, 2, 3)).toBeNaN();
  });

  it("twist four bonded particles to the target angle without changing bond lengths", () => {
    for (const target of [1, -0.6, 2.2]) {
      const s = createState(4);
      spawn(s, 0, 0, 1, 0);
      spawn(s, 1, 0, 0, 0);
      spawn(s, 2, 1, 0, 0);
      spawn(s, 3, 1, 0.6, 0.8);
      const t = createTopology(4);
      for (let i = 0; i < 3; i++) addSpring(t, i, i + 1, 1, 0.3);
      addSpring(t, 0, 2, Math.SQRT2, 0.3);
      addSpring(t, 1, 3, Math.SQRT2, 0.3);
      setTorsion(t, 0, target, 0.01);
      runSprings(s, t, plain(), 3000);
      expect(dihedral(s.pos, 0, 1, 2, 3)).toBeCloseTo(target, 2);
      expect(dist(s, 0, 1)).toBeCloseTo(1, 2);
      expect(dist(s, 2, 3)).toBeCloseTo(1, 2);
    }
  });

  it("wind a chain into a helix when every backbone quad has one", () => {
    const n = 24;
    const s = createState(n);
    for (let i = 0; i < n; i++) spawn(s, i, i * 0.7, Math.sin(i * 1.3) * 0.5, Math.cos(i * 0.9) * 0.5);
    const t = createTopology(n);
    for (let i = 0; i + 1 < n; i++) addSpring(t, i, i + 1, 1, 0.3);
    for (let i = 0; i + 2 < n; i++) addSpring(t, i, i + 2, 1.6, 0.3);
    runSprings(s, t, plain({ backbonetorsionspringforce: 0.01, backbonetorsionspringangle: 0.9 }), 6000);
    let worstAngle = 0;
    for (let i = 0; i + 3 < n; i++) worstAngle = Math.max(worstAngle, Math.abs(dihedral(s.pos, i, i + 1, i + 2, i + 3) - 0.9));
    expect(worstAngle).toBeLessThan(0.05);
  });
});

describe("matrix lines", () => {
  it("walks a helix line at constant offset and a sheet line at constant sum", () => {
    expect(lineCells(helixLine(2, 10, 4))).toEqual([
      [2, 6],
      [3, 7],
      [4, 8],
      [5, 9],
      [6, 10],
    ]);
    const [rung] = sheetLines(0, 12, 2, 3);
    // Strands 0..4 and 8..12 with a three-particle turn: antiparallel, so 0 pairs with 12.
    expect(rung).toEqual({ a0: 0, b0: 12, a1: 4, b1: 8 });
    expect(lineCells(rung!).every(([a, b]) => a + b === 12)).toBe(true);
    expect(sheetLines(0, 7, 2, 0)).toEqual([{ a0: 0, b0: 7, a1: 3, b1: 4 }]);
    expect(lineCells(rung!).length).toBe(5);
  });

  it("walks a shallow and a steep line one cell per step of the long axis", () => {
    expect(lineCells({ a0: 0, b0: 10, a1: 6, b1: 12 })).toHaveLength(7);
    const steep = lineCells({ a0: 3, b0: 0, a1: 5, b1: 8 });
    expect(steep.map(([, b]) => b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("snaps a dragged end onto the nearer diagonal", () => {
    expect(snapEnd(10, 20, 15, 27)).toEqual([17, 27]);
    expect(snapEnd(10, 20, 15, 13)).toEqual([17, 13]);
    expect(snapEnd(10, 20, 30, 21)).toEqual([30, 21]);
  });

  it("adds a spring per cell, replaces its own on re-apply, and leaves the backbone alone", () => {
    const sc = chainScenario({ n: 32 });
    const t = sc.topology;
    const first = applyLines(t, [helixLine(4, 16, 4), { a0: 6, b0: 7, a1: 6, b1: 7 }]);
    expect(first.pairs).toHaveLength(9);
    expect(getSpring(t, 4, 8)?.kind).toBe(3);
    expect(getSpring(t, 6, 7)?.kind).toBe(0);
    expect(first.torsions).toEqual([]);
    const second = applyLines(t, [helixLine(4, 10, 3)], undefined, first);
    expect(getSpring(t, 4, 8)).toBeUndefined();
    expect(second.pairs).toHaveLength(4);
    expect(second.torsions).toEqual([4, 5, 6, 7]);
    expect(bondCount(t)).toBe(31 + 4);
    expect(topologyDefect(t)).toBeUndefined();
    applyLines(t, [], undefined, second);
    expect(bondCount(t)).toBe(31);
    expect(t.torsion.every((v, k) => k % 2 === 0 || v === 0)).toBe(true);
  });

  it("folds a chain: a helix line pulls every tied pair to the spring's rest length", () => {
    const sc = chainScenario({ n: 40 });
    applyLines(sc.topology, [helixLine(5, 35, 4)]);
    sc.advance(3000);
    let worstGap = 0;
    for (let i = 5; i + 4 <= 35; i++) worstGap = Math.max(worstGap, Math.abs(dist(sc.state, i, i + 4) - 1.5));
    expect(worstGap).toBeLessThan(0.25);
  });
});

describe("protrusion and thrust", () => {
  /** A flat sheet of one species with a bump of three extra particles on top. */
  function sheetWithBump(): SpringState {
    const s = createState(80);
    let k = 0;
    for (let j = 0; j < 5; j++) for (let i = 0; i < 12; i++) spawn(s, k++, i + (j & 1) * 0.5, (j * Math.sqrt(3)) / 2, 0);
    for (const [x, y] of [
      [5.5, 4.33],
      [6.5, 4.33],
      [6, 5.2],
    ] as const)
      spawn(s, k++, x, y, 0);
    return s;
  }
  const moved = (s: SpringState, i: number, p: SpringParams) => {
    const y0 = s.pos[i * 4 + 1]!;
    stepSprings(s, createTopology(s.n), p);
    return s.pos[i * 4 + 1]! - y0;
  };

  it("cancel inside a tissue and point outward at its surface", () => {
    const p = plain({ protrusion: [0.01, 0, 0, 0], protrusionRange: 2, dims: 2 });
    // Particle 30 is in the middle row; 54 is in the top row away from the bump.
    expect(Math.abs(moved(sheetWithBump(), 30, p))).toBeLessThan(1e-6);
    expect(moved(sheetWithBump(), 50, p)).toBeGreaterThan(0.002);
  });

  it("protrusion pushes a bump harder than the flat surface beside it; thrust pushes both alike", () => {
    const tip = 62;
    const flat = 50;
    const p = plain({ protrusion: [0.01, 0, 0, 0], protrusionRange: 2, dims: 2 });
    const pushed = (i: number) => {
      const s = sheetWithBump();
      const before = [s.pos[i * 4]!, s.pos[i * 4 + 1]!];
      stepSprings(s, createTopology(s.n), p);
      return Math.hypot(s.pos[i * 4]! - before[0]!, s.pos[i * 4 + 1]! - before[1]!);
    };
    expect(pushed(tip)).toBeGreaterThan(pushed(flat) * 1.3);
    expect(pushed(tip)).toBeLessThanOrEqual(0.01);
    const q = plain({ thrust: [0.01, 0, 0, 0], protrusionRange: 2, dims: 2 });
    const a = sheetWithBump();
    const b = sheetWithBump();
    const t = createTopology(a.n);
    const before = [a.pos[tip * 4]!, a.pos[tip * 4 + 1]!, b.pos[flat * 4]!, b.pos[flat * 4 + 1]!];
    stepSprings(a, t, q);
    stepSprings(b, t, q);
    expect(Math.hypot(a.pos[tip * 4]! - before[0]!, a.pos[tip * 4 + 1]! - before[1]!)).toBeCloseTo(0.01, 5);
    expect(Math.hypot(b.pos[flat * 4]! - before[2]!, b.pos[flat * 4 + 1]! - before[3]!)).toBeCloseTo(0.01, 5);
  });

  it("counts other species as own tissue only through tissueMask", () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0, 0);
    spawn(s, 1, 1, 0, 0, 2);
    const off = plain({ thrust: [0.01, 0, 0, 0], protrusionRange: 2 });
    stepSprings(s, createTopology(2), off);
    expect(s.pos[0]).toBe(0);
    stepSprings(s, createTopology(2), { ...off, tissueMask: 1 << 2 });
    expect(s.pos[0]).toBeCloseTo(-0.01, 6);
  });
});

describe("leader selection", () => {
  const rule = { from: CELL, to: LEADER, tau: 1, maxNeighbours: 3, spacing: 4, revertNeighbours: 6 };

  it("picks exposed cells only, never two within the spacing, and turns a buried leader back", () => {
    const s = createState(64);
    lattice(s, 8, 8, CELL);
    const p = plain({ dims: 2 });
    expect(selectLeaders(s, p, rule)).toBeGreaterThan(0);
    const leaders: number[] = [];
    for (let i = 0; i < 64; i++) if (s.species[i] === LEADER) leaders.push(i);
    // Only the lattice's corners and edges have three or fewer neighbours.
    for (const i of leaders) {
      const col = i % 8;
      const row = Math.floor(i / 8);
      expect(col === 0 || col === 7 || row === 0 || row === 7).toBe(true);
    }
    for (const a of leaders) for (const b of leaders) if (a < b) expect(dist(s, a, b)).toBeGreaterThanOrEqual(4);
    // A leader put in the middle of the sheet has a full shell and reverts.
    s.species[27] = LEADER;
    s.step++;
    selectLeaders(s, p, rule);
    expect(s.species[27]).toBe(CELL);
  });
});

describe("3D readouts", () => {
  /** A ball of radius `r` with `arms` straight arms of length `len` along the axes. */
  function star(r: number, arms: number, len: number): SpringState {
    const s = createState(4000);
    let k = 0;
    for (let x = -r; x <= r; x++)
      for (let y = -r; y <= r; y++) for (let z = -r; z <= r; z++) if (Math.hypot(x, y, z) <= r) spawn(s, k++, x, y, z, CELL);
    const axes = [
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ] as const;
    for (let a = 0; a < arms; a++) {
      for (let d = r + 1; d <= r + len; d++) {
        for (const [u, v] of [
          [0, 0],
          [1, 0],
          [0, 1],
        ] as const) {
          const [ax, ay, az] = axes[a]!;
          spawn(s, k++, ax * d + (ax === 0 ? u : 0), ay * d + (ay === 0 ? (ax === 0 ? v : u) : 0), az * d + (az === 0 ? v : 0), CELL);
        }
      }
    }
    return s;
  }

  it("counts one piece per shell for a ball and one per arm for a star", () => {
    const ball = shollProfile(star(4, 0, 0), CELL, { coreRadius: 5 });
    expect(Math.max(...ball.pieces)).toBe(1);
    expect(ball.branches).toBe(0);
    for (const arms of [3, 5]) {
      const profile = shollProfile(star(4, arms, 8), CELL, { coreRadius: 5 });
      expect(profile.branches).toBe(arms);
      expect(profile.branchRadius).toBeGreaterThan(5);
    }
  });

  it("measures a ball as compact and isotropic, and a star as reaching far for its volume", () => {
    const ball = shape3d(star(4, 0, 0), CELL);
    expect(ball.anisotropy).toBeLessThan(0.01);
    expect(ball.maxRadius).toBeCloseTo(4.5, 1);
    const arms = shape3d(star(4, 6, 8), CELL);
    expect(arms.compactness).toBeLessThan(ball.compactness * 0.2);
    expect(arms.anisotropy).toBeLessThan(0.05);
    // One arm makes it lopsided; two species are measured together when asked.
    expect(shape3d(star(2, 1, 12), CELL).anisotropy).toBeGreaterThan(0.3);
    const mixed = star(4, 0, 0);
    mixed.species[0] = LEADER;
    expect(shape3d(mixed, [CELL, LEADER]).count).toBe(shape3d(mixed, CELL).count + 1);
  });
});

describe("adhesion", () => {
  it("repels overlap, attracts at mid range, and is continuous at both ends", () => {
    expect(adhesionForce(0.8, 1, 0.5, 0.1, 0.5)).toBeCloseTo(-0.1, 9);
    expect(adhesionForce(1, 1, 0.5, 0.1, 0.5)).toBe(0);
    expect(adhesionForce(1.25, 1, 0.5, 0.1, 0.5)).toBeCloseTo(0.1, 9);
    expect(adhesionForce(1.5, 1, 0.5, 0.1, 0.5)).toBe(0);
    expect(adhesionForce(1.4999, 1, 0.5, 0.1, 0.5)).toBeLessThan(1e-3);
  });

  it("settles two adhesive particles at contact", () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 1.3, 0, 0);
    const t = createTopology(2);
    const p = plain({ repulsion: speciesTable([[0, 0, 0.3]]), adhesion: speciesTable([[0, 0, 0.05]]) });
    runSprings(s, t, p, 400);
    expect(dist(s, 0, 1)).toBeCloseTo(1, 2);
  });

  it("gives the same step through the bucket grid as through all pairs", () => {
    const build = () => {
      const s = createState(40);
      for (let i = 0; i < 40; i++) spawn(s, i, rand01(i, 1, 1) * 6, rand01(i, 1, 2) * 6, rand01(i, 1, 3) * 6);
      return s;
    };
    const base = { repulsion: speciesTable([[0, 0, 0.3]]), adhesion: speciesTable([[0, 0, 0.05]]) };
    const a = build();
    const b = build();
    const t = createTopology(40);
    runSprings(a, t, plain(base), 5);
    runSprings(b, t, plain({ ...base, pairRange: 1.5 }), 5);
    let worst = 0;
    for (let k = 0; k < 40 * 4; k += 4)
      worst = Math.max(
        worst,
        Math.abs(a.pos[k]! - b.pos[k]!),
        Math.abs(a.pos[k + 1]! - b.pos[k + 1]!),
        Math.abs(a.pos[k + 2]! - b.pos[k + 2]!),
      );
    expect(worst).toBeLessThan(1e-6);
  });

  it("moves a heavy species less: mobility is an inverse viscosity", () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0, 0);
    spawn(s, 1, 0.8, 0, 0, 1);
    const t = createTopology(2);
    const mobility = [1, 0.25, 1, 1];
    stepSprings(s, t, plain({ repulsion: speciesTable([[0, 1, 0.5]]), mobility }));
    expect(-s.pos[0]!).toBeCloseTo(4 * (s.pos[4]! - 0.8), 5);
  });
});

describe("bond turnover", () => {
  it("breaks an over-stretched bond from both rows and leaves the rest", () => {
    const s = createState(3);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 1.5, 0, 0);
    spawn(s, 2, 0, 1.05, 0);
    const t = createTopology(3);
    addSpring(t, 0, 1, 1, 1, 0, 1);
    addSpring(t, 0, 2, 1, 1, 0, 1);
    expect(breakBonds(s, t, plain(), { kinds: 1 << 1, maxStrain: 0.25 })).toBe(1);
    expect(getSpring(t, 1, 0)).toBeUndefined();
    expect(getSpring(t, 2, 0)).toBeDefined();
    expect(topologyDefect(t)).toBeUndefined();
  });

  it("leaves kinds outside the rule alone", () => {
    const s = createState(2);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 9, 0, 0);
    const t = createTopology(2);
    addSpring(t, 0, 1, 1, 1, 0, 0);
    expect(breakBonds(s, t, plain(), { kinds: 1 << 1, maxStrain: 0.25 })).toBe(0);
  });

  it("forms a bond only between mutual nearest neighbours, at its current length", () => {
    const s = createState(3);
    spawn(s, 0, 0, 0, 0);
    spawn(s, 1, 1, 0, 0);
    spawn(s, 2, 2.1, 0, 0);
    const t = createTopology(3);
    const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => 1);
    const rule = { range: 1.3, kindOf, strength: [0, 0.4], rest: "current", speciesCount: MAX_SPECIES } as const;
    // 0 and 1 choose each other; 2 chooses 1 and is not chosen back.
    expect(formBonds(s, t, plain(), rule)).toBe(1);
    const b = getSpring(t, 0, 1);
    expect(b?.rest).toBeCloseTo(1, 6);
    expect(b?.strength).toBeCloseTo(0.4, 6);
    expect(getSpring(t, 1, 2)).toBeUndefined();
    // Next call: 0 and 1 are already bonded, so 1 and 2 pair up.
    expect(formBonds(s, t, plain(), rule)).toBe(1);
    expect(getSpring(t, 2, 1)?.rest).toBeCloseTo(1.1, 5);
    expect(topologyDefect(t)).toBeUndefined();
  });

  it("keeps the table symmetric through many rounds of random breaking and re-forming", () => {
    const s = createState(64);
    lattice(s, 8, 8);
    const t = createTopology(64, 6);
    const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => 1);
    const p = plain();
    for (let k = 0; k < 60; k++) {
      formBonds(s, t, p, { range: 1.1, kindOf, strength: [0, 0.1], rest: "current", speciesCount: MAX_SPECIES });
      breakBonds(s, t, p, { kinds: 2, tau: 10 });
      s.step++;
      const defect = topologyDefect(t);
      if (defect) expect.fail(`round ${k}: ${defect}`);
    }
    expect(bondCount(t)).toBeGreaterThan(40);
  });
});

describe("stress relaxation", () => {
  /** A stretched lattice with every particle pinned: the shape is held and only bonds change. */
  function strained(): { s: SpringState; t: ReturnType<typeof createTopology>; p: SpringParams } {
    const s = createState(400);
    const n = lattice(s, 20, 20);
    const t = createTopology(400, 6);
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (dist(s, a, b) < 1.01) addSpring(t, a, b, 1, 0.1, 0, 1);
    for (let i = 0; i < n; i++) {
      s.pos[i * 4] = s.pos[i * 4]! * 1.1;
      s.prev[i * 4] = s.pos[i * 4]!;
      setFix(t, i, s.pos[i * 4]!, s.pos[i * 4 + 1]!, 0);
    }
    return { s, t, p: plain() };
  }

  it("holds stress in an elastic network", () => {
    const { s, t, p } = strained();
    const before = bondReadout(s, t, p).virial[0];
    runSprings(s, t, p, 50);
    expect(bondReadout(s, t, p).virial[0]).toBeCloseTo(before, 6);
    expect(before).toBeGreaterThan(0);
  });

  it("relaxes with half-time τ·ln2 when bonds turn over at rate 1/τ", () => {
    const { s, t, p } = strained();
    const tau = 40;
    const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => 1);
    const start = bondReadout(s, t, p).virial[0];
    let half = -1;
    for (let k = 1; k <= 120 && half < 0; k++) {
      stepSprings(s, t, p);
      breakBonds(s, t, p, { kinds: 2, tau });
      // New bonds form at their current length, so they carry no stress.
      formBonds(s, t, p, { range: 1.2, kindOf, strength: [0, 0.1], rest: "current", speciesCount: MAX_SPECIES });
      if (bondReadout(s, t, p).virial[0] <= start / 2) half = k;
    }
    expect(half).toBeGreaterThan(tau * Math.LN2 * 0.8);
    expect(half).toBeLessThan(tau * Math.LN2 * 1.25);
  });
});

describe("growth", () => {
  it("divides uncrowded cells into dead slots and stops when full", () => {
    const s = createState(8);
    spawn(s, 0, 0, 0, 0, CELL);
    const p = plain({ dims: 2 });
    let divisions = 0;
    for (let k = 0; k < 200; k++) {
      divisions += divideCells(s, p, { species: CELL, tau: 5, maxDensity: 1e9, offset: 0.5 });
      s.step++;
    }
    expect(divisions).toBe(7);
    expect(liveCount(s, CELL)).toBe(8);
  });

  it("does not divide a crowded cell", () => {
    const s = createState(4);
    spawn(s, 0, 0, 0, 0, CELL);
    s.pos[3] = 20;
    const p = plain();
    let divisions = 0;
    for (let k = 0; k < 50; k++) {
      divisions += divideCells(s, p, { species: CELL, tau: 2, maxDensity: 10, offset: 0.5 });
      s.step++;
    }
    expect(divisions).toBe(0);
  });

  it("removes a killed particle's bonds from both ends", () => {
    const s = createState(3);
    for (let i = 0; i < 3; i++) spawn(s, i, i, 0, 0);
    const t = createTopology(3);
    addSpring(t, 0, 1);
    addSpring(t, 1, 2);
    killParticle(s, t, 1);
    expect(s.species[1]).toBe(DEAD);
    expect(bondCount(t)).toBe(0);
  });
});

describe("projectedShape", () => {
  it("measures a packed disc as round, with the right area", () => {
    const s = createState(400);
    let k = 0;
    for (let j = -12; j <= 12; j++) {
      for (let i = -12; i <= 12; i++) {
        const x = i + (j & 1) * 0.5;
        const y = (j * Math.sqrt(3)) / 2;
        if (Math.hypot(x, y) <= 8) spawn(s, k++, x, y, 0, CELL);
      }
    }
    const sh = projectedShape(s, CELL);
    expect(sh.count).toBe(k);
    expect(sh.circularity).toBeGreaterThan(0.93);
    expect(sh.area).toBeGreaterThan(Math.PI * 8 * 8 * 0.95);
    expect(sh.area).toBeLessThan(Math.PI * 8.5 * 8.5 * 1.05);
  });

  it("scores a cross far below a disc", () => {
    const s = createState(400);
    let k = 0;
    for (let i = -20; i <= 20; i++) {
      for (let w = -1; w <= 1; w++) {
        spawn(s, k++, i, w * 0.9, 0, CELL);
        if (Math.abs(i) > 1) spawn(s, k++, w * 0.9, i, 0, CELL);
      }
    }
    expect(projectedShape(s, CELL).circularity).toBeLessThan(0.35);
  });
});

describe("phase diagram", () => {
  it("runs every grid point and reports its shape", () => {
    const points = phaseDiagram({
      x: { key: "yieldStrain", values: [0.1, Number.POSITIVE_INFINITY] },
      y: { key: "motility", values: [0, 0.05] },
      base: { cellRadius: 2, matrixRadius: 7, maxCells: 120, divisionTime: 100 },
      steps: 300,
    });
    expect(points.map((q) => [q.ix, q.iy])).toEqual([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    for (const q of points) {
      expect(q.shape.circularity).toBeGreaterThan(0);
      expect(q.growth).toBeGreaterThan(0.9);
    }
    // The elastic column never breaks a bond.
    expect(points[1]!.brokenBonds).toBe(0);
    expect(points[0]!.brokenBonds).toBeGreaterThan(0);
  });
});

describe("scenarios", () => {
  it("relaxation: a clamped strip holds stress when elastic and sheds it with turnover", () => {
    const stress = (sc: Scenario) => bondReadout(sc.state, sc.topology, sc.params).virial[0];
    const elastic = relaxationScenario({ cols: 16, rows: 8 });
    const viscous = relaxationScenario({ cols: 16, rows: 8, bondLifetime: 30 });
    elastic.advance(60);
    viscous.advance(60);
    const settled = stress(elastic);
    elastic.advance(120);
    viscous.advance(120);
    expect(stress(elastic)).toBeGreaterThan(settled * 0.98);
    expect(stress(viscous)).toBeLessThan(settled * 0.05);
  });

  it("chromatin: recovers a known curve from its distance matrix", () => {
    const sc = chromatinScenario({ n: 60, mode: "distances" });
    const ref = sc.matrices!.dist!;
    expect(distanceCorrelation(distanceMatrix(sc.state, 60), ref, 60)).toBeLessThan(0.3);
    sc.advance(600);
    expect(distanceCorrelation(distanceMatrix(sc.state, 60), ref, 60)).toBeGreaterThan(0.99);
  });

  it("chromatin: recovers it from contacts alone", () => {
    const sc = chromatinScenario({ n: 60, mode: "contacts" });
    sc.advance(1500);
    expect(distanceCorrelation(distanceMatrix(sc.state, 60), sc.matrices!.dist!, 60)).toBeGreaterThan(0.98);
  });

  it("spheroid: settings change on a running tissue without rebuilding it", () => {
    const base = { cellRadius: 3, matrixRadius: 11, maxCells: 400, divisionTime: 150 };
    const sc = spheroidScenario({ ...base, yieldStrain: Number.POSITIVE_INFINITY });
    const state = sc.state;
    sc.advance(600);
    expect(sc.totals.broken).toBe(0);
    const stalled = liveCount(state, CELL);
    // Let the matrix yield: the same particles carry on, and now bonds break and cells divide on.
    // A key present but undefined leaves the setting alone.
    sc.configure({ yieldStrain: 0.1, stiffness: undefined });
    expect(sc.config.stiffness).toBe(0.3);
    expect(liveCount(spheroidScenario({ cellRadius: undefined, maxCells: 80 }).state, CELL)).toBe(61);
    expect(sc.config.yieldStrain).toBe(0.1);
    expect(sc.config.matrixRadius).toBe(11);
    sc.advance(900);
    expect(sc.state).toBe(state);
    expect(sc.totals.broken).toBeGreaterThan(0);
    expect(liveCount(state, CELL)).toBeGreaterThan(stalled);
    // Stiffness is a multiplier, so it reaches every existing bond at once.
    const before = bondReadout(state, sc.topology, sc.params, 1 << 1).virial[0];
    sc.configure({ stiffness: sc.config.stiffness * 2 });
    expect(bondReadout(state, sc.topology, sc.params, 1 << 1).virial[0]).toBeCloseTo(before * 2, 6);
    // Stopping division stops it.
    sc.configure({ divisionTime: Number.POSITIVE_INFINITY });
    const divisions = sc.totals.divisions;
    sc.advance(200);
    expect(sc.totals.divisions).toBe(divisions);
  });

  it("relaxation: turning bond turnover on mid-run starts the stress falling", () => {
    const sc = relaxationScenario({ cols: 16, rows: 8 });
    sc.advance(80);
    const held = bondReadout(sc.state, sc.topology, sc.params).virial[0];
    sc.configure({ bondLifetime: 30 });
    sc.advance(120);
    expect(bondReadout(sc.state, sc.topology, sc.params).virial[0]).toBeLessThan(held * 0.1);
  });

  it("spheroid in 3D: a close-packed ball in a bonded block, with disorder leaving it stress-free", () => {
    const ordered = spheroidScenario({ dims: 3, cellRadius: 2, matrixRadius: 5, maxCells: 100 });
    expect(ordered.params.dims).toBe(3);
    expect(ordered.config.maxDensity).toBe(22);
    // Face-centred cubic: an interior matrix particle has twelve bonds.
    const counts = new Map<number, number>();
    for (let i = 100; i < ordered.state.n; i++)
      counts.set(bondsOf(ordered.topology, i).length, (counts.get(bondsOf(ordered.topology, i).length) ?? 0) + 1);
    expect(Math.max(...counts.keys())).toBe(12);
    const scattered = spheroidScenario({ dims: 3, cellRadius: 2, matrixRadius: 5, maxCells: 100, matrixDisorder: 0.2 });
    expect(Math.abs(bondReadout(scattered.state, scattered.topology, scattered.params).meanStrain)).toBeLessThan(1e-6);
    expect(topologyDefect(scattered.topology)).toBeUndefined();
    scattered.advance(20);
    expect(liveCount(scattered.state, CELL)).toBeGreaterThan(20);
  });

  it("spheroid: settings are in model time, and the step turns them into per-step values", () => {
    const cfg = { divisionTime: 100, leaderTime: 40, leaderThrust: 0.5, bondLifetime: 50, reformProb: 0.1, motility: 0.04 };
    const sc = spheroidScenario({ ...cfg, dt: 0.5 });
    const { rules, params } = sc;
    expect([params.springrate, rules.division?.tau, rules.leaders?.tau, rules.breakRule.tau, params.motility[CELL]]).toEqual([
      0.5,
      200,
      80,
      100,
      0.04 / Math.sqrt(0.5),
    ]);
    // Two steps of half a unit re-bond as often as one step of a whole unit.
    expect(1 - (1 - (rules.formRule.prob ?? 0)) ** 2).toBeCloseTo(0.1, 12);
    expect(() => sc.configure({ dt: 0 })).toThrow(RangeError);
    expect(spheroidDefaults(2).dt).toBe(1);
  });

  it("spheroid in 3D: its default step is short enough that contacts do not flip every step", () => {
    /** Movement over two steps against the sum of the two single steps, over the run's last few
     *  steps: about 2 for a steady drift, near 0 for particles flipping back and forth. */
    const smoothness = (dt: number) => {
      const sc = spheroidScenario({ dims: 3, cellRadius: 1.5, matrixRadius: 4, matrixDisorder: 0.2, seed: 1 });
      sc.configure({ stiffness: 0.3, cellRepulsion: 0.5, cellAdhesion: 0.1, motility: 0.02, adhesionRange: 0.25, dt });
      const steps = Math.round(20 / dt);
      sc.advance(steps - 2);
      const frames = [sc.state.pos.slice()];
      for (let k = 0; k < 2; k++) {
        sc.advance(1);
        frames.push(sc.state.pos.slice());
      }
      const [a, b, c] = frames as [Float32Array, Float32Array, Float32Array];
      const by: Record<string, [number, number]> = { cell: [0, 0], matrix: [0, 0] };
      for (let i = 0; i < sc.state.n; i++) {
        const sp = sc.state.species[i];
        const row = sp === CELL ? by.cell : sp === MATRIX ? by.matrix : undefined;
        if (!row) continue;
        const d = (x: Float32Array, y: Float32Array) =>
          Math.hypot(x[i * 4]! - y[i * 4]!, x[i * 4 + 1]! - y[i * 4 + 1]!, x[i * 4 + 2]! - y[i * 4 + 2]!);
        row[0] += d(a, c);
        row[1] += d(a, b) + d(b, c);
      }
      return { cell: (2 * by.cell![0]) / by.cell![1], matrix: (2 * by.matrix![0]) / by.matrix![1] };
    };
    const long = smoothness(1);
    const short = smoothness(spheroidDefaults(3).dt);
    expect(
      { long: long.cell < 0.5 && long.matrix < 0.5, short: short.cell > 1.2 && short.matrix > 1.2 },
      JSON.stringify({ long, short }),
    ).toEqual({
      long: true,
      short: true,
    });
  });

  it("spheroid: a yielding matrix lets the tissue outgrow an elastic one", () => {
    const base = { cellRadius: 3, matrixRadius: 11, maxCells: 400, divisionTime: 150 };
    const elastic = spheroidScenario({ ...base, yieldStrain: Number.POSITIVE_INFINITY });
    const yielding = spheroidScenario({ ...base, yieldStrain: 0.1 });
    elastic.advance(1500);
    yielding.advance(1500);
    expect(elastic.totals.broken).toBe(0);
    expect(yielding.totals.broken).toBeGreaterThan(0);
    expect(topologyDefect(yielding.topology)).toBeUndefined();
    // Every break and division is reported once, with a position, and then the log is empty.
    const events = yielding.takeEvents();
    expect(events.filter((e) => e.type === "break").length).toBe(yielding.totals.broken);
    expect(events.filter((e) => e.type === "divide").length).toBe(yielding.totals.divisions);
    expect(events.every((e) => Math.hypot(e.x, e.y) < 12 && e.step > 0)).toBe(true);
    expect(yielding.takeEvents()).toEqual([]);
    expect(projectedShape(yielding.state, CELL).area).toBeGreaterThan(projectedShape(elastic.state, CELL).area * 1.15);
  });
});

describe("review fixes", () => {
  it("rejects a phase diagram whose axes vary the same setting", () => {
    const axis = { key: "motility", values: [0, 0.1] } as const;
    expect(() => phaseDiagram({ x: axis, y: axis, steps: 1 })).toThrow(/both axes/);
  });

  it("rejects a Sholl shell thickness that would never terminate", () => {
    const s = createState(3);
    spawn(s, 0, 0, 0, 0, CELL);
    spawn(s, 1, 2, 0, 0, CELL);
    spawn(s, 2, -2, 0, 0, CELL);
    expect(() => shollProfile(s, CELL, { thickness: 0 })).toThrow(RangeError);
  });

  it("turns leaders back into cells once leader selection is switched off", () => {
    const sc = spheroidScenario({ cellRadius: 2, matrixRadius: 4, leaderThrust: 0.5, seed: 3 });
    const cell = sc.state.species.indexOf(CELL);
    sc.state.species[cell] = LEADER;
    sc.configure({ leaderThrust: 0 });
    sc.advance(10);
    expect(liveCount(sc.state, LEADER)).toBe(0);
    expect(sc.state.species[cell]).toBe(CELL);
  });

  it("keeps structural settings out of a live update", () => {
    const sc = spheroidScenario({ cellRadius: 2, matrixRadius: 4, seed: 7 });
    const everything: Record<string, unknown> = { seed: 1, dims: 3, stiffness: 0.9 };
    sc.configure(everything);
    expect(sc.config.seed).toBe(7);
    expect(sc.config.dims).toBe(2);
    expect(sc.config.stiffness).toBe(0.9);
  });

  it("reaches far enough for the configured adhesion", () => {
    const sc = spheroidScenario({ cellRadius: 2, matrixRadius: 4, adhesionRange: 1 });
    expect(sc.params.pairRange).toBeGreaterThanOrEqual(2);
  });

  it("removes line springs by the kind they were installed with", () => {
    const t = createTopology(12, 8);
    const first = applyLines(t, [{ a0: 0, b0: 6, a1: 2, b1: 8 }], { ...DEFAULT_LINE_SPRINGS, kind: 3 });
    expect(first.pairs.length).toBe(3);
    applyLines(t, [], { ...DEFAULT_LINE_SPRINGS, kind: 4 }, first);
    expect(bondCount(t)).toBe(0);
  });
});
