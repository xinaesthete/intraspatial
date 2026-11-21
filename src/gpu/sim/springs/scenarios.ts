// Ready-made setups of the spring model. Each returns the state, topology and parameters plus
// an `advance` that runs the step and whatever turnover and growth the setup uses, so a caller
// (a test, a sweep, the playground) drives every scenario the same way.
import { divideCells, type DivisionRule, type LeaderRule, selectLeaders } from "./growth";
import { MAX_SPECIES, type SpringParams, speciesTable, springParams } from "./params";
import { rand01 } from "./rng";
import { createState, gatherNeighbours, neighbourGrid, type SpringMatrices, type SpringState, spawn } from "./state";
import { stepSprings } from "./step";
import { addSpring, createTopology, setFix, type SpringTopology } from "./topology";
import { type BreakRule, breakBonds, type FormRule, formBonds } from "./turnover";

/** Something that happened at a place and a step: what a sound engine, or any other listener
 *  for discrete events, triggers on. */
export interface SpringEvent {
  readonly type: "break" | "divide";
  readonly step: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** The settings of `o` that are present: a key given as `undefined` means "leave the default",
 *  not "set it to undefined", which a plain spread would do. */
function defined<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

/** Events a scenario holds before the oldest are dropped. */
const MAX_PENDING_EVENTS = 4000;

function eventLog(state: SpringState) {
  let pending: SpringEvent[] = [];
  const push = (type: SpringEvent["type"], a: number, b: number) => {
    if (pending.length >= MAX_PENDING_EVENTS) pending.shift();
    pending.push({
      type,
      step: state.step,
      x: (state.pos[a * 4]! + state.pos[b * 4]!) / 2,
      y: (state.pos[a * 4 + 1]! + state.pos[b * 4 + 1]!) / 2,
      z: (state.pos[a * 4 + 2]! + state.pos[b * 4 + 2]!) / 2,
    });
  };
  return {
    onBreak: (i: number, j: number) => push("break", i, j),
    onDivide: (parent: number, daughter: number) => push("divide", parent, daughter),
    take: (): SpringEvent[] => {
      const out = pending;
      pending = [];
      return out;
    },
  };
}

export interface Scenario {
  readonly kind: "chromatin" | "chain" | "spheroid" | "relaxation";
  readonly state: SpringState;
  readonly topology: SpringTopology;
  params: SpringParams;
  readonly matrices?: SpringMatrices;
  /** Run `steps` steps, with the scenario's turnover and growth. */
  advance(steps: number): void;
  /** Running totals since the scenario was made. */
  readonly totals: { broken: number; formed: number; divisions: number };
  /** Bond breaks and divisions since the last call, oldest first, each at the midpoint of the
   *  two particles involved. */
  takeEvents(): SpringEvent[];
}

// ── Chromatin: CSynth's own problem ───────────────────────────────────────────────────────────

export interface ChromatinConfig {
  /** Particles in the chain. */
  readonly n?: number;
  /** `distances` springs every pair to its target distance (`xyzforce`); `contacts` uses the
   *  contact attraction against a global repulsion, as CSynth does with Hi-C data. */
  readonly mode?: "distances" | "contacts";
  readonly seed?: number;
}

/** The curve the chromatin scenario tries to recover: a helix wound on a torus. */
export function referenceCurve(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const R = n / 14;
  const r = R * 0.45;
  for (let i = 0; i < n; i++) {
    const u = (i / n) * 2 * Math.PI;
    const v = u * 6;
    out[i * 3] = (R + r * Math.cos(v)) * Math.cos(u);
    out[i * 3 + 1] = (R + r * Math.cos(v)) * Math.sin(u);
    out[i * 3 + 2] = r * Math.sin(v);
  }
  return out;
}

/**
 * A chain that starts as a random tangle and is driven towards a known 3D curve, given only
 * that curve's distance (or contact) matrix. Because the answer is known, the reconstruction
 * can be scored: see `distanceCorrelation`.
 */
export function chromatinScenario(cfg: ChromatinConfig = {}): Scenario {
  const n = cfg.n ?? 200;
  const mode = cfg.mode ?? "distances";
  const seed = cfg.seed ?? 1;
  const ref = referenceCurve(n);
  const dist = new Float32Array(n * n);
  const contact = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = Math.hypot(ref[j * 3]! - ref[i * 3]!, ref[j * 3 + 1]! - ref[i * 3 + 1]!, ref[j * 3 + 2]! - ref[i * 3 + 2]!);
      dist[i * n + j] = d;
      contact[i * n + j] = 1 / d;
    }
  }
  const state = createState(n);
  const spread = n / 10;
  for (let i = 0; i < n; i++) {
    spawn(state, i, (rand01(i, seed, 1) - 0.5) * spread, (rand01(i, seed, 2) - 0.5) * spread, (rand01(i, seed, 3) - 0.5) * spread);
  }
  const topology = createTopology(n, 2);
  const common: Partial<SpringParams> = { seed, damp: 0.9, damptype: 1, springmaxvel: 1, pushapartlocalforce: 0, powBaseDist: 1 };
  const params =
    mode === "distances"
      ? springParams({ ...common, xyzforce: 0.4 / n, xyzpow: 0, xyzMaxDist: 1e9 })
      : // Attraction `c · contact · len` balances repulsion `p` where `len = p / (c · contact)`;
        // with `contact = 1 / d` and `p = c` that distance is the target `d`.
        springParams({ ...common, contactforcesc: 0.4 / n, pushapartforce: 0.4 / n, pushapartpow: 0, ignoreBackbone: 1 });
  const matrices: SpringMatrices = { size: n, dist, contact };
  const totals = { broken: 0, formed: 0, divisions: 0 };
  const scenario: Scenario = {
    kind: "chromatin",
    state,
    topology,
    params,
    matrices,
    totals,
    takeEvents: () => [],
    advance(steps) {
      for (let k = 0; k < steps; k++) stepSprings(state, topology, scenario.params, matrices);
    },
  };
  return scenario;
}

// ── Spheroid in a matrix: Elosegui-Artola et al. 2023 ─────────────────────────────────────────

/** Species in the spheroid scenario. */
export const CELL = 0;
export const MATRIX = 1;
/** A cell that has become a tip cell. */
export const LEADER = 2;
/** Bond kinds in the spheroid scenario. */
export const KIND_MATRIX = 1;
export const KIND_LIGAND = 2;

export interface SpheroidConfig {
  /** Radius of the initial ball of cells, in cell diameters. */
  readonly cellRadius?: number;
  /** Outer radius of the matrix; its outermost ring is pinned. */
  readonly matrixRadius?: number;
  /** Stiffness of matrix bonds: the model's storage modulus. */
  readonly stiffness?: number;
  /** A matrix bond stretched past this strain breaks; `Infinity` is an elastic matrix. */
  readonly yieldStrain?: number;
  /** Mean lifetime of a matrix bond in steps, regardless of load; `Infinity` is none. */
  readonly bondLifetime?: number;
  /** Matrix bonds re-form between particles closer than this. */
  readonly reformRange?: number;
  /** Chance per step that an eligible pair re-bonds. Low values leave broken matrix weak for a
   *  while, so damage stays where it happened. */
  readonly reformProb?: number;
  /** Stiffness of the contact repulsion between matrix particles: how hard the matrix is to
   *  compress, as opposed to shear. */
  readonly matrixRepulsion?: number;
  /** Mobility of matrix particles relative to cells: the inverse of the viscosity ratio. */
  readonly matrixMobility?: number;
  /** Stiffness of the contact repulsion between cells. It has to outweigh adhesion, or the
   *  tissue crushes itself. */
  readonly cellRepulsion?: number;
  /** Cell–cell adhesion strength. */
  readonly cellAdhesion?: number;
  /** How far past touching adhesion reaches, as a fraction of a cell diameter. With a long
   *  reach, second neighbours pull too, and strong adhesion crushes the tissue. */
  readonly adhesionRange?: number;
  /** Strength of cell–matrix ligand bonds (the paper's RGD); 0 is a matrix cells cannot grip. */
  readonly ligand?: number;
  /** Random force on cells: the paper's motility. */
  readonly motility?: number;
  /** Outward push of every cell at the tissue surface, stronger the more exposed the cell is,
   *  so a bump pushes harder than the flat around it; see `SpringParams.protrusion`. */
  readonly protrusion?: number;
  /** How far a cell looks for its own kind when working out which way is outward, in cell
   *  diameters. Larger values make broader branches. */
  readonly protrusionRange?: number;
  /** Force a leader cell pulls outward with; 0 is no leaders. Exposed surface cells become
   *  leaders, as tip cells do in a sprouting vessel. */
  readonly leaderThrust?: number;
  /** How much more strongly a leader holds on to its followers than cells hold each other.
   *  Too little and a leader tears free and leaves alone. */
  readonly leaderGrip?: number;
  /** Mean steps before an exposed cell becomes a leader. */
  readonly leaderTime?: number;
  /** No new leader forms within this distance of another: how far apart branches start. */
  readonly leaderSpacing?: number;
  /** Whether a matrix bond also breaks when squeezed past the yield strain. Off, the matrix
   *  only fails where it is pulled apart, which makes it crack rather than crumble. */
  readonly yieldInCompression?: boolean;
  /** How far matrix particles are scattered from their lattice sites, in diameters. Bonds take
   *  their scattered lengths as rest lengths, so the matrix starts stress-free. A perfect lattice
   *  yields along its crystal planes and the tissue grows faceted; disorder removes that. */
  readonly matrixDisorder?: number;
  /** 2 for a sheet, 3 for a ball in a block of matrix. */
  readonly dims?: 2 | 3;
  /** Mean steps between divisions of an uncrowded cell; `Infinity` is no growth. */
  readonly divisionTime?: number;
  /** Density above which a cell does not divide. Default 9.5 in 2D and 22 in 3D: both are a
   *  close-packed cell squeezed about 14% closer to its neighbours. */
  readonly maxDensity?: number;
  /** Most cells the run can hold. */
  readonly maxCells?: number;
  readonly seed?: number;
}

/** Settings that fix the particles a spheroid is built from. Changing one needs a new scenario. */
export type SpheroidStructure = "cellRadius" | "matrixRadius" | "maxCells" | "seed" | "dims" | "matrixDisorder";
/** Settings that can change while a spheroid runs. */
export type SpheroidLiveConfig = Omit<SpheroidConfig, SpheroidStructure>;

export interface SpheroidScenario extends Scenario {
  /** The settings in force now. */
  readonly config: Required<SpheroidConfig>;
  /** The turnover and growth rules those settings give, for a driver that runs the steps itself. */
  readonly rules: SpheroidRules;
  /** Change settings on the running tissue. Takes effect from the next step: existing bonds get
   *  the new stiffness, and the new yield, healing and division rules apply to what is there. */
  configure(change: SpheroidLiveConfig): void;
}

/** Defaults sit in the regime where the outline breaks symmetry: a stiff matrix that yields at
 *  a high strain and heals slowly, and motile cells. */
export const DEFAULT_SPHEROID: Required<SpheroidConfig> = {
  cellRadius: 4,
  matrixRadius: 18,
  stiffness: 0.3,
  yieldStrain: 0.3,
  bondLifetime: Number.POSITIVE_INFINITY,
  reformRange: 1.2,
  reformProb: 0.0005,
  matrixRepulsion: 0.05,
  matrixMobility: 1,
  cellRepulsion: 0.3,
  cellAdhesion: 0.02,
  adhesionRange: 0.5,
  ligand: 0.02,
  motility: 0.08,
  protrusion: 0,
  protrusionRange: 1.5,
  leaderThrust: 0,
  leaderGrip: 3,
  leaderTime: 200,
  leaderSpacing: 5,
  yieldInCompression: true,
  matrixDisorder: 0,
  dims: 2,
  divisionTime: 400,
  maxDensity: 9.5,
  maxCells: 600,
  seed: 1,
};

/**
 * A 2D ball of dividing, motile, mutually adhesive cells inside a bonded matrix. Matrix
 * particles sit on a triangular lattice and are bonded to their neighbours; those bonds break
 * past a yield strain (and, optionally, at random) and re-form stress-free with whoever is then
 * nearby. With breaking off the matrix is elastic and confines the cells; with it on the matrix
 * creeps, and the cells can push into it.
 */
export function spheroidScenario(cfg: SpheroidConfig = {}): SpheroidScenario {
  let c = { ...DEFAULT_SPHEROID, ...defined(cfg) };
  // A close-packed cell has 6 touching neighbours in 2D and 12 in 3D, so the same squeeze reads
  // as a higher density in 3D.
  if (cfg.maxDensity === undefined && c.dims === 3) c = { ...c, maxDensity: 22 };
  // Close-packed sites one diameter apart: a triangular lattice in 2D, face-centred cubic in 3D.
  const sites: [number, number, number][] = [];
  const reach = Math.ceil(c.matrixRadius) + 1;
  if (c.dims === 2) {
    const rows = Math.ceil(c.matrixRadius / (Math.sqrt(3) / 2));
    for (let j = -rows; j <= rows; j++) {
      for (let i = -reach; i <= reach; i++) sites.push([i + (j & 1) * 0.5, (j * Math.sqrt(3)) / 2, 0]);
    }
  } else {
    const a = Math.SQRT1_2;
    const span = Math.ceil(reach / a);
    for (let k = -span; k <= span; k++) {
      for (let j = -span; j <= span; j++) {
        for (let i = -span; i <= span; i++) if ((i + j + k) % 2 === 0) sites.push([i * a, j * a, k * a]);
      }
    }
  }
  const inside = sites.filter(([x, y, z]) => Math.hypot(x, y, z) <= c.matrixRadius);
  const isCell = ([x, y, z]: [number, number, number]) => Math.hypot(x, y, z) <= c.cellRadius;
  const cells = inside.filter(isCell);
  const matrix = inside.filter((p) => !isCell(p));
  const cellCap = Math.max(c.maxCells, cells.length);
  const n = cellCap + matrix.length;

  const state = createState(n);
  // Room for a full shell of lattice neighbours (6 or 12) plus re-formed and ligand bonds.
  const topology = createTopology(n, c.dims === 2 ? 8 : 16);
  cells.forEach(([x, y, z], k) => spawn(state, k, x, y, z, CELL));
  matrix.forEach(([x, y, z], k) => {
    const i = cellCap + k;
    const jitter = (salt: number) => (rand01(i, c.seed, salt) - 0.5) * 2 * c.matrixDisorder;
    const px = x + jitter(11);
    const py = y + jitter(12);
    const pz = c.dims === 3 ? z + jitter(13) : 0;
    spawn(state, i, px, py, pz, MATRIX);
    if (Math.hypot(x, y, z) > c.matrixRadius - 1) setFix(topology, i, px, py, pz);
  });
  // Bond each matrix particle to those within reach: lattice neighbours, however scattered.
  const bondReach = 1.01 + 2 * c.matrixDisorder;
  const grid = neighbourGrid(state, bondReach, c.dims);
  const near = new Int32Array(n);
  for (let a = cellCap; a < n; a++) {
    const count = gatherNeighbours(grid, state.pos[a * 4]!, state.pos[a * 4 + 1]!, state.pos[a * 4 + 2]!, near);
    for (let q = 0; q < count; q++) {
      const b = near[q]!;
      if (b <= a || state.species[b] !== MATRIX) continue;
      const d = Math.hypot(
        state.pos[a * 4]! - state.pos[b * 4]!,
        state.pos[a * 4 + 1]! - state.pos[b * 4 + 1]!,
        state.pos[a * 4 + 2]! - state.pos[b * 4 + 2]!,
      );
      if (d < bondReach) addSpring(topology, a, b, d, 1, 0, KIND_MATRIX);
    }
  }

  const base = springParams({
    seed: c.seed,
    dims: c.dims,
    // No inertia: velocity is force times mobility, the overdamped limit the paper models.
    damp: 0,
    damptype: 1,
    springmaxvel: 0.2,
    springforce: 1,
    backboneStrength: 1,
    pushapartlocalforce: 0,
    // A soft wall at the edge of the matrix: past it there is nothing for a cell to push on.
    modelSphereForce: 0.5,
    modelSphereRadius: c.matrixRadius - 0.5,
    // Density is the crowding signal for division, so it must not change with `pairRange`.
    densityRange: 1.5,
  });

  const totals = { broken: 0, formed: 0, divisions: 0 };
  const events = eventLog(state);
  let rules = spheroidRules(c);
  const scenario: SpheroidScenario = {
    kind: "spheroid",
    state,
    topology,
    params: { ...base, ...spheroidParams(c) },
    totals,
    takeEvents: events.take,
    get config() {
      return c;
    },
    get rules() {
      return rules;
    },
    configure(change) {
      c = { ...c, ...defined(change) };
      rules = spheroidRules(c);
      scenario.params = { ...scenario.params, ...spheroidParams(c) };
    },
    advance(steps) {
      for (let k = 0; k < steps; k++) {
        stepSprings(state, topology, scenario.params);
        if (rules.turnover) totals.broken += breakBonds(state, topology, scenario.params, rules.breakRule, events.onBreak);
        if (rules.turnover || c.ligand > 0) totals.formed += formBonds(state, topology, scenario.params, rules.formRule);
        if (rules.division) totals.divisions += divideCells(state, scenario.params, rules.division, events.onDivide);
        // Selection needs a neighbour search, so it runs every few steps at a rate to match.
        if (rules.leaders && state.step % LEADER_EVERY === 0)
          selectLeaders(state, scenario.params, { ...rules.leaders, tau: Math.max(1, rules.leaders.tau / LEADER_EVERY) });
      }
    },
  };
  return scenario;
}

/** The part of `SpringParams` the spheroid's settings decide. Bond strengths live here, as
 *  per-kind multipliers, so that stiffness can change without touching a single bond. */
function spheroidParams(c: Required<SpheroidConfig>): Partial<SpringParams> {
  const mobility = Array.from({ length: MAX_SPECIES }, () => 1);
  mobility[MATRIX] = c.matrixMobility;
  const motility = Array.from({ length: MAX_SPECIES }, () => 0);
  motility[CELL] = c.motility;
  const roleforces = Array.from({ length: 8 }, () => 1);
  roleforces[KIND_MATRIX] = c.stiffness;
  roleforces[KIND_LIGAND] = c.ligand;
  const protrusion = Array.from({ length: MAX_SPECIES }, () => 0);
  protrusion[CELL] = c.protrusion;
  const thrust = Array.from({ length: MAX_SPECIES }, () => 0);
  thrust[LEADER] = c.leaderThrust;
  motility[LEADER] = c.motility;
  return {
    pairRange: Math.max(1.5, c.protrusion !== 0 || c.leaderThrust !== 0 ? c.protrusionRange : 0),
    adhesionRange: c.adhesionRange,
    protrusion,
    // Summed weight of a full shell of neighbours out to two diameters, in each packing.
    protrusionFull: c.dims === 2 ? 3.8 : 11,
    thrust,
    tissueMask: (1 << CELL) | (1 << LEADER),
    protrusionRange: c.protrusionRange,
    // A leader is a cell: it repels and adheres as one.
    repulsion: speciesTable([
      [CELL, CELL, c.cellRepulsion],
      [CELL, LEADER, c.cellRepulsion],
      [LEADER, LEADER, c.cellRepulsion],
      [CELL, MATRIX, 0.3],
      [LEADER, MATRIX, 0.3],
      [MATRIX, MATRIX, c.matrixRepulsion],
    ]),
    adhesion: speciesTable([
      [CELL, CELL, c.cellAdhesion],
      [CELL, LEADER, c.cellAdhesion * c.leaderGrip],
      [LEADER, LEADER, c.cellAdhesion * c.leaderGrip],
    ]),
    mobility,
    motility,
    roleforces,
  };
}

export interface SpheroidRules {
  readonly breakRule: BreakRule;
  readonly formRule: FormRule;
  readonly division: DivisionRule | undefined;
  readonly leaders: LeaderRule | undefined;
  /** Whether any bond can break. */
  readonly turnover: boolean;
}

/** The turnover and growth rules the spheroid's settings decide. */
function spheroidRules(c: Required<SpheroidConfig>): SpheroidRules {
  const breakRule: BreakRule = {
    kinds: (1 << KIND_MATRIX) | (1 << KIND_LIGAND),
    maxStrain: Number.isFinite(c.yieldStrain) ? c.yieldStrain : undefined,
    tensionOnly: !c.yieldInCompression,
    tau: Number.isFinite(c.bondLifetime) ? c.bondLifetime : undefined,
  };
  const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => -1);
  kindOf[MATRIX * MAX_SPECIES + MATRIX] = KIND_MATRIX;
  if (c.ligand > 0) {
    for (const cell of [CELL, LEADER]) {
      kindOf[cell * MAX_SPECIES + MATRIX] = KIND_LIGAND;
      kindOf[MATRIX * MAX_SPECIES + cell] = KIND_LIGAND;
    }
  }
  // Every bond has strength 1; its kind's multiplier in `roleforces` is the real strength.
  const formRule: FormRule = {
    range: c.reformRange,
    kindOf,
    strength: UNIT_STRENGTHS,
    rest: "current",
    prob: c.reformProb,
    speciesCount: MAX_SPECIES,
  };
  const division: DivisionRule | undefined = Number.isFinite(c.divisionTime)
    ? { species: CELL, tau: c.divisionTime, maxDensity: c.maxDensity, offset: 0.5 }
    : undefined;
  // A full shell is 6 neighbours in 2D and 12 in 3D. A surface cell of a packed tissue has about
  // two thirds of that, so that is the limit for becoming a leader; only a leader with a full
  // shell, buried in the tissue, goes back.
  const shell = c.dims === 2 ? 6 : 12;
  const leaders: LeaderRule | undefined =
    c.leaderThrust !== 0
      ? { from: CELL, to: LEADER, tau: c.leaderTime, maxNeighbours: (shell * 2) / 3, spacing: c.leaderSpacing, revertNeighbours: shell }
      : undefined;
  return { breakRule, formRule, division, leaders, turnover: breakRule.maxStrain !== undefined || breakRule.tau !== undefined };
}

/** Steps between rounds of leader selection on the CPU path. */
const LEADER_EVERY = 10;

const UNIT_STRENGTHS: readonly number[] = Array.from({ length: 8 }, () => 1);

// ── Stress relaxation: the rheometer test ─────────────────────────────────────────────────────

export interface RelaxationConfig {
  readonly cols?: number;
  readonly rows?: number;
  /** Stretch applied along x at the start and then held. */
  readonly strain?: number;
  readonly stiffness?: number;
  /** Mean bond lifetime in steps; `Infinity` is an elastic network. */
  readonly bondLifetime?: number;
  /** Strain past which a bond breaks; `Infinity` is never. */
  readonly yieldStrain?: number;
  readonly seed?: number;
}

/** The relaxation test's settings that can change while it runs. */
export type RelaxationLiveConfig = Pick<RelaxationConfig, "stiffness" | "bondLifetime" | "yieldStrain">;

export interface RelaxationScenario extends Scenario {
  readonly config: Required<RelaxationLiveConfig>;
  /** Change settings on the running strip, from the next step. */
  configure(change: RelaxationLiveConfig): void;
}

/**
 * A strip of bonded matrix, stretched and held by clamps at both ends: the test the paper runs
 * on its gels to measure how fast they relax. The stress the strip carries is
 * `bondReadout(...).virial[0]`. With no turnover it stays put; with bonds of lifetime `tau` it
 * halves every `tau · ln 2` steps.
 */
export function relaxationScenario(cfg: RelaxationConfig = {}): RelaxationScenario {
  const cols = cfg.cols ?? 28;
  const rows = cfg.rows ?? 14;
  const strain = cfg.strain ?? 0.12;
  let live: Required<RelaxationLiveConfig> = {
    stiffness: cfg.stiffness ?? 0.1,
    bondLifetime: cfg.bondLifetime ?? Number.POSITIVE_INFINITY,
    yieldStrain: cfg.yieldStrain ?? Number.POSITIVE_INFINITY,
  };
  const n = cols * rows;
  const state = createState(n);
  const topology = createTopology(n, 8);
  const at = (i: number, j: number): [number, number] => [i + (j & 1) * 0.5 - cols / 2, ((j - rows / 2) * Math.sqrt(3)) / 2];
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) spawn(state, j * cols + i, ...at(i, j), 0, MATRIX);
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      const d = Math.hypot(state.pos[a * 4]! - state.pos[b * 4]!, state.pos[a * 4 + 1]! - state.pos[b * 4 + 1]!);
      if (d < 1.01) addSpring(topology, a, b, d, 1, 0, KIND_MATRIX);
    }
  }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      const x = state.pos[k * 4]! * (1 + strain);
      state.pos[k * 4] = x;
      state.prev[k * 4] = x;
      if (i < 2 || i >= cols - 2) setFix(topology, k, x, state.pos[k * 4 + 1]!, 0);
    }
  }
  const stiffnessParams = (stiffness: number): Partial<SpringParams> => {
    const roleforces = Array.from({ length: 8 }, () => 1);
    roleforces[KIND_MATRIX] = stiffness;
    return { roleforces };
  };
  const kindOf = Array.from({ length: MAX_SPECIES * MAX_SPECIES }, () => -1);
  kindOf[MATRIX * MAX_SPECIES + MATRIX] = KIND_MATRIX;
  const formRule: FormRule = { range: 1.25, kindOf, strength: UNIT_STRENGTHS, rest: "current", speciesCount: MAX_SPECIES };
  const breakRuleFor = (c: Required<RelaxationLiveConfig>): BreakRule => ({
    kinds: 1 << KIND_MATRIX,
    maxStrain: Number.isFinite(c.yieldStrain) ? c.yieldStrain : undefined,
    tau: Number.isFinite(c.bondLifetime) ? c.bondLifetime : undefined,
  });
  let breakRule = breakRuleFor(live);
  const totals = { broken: 0, formed: 0, divisions: 0 };
  const events = eventLog(state);
  const scenario: RelaxationScenario = {
    kind: "relaxation",
    state,
    topology,
    params: springParams({
      seed: cfg.seed ?? 1,
      dims: 2,
      damp: 0,
      damptype: 1,
      springmaxvel: 0.2,
      springforce: 1,
      backboneStrength: 1,
      pushapartlocalforce: 0,
      pairRange: 1.5,
      repulsion: speciesTable([[MATRIX, MATRIX, 0.05]]),
      ...stiffnessParams(live.stiffness),
    }),
    totals,
    takeEvents: events.take,
    get config() {
      return live;
    },
    configure(change) {
      live = { ...live, ...defined(change) };
      breakRule = breakRuleFor(live);
      scenario.params = { ...scenario.params, ...stiffnessParams(live.stiffness) };
    },
    advance(steps) {
      for (let k = 0; k < steps; k++) {
        stepSprings(state, topology, scenario.params);
        if (breakRule.maxStrain === undefined && breakRule.tau === undefined) continue;
        totals.broken += breakBonds(state, topology, scenario.params, breakRule, events.onBreak);
        totals.formed += formBonds(state, topology, scenario.params, formRule);
      }
    },
  };
  return scenario;
}

// ── Free chain: something to draw structure onto ──────────────────────────────────────────────

export interface ChainConfig {
  /** Particles in the chain. */
  readonly n?: number;
  readonly seed?: number;
}

/** Radius of the spiral `chainScenario` starts as: neighbours sit about 0.8 apart. */
export function chainRadius(n: number): number {
  return (n * 0.8) / (2.5 * Math.PI);
}

/**
 * A chain held together only by its backbone, with particles that cannot overlap and a little
 * noise. It has no target: it is what `applyLines` draws helices and sheets onto.
 */
export function chainScenario(cfg: ChainConfig = {}): Scenario {
  const n = cfg.n ?? 64;
  const seed = cfg.seed ?? 1;
  const state = createState(n);
  // An open spiral of one and a quarter turns, jittered so that nothing starts exactly in a
  // plane. A spiral shows its length from every side, where a straight chain seen end-on is a dot.
  const radius = chainRadius(n);
  for (let i = 0; i < n; i++) {
    const u = (i / n) * 2.5 * Math.PI;
    const j = (salt: number) => (rand01(i, seed, salt) - 0.5) * 0.4;
    spawn(state, i, radius * Math.cos(u) + j(1), (i / n - 0.5) * radius + j(2), radius * Math.sin(u) + j(3));
  }
  const topology = createTopology(n, 8);
  for (let i = 0; i + 1 < n; i++) addSpring(topology, i, i + 1, 1, 0.4);
  const params = springParams({
    seed,
    damp: 0.85,
    damptype: 1,
    springforce: 1,
    springmaxvel: 0.3,
    backboneStrength: 1,
    powBaseDist: 1,
    // CSynth's local pushapart keeps particles from passing through each other.
    pushapartlocalforce: 0.1,
    nonBackboneLen: 1,
    noiseforce: 0.02,
  });
  const scenario: Scenario = {
    kind: "chain",
    state,
    topology,
    params,
    totals: { broken: 0, formed: 0, divisions: 0 },
    takeEvents: () => [],
    advance(steps) {
      for (let k = 0; k < steps; k++) stepSprings(state, topology, scenario.params);
    },
  };
  return scenario;
}
