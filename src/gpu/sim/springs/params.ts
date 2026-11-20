// Parameters of the spring model. The first block keeps CSynth's names and defaults (the
// `addGenePerm_withID` calls in organic/TS/springsfs.ts) so a CSynth setup carries over by name.
// The second block is new here: per-species adhesion, mobility and motility.
//
// In CSynth a "force" is a velocity increment per step: `vel = velold * use + force` and
// `pos += vel * springrate`. That convention is kept.

/** Maximum number of bond kinds (CSynth's ROLEFORCESLENGTH). */
export const MAX_KINDS = 8;
/** Maximum number of species. */
export const MAX_SPECIES = 4;

export interface SpringParams {
  // ── Integrator ──────────────────────────────────────────────────────────────────────────────
  /** Damping. Its meaning depends on `damptype`; negative forces `use = -damp`. */
  damp: number;
  /** 1 exponential, 2 direct 4th power, 3 and 4 the `t^(-1/3)` forms. CSynth runs 4. */
  damptype: 1 | 2 | 3 | 4;
  /** Step length: positions move by `vel * springrate`. */
  springrate: number;
  /** Soft cap on speed. */
  springmaxvel: number;
  gravity: number;
  /** Velocity drag `visc * v * |v|^(viscpow - 1)`. */
  visc: number;
  viscpow: number;
  /** Pull every particle towards `springCentre` each step; 1 leaves positions alone. */
  springCentreDamp: number;
  springCentre: readonly [number, number, number];
  /** Soft spherical container about the origin. */
  modelSphereForce: number;
  modelSphereRadius: number;
  /** Chance per particle per step of a random kick of size `noiseforce`. */
  noiseprob: number;
  noiseforce: number;

  // ── Bonded springs ──────────────────────────────────────────────────────────────────────────
  springforce: number;
  /** Effective rest length is `rest * springlen + springlenfix`. */
  springlen: number;
  springlenfix: number;
  /** Extra distance power on springs and on the global pair terms. */
  springpow: number;
  powBaseDist: number;
  /** Per-kind strength: `strength * (springforce * roleforces[kind] + roleforcesFix[kind])`. */
  roleforces: readonly number[];
  roleforcesFix: readonly number[];
  /** Multiplier for springs between index neighbours (`|i - j| = 1`). */
  backboneStrength: number;
  pullspringforce: number;
  pullspringpow: number;
  /** After the step, blend the position towards its pull target by this fraction. */
  pullfixdamp: number;
  /** Multiplier on every explicit torsion spring's stiffness. */
  torsionspringforce: number;
  /** Non-zero gives every four consecutive particles a torsion spring of this stiffness, where
   *  no explicit one is set. */
  backbonetorsionspringforce: number;
  /** Target angle of those automatic torsion springs, in radians. */
  backbonetorsionspringangle: number;
  /** A torsion spring is ignored while any of its three bonds is longer than this. */
  torsionMaxLen: number;

  // ── Pair forces (CSynth `pairforces`) ───────────────────────────────────────────────────────
  /** Expected distance between index neighbours. */
  backboneScale: number;
  /** Closest approach for non-neighbours, relative to `backboneScale`. */
  nonBackboneLen: number;
  /** Spring holding index neighbours at `backboneScale`. */
  backboneforce: number;
  backboneContactStrength: number;
  /** Short-range repulsion, strongest inside half the push distance. */
  pushapartlocalforce: number;
  pushapartdelta: number;
  /** Long-range repulsion `pushapartforce * (len / powBaseDist)^pushapartpow`. */
  pushapartforce: number;
  pushapartpow: number;
  pushapartDensityFactor: number;
  pushapartDensityPow: number;
  /** Non-zero applies the global repulsion to index neighbours too. */
  ignoreBackbone: number;
  fractforce: number;
  fractpow: number;
  /** Spring towards a target distance from the distance matrix. */
  xyzforce: number;
  xyzpow: number;
  xyzMaxDist: number;
  /** Attraction proportional to the contact matrix entry times distance. */
  contactforcesc: number;
  contactthreshold: number;
  /** Contact value assumed for near index neighbours whose data is missing (Lorentzian model). */
  patchval: number;
  patchwidth: number;
  /** Lorentzian fade of contact and repulsion away from their balance distance; 0 is off. */
  wrongfade: number;
  /** Pair terms that depend on index distance only apply within this fraction of the chain. */
  maxBackboneDist: number;
  /** Lorentzian contact model (Trieu and Cheng 2017): target `m_k * contact^-m_alpha`. */
  m_force: number;
  m_k: number;
  m_alpha: number;
  m_c: number;
  representativeContact: number;
  /** Largest contact value, used for index neighbours in the Lorentzian model. */
  maxv: number;
  boidalignment: number;
  boidcohesion: number;
  boidseparation: number;
  boidseparationpow: number;
  boidrad: number;

  // ── New here ────────────────────────────────────────────────────────────────────────────────
  /** Only pairs closer than this are evaluated, through a bucket grid. `undefined` is every
   *  pair, as in CSynth. Long-range terms are truncated when it is set. */
  pairRange: number | undefined;
  /** Bitmask of species the CSynth pair terms above apply to. */
  pairSpeciesMask: number;
  /** Species × species repulsion stiffness when two particles overlap (`len < r_i + r_j`). */
  repulsion: readonly number[];
  /** Species × species adhesion strength: attraction between touching distance and
   *  `(r_i + r_j) * (1 + adhesionRange)`, strongest midway. */
  adhesion: readonly number[];
  adhesionRange: number;
  /** Per-species multiplier on the total force: the inverse of a viscosity. */
  mobility: readonly number[];
  /** Per-species random kick applied every step. */
  motility: readonly number[];
  /** Seed for every random stream. */
  seed: number;
  /** 2 keeps noise, motility and division in the xy plane. */
  dims: 2 | 3;
}

function fill(n: number, v: number): number[] {
  return Array.from({ length: n }, () => v);
}

/** CSynth's defaults, with the new terms switched off. */
export const DEFAULT_SPRING_PARAMS: SpringParams = {
  damp: 0.995,
  damptype: 4,
  springrate: 1,
  springmaxvel: 0.25,
  gravity: 0,
  visc: 0,
  viscpow: 1,
  springCentreDamp: 1,
  springCentre: [0, 0, 0],
  modelSphereForce: 0,
  modelSphereRadius: 1,
  noiseprob: 1,
  noiseforce: 0,

  springforce: 0.001,
  springlen: 1,
  springlenfix: 0,
  springpow: 0,
  powBaseDist: 100,
  roleforces: fill(MAX_KINDS, 1),
  roleforcesFix: fill(MAX_KINDS, 0),
  backboneStrength: 10,
  pullspringforce: 0,
  pullspringpow: 0,
  pullfixdamp: 0,
  torsionspringforce: 1,
  backbonetorsionspringforce: 0,
  backbonetorsionspringangle: -0.6,
  torsionMaxLen: Math.sqrt(10),

  backboneScale: 1,
  nonBackboneLen: 1,
  backboneforce: 0,
  backboneContactStrength: 1,
  pushapartlocalforce: 0.2,
  pushapartdelta: 0,
  pushapartforce: 0,
  pushapartpow: 0,
  pushapartDensityFactor: 0,
  pushapartDensityPow: -1,
  ignoreBackbone: 0,
  fractforce: 0,
  fractpow: -2,
  xyzforce: 0,
  xyzpow: 0.5,
  xyzMaxDist: 5,
  contactforcesc: 0,
  contactthreshold: 0,
  patchval: 1,
  patchwidth: 1.5,
  wrongfade: 0,
  maxBackboneDist: 1,
  m_force: 0,
  m_k: 10,
  m_alpha: 1,
  m_c: 20,
  representativeContact: 1,
  maxv: 1,
  boidalignment: 0,
  boidcohesion: 0,
  boidseparation: 0,
  boidseparationpow: -2,
  boidrad: 0.1,

  pairRange: undefined,
  pairSpeciesMask: 0xff,
  repulsion: fill(MAX_SPECIES * MAX_SPECIES, 0),
  adhesion: fill(MAX_SPECIES * MAX_SPECIES, 0),
  adhesionRange: 0.5,
  mobility: fill(MAX_SPECIES, 1),
  motility: fill(MAX_SPECIES, 0),
  seed: 1,
  dims: 3,
};

/** Defaults with overrides. */
export function springParams(over: Partial<SpringParams> = {}): SpringParams {
  return { ...DEFAULT_SPRING_PARAMS, ...over };
}

/** A symmetric species × species table from `[a, b, value]` entries. */
export function speciesTable(entries: readonly (readonly [number, number, number])[]): number[] {
  const t = fill(MAX_SPECIES * MAX_SPECIES, 0);
  for (const [a, b, v] of entries) {
    t[a * MAX_SPECIES + b] = v;
    t[b * MAX_SPECIES + a] = v;
  }
  return t;
}
