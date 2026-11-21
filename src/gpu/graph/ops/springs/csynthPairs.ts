// CSynth's remaining pair forces as graph ops, one per term (`pairforcesfull` in organic's
// springsfs.ts; `pushApart` is its local push-apart):
//
//   pushApartGlobal    long-range repulsion        −f · (max(len, backbone) / d0)^pow
//   pushApartDensity   crowded particles repel     −f · ρ_i ρ_j · (len / d0)^pow
//   fractalForce       pull by chain separation    f · |i − j|^pow · len
//   backboneForce      chain neighbours' spring    f · (len − backbone)
//   distanceForce      towards a target distance   f · (len − target) / target^pow
//   contactForce       attraction where a contact map says the pair touch
//   lorentzianContact  a contact model whose pull peaks at a distance set by the contact
//   boids              cohesion, alignment and separation among near neighbours
//
// Every one walks all pairs within `pairRange`, between species in `speciesMask`.
//
// THE GATE. In CSynth a pair whose contact is marked missing (≤ −9) or excluded (< −1000)
// switches other terms off for that pair. Here that is a gate each term works out for itself
// from the contact map, so no term reaches into another:
//
//   before  0 when contacts or the Lorentzian model are on and the pair is missing
//           (density push-apart, distance targets)
//   chain   0 when contacts are on and the pair is excluded (fractal, backbone)
//   after   0 when the Lorentzian model is on and the pair is missing (global push-apart,
//           boid separation)
//
// A model with no contact map passes a 1 × 1 one, which gates nothing.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { ParamSpec, Params } from "../../op";
import { cutoff, DEAD_SPECIES, numbers, PAIR_RANGE } from "./common";
import { KERNEL_WORKGROUP, type KernelPort, kernelOp } from "./kernelOp";

const WG = KERNEL_WORKGROUP;
const vec4s = (n: number) => d.arrayOf(d.vec4f, n);
const u32s = (n: number) => d.arrayOf(d.u32, n);
const f32s = (n: number) => d.arrayOf(d.f32, n);

/** What every one of these ops needs to walk the pairs and work out the gate. */
const Common = d.struct({
  n: d.u32,
  speciesMask: d.u32,
  /** Side of the model's matrices: particles below it have a row. */
  msize: d.u32,
  /** Side of the contact map bound (1 when the model has none). */
  csize: d.u32,
  /** Bit 0: contacts are on. Bit 1: the Lorentzian model is on. */
  flags: d.u32,
  pairRange: d.f32,
  maxBackboneDist: d.f32,
  powBaseDist: d.f32,
  springpow: d.f32,
});

/** (before, chain, contacts on for this pair, after): see THE GATE above. */
const gateGpu = tgpu.fn(
  [d.u32, d.u32, d.u32, d.u32, d.u32, d.f32, d.f32],
  d.vec4f,
)((i, j, n, msize, flags, maxBackboneDist, raw) => {
  "use gpu";
  const along = std.abs(d.f32(i) - d.f32(j));
  const inMatrix = i < msize && j < msize;
  const contactsOn = (flags & 1) === 1 && inMatrix && along / d.f32(n) < maxBackboneDist;
  const lorentzOn = (flags & 2) === 2 && inMatrix;
  let before = d.f32(1);
  let chain = d.f32(1);
  let after = d.f32(1);
  let on = d.f32(0);
  if (contactsOn) {
    on = 1;
    if (raw <= -9) {
      before = 0;
    }
    if (raw < -1000) {
      chain = 0;
    }
  }
  if (lorentzOn && raw <= -9) {
    before = 0;
    after = 0;
  }
  return d.vec4f(before, chain, on, after);
});

/** CSynth's fade of a pair that sits far from where its contact and the repulsion balance. */
const wrongFadeGpu = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((contact, len, contactforcesc, pushapartforce, pushapartpow, powBaseDist, wrongfade) => {
  "use gpu";
  let fade = d.f32(1);
  if (wrongfade !== 0 && pushapartforce !== 0) {
    // With no contact the balance distance is infinite. Say so, instead of leaving it to
    // `pow(0, negative)`, which WGSL does not define and devices answer differently.
    const strength = (contactforcesc * contact * powBaseDist) / pushapartforce;
    let q = d.f32(1);
    if (strength > 0) {
      q = 1 - len / (std.pow(strength, 1 / (pushapartpow - 1)) * powBaseDist);
    }
    const cc = 1 / (wrongfade * wrongfade);
    const f = cc + q * q;
    fade = (cc * cc) / (f * f);
  }
  return fade;
});

// ── Host side: the same walk, once ─────────────────────────────────────────────────────────────

const COMMON_PARAMS: ParamSpec[] = [
  { name: "speciesMask", type: "int", default: 0xff, min: 0, max: 0xff, describe: "Bit per species that takes part." },
  PAIR_RANGE,
  { name: "matrixSize", type: "int", default: 0, min: 0, describe: "Particles below this index have a row in the model's matrices." },
  { name: "contactsOn", type: "bool", default: false, describe: "The model's contact force is on, so missing contacts gate this term." },
  {
    name: "lorentzianOn",
    type: "bool",
    default: false,
    describe: "The Lorentzian contact model is on, so missing contacts gate this term.",
  },
  {
    name: "maxBackboneDist",
    type: "number",
    default: 1,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Largest chain separation, as a fraction of the chain, that chain terms and contacts act over.",
  },
  { name: "powBaseDist", type: "number", default: 100, min: 1e-6, max: 1000, describe: "The distance at which power factors are 1." },
  { name: "springpow", type: "number", default: 0, min: -4, max: 4, step: 0.1, describe: "Extra distance power on every long-range term." },
];

interface Pair {
  readonly i: number;
  readonly j: number;
  readonly dx: number;
  readonly dy: number;
  readonly dz: number;
  readonly len: number;
  /** Separation along the chain. */
  readonly along: number;
  readonly inRange: boolean;
  readonly inMatrix: boolean;
  /** The contact map's value for the pair, 0 outside it. */
  readonly raw: number;
  readonly before: number;
  readonly chain: number;
  readonly contactsOn: boolean;
  readonly after: number;
  /** `(len / powBaseDist)^springpow`. */
  readonly sp: number;
}

interface CommonValues {
  readonly n: number;
  readonly speciesMask: number;
  readonly msize: number;
  readonly csize: number;
  readonly flags: number;
  readonly pairRange: number;
  readonly maxBackboneDist: number;
  readonly powBaseDist: number;
  readonly springpow: number;
}

function commonValues(n: number, p: Params, num: (p: Params, name: string) => number, csize: number): CommonValues {
  return {
    n,
    speciesMask: num(p, "speciesMask"),
    msize: num(p, "matrixSize"),
    csize,
    flags: (p.contactsOn ? 1 : 0) | (p.lorentzianOn ? 2 : 0),
    pairRange: cutoff(num(p, "pairRange")),
    maxBackboneDist: num(p, "maxBackboneDist"),
    powBaseDist: num(p, "powBaseDist"),
    springpow: num(p, "springpow"),
  };
}

/** Walk the pairs of particle `i` as the kernels do, calling `each` for every one that counts. */
function walk(
  i: number,
  pos: ArrayLike<number>,
  species: ArrayLike<number>,
  contact: ArrayLike<number>,
  c: CommonValues,
  each: (pair: Pair) => void,
): void {
  const takesPart = (s: number) => s !== DEAD_SPECIES && ((c.speciesMask >> s) & 1) === 1;
  if (!takesPart(species[i]!)) return;
  for (let j = 0; j < c.n; j++) {
    if (j === i || !takesPart(species[j]!)) continue;
    const dx = pos[j * 4]! - pos[i * 4]!;
    const dy = pos[j * 4 + 1]! - pos[i * 4 + 1]!;
    const dz = pos[j * 4 + 2]! - pos[i * 4 + 2]!;
    const len = Math.hypot(dx, dy, dz);
    if (len === 0 || len > c.pairRange) continue;
    const along = Math.abs(i - j);
    const inMatrix = i < c.msize && j < c.msize;
    const inRange = along / c.n < c.maxBackboneDist;
    const raw = i < c.csize && j < c.csize ? contact[i * c.csize + j]! : 0;
    const contactsOn = (c.flags & 1) === 1 && inMatrix && inRange;
    const lorentzOn = (c.flags & 2) === 2 && inMatrix;
    const missing = raw <= -9;
    each({
      i,
      j,
      dx,
      dy,
      dz,
      len,
      along,
      inRange,
      inMatrix,
      raw,
      before: (contactsOn || lorentzOn) && missing ? 0 : 1,
      chain: contactsOn && raw < -1000 ? 0 : 1,
      contactsOn,
      after: lorentzOn && missing ? 0 : 1,
      sp: (len / c.powBaseDist) ** c.springpow,
    });
  }
}

/** The CPU golden of a term that is a scalar force along each pair. */
function alongCpu(
  n: number,
  pos: ArrayLike<number>,
  species: ArrayLike<number>,
  contact: ArrayLike<number>,
  c: CommonValues,
  term: (pair: Pair) => number,
): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    let fx = 0;
    let fy = 0;
    let fz = 0;
    walk(i, pos, species, contact, c, (pair) => {
      const sc = term(pair) / pair.len;
      fx += pair.dx * sc;
      fy += pair.dy * sc;
      fz += pair.dz * sc;
    });
    out[i * 4] = fx;
    out[i * 4 + 1] = fy;
    out[i * 4 + 2] = fz;
  }
  return out;
}

function wrongFade(
  contact: number,
  len: number,
  contactforcesc: number,
  pushapartforce: number,
  pushapartpow: number,
  powBaseDist: number,
  wrongfade: number,
): number {
  if (wrongfade === 0 || pushapartforce === 0) return 1;
  // No contact: the balance distance is infinite, as in the kernel.
  const strength = (contactforcesc * contact * powBaseDist) / pushapartforce;
  const q = strength > 0 ? 1 - len / (strength ** (1 / (pushapartpow - 1)) * powBaseDist) : 1;
  const cc = 1 / (wrongfade * wrongfade);
  const f = cc + q * q;
  return (cc * cc) / (f * f);
}

/** The side of a square contact map, from its cell count. Both paths index it as square. */
const side = (count: number) => {
  const s = Math.round(Math.sqrt(count));
  if (s * s !== count) throw new Error(`contact map: ${count} cells are not a square`);
  return s;
};
const POS: KernelPort = { name: "pos", kind: "points", dtype: "f32", lanes: 4 };
const SPECIES: KernelPort = { name: "species", kind: "points", dtype: "u32" };
const CONTACT: KernelPort = { name: "contact", kind: "matrix", dtype: "f32" };
const FORCE = { name: "force", dtype: "f32", lanes: 4 } as const;
const matrixSide = (shape: { kind: string; rows?: number; cols?: number }) => {
  if (shape.kind !== "matrix") return 1;
  if (shape.rows !== shape.cols) throw new Error(`contact map: expected a square matrix, got ${shape.rows} × ${shape.cols}`);
  return shape.rows ?? 1;
};

// ── pushApartGlobal ────────────────────────────────────────────────────────────────────────────

const GlobalParams = d.struct({
  c: Common,
  force: d.f32,
  power: d.f32,
  backboneScale: d.f32,
  ignoreBackbone: d.f32,
  contactforcesc: d.f32,
  contactthreshold: d.f32,
  backboneContactStrength: d.f32,
  wrongfade: d.f32,
});
const globalLayout = tgpu
  .bindGroupLayout({
    params: { uniform: GlobalParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    contact: { storage: f32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("pushApartGlobal");

const globalFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = globalLayout.$.params;
    if (i < P.c.n) {
      let f = d.vec3f(0, 0, 0);
      const si = globalLayout.$.species[i]!;
      if (si !== 255 && ((P.c.speciesMask >> si) & 1) === 1) {
        const me = globalLayout.$.pos[i]!.xyz;
        for (let j = d.u32(0); j < P.c.n; j++) {
          const sj = globalLayout.$.species[j]!;
          if (j !== i && sj !== 255 && ((P.c.speciesMask >> sj) & 1) === 1) {
            const dd = std.sub(globalLayout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            const chainGap = std.abs(d.f32(i) - d.f32(j));
            if (len > 0 && len <= P.c.pairRange && (chainGap > 1.5 || P.ignoreBackbone !== 0)) {
              let raw = d.f32(0);
              if (i < P.c.csize && j < P.c.csize) {
                raw = globalLayout.$.contact[i * P.c.csize + j]!;
              }
              const gate = gateGpu(i, j, P.c.n, P.c.msize, P.c.flags, P.c.maxBackboneDist, raw);
              let fade = d.f32(1);
              if (gate.z === 1) {
                fade = 0;
                if (raw > -9) {
                  let contact = std.max(0, raw - P.contactthreshold);
                  if (chainGap <= 1.5) {
                    contact = contact * P.backboneContactStrength;
                  }
                  fade = wrongFadeGpu(contact, len, P.contactforcesc, P.force, P.power, P.c.powBaseDist, P.wrongfade);
                }
              }
              const sp = std.pow(len / P.c.powBaseDist, P.c.springpow);
              const along = -P.force * std.pow(std.max(len, P.backboneScale) / P.c.powBaseDist, P.power) * fade * gate.w * sp;
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
      }
      globalLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("pushApartGlobal");

const CONTACT_COUPLING: ParamSpec[] = [
  {
    name: "contactforcesc",
    type: "number",
    default: 0,
    min: 0,
    max: 100,
    describe: "The model's contact force strength (for the balance distance).",
  },
  { name: "contactthreshold", type: "number", default: 0, min: 0, max: 10, describe: "Contacts below this count as none." },
  {
    name: "backboneContactStrength",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    describe: "Extra factor on the contact between chain neighbours.",
  },
  {
    name: "wrongfade",
    type: "number",
    default: 0,
    min: 0,
    max: 10,
    describe: "How sharply a pair far from its balance distance is faded out; 0 is off.",
  },
];
const GLOBAL_PARAMS: ParamSpec[] = [
  { name: "pushapartforce", type: "number", default: 0, min: 0, max: 10, step: 0.0001, describe: "Strength of the repulsion." },
  { name: "pushapartpow", type: "number", default: 0, min: -6, max: 2, step: 0.1, describe: "Power of distance it falls off with." },
  {
    name: "backboneScale",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    step: 0.01,
    describe: "Spacing of chain neighbours; distances below it count as it.",
  },
  { name: "ignoreBackbone", type: "number", default: 0, min: 0, max: 1, step: 1, describe: "Non-zero also repels chain neighbours." },
  ...CONTACT_COUPLING,
  ...COMMON_PARAMS,
];
const globalNum = numbers(GLOBAL_PARAMS, "pushApartGlobal");

export const pushApartGlobalOp = kernelOp<d.Infer<typeof GlobalParams>>({
  name: "pushApartGlobal",
  label: "Push apart (global)",
  describe: "Long-range repulsion between all particles.",
  inputs: [POS, SPECIES, CONTACT],
  output: FORCE,
  params: GLOBAL_PARAMS,
  layout: globalLayout,
  kernel: globalFn,
  uniform: (root) => root.createBuffer(GlobalParams).$usage("uniform").$name("pushApartGlobal:params"),
  uniforms: (n, p, shapes) => ({
    c: commonValues(n, p, globalNum, matrixSide(shapes[2]!)),
    force: globalNum(p, "pushapartforce"),
    power: globalNum(p, "pushapartpow"),
    backboneScale: globalNum(p, "backboneScale"),
    ignoreBackbone: globalNum(p, "ignoreBackbone"),
    contactforcesc: globalNum(p, "contactforcesc"),
    contactthreshold: globalNum(p, "contactthreshold"),
    backboneContactStrength: globalNum(p, "backboneContactStrength"),
    wrongfade: globalNum(p, "wrongfade"),
  }),
  cpu([pos, species, contact], n, p) {
    const c = commonValues(n, p, globalNum, side(contact!.length));
    const force = globalNum(p, "pushapartforce");
    const power = globalNum(p, "pushapartpow");
    return alongCpu(n, pos!, species!, contact!, c, (q) => {
      if (!(q.along > 1.5 || globalNum(p, "ignoreBackbone") !== 0)) return 0;
      let fade = 1;
      if (q.contactsOn) {
        fade = 0;
        if (q.raw > -9) {
          let ct = Math.max(0, q.raw - globalNum(p, "contactthreshold"));
          if (q.along <= 1.5) ct *= globalNum(p, "backboneContactStrength");
          fade = wrongFade(ct, q.len, globalNum(p, "contactforcesc"), force, power, c.powBaseDist, globalNum(p, "wrongfade"));
        }
      }
      return -force * (Math.max(q.len, globalNum(p, "backboneScale")) / c.powBaseDist) ** power * fade * q.after * q.sp;
    });
  },
});

// ── The simple terms: a scalar along the pair from a few numbers ───────────────────────────────

const SimpleParams = d.struct({
  c: Common,
  /** Which term: 0 density push-apart, 1 fractal, 2 backbone. */
  term: d.u32,
  force: d.f32,
  power: d.f32,
  backboneScale: d.f32,
});
const simpleLayout = tgpu
  .bindGroupLayout({
    params: { uniform: SimpleParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    contact: { storage: f32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("csynthSimplePair");

const simpleFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = simpleLayout.$.params;
    if (i < P.c.n) {
      let f = d.vec3f(0, 0, 0);
      const si = simpleLayout.$.species[i]!;
      if (si !== 255 && ((P.c.speciesMask >> si) & 1) === 1) {
        const mine = simpleLayout.$.pos[i]!;
        for (let j = d.u32(0); j < P.c.n; j++) {
          const sj = simpleLayout.$.species[j]!;
          if (j !== i && sj !== 255 && ((P.c.speciesMask >> sj) & 1) === 1) {
            const theirs = simpleLayout.$.pos[j]!;
            const dd = std.sub(theirs.xyz, mine.xyz);
            const len = std.length(dd);
            if (len > 0 && len <= P.c.pairRange) {
              const chainGap = std.abs(d.f32(i) - d.f32(j));
              let raw = d.f32(0);
              if (i < P.c.csize && j < P.c.csize) {
                raw = simpleLayout.$.contact[i * P.c.csize + j]!;
              }
              const gate = gateGpu(i, j, P.c.n, P.c.msize, P.c.flags, P.c.maxBackboneDist, raw);
              let along = d.f32(0);
              if (P.term === 0) {
                const sp = std.pow(len / P.c.powBaseDist, P.c.springpow);
                along = -P.force * mine.w * theirs.w * std.pow(len / P.c.powBaseDist, P.power) * gate.x * sp;
              }
              if (P.term === 1 && chainGap / d.f32(P.c.n) < P.c.maxBackboneDist) {
                along = P.force * std.pow(chainGap, P.power) * len * gate.y;
              }
              if (P.term === 2 && chainGap <= 1.5) {
                along = (len - P.backboneScale) * P.force * gate.y;
              }
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
      }
      simpleLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("csynthSimplePair");

/** One of the three simple terms as its own op. They share a kernel that a uniform selects from,
 *  which is the fusion the note describes, in miniature: one loop, a term chosen by data. */
function simpleOp(cfg: {
  name: string;
  label: string;
  describe: string;
  term: 0 | 1 | 2;
  force: ParamSpec;
  extra: ParamSpec[];
  power?: string;
  cpu(q: Pair, pos: ArrayLike<number>, v: (name: string) => number, c: CommonValues): number;
}) {
  const params = [cfg.force, ...cfg.extra, ...COMMON_PARAMS];
  const num = numbers(params, cfg.name);
  const has = (name: string) => params.some((s) => s.name === name);
  return kernelOp<d.Infer<typeof SimpleParams>>({
    name: cfg.name,
    label: cfg.label,
    describe: cfg.describe,
    inputs: [POS, SPECIES, CONTACT],
    output: FORCE,
    params,
    layout: simpleLayout,
    kernel: simpleFn,
    entryPoint: "csynthSimplePair",
    uniform: (root) => root.createBuffer(SimpleParams).$usage("uniform").$name(`${cfg.name}:params`),
    uniforms: (n, p, shapes) => ({
      c: commonValues(n, p, num, matrixSide(shapes[2]!)),
      term: cfg.term,
      force: num(p, cfg.force.name),
      power: cfg.power ? num(p, cfg.power) : 0,
      backboneScale: has("backboneScale") ? num(p, "backboneScale") : 1,
    }),
    cpu([pos, species, contact], n, p) {
      const c = commonValues(n, p, num, side(contact!.length));
      return alongCpu(n, pos!, species!, contact!, c, (q) => cfg.cpu(q, pos!, (name) => num(p, name), c));
    },
  });
}

export const pushApartDensityOp = simpleOp({
  name: "pushApartDensity",
  label: "Push apart (density)",
  describe: "Crowded particles repel each other more: the product of their two densities.",
  term: 0,
  force: { name: "pushapartDensityFactor", type: "number", default: 0, min: 0, max: 10, step: 0.0001, describe: "Strength." },
  extra: [{ name: "pushapartDensityPow", type: "number", default: -1, min: -6, max: 2, step: 0.1, describe: "Power of distance." }],
  power: "pushapartDensityPow",
  cpu: (q, pos, v, c) =>
    -v("pushapartDensityFactor") *
    pos[q.i * 4 + 3]! *
    pos[q.j * 4 + 3]! *
    (q.len / c.powBaseDist) ** v("pushapartDensityPow") *
    q.before *
    q.sp,
});

export const fractalForceOp = simpleOp({
  name: "fractalForce",
  label: "Fractal force",
  describe: "A pull between particles that depends on how far apart they are along the chain.",
  term: 1,
  force: { name: "fractforce", type: "number", default: 0, min: 0, max: 10, step: 0.0001, describe: "Strength." },
  extra: [{ name: "fractpow", type: "number", default: -2, min: -6, max: 2, step: 0.1, describe: "Power of chain separation." }],
  power: "fractpow",
  cpu: (q, _pos, v) => (q.inRange ? v("fractforce") * q.along ** v("fractpow") * q.len * q.chain : 0),
});

export const backboneForceOp = simpleOp({
  name: "backboneForce",
  label: "Backbone force",
  describe: "A spring holding chain neighbours at the backbone spacing.",
  term: 2,
  force: { name: "backboneforce", type: "number", default: 0, min: 0, max: 10, step: 0.001, describe: "Strength." },
  extra: [{ name: "backboneScale", type: "number", default: 1, min: 0, max: 10, step: 0.01, describe: "Spacing of chain neighbours." }],
  cpu: (q, _pos, v) => (q.along <= 1.5 ? (q.len - v("backboneScale")) * v("backboneforce") * q.chain : 0),
});

// ── distanceForce ──────────────────────────────────────────────────────────────────────────────

const DistanceParams = d.struct({ c: Common, force: d.f32, power: d.f32, maxDist: d.f32 });
const distanceLayout = tgpu
  .bindGroupLayout({
    params: { uniform: DistanceParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    contact: { storage: f32s, access: "readonly" },
    dist: { storage: f32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("distanceForce");

const distanceFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = distanceLayout.$.params;
    if (i < P.c.n) {
      let f = d.vec3f(0, 0, 0);
      const si = distanceLayout.$.species[i]!;
      if (si !== 255 && ((P.c.speciesMask >> si) & 1) === 1 && i < P.c.msize) {
        const me = distanceLayout.$.pos[i]!.xyz;
        for (let j = d.u32(0); j < P.c.msize; j++) {
          const sj = distanceLayout.$.species[j]!;
          if (j !== i && sj !== 255 && ((P.c.speciesMask >> sj) & 1) === 1) {
            const dd = std.sub(distanceLayout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            const chainGap = std.abs(d.f32(i) - d.f32(j));
            if (len > 0 && len <= P.c.pairRange && chainGap / d.f32(P.c.n) < P.c.maxBackboneDist) {
              const target = distanceLayout.$.dist[i * P.c.msize + j]!;
              // A target of zero is a missing entry, not a wish to coincide: the law divides by it.
              if (target > 0 && target < P.maxDist) {
                let raw = d.f32(0);
                if (i < P.c.csize && j < P.c.csize) {
                  raw = distanceLayout.$.contact[i * P.c.csize + j]!;
                }
                const gate = gateGpu(i, j, P.c.n, P.c.msize, P.c.flags, P.c.maxBackboneDist, raw);
                const sp = std.pow(len / P.c.powBaseDist, P.c.springpow);
                const along = (((len - target) * P.force) / std.pow(target, P.power)) * gate.x * sp;
                f = std.add(f, std.mul(along / len, dd));
              }
            }
          }
        }
      }
      distanceLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("distanceForce");

const DISTANCE_PARAMS: ParamSpec[] = [
  { name: "xyzforce", type: "number", default: 0, min: 0, max: 10, step: 0.001, describe: "Strength." },
  {
    name: "xyzpow",
    type: "number",
    default: 0.5,
    min: -2,
    max: 4,
    step: 0.1,
    describe: "Power of the target distance the pull is divided by: far targets pull less.",
  },
  { name: "xyzMaxDist", type: "number", default: 5, min: 0, max: 1000, describe: "Targets at or beyond this are ignored." },
  ...COMMON_PARAMS,
];
const distanceNum = numbers(DISTANCE_PARAMS, "distanceForce");

export const distanceForceOp = kernelOp<d.Infer<typeof DistanceParams>>({
  name: "distanceForce",
  label: "Distance targets",
  describe: "Pulls each pair towards a target distance from a matrix.",
  inputs: [POS, SPECIES, CONTACT, { name: "dist", kind: "matrix", dtype: "f32" }],
  output: FORCE,
  params: DISTANCE_PARAMS,
  layout: distanceLayout,
  kernel: distanceFn,
  uniform: (root) => root.createBuffer(DistanceParams).$usage("uniform").$name("distanceForce:params"),
  uniforms: (n, p, shapes) => ({
    c: commonValues(n, p, distanceNum, matrixSide(shapes[2]!)),
    force: distanceNum(p, "xyzforce"),
    power: distanceNum(p, "xyzpow"),
    maxDist: distanceNum(p, "xyzMaxDist"),
  }),
  cpu([pos, species, contact, dist], n, p) {
    const c = commonValues(n, p, distanceNum, side(contact!.length));
    return alongCpu(n, pos!, species!, contact!, c, (q) => {
      if (!(q.inRange && q.inMatrix)) return 0;
      const target = dist![q.i * c.msize + q.j]!;
      if (!(target > 0 && target < distanceNum(p, "xyzMaxDist"))) return 0;
      return (((q.len - target) * distanceNum(p, "xyzforce")) / target ** distanceNum(p, "xyzpow")) * q.before * q.sp;
    });
  },
});

// ── contactForce and lorentzianContact ─────────────────────────────────────────────────────────

const ContactParams = d.struct({
  c: Common,
  /** 0: CSynth's contact force. 1: the Lorentzian model. */
  model: d.u32,
  force: d.f32,
  threshold: d.f32,
  backboneContactStrength: d.f32,
  wrongfade: d.f32,
  pushapartforce: d.f32,
  pushapartpow: d.f32,
  patchval: d.f32,
  patchwidth: d.f32,
  representative: d.f32,
  maxv: d.f32,
  mK: d.f32,
  mAlpha: d.f32,
  mC: d.f32,
});
const contactLayout = tgpu
  .bindGroupLayout({
    params: { uniform: ContactParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    contact: { storage: f32s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("contactForce");

const contactFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = contactLayout.$.params;
    if (i < P.c.n) {
      let f = d.vec3f(0, 0, 0);
      const si = contactLayout.$.species[i]!;
      if (si !== 255 && ((P.c.speciesMask >> si) & 1) === 1 && i < P.c.msize) {
        const me = contactLayout.$.pos[i]!.xyz;
        for (let j = d.u32(0); j < P.c.msize; j++) {
          const sj = contactLayout.$.species[j]!;
          if (j !== i && sj !== 255 && ((P.c.speciesMask >> sj) & 1) === 1) {
            const dd = std.sub(contactLayout.$.pos[j]!.xyz, me);
            const len = std.length(dd);
            if (len > 0 && len <= P.c.pairRange) {
              const chainGap = std.abs(d.f32(i) - d.f32(j));
              // A pair outside the contact map has no contact, as on the host.
              let raw = d.f32(0);
              if (i < P.c.csize && j < P.c.csize) {
                raw = contactLayout.$.contact[i * P.c.csize + j]!;
              }
              const sp = std.pow(len / P.c.powBaseDist, P.c.springpow);
              let along = d.f32(0);
              if (P.model === 0 && chainGap / d.f32(P.c.n) < P.c.maxBackboneDist && raw > -9) {
                let contact = std.max(0, raw - P.threshold);
                if (chainGap <= 1.5) {
                  contact = contact * P.backboneContactStrength;
                }
                const fade = wrongFadeGpu(contact, len, P.force, P.pushapartforce, P.pushapartpow, P.c.powBaseDist, P.wrongfade);
                along = P.force * contact * len * fade * sp;
              }
              if (P.model === 1) {
                let contact = raw;
                if (raw <= -9 && chainGap <= P.patchwidth) {
                  contact = P.patchval;
                }
                contact = contact / P.representative;
                if (chainGap < 1.5) {
                  contact = (P.maxv / P.representative) * P.backboneContactStrength;
                }
                if (contact > 0) {
                  const gap = P.mK * std.pow(contact, -P.mAlpha) - len;
                  const dem = P.mC * P.mC + gap * gap;
                  along = ((P.force * contact * -2 * P.mC * P.mC * gap) / (dem * dem)) * sp;
                }
              }
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
      }
      contactLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("contactForce");

const SHARED_CONTACT: ParamSpec[] = [
  {
    name: "backboneContactStrength",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    describe: "Extra factor on the contact between chain neighbours.",
  },
  ...COMMON_PARAMS,
];
const CONTACT_PARAMS: ParamSpec[] = [
  { name: "contactforcesc", type: "number", default: 0, min: 0, max: 100, step: 0.01, describe: "Strength." },
  { name: "contactthreshold", type: "number", default: 0, min: 0, max: 10, describe: "Contacts below this count as none." },
  {
    name: "wrongfade",
    type: "number",
    default: 0,
    min: 0,
    max: 10,
    describe: "How sharply a pair far from its balance distance is faded out; 0 is off.",
  },
  {
    name: "pushapartforce",
    type: "number",
    default: 0,
    min: 0,
    max: 10,
    describe: "The model's global repulsion (for the balance distance).",
  },
  { name: "pushapartpow", type: "number", default: 0, min: -6, max: 2, describe: "Its distance power." },
  ...SHARED_CONTACT,
];
const LORENTZ_PARAMS: ParamSpec[] = [
  { name: "m_force", type: "number", default: 0, min: 0, max: 100, step: 0.01, describe: "Strength." },
  { name: "m_k", type: "number", default: 10, min: 0, max: 1000, describe: "Distance of the peak for a contact of 1." },
  {
    name: "m_alpha",
    type: "number",
    default: 1,
    min: 0,
    max: 4,
    step: 0.1,
    describe: "How fast that distance shrinks as the contact grows.",
  },
  { name: "m_c", type: "number", default: 20, min: 1e-6, max: 1000, describe: "Width of the peak." },
  { name: "representativeContact", type: "number", default: 1, min: 1e-9, describe: "Contacts are divided by this." },
  { name: "maxv", type: "number", default: 1, min: 0, describe: "The contact given to chain neighbours." },
  { name: "patchval", type: "number", default: 1, min: 0, describe: "The contact given to a missing pair near the diagonal." },
  { name: "patchwidth", type: "number", default: 1.5, min: 0, describe: "How near the diagonal a missing pair is patched." },
  ...SHARED_CONTACT,
];

function contactOp(model: 0 | 1, name: string, label: string, describe: string, params: ParamSpec[]) {
  const num = numbers(params, name);
  const v = (p: Params, key: string) => (params.some((s) => s.name === key) ? num(p, key) : 0);
  return kernelOp<d.Infer<typeof ContactParams>>({
    name,
    label,
    describe,
    inputs: [POS, SPECIES, CONTACT],
    output: FORCE,
    params,
    layout: contactLayout,
    kernel: contactFn,
    entryPoint: "contactForce",
    uniform: (root) => root.createBuffer(ContactParams).$usage("uniform").$name(`${name}:params`),
    check(shapes) {
      const contact = shapes[2]!;
      if (contact.kind !== "matrix" || contact.rows !== contact.cols) throw new Error(`${name}: the contact map must be square`);
    },
    uniforms: (n, p, shapes) => ({
      c: commonValues(n, p, num, matrixSide(shapes[2]!)),
      model,
      force: model === 0 ? num(p, "contactforcesc") : num(p, "m_force"),
      threshold: v(p, "contactthreshold"),
      backboneContactStrength: num(p, "backboneContactStrength"),
      wrongfade: v(p, "wrongfade"),
      pushapartforce: v(p, "pushapartforce"),
      pushapartpow: v(p, "pushapartpow"),
      patchval: v(p, "patchval"),
      patchwidth: v(p, "patchwidth"),
      representative: model === 1 ? num(p, "representativeContact") : 1,
      maxv: v(p, "maxv"),
      mK: v(p, "m_k"),
      mAlpha: v(p, "m_alpha"),
      mC: model === 1 ? num(p, "m_c") : 1,
    }),
    cpu([pos, species, contact], n, p) {
      const c = commonValues(n, p, num, side(contact!.length));
      const bcs = num(p, "backboneContactStrength");
      return alongCpu(n, pos!, species!, contact!, c, (q) => {
        if (!q.inMatrix) return 0;
        if (model === 0) {
          if (!(q.inRange && q.raw > -9)) return 0;
          let ct = Math.max(0, q.raw - num(p, "contactthreshold"));
          if (q.along <= 1.5) ct *= bcs;
          const force = num(p, "contactforcesc");
          return (
            force *
            ct *
            q.len *
            wrongFade(ct, q.len, force, num(p, "pushapartforce"), num(p, "pushapartpow"), c.powBaseDist, num(p, "wrongfade")) *
            q.sp
          );
        }
        let ct = q.raw <= -9 && q.along <= num(p, "patchwidth") ? num(p, "patchval") : q.raw;
        ct /= num(p, "representativeContact");
        if (q.along < 1.5) ct = (num(p, "maxv") / num(p, "representativeContact")) * bcs;
        if (!(ct > 0)) return 0;
        const mc = num(p, "m_c");
        const gap = num(p, "m_k") * ct ** -num(p, "m_alpha") - q.len;
        const dem = mc * mc + gap * gap;
        return ((num(p, "m_force") * ct * -2 * mc * mc * gap) / (dem * dem)) * q.sp;
      });
    },
  });
}

export const contactForceOp = contactOp(
  0,
  "contactForce",
  "Contacts",
  "Pulls together the pairs a contact map says touch, in proportion to the contact.",
  CONTACT_PARAMS,
);
export const lorentzianContactOp = contactOp(
  1,
  "lorentzianContact",
  "Contacts (Lorentzian)",
  "A contact model whose pull peaks at a distance set by the contact and fades either side.",
  LORENTZ_PARAMS,
);

// ── boids ──────────────────────────────────────────────────────────────────────────────────────

const BoidParams = d.struct({
  c: Common,
  alignment: d.f32,
  cohesion: d.f32,
  separation: d.f32,
  separationPow: d.f32,
  radius: d.f32,
  rate: d.f32,
});
const boidLayout = tgpu
  .bindGroupLayout({
    params: { uniform: BoidParams },
    pos: { storage: vec4s, access: "readonly" },
    species: { storage: u32s, access: "readonly" },
    contact: { storage: f32s, access: "readonly" },
    prev: { storage: vec4s, access: "readonly" },
    force: { storage: vec4s, access: "mutable" },
  })
  .$name("boids");

const boidFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = boidLayout.$.params;
    if (i < P.c.n) {
      let f = d.vec3f(0, 0, 0);
      const si = boidLayout.$.species[i]!;
      if (si !== 255 && ((P.c.speciesMask >> si) & 1) === 1) {
        const me = boidLayout.$.pos[i]!.xyz;
        const myVel = std.div(std.sub(me, boidLayout.$.prev[i]!.xyz), P.rate);
        let centre = d.vec3f(0, 0, 0);
        let heading = d.vec3f(0, 0, 0);
        let flock = d.f32(0);
        for (let j = d.u32(0); j < P.c.n; j++) {
          const sj = boidLayout.$.species[j]!;
          if (j !== i && sj !== 255 && ((P.c.speciesMask >> sj) & 1) === 1) {
            const theirs = boidLayout.$.pos[j]!.xyz;
            const dd = std.sub(theirs, me);
            const len = std.length(dd);
            if (len > 0 && len <= P.c.pairRange && len < P.radius) {
              centre = std.add(centre, theirs);
              heading = std.add(heading, std.div(std.sub(theirs, boidLayout.$.prev[j]!.xyz), P.rate));
              flock += 1;
              let raw = d.f32(0);
              if (i < P.c.csize && j < P.c.csize) {
                raw = boidLayout.$.contact[i * P.c.csize + j]!;
              }
              const gate = gateGpu(i, j, P.c.n, P.c.msize, P.c.flags, P.c.maxBackboneDist, raw);
              const sp = std.pow(len / P.c.powBaseDist, P.c.springpow);
              const along = -P.separation * std.pow(std.max(len, 0.01) / P.c.powBaseDist, P.separationPow) * gate.w * sp;
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
        if (flock > 0) {
          f = std.add(f, std.mul(P.cohesion, std.sub(std.div(centre, flock), me)));
          f = std.add(f, std.mul(P.alignment, std.sub(std.div(heading, flock), myVel)));
        }
      }
      boidLayout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("boids");

const BOID_PARAMS: ParamSpec[] = [
  {
    name: "boidalignment",
    type: "number",
    default: 0,
    min: 0,
    max: 1,
    step: 0.001,
    describe: "Pull of a particle's velocity towards its neighbours' mean.",
  },
  { name: "boidcohesion", type: "number", default: 0, min: 0, max: 1, step: 0.001, describe: "Pull towards its neighbours' centre." },
  { name: "boidseparation", type: "number", default: 0, min: 0, max: 1, step: 0.001, describe: "Push away from each neighbour." },
  {
    name: "boidseparationpow",
    type: "number",
    default: -2,
    min: -6,
    max: 2,
    step: 0.1,
    describe: "Power of distance the push falls off with.",
  },
  { name: "boidrad", type: "number", default: 0.1, min: 0, max: 100, step: 0.01, describe: "How far a neighbour is." },
  {
    name: "springrate",
    type: "number",
    default: 1,
    min: 1e-6,
    max: 4,
    step: 0.01,
    describe: "Step length, to turn the two positions into a velocity.",
  },
  ...COMMON_PARAMS,
];
const boidNum = numbers(BOID_PARAMS, "boids");

export const boidsOp = kernelOp<d.Infer<typeof BoidParams>>({
  name: "boids",
  label: "Boids",
  describe: "Flocking: each particle steers towards its near neighbours' centre and heading, and away from each of them.",
  inputs: [POS, SPECIES, CONTACT, { name: "prev", kind: "points", dtype: "f32", lanes: 4 }],
  output: FORCE,
  params: BOID_PARAMS,
  layout: boidLayout,
  kernel: boidFn,
  uniform: (root) => root.createBuffer(BoidParams).$usage("uniform").$name("boids:params"),
  uniforms: (n, p, shapes) => ({
    c: commonValues(n, p, boidNum, matrixSide(shapes[2]!)),
    alignment: boidNum(p, "boidalignment"),
    cohesion: boidNum(p, "boidcohesion"),
    separation: boidNum(p, "boidseparation"),
    separationPow: boidNum(p, "boidseparationpow"),
    radius: boidNum(p, "boidrad"),
    rate: boidNum(p, "springrate"),
  }),
  cpu([pos, species, contact, prev], n, p) {
    const c = commonValues(n, p, boidNum, side(contact!.length));
    const out = new Float32Array(n * 4);
    const rate = boidNum(p, "springrate");
    const vel = (k: number, axis: number) => (pos![k * 4 + axis]! - prev![k * 4 + axis]!) / rate;
    for (let i = 0; i < n; i++) {
      const f = [0, 0, 0];
      const centre = [0, 0, 0];
      const heading = [0, 0, 0];
      let flock = 0;
      walk(i, pos!, species!, contact!, c, (q) => {
        if (!(q.len < boidNum(p, "boidrad"))) return;
        flock++;
        const along =
          -boidNum(p, "boidseparation") * (Math.max(q.len, 0.01) / c.powBaseDist) ** boidNum(p, "boidseparationpow") * q.after * q.sp;
        const delta = [q.dx, q.dy, q.dz];
        for (let k = 0; k < 3; k++) {
          centre[k] = centre[k]! + pos![q.j * 4 + k]!;
          heading[k] = heading[k]! + vel(q.j, k);
          f[k] = f[k]! + (delta[k]! * along) / q.len;
        }
      });
      for (let k = 0; k < 3; k++) {
        if (flock > 0) {
          f[k] =
            f[k]! +
            (centre[k]! / flock - pos![i * 4 + k]!) * boidNum(p, "boidcohesion") +
            (heading[k]! / flock - vel(i, k)) * boidNum(p, "boidalignment");
        }
        out[i * 4 + k] = f[k]!;
      }
    }
    return out;
  },
});
