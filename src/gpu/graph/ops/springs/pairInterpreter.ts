// `pairInterpreter`: several pair terms in one pass over the pairs, chosen by a program that is
// data (docs/spring-model-as-ops.md, "fusion"). It is the same maths as the separate ops
// `pushApart`, `adhesion`, `outward` and `pairDensity`, which stay as the definition; a graph
// may use either, and the two must agree.
//
// The kernel is fixed. What it does is set by a program: a list of instructions, each an opcode
// and where its parameters start. Each particle decodes the program once into registers, then
// walks its pairs adding every term's share; terms that need the whole neighbourhood (the
// outward push) gather into registers and are finished after the loop. A term may appear once. Changing which terms run, or their settings, is a buffer write,
// never a new pipeline. This is the principle of organic's `interp.vs`.
//
// It reads a PACKED state and a PACKED inputs bundle: each is one buffer bound once, as raw
// words, with the parts found through offsets passed as data. That is what keeps it to four
// storage buffers however many terms the program has.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { submitDispatch } from "#src/gpu/device";
import { typegpuComputePipeline } from "#src/gpu/typegpuPipeline";
import type { Root } from "../../backend";
import type { FieldValue, ResidentBuffer, Shape } from "../../handle";
import type { OpType, ParamSpec } from "../../op";
import { rawBindGroup } from "../../residentBind";
import { contactForce } from "./adhesion";
import { VEC4 } from "./bondTable";
import { cutoff, DEAD_SPECIES, numbers, VEC2 } from "./common";

const WG = 64;

/** Opcodes. A program is `[opcode, paramBase]` pairs of u32; `paramBase` indexes the f32 params. */
export const PAIR_OP = { pushApart: 1, adhesion: 2, outward: 3, density: 4 } as const;
/** Words per instruction. */
export const PAIR_INSTRUCTION_WORDS = 2;

/** What each instruction's parameter block holds, in order. */
export const PAIR_PARAMS = {
  pushApart: ["force", "nonBackboneLen", "backboneScale", "delta", "pairRange", "speciesMask"],
  adhesion: ["range", "pairRange"],
  outward: [
    "range",
    "full",
    "pairRange",
    "tissueMask",
    "protrusion0",
    "protrusion1",
    "protrusion2",
    "protrusion3",
    "thrust0",
    "thrust1",
    "thrust2",
    "thrust3",
  ],
  density: ["pairRange", "densityRange"],
} as const;

export interface PairProgram {
  readonly program: Uint32Array;
  readonly params: Float32Array;
}

/** Assemble a program from terms and their settings, in the order given. */
export function pairProgram(terms: readonly { op: keyof typeof PAIR_OP; values: Readonly<Record<string, number>> }[]): PairProgram {
  const program: number[] = [];
  const params: number[] = [];
  for (const term of terms) {
    program.push(PAIR_OP[term.op], params.length);
    for (const name of PAIR_PARAMS[term.op]) {
      const v = term.values[name];
      if (v === undefined) throw new Error(`pairProgram: ${term.op} needs "${name}"`);
      params.push(name === "pairRange" || name === "densityRange" ? cutoff(v) : v);
    }
  }
  return { program: Uint32Array.from(program), params: Float32Array.from(params.length > 0 ? params : [0]) };
}

const InterpParams = d.struct({
  n: d.u32,
  instructions: d.u32,
  speciesCount: d.u32,
  // Where each part starts, in words: four in the state buffer, four in the inputs buffer.
  pos: d.u32,
  prev: d.u32,
  species: d.u32,
  radius: d.u32,
  program: d.u32,
  params: d.u32,
  repulsion: d.u32,
  adhesion: d.u32,
  rate: d.f32,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: InterpParams },
    state: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    inputs: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
    density: { storage: (n: number) => d.arrayOf(d.vec2f, n), access: "mutable" },
  })
  .$name("pairInterpreter");

const interpFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [WG] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const P = layout.$.params;
    if (i < P.n) {
      let f = d.vec3f(0, 0, 0);
      let density = d.f32(0);
      let directional = d.f32(0);
      // Registers for terms that need the whole neighbourhood before they can act.
      let away = d.vec3f(0, 0, 0);
      let awayWeight = d.f32(0);
      const si = layout.$.state[P.species + i]!;

      // Decode the program once, into registers. The loop over pairs below then reads no
      // program and no parameters from memory: doing so per pair, per instruction, cost more
      // than the separate passes it was meant to replace. A term appears at most once.
      let pushOn = false;
      let pushForce = d.f32(0);
      let pushLen = d.f32(0);
      let pushScale = d.f32(0);
      let pushDelta = d.f32(0);
      let pushRange = d.f32(0);
      let pushMask = d.u32(0);
      let stickOn = false;
      let stickReach = d.f32(0);
      let stickRange = d.f32(0);
      let outOn = false;
      let outReach = d.f32(0);
      let outFull = d.f32(1);
      let outRange = d.f32(0);
      let outTissue = d.u32(0);
      let outProtrude = d.f32(0);
      let outThrust = d.f32(0);
      let crowdOn = false;
      let crowdRange = d.f32(0);
      let crowdReach = d.f32(0);
      for (let k = d.u32(0); k < P.instructions; k++) {
        const op = layout.$.inputs[P.program + k * 2]!;
        const base = P.params + layout.$.inputs[P.program + k * 2 + 1]!;
        if (op === 1) {
          pushOn = true;
          pushForce = std.bitcastU32toF32(layout.$.inputs[base]!);
          pushLen = std.bitcastU32toF32(layout.$.inputs[base + 1]!);
          pushScale = std.bitcastU32toF32(layout.$.inputs[base + 2]!);
          pushDelta = std.bitcastU32toF32(layout.$.inputs[base + 3]!);
          pushRange = std.bitcastU32toF32(layout.$.inputs[base + 4]!);
          pushMask = d.u32(std.bitcastU32toF32(layout.$.inputs[base + 5]!));
        }
        if (op === 2) {
          stickOn = true;
          stickReach = std.bitcastU32toF32(layout.$.inputs[base]!);
          stickRange = std.bitcastU32toF32(layout.$.inputs[base + 1]!);
        }
        if (op === 3) {
          outOn = true;
          outReach = std.bitcastU32toF32(layout.$.inputs[base]!);
          outFull = std.bitcastU32toF32(layout.$.inputs[base + 1]!);
          outRange = std.bitcastU32toF32(layout.$.inputs[base + 2]!);
          outTissue = d.u32(std.bitcastU32toF32(layout.$.inputs[base + 3]!));
          if (si < 4) {
            outProtrude = std.bitcastU32toF32(layout.$.inputs[base + 4 + si]!);
            outThrust = std.bitcastU32toF32(layout.$.inputs[base + 8 + si]!);
          }
        }
        if (op === 4) {
          crowdOn = true;
          crowdRange = std.bitcastU32toF32(layout.$.inputs[base]!);
          crowdReach = std.bitcastU32toF32(layout.$.inputs[base + 1]!);
        }
      }
      // A term this particle takes no part in is switched off for all its pairs at once.
      if (pushOn && ((pushMask >> si) & 1) !== 1) {
        pushOn = false;
      }
      if (outOn && outProtrude === 0 && outThrust === 0) {
        outOn = false;
      }

      if (si !== 255) {
        const at = P.pos + i * 4;
        const me = d.vec3f(
          std.bitcastU32toF32(layout.$.state[at]!),
          std.bitcastU32toF32(layout.$.state[at + 1]!),
          std.bitcastU32toF32(layout.$.state[at + 2]!),
        );
        const was = P.prev + i * 4;
        const v = std.div(
          std.sub(
            me,
            d.vec3f(
              std.bitcastU32toF32(layout.$.state[was]!),
              std.bitcastU32toF32(layout.$.state[was + 1]!),
              std.bitcastU32toF32(layout.$.state[was + 2]!),
            ),
          ),
          P.rate,
        );
        const ri = std.bitcastU32toF32(layout.$.state[P.radius + i]!);
        for (let j = d.u32(0); j < P.n; j++) {
          const sj = layout.$.state[P.species + j]!;
          if (j !== i && sj !== 255) {
            const their = P.pos + j * 4;
            const dd = std.sub(
              d.vec3f(
                std.bitcastU32toF32(layout.$.state[their]!),
                std.bitcastU32toF32(layout.$.state[their + 1]!),
                std.bitcastU32toF32(layout.$.state[their + 2]!),
              ),
              me,
            );
            const len = std.length(dd);
            if (len > 0) {
              let along = d.f32(0);
              if (pushOn && len <= pushRange && ((pushMask >> sj) & 1) === 1) {
                const chainGap = std.abs(d.f32(i) - d.f32(j));
                const t = std.clamp((len / (std.min(chainGap, pushLen) * pushScale + pushDelta) - 0.5) / 0.5, 0, 1);
                along += -pushForce * (1 - t * t * (3 - 2 * t));
              }
              if (stickOn && len <= stickRange) {
                const pair = si * P.speciesCount + sj;
                const sigma = ri + std.bitcastU32toF32(layout.$.state[P.radius + j]!);
                if (len < sigma) {
                  along += -std.bitcastU32toF32(layout.$.inputs[P.repulsion + pair]!) * (sigma - len);
                } else {
                  const reach = sigma * stickReach;
                  const stick = std.bitcastU32toF32(layout.$.inputs[P.adhesion + pair]!);
                  if (stick !== 0 && len < sigma + reach) {
                    const u = (len - sigma) / reach;
                    along += stick * 4 * u * (1 - u);
                  }
                }
              }
              if (outOn && len <= outRange && len < outReach && (sj === si || ((outTissue >> sj) & 1) === 1)) {
                const w = 1 - len / outReach;
                awayWeight += w;
                away = std.sub(away, std.mul(w / len, dd));
              }
              if (crowdOn && len <= crowdRange && len < crowdReach) {
                density += 1 / (len * len * len);
                directional += std.max(0, std.dot(dd, v) / len) / (len * len);
              }
              f = std.add(f, std.mul(along / len, dd));
            }
          }
        }
        // Finish the term that gathered.
        const awayLen = std.length(away);
        if (outOn && awayLen > 1e-6) {
          f = std.add(f, std.mul((outProtrude * std.max(0, 1 - awayWeight / outFull) + outThrust) / awayLen, away));
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
      layout.$.density[i] = d.vec2f(density, directional);
    }
  })
  .$name("pairInterpreter");

const makeUniform = (root: Root) => root.createBuffer(InterpParams).$usage("uniform").$name("pairInterpreter:params");
const pipes = new WeakMap<object, { pipeline: GPUComputePipeline; uniform: ReturnType<typeof makeUniform> }>();

const PARAMS: ParamSpec[] = [
  {
    name: "springrate",
    type: "number",
    default: 1,
    min: 1e-6,
    max: 4,
    step: 0.01,
    describe: "Step length, to turn the two positions into a velocity.",
  },
];
const num = numbers(PARAMS, "pairInterpreter");

function part(bundle: FieldValue, name: string, who: string): FieldValue {
  const v = bundle.parts?.[name];
  if (!v) throw new Error(`pairInterpreter: ${who} has no "${name}" part`);
  return v;
}

/** The one buffer a packed bundle's parts are ranges of. */
function rootOf(bundle: FieldValue, who: string): ResidentBuffer {
  const roots = new Set(Object.values(bundle.parts ?? {}).map((v) => v.buffer?.sliceOf));
  const [root] = roots;
  if (roots.size !== 1 || !root)
    throw new Error(`pairInterpreter: ${who} must be a packed bundle on the device (its parts ranges of one buffer)`);
  return root;
}

/** A part's start in its packed buffer, in words. */
const wordOffset = (bundle: FieldValue, name: string, who: string) => (part(bundle, name, who).buffer?.offset ?? 0) / 4;

function count(shape: Shape): number {
  return shape.kind === "points" ? shape.n : 0;
}

/** The host version: the same walk, on the bundles' separate arrays. */
function interpretCpu(state: FieldValue, inputs: FieldValue, rate: number): { force: Float32Array; density: Float32Array } {
  const data = (bundle: FieldValue, name: string, who: string) => {
    const v = part(bundle, name, who).data;
    if (!v) throw new Error(`pairInterpreter: cpu golden needs "${name}" on the host`);
    return v;
  };
  const pos = data(state, "pos", "the state");
  const prev = data(state, "prev", "the state");
  const species = data(state, "species", "the state");
  const radius = data(state, "radius", "the state");
  const program = data(inputs, "program", "the inputs");
  const params = data(inputs, "programParams", "the inputs");
  const n = species.length;
  const instructions = program.length / PAIR_INSTRUCTION_WORDS;
  const has = (op: number) => Array.from({ length: instructions }, (_, k) => program[k * 2]).includes(op);
  const repulsion = has(PAIR_OP.adhesion) ? data(inputs, "repulsion", "the inputs") : undefined;
  const adhesion = has(PAIR_OP.adhesion) ? data(inputs, "adhesion", "the inputs") : undefined;
  const speciesCount = repulsion ? Math.round(Math.sqrt(repulsion.length)) : 0;
  const force = new Float32Array(n * 4);
  const density = new Float32Array(n * 2);
  const smooth = (x: number) => {
    const t = Math.min(1, Math.max(0, (x - 0.5) / 0.5));
    return t * t * (3 - 2 * t);
  };
  for (let i = 0; i < n; i++) {
    const si = species[i]!;
    if (si === DEAD_SPECIES) continue;
    const vel = [0, 1, 2].map((c) => (pos[i * 4 + c]! - prev[i * 4 + c]!) / rate);
    const f = [0, 0, 0];
    const away = [0, 0, 0];
    let awayWeight = 0;
    for (let j = 0; j < n; j++) {
      const sj = species[j]!;
      if (j === i || sj === DEAD_SPECIES) continue;
      const dd = [0, 1, 2].map((c) => pos[j * 4 + c]! - pos[i * 4 + c]!);
      const len = Math.hypot(dd[0]!, dd[1]!, dd[2]!);
      if (len === 0) continue;
      let along = 0;
      for (let k = 0; k < instructions; k++) {
        const op = program[k * 2]!;
        const base = program[k * 2 + 1]!;
        const p = (o: number) => params[base + o]!;
        if (op === PAIR_OP.pushApart) {
          const mask = p(5);
          if (len <= p(4) && ((mask >> si) & 1) === 1 && ((mask >> sj) & 1) === 1) {
            along += -p(0) * (1 - smooth(len / (Math.min(Math.abs(i - j), p(1)) * p(2) + p(3))));
          }
        } else if (op === PAIR_OP.adhesion && repulsion && adhesion) {
          if (len <= p(1))
            along += contactForce(
              len,
              radius[i]! + radius[j]!,
              repulsion[si * speciesCount + sj]!,
              adhesion[si * speciesCount + sj]!,
              p(0),
            );
        } else if (op === PAIR_OP.outward) {
          if (len <= p(2) && len < p(0) && (sj === si || ((p(3) >> sj) & 1) === 1)) {
            const w = 1 - len / p(0);
            awayWeight += w;
            for (let c = 0; c < 3; c++) away[c] = away[c]! - (dd[c]! * w) / len;
          }
        } else if (op === PAIR_OP.density) {
          if (len <= p(0) && len < p(1)) {
            density[i * 2] = density[i * 2]! + 1 / (len * len * len);
            density[i * 2 + 1] =
              density[i * 2 + 1]! + Math.max(0, (dd[0]! * vel[0]! + dd[1]! * vel[1]! + dd[2]! * vel[2]!) / len) / (len * len);
          }
        }
      }
      for (let c = 0; c < 3; c++) f[c] = f[c]! + (dd[c]! * along) / len;
    }
    for (let k = 0; k < instructions; k++) {
      if (program[k * 2] !== PAIR_OP.outward || si >= 4) continue;
      const base = program[k * 2 + 1]!;
      const protrude = params[base + 4 + si]!;
      const thrust = params[base + 8 + si]!;
      const awayLen = Math.hypot(away[0]!, away[1]!, away[2]!);
      if ((protrude === 0 && thrust === 0) || !(awayLen > 1e-6)) continue;
      const sc = (protrude * Math.max(0, 1 - awayWeight / params[base + 1]!) + thrust) / awayLen;
      for (let c = 0; c < 3; c++) f[c] = f[c]! + away[c]! * sc;
    }
    for (let c = 0; c < 3; c++) force[i * 4 + c] = f[c]!;
  }
  return { force, density };
}

export const pairInterpreterOp: OpType = {
  name: "pairInterpreter",
  label: "Pair forces (interpreted)",
  describe: "Several pair terms in one pass over the pairs, chosen by a program held as data.",
  category: "Springs",
  help: {
    detail:
      "The separate pair ops each walk every pair of particles. This walks them once and, for each " +
      "pair, runs a short list of instructions: one per term. The list is data, so switching a term on " +
      "or off, or changing its settings, never rebuilds anything. It needs the state and the inputs " +
      "packed, each in one buffer.",
  },
  inputs: [
    { name: "state", kind: "bundle" },
    { name: "inputs", kind: "bundle" },
  ],
  outputs: [
    { name: "force", kind: "points", dtype: "f32" },
    { name: "density", kind: "points", dtype: "f32" },
  ],
  params: PARAMS,
  inferShapes(shapes) {
    const state = shapes[0]!;
    const species = state.kind === "bundle" ? state.parts.species : undefined;
    if (!species || count(species) === 0) throw new Error("pairInterpreter.state: expected a spring state bundle");
    const inputs = shapes[1]!;
    if (inputs.kind !== "bundle" || !inputs.parts.program || !inputs.parts.programParams) {
      throw new Error("pairInterpreter.inputs: expected a bundle with `program` and `programParams` parts");
    }
    return [species, species];
  },
  inferElements() {
    return [VEC4, VEC2];
  },
  resident: true,
  slices: true,
  async execute(ctx, inputs, params) {
    const [state, fixed] = inputs as [FieldValue, FieldValue];
    const n = count(part(state, "species", "the state").shape);
    const stateRoot = rootOf(state, "the state");
    const inputsRoot = rootOf(fixed, "the inputs");
    const tables = fixed.parts?.repulsion && fixed.parts.adhesion;
    const repulsionShape = fixed.parts?.repulsion?.shape;
    // Gather the uniform before either lease. Two graphs share it per device, so the write has
    // to sit with the bind and the submit, after the awaits, or the other graph can overwrite
    // `n` before this one submits.
    const uniform = {
      n,
      instructions: count(part(fixed, "program", "the inputs").shape) / PAIR_INSTRUCTION_WORDS,
      speciesCount: repulsionShape?.kind === "matrix" ? repulsionShape.rows : 0,
      pos: wordOffset(state, "pos", "the state"),
      prev: wordOffset(state, "prev", "the state"),
      species: wordOffset(state, "species", "the state"),
      radius: wordOffset(state, "radius", "the state"),
      program: wordOffset(fixed, "program", "the inputs"),
      params: wordOffset(fixed, "programParams", "the inputs"),
      repulsion: tables ? wordOffset(fixed, "repulsion", "the inputs") : 0,
      adhesion: tables ? wordOffset(fixed, "adhesion", "the inputs") : 0,
      rate: num(params, "springrate"),
    };
    const device = await ctx.backend.getDevice();
    const root = await ctx.backend.getRoot();
    let pipe = pipes.get(root);
    if (!pipe) {
      const resolved = tgpu.resolveWithContext([interpFn], { names: "strict" });
      pipe = { pipeline: typegpuComputePipeline(device, root, resolved, "pairInterpreter", "pairInterpreter"), uniform: makeUniform(root) };
      pipes.set(root, pipe);
    }
    const force = await ctx.backend.lease(n * 16);
    try {
      const density = await ctx.backend.lease(n * 8);
      try {
        pipe.uniform.write(uniform);
        const bind = rawBindGroup(
          device,
          root,
          layout,
          [pipe.uniform, stateRoot, inputsRoot, force.buffer, density.buffer],
          "pairInterpreter",
        );
        submitDispatch(device, "pairInterpreter", pipe.pipeline, bind, Math.ceil(n / WG));
        const shape: Shape = { kind: "points", n };
        return [
          { shape, dtype: "f32", element: VEC4, buffer: force },
          { shape, dtype: "f32", element: VEC2, buffer: density },
        ];
      } catch (e) {
        ctx.backend.release(density);
        throw e;
      }
    } catch (e) {
      ctx.backend.release(force);
      throw e;
    }
  },
  cpuGolden(inputs, params) {
    const [state, fixed] = inputs as [FieldValue, FieldValue];
    const { force, density } = interpretCpu(state, fixed, num(params, "springrate"));
    const shape: Shape = { kind: "points", n: force.length / 4 };
    return [
      { shape, dtype: "f32", element: VEC4, data: force },
      { shape, dtype: "f32", element: VEC2, data: density },
    ];
  },
};
