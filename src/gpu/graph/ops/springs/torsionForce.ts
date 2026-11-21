// `torsionForce`: torsion springs as a graph op. A torsion spring holds the twist (dihedral
// angle) of four consecutive particles a, b, c, d at a target. Every particle is in up to four
// of them, as a, b, c or d, and sums its share of each.
//
// The forces are built so that each spring exerts no net force and no net torque (FoldSynth's
// `TorsionSpring.java`). A spring is indexed by its first particle: `torsion[a] = (angle,
// strength)`. Where a strength is zero the backbone's own setting applies, if any.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { TORSION_SCALE, torsionForces } from "#src/gpu/sim/springs/torsion";
import type { ParamSpec } from "../../op";
import { DEAD_SPECIES, numbers } from "./common";
import { KERNEL_WORKGROUP, kernelOp } from "./kernelOp";

const TorsionParams = d.struct({
  n: d.u32,
  own: d.f32,
  backbone: d.f32,
  backboneAngle: d.f32,
  maxLenSq: d.f32,
});

const layout = tgpu
  .bindGroupLayout({
    params: { uniform: TorsionParams },
    pos: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "readonly" },
    species: { storage: (n: number) => d.arrayOf(d.u32, n), access: "readonly" },
    torsion: { storage: (n: number) => d.arrayOf(d.vec2f, n), access: "readonly" },
    force: { storage: (n: number) => d.arrayOf(d.vec4f, n), access: "mutable" },
  })
  .$name("torsionForce");

/** The force one torsion spring puts on the particle in position `role` (0 = a … 3 = d). */
const quadForce = tgpu.fn(
  [d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.u32, d.f32, d.f32, d.f32],
  d.vec3f,
)((pa, pb, pc, pd, role, target, strength, maxLenSq) => {
  "use gpu";
  let out = d.vec3f(0, 0, 0);
  // With b at the origin.
  const a = std.sub(pa, pb);
  const c = std.sub(pc, pb);
  const dv = std.sub(pd, pb);
  const vcd = std.sub(dv, c);
  const bcSq = std.dot(c, c);
  if (std.dot(a, a) <= maxLenSq && bcSq <= maxLenSq && std.dot(vcd, vcd) <= maxLenSq) {
    const lenBC = std.sqrt(bcSq);
    const uRaw = std.cross(c, std.neg(a));
    const wRaw = std.cross(vcd, c);
    const lenABC = std.length(uRaw);
    const lenBCD = std.length(wRaw);
    if (lenBC > 1e-9 && lenABC > 1e-9 && lenBCD > 1e-9) {
      const u = std.div(uRaw, lenABC);
      const w = std.div(wRaw, lenBCD);
      const angle = std.atan2(std.dot(std.cross(u, w), c) / lenBC, std.dot(u, w));
      // Scale down near degenerate, never up.
      const ss = std.min(1, lenBCD * lenABC);
      const fa = std.mul(lenBCD * ss, u);
      const fd = std.mul(-lenABC * ss, w);
      const k1 = std.neg(std.add(fa, fd));
      const k3 = std.add(std.add(std.cross(k1, c), std.cross(fa, a)), std.cross(fd, dv));
      const fb = std.div(std.cross(c, k3), bcSq);
      const fc = std.sub(k1, fb);

      let stress = target - angle;
      if (stress > 3.141592653589793) {
        stress -= 6.283185307179586;
      }
      if (stress < -3.141592653589793) {
        stress += 6.283185307179586;
      }
      // Fade past a quarter turn, so the drive is zero at ±π instead of flipping sign there.
      if (stress < -1.5707963267948966) {
        stress = -3.141592653589793 - stress;
      }
      if (stress > 1.5707963267948966) {
        stress = 3.141592653589793 - stress;
      }
      const over = (std.dot(fa, fa) + std.dot(fb, fb) + std.dot(fc, fc) + std.dot(fd, fd)) * strength * strength;
      if (over <= 1000) {
        let clamp = d.f32(1);
        if (over > 1) {
          clamp = 1 / std.sqrt(over);
        }
        const drive = stress * strength * 10 * clamp;
        let mine = d.vec3f(fa);
        if (role === 1) {
          mine = d.vec3f(fb);
        }
        if (role === 2) {
          mine = d.vec3f(fc);
        }
        if (role === 3) {
          mine = d.vec3f(fd);
        }
        out = std.mul(drive, mine);
      }
    }
  }
  return out;
});

const torsionFn = tgpu
  .computeFn({ in: { gid: d.builtin.globalInvocationId }, workgroupSize: [KERNEL_WORKGROUP] })((input) => {
    "use gpu";
    const i = input.gid.x;
    const n = layout.$.params.n;
    if (i < n) {
      let f = d.vec3f(0, 0, 0);
      if (layout.$.species[i]! !== 255) {
        for (let role = d.u32(0); role < 4; role++) {
          if (i >= role && i - role + 3 < n) {
            const a = i - role;
            const live =
              layout.$.species[a]! !== 255 &&
              layout.$.species[a + 1]! !== 255 &&
              layout.$.species[a + 2]! !== 255 &&
              layout.$.species[a + 3]! !== 255;
            const spring = layout.$.torsion[a]!;
            let strength = layout.$.params.backbone;
            let target = layout.$.params.backboneAngle;
            if (spring.y !== 0) {
              strength = spring.y * layout.$.params.own;
              target = spring.x;
            }
            if (live && strength !== 0) {
              f = std.add(
                f,
                quadForce(
                  layout.$.pos[a]!.xyz,
                  layout.$.pos[a + 1]!.xyz,
                  layout.$.pos[a + 2]!.xyz,
                  layout.$.pos[a + 3]!.xyz,
                  role,
                  target,
                  strength,
                  layout.$.params.maxLenSq,
                ),
              );
            }
          }
        }
      }
      layout.$.force[i] = d.vec4f(f, 0);
    }
  })
  .$name("torsionForce");

const PARAMS: ParamSpec[] = [
  {
    name: "torsionspringforce",
    type: "number",
    default: 1,
    min: 0,
    max: 10,
    step: 0.01,
    describe: "Scales the strength of every explicitly set torsion spring.",
  },
  {
    name: "backbonetorsionspringforce",
    type: "number",
    default: 0,
    min: 0,
    max: 10,
    step: 0.01,
    describe: "Strength of a torsion spring on every four consecutive particles that has none set.",
  },
  {
    name: "backbonetorsionspringangle",
    type: "number",
    default: -0.6,
    min: -3.1416,
    max: 3.1416,
    step: 0.01,
    units: "rad",
    describe: "Target twist of those backbone springs.",
  },
  {
    name: "torsionMaxLen",
    type: "number",
    default: Math.sqrt(10),
    min: 0,
    max: 1000,
    describe: "A spring is off while any of its three bonds is longer than this.",
  },
];
const num = numbers(PARAMS, "torsionForce");
// The kernel's literal must be the model's scale.
if (TORSION_SCALE !== 10) throw new Error("torsionForce: the kernel assumes TORSION_SCALE is 10");

export const torsionForceOp = kernelOp<d.Infer<typeof TorsionParams>>({
  name: "torsionForce",
  label: "Torsion springs",
  describe: "Holds the twist of each four consecutive particles at a target angle.",
  help: {
    detail:
      "Take four particles in a row. The first three lie in one plane and the last three in another. " +
      "The twist is the angle between those planes. Each torsion spring pushes its four particles so " +
      "that the twist moves towards a target, without pushing or spinning the four as a whole.",
  },
  inputs: [
    { name: "pos", kind: "points", dtype: "f32", lanes: 4 },
    { name: "species", kind: "points", dtype: "u32" },
    { name: "torsion", kind: "points", dtype: "f32", lanes: 2 },
  ],
  output: { name: "force", dtype: "f32", lanes: 4 },
  params: PARAMS,
  layout,
  kernel: torsionFn,
  uniform: (root) => root.createBuffer(TorsionParams).$usage("uniform").$name("torsionForce:params"),
  uniforms(n, p) {
    const maxLen = num(p, "torsionMaxLen");
    return {
      n,
      own: num(p, "torsionspringforce"),
      backbone: num(p, "backbonetorsionspringforce"),
      backboneAngle: num(p, "backbonetorsionspringangle"),
      maxLenSq: Number.isFinite(maxLen) ? maxLen * maxLen : 3.0e38,
    };
  },
  cpu([pos, species, torsion], n, p) {
    const out = new Float32Array(n * 4);
    const quad = new Float64Array(12);
    const maxLen = num(p, "torsionMaxLen");
    const dead = (k: number) => species![k] === DEAD_SPECIES;
    for (let i = 0; i < n; i++) {
      if (dead(i)) continue;
      for (let role = 0; role < 4; role++) {
        const a = i - role;
        if (a < 0 || a + 3 >= n || dead(a) || dead(a + 1) || dead(a + 2) || dead(a + 3)) continue;
        const own = torsion![a * 2 + 1]!;
        const strength = own !== 0 ? own * num(p, "torsionspringforce") : num(p, "backbonetorsionspringforce");
        const angle = own !== 0 ? torsion![a * 2]! : num(p, "backbonetorsionspringangle");
        if (!torsionForces(pos!, a, a + 1, a + 2, a + 3, angle, strength, quad, maxLen * maxLen)) continue;
        for (let k = 0; k < 3; k++) out[i * 4 + k] = out[i * 4 + k]! + quad[role * 3 + k]!;
      }
    }
    return out;
  },
});
