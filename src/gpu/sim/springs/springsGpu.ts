// The spring model on the device: `stepSprings`, `breakBonds` and `formBonds` as compute kernels.
// `step.ts` and `turnover.ts` are the reference; each kernel here is the same arithmetic in the
// same order and is tested against them (`springsGpu.gpu.test.ts`).
//
// Every kernel is a gather: a thread reads whatever it likes and writes only its own particle's
// row, so there are no atomics and no ordering between threads.
//
// WGSL templates rather than `"use gpu"` TGSL: the step is one long function over a parameter
// block that is generated from a list of names, which a template expresses directly.
//
// Pairs are brute force, O(n²) per step, with `pairRange` applied as a cutoff. Walking the
// uniform-grid index instead is an optimisation that changes no result, and is not done yet.
// Growth (`growth.ts`) stays on the host: `download`, divide, `upload`.
import { compileShader, deviceScoped, readBackBytes, writeView } from "../../device";
import { MAX_KINDS, MAX_SPECIES, type SpringParams } from "./params";
import { clearLeaders, divideCells, selectLeaders } from "./growth";
import { probThreshold, SALT } from "./rng";
import { CELL, LEADER, type SpheroidScenario } from "./scenarios";
import { DEAD, type SpringMatrices, type SpringState } from "./state";
import { SLOT_WORDS, type SpringTopology } from "./topology";
import { TORSION_SCALE } from "./torsion";
import { pairTermsActive } from "./step";
import type { BreakRule, FormRule } from "./turnover";

const WG = 64;

/** The f32 scalars of `SpringParams`, in the order they sit in the uniform block. */
const SCALARS = [
  "damp",
  "springrate",
  "springmaxvel",
  "gravity",
  "visc",
  "viscpow",
  "springCentreDamp",
  "modelSphereForce",
  "modelSphereRadius",
  "noiseprob",
  "noiseforce",
  "springforce",
  "springlen",
  "springlenfix",
  "springpow",
  "powBaseDist",
  "backboneStrength",
  "pullspringforce",
  "pullspringpow",
  "pullfixdamp",
  "torsionspringforce",
  "backbonetorsionspringforce",
  "backbonetorsionspringangle",
  "torsionMaxLen",
  "backboneScale",
  "nonBackboneLen",
  "backboneforce",
  "backboneContactStrength",
  "pushapartlocalforce",
  "pushapartdelta",
  "pushapartforce",
  "pushapartpow",
  "pushapartDensityFactor",
  "pushapartDensityPow",
  "ignoreBackbone",
  "fractforce",
  "fractpow",
  "xyzforce",
  "xyzpow",
  "xyzMaxDist",
  "contactforcesc",
  "contactthreshold",
  "patchval",
  "patchwidth",
  "wrongfade",
  "maxBackboneDist",
  "m_force",
  "m_k",
  "m_alpha",
  "m_c",
  "representativeContact",
  "maxv",
  "boidalignment",
  "boidcohesion",
  "boidseparation",
  "boidseparationpow",
  "boidrad",
  "adhesionRange",
  "protrusionRange",
  "protrusionFull",
] as const satisfies readonly (keyof SpringParams)[];

/** The u32 fields that follow the vector tables in the uniform block. */
const UINTS = [
  "n",
  "K",
  "step",
  "seed",
  "damptype",
  "dims",
  "msize",
  "mask",
  "pairs",
  "hasContact",
  "hasDist",
  "torsion",
  "tissueMask",
] as const;
const VEC_BYTES = 272;
const UNI_BYTES = Math.ceil((VEC_BYTES + UINTS.length * 4 + (SCALARS.length + 2) * 4) / 16) * 16;

const COMMON = /* wgsl */ `
const NO_BOND = 0xffffffffu;
const DEAD = ${DEAD}.0;
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn hash3(a: u32, b: u32, c: u32) -> u32 { return pcg(a ^ pcg(b ^ pcg(c))); }
fn rand01(a: u32, b: u32, c: u32) -> f32 { return f32(hash3(a, b, c)) / 4294967296.0; }
fn pairHash(i: u32, j: u32, step: u32, salt: u32) -> u32 {
  return hash3(min(i, j), (max(i, j) * 0x9e3779b1u) ^ salt, step);
}
`;

const STEP_WGSL = /* wgsl */ `
${COMMON}
struct P {
  repulsion: array<vec4f, 4>,
  adhesion: array<vec4f, 4>,
  roleforces: array<vec4f, 2>,
  roleforcesFix: array<vec4f, 2>,
  mobility: vec4f,
  motility: vec4f,
  protrusion: vec4f,
  thrust: vec4f,
  centre: vec4f,
${UINTS.map((u) => `  ${u}: u32,`).join("\n")}
${SCALARS.map((s) => `  ${s}: f32,`).join("\n")}
  pairRange: f32,
  densityRange: f32,
}
@group(0) @binding(0) var<uniform> p: P;
// [pos n | prev n], each [x, y, z, density].
@group(0) @binding(1) var<storage, read> state: array<vec4f>;
// Bond slots as raw words: word 0 is the partner and kind, words 1-3 are f32 bit patterns.
// (Bound as u32 because an empty slot's word 0 is a NaN pattern, which f32 need not preserve.)
@group(0) @binding(2) var<storage, read> slots: array<vec4u>;
// Per particle: rod, fix, pull, [species, radius, torsion angle, torsion stiffness].
@group(0) @binding(3) var<storage, read> attrs: array<vec4f>;
// [contact msize² | dist msize²].
@group(0) @binding(4) var<storage, read> mats: array<f32>;
@group(0) @binding(5) var<storage, read_write> outState: array<vec4f>;

fn soft(rr: f32) -> f32 {
  if (rr <= 0.7) { return 1.0; }
  let ss = 1.0 - (0.3 * 0.3) / (rr + 1.0 - 1.4);
  return ss / rr;
}

fn adhesionForce(len: f32, sigma: f32, repulsion: f32, adhesion: f32, range: f32) -> f32 {
  if (len < sigma) { return -repulsion * (sigma - len); }
  let reach = sigma * range;
  if (adhesion == 0.0 || len >= sigma + reach) { return 0.0; }
  let u = (len - sigma) / reach;
  return adhesion * 4.0 * u * (1.0 - u);
}

// Force of the torsion spring over particles a .. a+3 on the one in position \`role\`. The same
// arithmetic as \`torsionForces\` in torsion.ts, about b as origin.
fn torsion(a: u32, role: u32) -> vec3f {
  var result = vec3f(0.0);
  let own = attrs[a * 4u + 3u];
  var strength = p.backbonetorsionspringforce;
  var target_ = p.backbonetorsionspringangle;
  if (own.w != 0.0) {
    strength = own.w * p.torsionspringforce;
    target_ = own.z;
  }
  let pb = state[a + 1u].xyz;
  let pa = state[a].xyz - pb;
  let pc = state[a + 2u].xyz - pb;
  let pd = state[a + 3u].xyz - pb;
  let vab = -pa;
  let vcd = pd - pc;
  let maxSq = p.torsionMaxLen * p.torsionMaxLen;
  let bcSq = dot(pc, pc);
  var u = cross(pc, vab);
  var w = cross(vcd, pc);
  let lenABC = length(u);
  let lenBCD = length(w);
  let alive = attrs[a * 4u + 3u].x != DEAD && attrs[a * 4u + 7u].x != DEAD && attrs[a * 4u + 11u].x != DEAD && attrs[a * 4u + 15u].x != DEAD;
  let sane = dot(pa, pa) <= maxSq && bcSq <= maxSq && dot(vcd, vcd) <= maxSq && bcSq > 1e-18 && lenABC > 1e-9 && lenBCD > 1e-9;
  if (alive && sane && strength != 0.0) {
    u /= lenABC;
    w /= lenBCD;
    let angle = atan2(dot(cross(u, w), pc) / sqrt(bcSq), dot(u, w));
    var stress = target_ - angle;
    if (stress > 3.14159265358979) { stress -= 6.28318530717959; }
    if (stress < -3.14159265358979) { stress += 6.28318530717959; }
    if (stress < -1.5707963267949) { stress = -3.14159265358979 - stress; }
    if (stress > 1.5707963267949) { stress = 3.14159265358979 - stress; }
    let ss = min(1.0, lenBCD * lenABC);
    let fa = u * (lenBCD * ss);
    let fd = -w * (lenABC * ss);
    let k1 = -(fa + fd);
    let k3 = cross(k1, pc) + cross(fa, pa) + cross(fd, pd);
    let fb = cross(pc, k3) / bcSq;
    let fc = k1 - fb;
    let over = (dot(fa, fa) + dot(fb, fb) + dot(fc, fc) + dot(fd, fd)) * strength * strength;
    if (over <= 1000.0) {
      var clampBy = 1.0;
      if (over > 1.0) { clampBy = 1.0 / sqrt(over); }
      let dd = stress * strength * ${TORSION_SCALE}.0 * clampBy;
      if (role == 0u) { result = fa * dd; }
      if (role == 1u) { result = fb * dd; }
      if (role == 2u) { result = fc * dd; }
      if (role == 3u) { result = fd * dd; }
    }
  }
  return result;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = p.n;
  if (i >= n) { return; }
  let old4 = state[i];
  let mine = attrs[i * 4u + 3u];
  // The previous position rides along so the next step can difference it for velocity.
  outState[n + i] = old4;
  if (mine.x == DEAD) { outState[i] = old4; return; }
  let si = u32(mine.x);
  let old = old4.xyz;
  let olddensity = old4.w;
  let rate = p.springrate;
  var vel = (old - state[n + i].xyz) / rate;
  var force = vec3f(0.0);

  // Bonded springs.
  for (var k = 0u; k < p.K; k++) {
    let spr = slots[i * p.K + k];
    let bits = spr.x;
    if (bits == NO_BOND) { continue; }
    let j = bits & 0xffffffu;
    let kind = bits >> 24u;
    let dir = state[j].xyz - old;
    let len = length(dir);
    if (len == 0.0) { continue; }
    let lspringlen = bitcast<f32>(spr.y) * p.springlen + p.springlenfix;
    let lspringforce = bitcast<f32>(spr.z) * (p.springforce * p.roleforces[kind / 4u][kind % 4u] + p.roleforcesFix[kind / 4u][kind % 4u]);
    let lspringpow = bitcast<f32>(spr.w) + p.springpow;
    var sforce = (len - lspringlen) * lspringforce * min(1000.0, pow(len / p.powBaseDist, lspringpow));
    let bdj = max(i, j) - min(i, j);
    if (bdj == 1u) { sforce *= p.backboneStrength; }
    force += dir * min(sforce / len, 0.5 / rate);
  }

  // Torsion springs: this particle is one of four in up to four of them.
  if (p.torsion == 1u) {
    for (var role = 0u; role < 4u; role++) {
      if (i >= role && i - role + 3u < n) { force += torsion(i - role, role); }
    }
  }

  // Pull towards a point.
  let pull = attrs[i * 4u + 2u];
  let hasPull = pull.w != 0.0;
  if (hasPull && p.pullspringforce != 0.0) {
    let kk = min(p.pullspringforce * abs(pull.w), 1.0 / rate);
    let v = pull.xyz - old;
    let vl = length(v);
    if (vl > 0.0) { force += v * (kk * pow(vl, p.pullspringpow)); }
  }

  // Pair forces.
  var density = 0.0;
  var dirdensity = 0.0;
  if (p.pairs == 1u) {
    var boidpos = vec3f(0.0);
    var boidvel = vec3f(0.0);
    var boidn = 0.0;
    let csynthI = ((p.mask >> si) & 1u) == 1u;
    let protrude = p.protrusion[si];
    let thrust = p.thrust[si];
    let outward = protrude != 0.0 || thrust != 0.0;
    var away = vec3f(0.0);
    var awayWeight = 0.0;
    for (var j = 0u; j < n; j++) {
      if (j == i) { continue; }
      let metaJ = attrs[j * 4u + 3u];
      if (metaJ.x == DEAD) { continue; }
      let sj = u32(metaJ.x);
      let other = state[j];
      let dir = other.xyz - old;
      let len = length(dir);
      if (len == 0.0 || len > p.pairRange) { continue; }

      if (len < p.densityRange) {
        density += 1.0 / (len * len * len);
        dirdensity += max(0.0, dot(dir, vel) / len) / (len * len);
      }
      if (outward && (sj == si || ((p.tissueMask >> sj) & 1u) == 1u) && len < p.protrusionRange) {
        let w = 1.0 - len / p.protrusionRange;
        awayWeight += w;
        away -= dir * (w / len);
      }

      var lforce = 0.0;
      var gforce = 0.0;
      var bforce = 0.0;

      if (csynthI && ((p.mask >> sj) & 1u) == 1u) {
        let bdu = max(i, j) - min(i, j);
        let bd = f32(bdu);
        let inRange = bd / f32(n) < p.maxBackboneDist;
        let inMatrix = i < p.msize && j < p.msize;
        let m = i * p.msize + j;

        let pushapartuse = min(bd, p.nonBackboneLen) * p.backboneScale + p.pushapartdelta;
        lforce += -p.pushapartlocalforce * (1.0 - smoothstep(0.5, 1.0, len / pushapartuse));
        if (p.pushapartDensityFactor != 0.0) {
          gforce -= p.pushapartDensityFactor * olddensity * other.w * pow(len / p.powBaseDist, p.pushapartDensityPow);
        }
        if (p.fractforce != 0.0 && inRange) { bforce += p.fractforce * pow(bd, p.fractpow) * len; }
        if (p.xyzforce != 0.0 && inRange && inMatrix && p.hasDist == 1u) {
          let targlen = mats[p.msize * p.msize + m];
          if (targlen < p.xyzMaxDist) { gforce += (len - targlen) * p.xyzforce / pow(targlen, p.xyzpow); }
        }
        if (p.backboneforce != 0.0 && bd <= 1.5) { bforce += (len - p.backboneScale) * p.backboneforce; }

        var uwrongfade = 1.0;
        var rawContact = 0.0;
        if (inMatrix && p.hasContact == 1u) { rawContact = mats[m]; }
        if (p.contactforcesc != 0.0 && inRange && inMatrix && p.hasContact == 1u) {
          var contact = rawContact;
          if (contact < -1000.0) {
            bforce = 0.0;
            gforce = 0.0;
            uwrongfade = 0.0;
          } else if (contact <= -9.0) {
            gforce = 0.0;
            uwrongfade = 0.0;
          } else {
            contact = max(0.0, contact - p.contactthreshold);
            if (bd <= 1.5) { contact *= p.backboneContactStrength; }
            if (p.wrongfade != 0.0 && p.pushapartforce != 0.0) {
              let tlen = pow(p.contactforcesc * contact * p.powBaseDist / p.pushapartforce, 1.0 / (p.pushapartpow - 1.0)) * p.powBaseDist;
              let cc = 1.0 / (p.wrongfade * p.wrongfade);
              let q = 1.0 - len / tlen;
              let f = cc + q * q;
              uwrongfade = cc * cc / (f * f);
            }
            gforce += p.contactforcesc * contact * len * uwrongfade;
          }
        }

        if (bd > 1.5 || p.ignoreBackbone != 0.0) {
          gforce += -p.pushapartforce * pow(max(len, p.backboneScale) / p.powBaseDist, p.pushapartpow) * uwrongfade;
        }

        if (p.boidalignment != 0.0 && len < p.boidrad) {
          boidpos += other.xyz;
          boidvel += (other.xyz - state[n + j].xyz) / rate;
          boidn += 1.0;
          gforce -= p.boidseparation * pow(max(len, 0.01) / p.powBaseDist, p.boidseparationpow);
        }

        if (p.m_force != 0.0 && inMatrix && p.hasContact == 1u) {
          var contact = rawContact;
          if (contact <= -9.0) {
            gforce = 0.0;
            if (bd <= p.patchwidth) { contact = p.patchval; }
          }
          contact /= p.representativeContact;
          if (bd < 1.5) { contact = p.maxv / p.representativeContact * p.backboneContactStrength; }
          if (contact > 0.0) {
            let dd = p.m_k * pow(contact, -p.m_alpha) - len;
            let dem = p.m_c * p.m_c + dd * dd;
            gforce += p.m_force * contact * -2.0 * p.m_c * p.m_c * dd / (dem * dem);
          }
        }

        gforce *= pow(len / p.powBaseDist, p.springpow);
      }

      lforce += adhesionForce(len, mine.y + metaJ.y, p.repulsion[si][sj], p.adhesion[si][sj], p.adhesionRange);
      force += dir * ((gforce + lforce + bforce) / len);
    }
    let awayLen = length(away);
    var outwardScale = 0.0;
    if (awayLen > 1e-6) { outwardScale = (protrude * max(0.0, 1.0 - awayWeight / p.protrusionFull) + thrust) / awayLen; }
    force += outwardScale * away;
    if (boidn > 0.0) {
      force += (boidpos / boidn - old) * p.boidcohesion + (boidvel / boidn - vel) * p.boidalignment;
    }
  }

  // Viscosity.
  if (p.visc != 0.0) {
    let ll = dot(vel, vel);
    if (ll > 0.0) { force -= p.visc * vel * pow(ll, p.viscpow * 0.5 - 0.5); }
  }

  // Soft spherical container.
  if (p.modelSphereForce > 0.0) {
    let r = length(old);
    let xdist = r - p.modelSphereRadius;
    if (xdist > 0.0) { force -= old * (p.modelSphereForce * xdist / r); }
  }

  // Mobility, then the random terms.
  force *= p.mobility[si];
  let key = p.seed ^ (p.step * 0x9e3779b1u);
  if (p.noiseforce != 0.0 && rand01(i, key, ${SALT.noiseGate}u) < p.noiseprob) {
    force.x += (rand01(i, key, ${SALT.noiseX}u) - 0.5) * p.noiseforce;
    force.y += (rand01(i, key, ${SALT.noiseY}u) - 0.5) * p.noiseforce;
    if (p.dims == 3u) { force.z += (rand01(i, key, ${SALT.noiseZ}u) - 0.5) * p.noiseforce; }
  }
  let mot = p.motility[si];
  if (mot != 0.0) {
    force.x += (rand01(i, key, ${SALT.motileX}u) - 0.5) * 2.0 * mot;
    force.y += (rand01(i, key, ${SALT.motileY}u) - 0.5) * 2.0 * mot;
    if (p.dims == 3u) { force.z += (rand01(i, key, ${SALT.motileZ}u) - 0.5) * 2.0 * mot; }
  }

  // Damping.
  var use_ = p.damp;
  let speed = length(vel);
  if (p.damptype == 2u) {
    let velsq = speed * speed;
    use_ = max(0.0, 1.0 - p.damp * velsq * velsq);
  } else if (p.damptype == 3u) {
    let v = max(1e-14, speed);
    use_ = pow(pow(v, -3.0) + p.damp, -1.0 / 3.0) / v;
  } else if (p.damptype == 4u) {
    let v = max(1e-14, speed * (1.0 + dirdensity));
    use_ = clamp(pow(pow(v * p.damp, -3.0) + 1.0, -1.0 / 3.0) / v, 0.0, 1.0);
  }
  if (p.damp < 0.0) { use_ = -p.damp; }

  // Integrate.
  vel = vel * use_ + force;
  vel.y -= p.gravity;
  vel *= soft(length(vel) / p.springmaxvel);
  let vsum = vel.x + vel.y + vel.z;
  if (!(abs(vsum) < 3.0e38)) { vel = vec3f(0.0); }
  var mypos = old + vel * rate;
  if (dot(vel, force) < 0.0) { mypos = (old + mypos) * 0.5; }

  // Rod.
  let rod = attrs[i * 4u];
  if (rod.x >= 0.0 && rod.w >= 0.0) {
    let otherpos = state[u32(rod.x)].xyz;
    let dir = otherpos - mypos;
    let len = length(dir);
    if (len >= 0.001) {
      let lo = rod.y * p.springlen + p.springlenfix;
      let hi = rod.z * p.springlen + p.springlenfix;
      let want = clamp(len, lo, hi) * rod.w + len * (1.0 - rod.w);
      mypos = otherpos - dir * (want / len);
    }
  }

  // Centring, fix, pull-as-fix.
  mypos = p.centre.xyz + (mypos - p.centre.xyz) * p.springCentreDamp;
  let fix = attrs[i * 4u + 1u];
  if (fix.x == 1.0) { mypos = fix.yzw; }
  if (p.pullfixdamp != 0.0 && hasPull) { mypos = p.pullfixdamp * pull.xyz + (1.0 - p.pullfixdamp) * mypos; }

  outState[i] = vec4f(mypos, density);
}
`;

const TURNOVER_WGSL = /* wgsl */ `
${COMMON}
struct T {
  kindOf: array<vec4i, 4>,
  strength: array<vec4f, 2>,
  n: u32,
  K: u32,
  step: u32,
  seed: u32,
  kinds: u32,
  useStrain: u32,
  breakThreshold: u32,
  formThreshold: u32,
  restCurrent: u32,
  tensionOnly: u32,
  maxStrain: f32,
  springlen: f32,
  springlenfix: f32,
  range: f32,
  rest: f32,
}
@group(0) @binding(0) var<uniform> t: T;
@group(0) @binding(1) var<storage, read> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> slots: array<vec4u>;
@group(0) @binding(3) var<storage, read> attrs: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> proposal: array<u32>;

fn species(i: u32) -> f32 { return attrs[i * 4u + 3u].x; }

// Break: each thread clears the doomed slots of its own row. Both ends reach the same verdict.
@compute @workgroup_size(${WG})
fn breakMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= t.n) { return; }
  for (var k = 0u; k < t.K; k++) {
    let o = i * t.K + k;
    let spr = slots[o];
    let bits = spr.x;
    if (bits == NO_BOND) { continue; }
    let j = bits & 0xffffffu;
    var doomed = species(i) == DEAD || species(j) == DEAD;
    if (!doomed && ((t.kinds >> (bits >> 24u)) & 1u) == 1u) {
      if (t.useStrain == 1u) {
        let len = length(state[j].xyz - state[i].xyz);
        let rest = bitcast<f32>(spr.y) * t.springlen + t.springlenfix;
        var strain = len / rest - 1.0;
        if (t.tensionOnly == 0u) { strain = abs(strain); }
        if (strain > t.maxStrain) { doomed = true; }
      }
      if (pairHash(i, j, t.step, t.seed ^ ${SALT.bondBreak}u) < t.breakThreshold) { doomed = true; }
    }
    if (doomed) { slots[o].x = NO_BOND; }
  }
}

fn hasFree(i: u32) -> bool {
  var found = false;
  for (var k = 0u; k < t.K; k++) {
    found = found || slots[i * t.K + k].x == NO_BOND;
  }
  return found;
}

fn bonded(i: u32, j: u32) -> bool {
  var found = false;
  for (var k = 0u; k < t.K; k++) {
    let bits = slots[i * t.K + k].x;
    found = found || (bits != NO_BOND && (bits & 0xffffffu) == j);
  }
  return found;
}

// Propose: the nearest eligible neighbour, lowest index on a tie.
@compute @workgroup_size(${WG})
fn proposeMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= t.n) { return; }
  var best = NO_BOND;
  var bestLen = 3.4e38;
  let si = species(i);
  if (si != DEAD && hasFree(i)) {
    let here = state[i].xyz;
    for (var j = 0u; j < t.n; j++) {
      let sj = species(j);
      if (j == i || sj == DEAD) { continue; }
      if (t.kindOf[u32(si)][u32(sj)] < 0) { continue; }
      let len = length(state[j].xyz - here);
      if (len > t.range || len == 0.0) { continue; }
      if (len > bestLen || (len == bestLen && j > best)) { continue; }
      if (bonded(i, j) || !hasFree(j)) { continue; }
      if (pairHash(i, j, t.step, t.seed ^ ${SALT.bondForm}u) >= t.formThreshold) { continue; }
      best = j;
      bestLen = len;
    }
  }
  proposal[i] = best;
}

// Accept: a bond forms where two particles proposed to each other.
@compute @workgroup_size(${WG})
fn acceptMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= t.n) { return; }
  let j = proposal[i];
  if (j == NO_BOND || proposal[j] != i) { return; }
  for (var k = 0u; k < t.K; k++) {
    let o = i * t.K + k;
    if (slots[o].x != NO_BOND) { continue; }
    let kind = u32(t.kindOf[u32(species(i))][u32(species(j))]);
    var rest = t.rest;
    if (t.restCurrent == 1u) { rest = (length(state[j].xyz - state[i].xyz) - t.springlenfix) / t.springlen; }
    slots[o] = vec4u((kind << 24u) | j, bitcast<u32>(rest), bitcast<u32>(t.strength[kind / 4u][kind % 4u]), 0u);
    return;
  }
}
`;

const T_BYTES = 160;

interface Pipes {
  readonly device: GPUDevice;
  readonly step: GPUComputePipeline;
  readonly breakP: GPUComputePipeline;
  readonly propose: GPUComputePipeline;
  readonly accept: GPUComputePipeline;
}

const getPipes = deviceScoped(async (device: GPUDevice): Promise<Pipes> => {
  const stepModule = await compileShader(device, STEP_WGSL, "springs:step");
  const turnModule = await compileShader(device, TURNOVER_WGSL, "springs:turnover");
  // One explicit layout for the three turnover entry points, so they share a bind group even
  // though each uses a different subset of the bindings.
  const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  });
  const turnLayout = device.createBindGroupLayout({
    label: "springs:turnover",
    entries: [
      storage(0, "uniform"),
      storage(1, "read-only-storage"),
      storage(2, "storage"),
      storage(3, "read-only-storage"),
      storage(4, "storage"),
    ],
  });
  const layout = device.createPipelineLayout({ label: "springs:turnover", bindGroupLayouts: [turnLayout] });
  const turn = (entryPoint: string) =>
    device.createComputePipeline({ label: `springs:${entryPoint}`, layout, compute: { module: turnModule, entryPoint } });
  return {
    device,
    step: device.createComputePipeline({ label: "springs:step", layout: "auto", compute: { module: stepModule, entryPoint: "main" } }),
    breakP: turn("breakMain"),
    propose: turn("proposeMain"),
    accept: turn("acceptMain"),
  };
});

function packStepUniform(
  p: SpringParams,
  n: number,
  K: number,
  step: number,
  m: SpringMatrices | undefined,
  pairs: boolean,
  torsion: boolean,
): ArrayBuffer {
  // This fused kernel predates traction and would apply thrust as a force from nowhere.
  if (p.tractionKinds !== 0 && p.thrust.some((v) => v !== 0)) {
    throw new Error("springsGpu: thrust as traction (tractionKinds) is not supported here; run the scenario as a graph (createSpringSim)");
  }
  const buf = new ArrayBuffer(UNI_BYTES);
  const dv = new DataView(buf);
  const f32s = (offset: number, values: readonly number[], count: number) => {
    for (let k = 0; k < count; k++) dv.setFloat32(offset + k * 4, values[k] ?? 0, true);
  };
  f32s(0, p.repulsion, MAX_SPECIES * MAX_SPECIES);
  f32s(64, p.adhesion, MAX_SPECIES * MAX_SPECIES);
  f32s(128, p.roleforces, MAX_KINDS);
  f32s(160, p.roleforcesFix, MAX_KINDS);
  f32s(192, p.mobility, MAX_SPECIES);
  f32s(208, p.motility, MAX_SPECIES);
  f32s(224, p.protrusion, MAX_SPECIES);
  f32s(240, p.thrust, MAX_SPECIES);
  f32s(256, p.springCentre, 3);
  const uints: Record<(typeof UINTS)[number], number> = {
    n,
    K,
    step,
    seed: p.seed >>> 0,
    damptype: p.damptype,
    dims: p.dims,
    msize: m?.size ?? 0,
    mask: p.pairSpeciesMask,
    pairs: pairs ? 1 : 0,
    hasContact: m?.contact ? 1 : 0,
    hasDist: m?.dist ? 1 : 0,
    torsion: torsion ? 1 : 0,
    tissueMask: p.tissueMask,
  };
  UINTS.forEach((u, k) => dv.setUint32(VEC_BYTES + k * 4, uints[u], true));
  const base = VEC_BYTES + UINTS.length * 4;
  SCALARS.forEach((s, k) => dv.setFloat32(base + k * 4, p[s], true));
  dv.setFloat32(base + SCALARS.length * 4, p.pairRange ?? 3.0e38, true);
  dv.setFloat32(base + SCALARS.length * 4 + 4, Number.isFinite(p.densityRange) ? p.densityRange : 3.0e38, true);
  return buf;
}

export interface SpringsGpu {
  readonly n: number;
  readonly K: number;
  /** Steps taken since the last `upload`, counted from the uploaded state's own step. */
  readonly stepIndex: number;
  /** Send a state, its topology and (optionally) its target matrices to the device. */
  upload(s: SpringState, t: SpringTopology, m?: SpringMatrices): void;
  /** Advance by `steps` steps. */
  step(p: SpringParams, steps?: number): void;
  /** `breakBonds` on the device. */
  breakBonds(p: SpringParams, rule: BreakRule): void;
  /** `formBonds` on the device. */
  formBonds(p: SpringParams, rule: FormRule): void;
  /** Read positions, previous positions and the bond table back into host objects, as they
   *  were when this was called: the state is snapshotted on the device at the call, so the
   *  model may go on stepping while the read completes. One download at a time. */
  download(s: SpringState, t: SpringTopology): Promise<void>;
  /** The current positions only, `xyz` + density per particle, as they were when this was
   *  called: snapshotted on the device at the call, like `download`. One read at a time, shared
   *  with `download`. For a host that needs positions now and then (picking, a follow camera)
   *  without bringing the bond table back. */
  snapshotPositions(): Promise<Float32Array>;
  /** The buffer holding the current `[pos | prev]`, for a renderer to bind. */
  positions(): GPUBuffer;
}

let nextId = 0;

/** Device buffers and kernels for a spring model of `n` particles with `K` bond slots. */
export async function createSpringsGpu(n: number, K: number, opts: { device?: GPUDevice } = {}): Promise<SpringsGpu> {
  const pipes = await getPipes(opts.device);
  const { device } = pipes;
  const id = nextId++;
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const make = (label: string, bytes: number, usage = STORAGE) =>
    device.createBuffer({ label: `springs${id}:${label}`, size: Math.max(16, bytes), usage });
  let cur = make("stateA", n * 2 * 16);
  let nxt = make("stateB", n * 2 * 16);
  const slots = make("slots", n * K * 16);
  const attrs = make("attrs", n * 4 * 16);
  const proposal = make("proposal", n * 4);
  const stepUni = make("stepUni", UNI_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const turnUni = make("turnUni", T_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  let mats = make("mats", 16);
  let matrices: SpringMatrices | undefined;
  let stepIndex = 0;
  // A download first copies the state into these, in a submit of its own made at the call. The
  // slow part (mapping) then reads the copies, so the model is free to carry on stepping and the
  // host still gets exactly the state of the step it asked at.
  const snapState = make("snapState", n * 2 * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const snapSlots = make("snapSlots", n * K * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  let downloading = false;
  let explicitTorsion = false;
  const groups = Math.ceil(n / WG);

  const run = (pipeline: GPUComputePipeline, bind: GPUBindGroup, label: string) => {
    const enc = device.createCommandEncoder({ label });
    const pass = enc.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  const turnBind = () =>
    device.createBindGroup({
      label: "springs:turnover",
      layout: pipes.breakP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: turnUni } },
        { binding: 1, resource: { buffer: cur } },
        { binding: 2, resource: { buffer: slots } },
        { binding: 3, resource: { buffer: attrs } },
        { binding: 4, resource: { buffer: proposal } },
      ],
    });

  const writeTurnUniform = (p: SpringParams, br: BreakRule | undefined, fr: FormRule | undefined) => {
    const buf = new ArrayBuffer(T_BYTES);
    const dv = new DataView(buf);
    for (let a = 0; a < MAX_SPECIES; a++) {
      for (let b = 0; b < MAX_SPECIES; b++) dv.setInt32((a * 4 + b) * 4, fr ? (fr.kindOf[a * fr.speciesCount + b] ?? -1) : -1, true);
    }
    for (let k = 0; k < MAX_KINDS; k++) dv.setFloat32(64 + k * 4, fr?.strength[k] ?? 0, true);
    const u = [
      n,
      K,
      stepIndex,
      p.seed >>> 0,
      br?.kinds ?? 0,
      br?.maxStrain !== undefined ? 1 : 0,
      br?.tau !== undefined ? probThreshold(1 / br.tau) : 0,
      probThreshold(fr?.prob ?? 1),
      fr?.rest === "current" ? 1 : 0,
      br?.tensionOnly ? 1 : 0,
    ];
    u.forEach((v, k) => dv.setUint32(96 + k * 4, v, true));
    const f = [br?.maxStrain ?? 0, p.springlen, p.springlenfix, fr?.range ?? 0, typeof fr?.rest === "number" ? fr.rest : 0];
    f.forEach((v, k) => dv.setFloat32(136 + k * 4, v, true));
    device.queue.writeBuffer(turnUni, 0, buf);
  };

  return {
    n,
    K,
    get stepIndex() {
      return stepIndex;
    },
    upload(s, t, m) {
      if (s.n !== n || t.n !== n || t.K !== K) throw new Error(`springsGpu: built for n=${n}, K=${K}; got n=${s.n}, K=${t.K}`);
      const st = new Float32Array(n * 8);
      st.set(s.pos, 0);
      st.set(s.prev, n * 4);
      writeView(device.queue, cur, st);
      writeView(device.queue, slots, t.slots);
      const at = new Float32Array(n * 16);
      for (let i = 0; i < n; i++) {
        at.set(t.rod.subarray(i * 4, i * 4 + 4), i * 16);
        at.set(t.fix.subarray(i * 4, i * 4 + 4), i * 16 + 4);
        at.set(t.pull.subarray(i * 4, i * 4 + 4), i * 16 + 8);
        at[i * 16 + 12] = s.species[i]!;
        at[i * 16 + 13] = s.radius[i]!;
        at[i * 16 + 14] = t.torsion[i * 2]!;
        at[i * 16 + 15] = t.torsion[i * 2 + 1]!;
      }
      writeView(device.queue, attrs, at);
      matrices = m;
      explicitTorsion = t.torsion.some((v, k) => k % 2 === 1 && v !== 0);
      if (m) {
        const mm = m.size * m.size;
        const packed = new Float32Array(mm * 2);
        if (m.contact) packed.set(m.contact, 0);
        if (m.dist) packed.set(m.dist, mm);
        if (mats.size < packed.byteLength) mats = make("mats", packed.byteLength);
        writeView(device.queue, mats, packed);
      }
      stepIndex = s.step;
    },
    step(p, steps = 1) {
      const pairs = pairTermsActive(p);
      const torsion = p.backbonetorsionspringforce !== 0 || (explicitTorsion && p.torsionspringforce !== 0);
      for (let k = 0; k < steps; k++) {
        device.queue.writeBuffer(stepUni, 0, packStepUniform(p, n, K, stepIndex, matrices, pairs, torsion));
        const bind = device.createBindGroup({
          label: "springs:step",
          layout: pipes.step.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stepUni } },
            { binding: 1, resource: { buffer: cur } },
            { binding: 2, resource: { buffer: slots } },
            { binding: 3, resource: { buffer: attrs } },
            { binding: 4, resource: { buffer: mats } },
            { binding: 5, resource: { buffer: nxt } },
          ],
        });
        run(pipes.step, bind, "springs:step");
        [cur, nxt] = [nxt, cur];
        stepIndex++;
      }
    },
    breakBonds(p, rule) {
      writeTurnUniform(p, rule, undefined);
      run(pipes.breakP, turnBind(), "springs:break");
    },
    formBonds(p, rule) {
      writeTurnUniform(p, undefined, rule);
      const bind = turnBind();
      run(pipes.propose, bind, "springs:propose");
      run(pipes.accept, bind, "springs:accept");
    },
    async download(s, t) {
      // One at a time: a second would overwrite the snapshot the first has yet to read.
      if (downloading) throw new Error("springsGpu: download called while another download is pending");
      downloading = true;
      const step = stepIndex;
      const stateBytes = n * 2 * 16;
      const slotBytes = n * K * SLOT_WORDS * 4;
      const enc = device.createCommandEncoder({ label: `springs${id}:snapshot` });
      enc.copyBufferToBuffer(cur, 0, snapState, 0, stateBytes);
      enc.copyBufferToBuffer(slots, 0, snapSlots, 0, slotBytes);
      device.queue.submit([enc.finish()]);
      try {
        // The staging keys carry no instance id: the pool never frees a buffer, and reads that
        // share a key take turns. The host objects change only once both have arrived.
        const [stBytes, slBytes] = await Promise.all([
          readBackBytes(device, "springs:state", snapState, 0, stateBytes),
          readBackBytes(device, "springs:slots", snapSlots, 0, slotBytes),
        ]);
        const st = new Float32Array(stBytes);
        s.pos.set(st.subarray(0, n * 4));
        s.prev.set(st.subarray(n * 4));
        s.step = step;
        t.slotBits.set(new Uint32Array(slBytes));
      } finally {
        downloading = false;
      }
    },
    async snapshotPositions() {
      if (downloading) throw new Error("springsGpu: snapshotPositions called while another download is pending");
      downloading = true;
      const enc = device.createCommandEncoder({ label: `springs${id}:snapshotPositions` });
      enc.copyBufferToBuffer(cur, 0, snapState, 0, n * 16);
      device.queue.submit([enc.finish()]);
      try {
        return new Float32Array(await readBackBytes(device, "springs:state", snapState, 0, n * 16));
      } finally {
        downloading = false;
      }
    },
    positions() {
      return cur;
    },
  };
}

/**
 * Run a spheroid on the device. The step and bond turnover are kernels; division needs a free
 * slot per daughter, so every `syncEvery` steps the state comes back to the host, cells divide
 * there (with the division rate scaled to cover the steps in between), and it goes back up.
 * After each `advance` the scenario's `state` and `topology` are current.
 *
 * Not tracked on this path: `totals.broken`, `totals.formed` and `takeEvents()`.
 */
export async function createSpheroidGpuRunner(
  sc: SpheroidScenario,
  opts: { device?: GPUDevice; syncEvery?: number } = {},
): Promise<{ advance(steps: number): Promise<void> }> {
  const syncEvery = opts.syncEvery ?? 10;
  const gpu = await createSpringsGpu(sc.state.n, sc.topology.K, opts);
  gpu.upload(sc.state, sc.topology);
  let sinceSync = 0;
  const sync = async () => {
    await gpu.download(sc.state, sc.topology);
    let changed = 0;
    const division = sc.rules.division;
    if (division && sinceSync > 0) {
      const born = divideCells(sc.state, sc.params, { ...division, tau: Math.max(1, division.tau / sinceSync) });
      sc.totals.divisions += born;
      changed += born;
    }
    const leaders = sc.rules.leaders;
    if (leaders && sinceSync > 0) changed += selectLeaders(sc.state, sc.params, { ...leaders, tau: Math.max(1, leaders.tau / sinceSync) });
    // With leaders switched off nothing else would turn the existing ones back into cells.
    if (!leaders) changed += clearLeaders(sc.state, LEADER, CELL);
    if (changed > 0) gpu.upload(sc.state, sc.topology);
    sinceSync = 0;
  };
  let advancing = false;
  return {
    async advance(steps) {
      // A second call while one is awaiting its sync would share `sinceSync` and upload growth
      // computed from an older snapshot.
      if (advancing) throw new Error("createSpheroidGpuRunner: advance called while a previous advance is still running");
      advancing = true;
      try {
        for (let k = 0; k < steps; k++) {
          const rules = sc.rules;
          gpu.step(sc.params);
          if (rules.turnover) gpu.breakBonds(sc.params, rules.breakRule);
          if (rules.turnover || sc.config.ligand > 0) gpu.formBonds(sc.params, rules.formRule);
          if (++sinceSync >= syncEvery) await sync();
        }
        // Already current when the batch ended exactly on a sync.
        if (sinceSync > 0) await sync();
      } finally {
        advancing = false;
      }
    },
  };
}
