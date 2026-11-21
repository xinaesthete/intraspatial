// One step of the spring model on the CPU: the reference the GPU kernel is tested against.
//
// A port of the fragment shader in organic/TS/springsfs.ts (`spring`, `pull`, `pairforcesfull`,
// `rod` and `main`), in the same order, with the same parameter names. Every particle reads the
// previous state and writes its own new position, so the update is order-independent.
//
// Left out of the port, on purpose:
//   - `boost` (interactive editing of a contact-map region) and `pullspringmat`;
//   - `minActive` / `maxActive` (partial activation of a chain) and `regionBoundary`;
//   - `gradforce` and `noisefieldforce` (sampled 3D fields; here those are graph fields);
//   - `springminvel`, whose shader line normalises the velocity to length 1 rather than to the
//     minimum;
//   - the sampled pair loop (`PUSHSAMPLES`): `pairRange` and a bucket grid replace it.
// CSynth's boid terms only act between every eighth particle (tadpole heads); here they act
// between all particles the pair terms cover.
// A contact value marked missing (≤ -9) zeroes the global terms for that pair. CSynth also
// assigns `patchval` there but never uses it on the `contactforcesc` path; the Lorentzian path
// does use it, and so does this port.
//
// Torsion springs come from FoldSynth's Java instead (`torsion.ts`), because CSynth's shader for
// them is not in the organic repo.
//
// New terms, all off by default: per-species repulsion, adhesion, mobility, motility and
// protrusion.
import type { BucketGrid } from "#src/spatial/bucketGrid";
import { MAX_SPECIES, type SpringParams } from "./params";
import { rand01, SALT } from "./rng";
import { DEAD, gatherNeighbours, neighbourGrid, type SpringMatrices, type SpringState } from "./state";
import { NO_BOND, SLOT_WORDS, type SpringTopology, slotKind, slotPartner } from "./topology";
import { torsionForces } from "./torsion";

/** CSynth's `soft`: the multiplier that soft-clips a ratio in `[0, ∞)` to below 1. */
export function soft(rr: number): number {
  const softt = 0.7;
  if (rr <= softt) return 1;
  const ss = 1 - ((1 - softt) * (1 - softt)) / (rr + 1 - 2 * softt);
  return ss / rr;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * The new short-range term: scalar force along the line to the other particle (positive pulls
 * together). Overlapping particles repel in proportion to the overlap. Between touching
 * (`sigma`) and `sigma * (1 + range)` they attract, strongest midway and zero at both ends, so
 * the force is continuous.
 */
export function adhesionForce(len: number, sigma: number, repulsion: number, adhesion: number, range: number): number {
  if (len < sigma) return -repulsion * (sigma - len);
  const reach = sigma * range;
  if (adhesion === 0 || len >= sigma + reach) return 0;
  const u = (len - sigma) / reach;
  return adhesion * 4 * u * (1 - u);
}

/** Whether any pairwise term is switched on; when none is, the pair loop is skipped. */
export function pairTermsActive(p: SpringParams): boolean {
  const csynth =
    p.pushapartlocalforce !== 0 ||
    p.pushapartforce !== 0 ||
    p.pushapartDensityFactor !== 0 ||
    p.fractforce !== 0 ||
    p.xyzforce !== 0 ||
    p.backboneforce !== 0 ||
    p.contactforcesc !== 0 ||
    p.m_force !== 0 ||
    p.boidalignment !== 0;
  return (
    csynth ||
    p.repulsion.some((v) => v !== 0) ||
    p.adhesion.some((v) => v !== 0) ||
    p.protrusion.some((v) => v !== 0) ||
    p.thrust.some((v) => v !== 0)
  );
}

/** Advance the state by one step, in place. */
export function stepSprings(s: SpringState, t: SpringTopology, p: SpringParams, m?: SpringMatrices): void {
  const { n, pos, prev, species, radius } = s;
  const next = s.scratch;
  const rate = p.springrate;
  const pairs = pairTermsActive(p);
  const grid: BucketGrid | undefined = pairs && p.pairRange !== undefined ? neighbourGrid(s, p.pairRange, p.dims) : undefined;
  const cand = grid ? new Int32Array(n) : undefined;
  const msize = m?.size ?? 0;
  const step = s.step;
  const torsionOn = p.backbonetorsionspringforce !== 0 || (p.torsionspringforce !== 0 && t.torsion.some((v, k) => k % 2 === 1 && v !== 0));
  const torsionMaxSq = p.torsionMaxLen * p.torsionMaxLen;
  const quadOut = new Float64Array(13);

  for (let i = 0; i < n; i++) {
    const i4 = i * 4;
    const si = species[i]!;
    if (si === DEAD) {
      next.set(pos.subarray(i4, i4 + 4), i4);
      continue;
    }
    const ox = pos[i4]!;
    const oy = pos[i4 + 1]!;
    const oz = pos[i4 + 2]!;
    const olddensity = pos[i4 + 3]!;
    let vx = (ox - prev[i4]!) / rate;
    let vy = (oy - prev[i4 + 1]!) / rate;
    let vz = (oz - prev[i4 + 2]!) / rate;
    let fx = 0;
    let fy = 0;
    let fz = 0;

    // ── Bonded springs: `dospringset` / `spring` ──
    const base = i * t.K * SLOT_WORDS;
    for (let k = 0; k < t.K; k++) {
      const o = base + k * SLOT_WORDS;
      const bits = t.slotBits[o]!;
      if (bits === NO_BOND) continue;
      const j = slotPartner(bits);
      const kind = slotKind(bits);
      const dx = pos[j * 4]! - ox;
      const dy = pos[j * 4 + 1]! - oy;
      const dz = pos[j * 4 + 2]! - oz;
      const len = Math.hypot(dx, dy, dz);
      if (len === 0) continue;
      const lspringlen = t.slots[o + 1]! * p.springlen + p.springlenfix;
      const lspringforce = t.slots[o + 2]! * (p.springforce * p.roleforces[kind]! + p.roleforcesFix[kind]!);
      const lspringpow = t.slots[o + 3]! + p.springpow;
      let sforce = (len - lspringlen) * lspringforce * Math.min(1000, (len / p.powBaseDist) ** lspringpow);
      if (Math.abs(i - j) === 1) sforce *= p.backboneStrength;
      // Cap so that a bond cannot overshoot: half, because the other end moves too.
      const kk = Math.min(sforce / len, 0.5 / rate);
      fx += dx * kk;
      fy += dy * kk;
      fz += dz * kk;
    }

    // ── Torsion springs: this particle is one of four in up to four of them ──
    if (torsionOn) {
      for (let role = 0; role < 4; role++) {
        const a = i - role;
        if (a < 0 || a + 3 >= n) continue;
        if (species[a] === DEAD || species[a + 1] === DEAD || species[a + 2] === DEAD || species[a + 3] === DEAD) continue;
        const own = t.torsion[a * 2 + 1]!;
        const strength = own !== 0 ? own * p.torsionspringforce : p.backbonetorsionspringforce;
        const angle = own !== 0 ? t.torsion[a * 2]! : p.backbonetorsionspringangle;
        if (!torsionForces(pos, a, a + 1, a + 2, a + 3, angle, strength, quadOut, torsionMaxSq)) continue;
        fx += quadOut[role * 3]!;
        fy += quadOut[role * 3 + 1]!;
        fz += quadOut[role * 3 + 2]!;
      }
    }

    // ── Pull towards a point: `pull` ──
    const pw = t.pull[i4 + 3]!;
    const hasPull = pw !== 0;
    if (hasPull && p.pullspringforce !== 0) {
      const kk = Math.min(p.pullspringforce * Math.abs(pw), 1 / rate);
      const dx = t.pull[i4]! - ox;
      const dy = t.pull[i4 + 1]! - oy;
      const dz = t.pull[i4 + 2]! - oz;
      const sc = kk * Math.hypot(dx, dy, dz) ** p.pullspringpow;
      if (Number.isFinite(sc)) {
        fx += dx * sc;
        fy += dy * sc;
        fz += dz * sc;
      }
    }

    // ── Pair forces: `pairforcesfull`, plus adhesion ──
    let density = 0;
    let dirdensity = 0;
    if (pairs) {
      let bpx = 0;
      let bpy = 0;
      let bpz = 0;
      let bvx = 0;
      let bvy = 0;
      let bvz = 0;
      let boidn = 0;
      const protrude = p.protrusion[si]!;
      const thrust = p.thrust[si]!;
      const outward = protrude !== 0 || thrust !== 0;
      let awayWeight = 0;
      let awayX = 0;
      let awayY = 0;
      let awayZ = 0;
      const count = grid && cand ? gatherNeighbours(grid, ox, oy, oz, cand) : n;
      const csynthI = ((p.pairSpeciesMask >> si) & 1) === 1;
      for (let c = 0; c < count; c++) {
        const j = cand ? cand[c]! : c;
        if (j === i) continue;
        const sj = species[j]!;
        if (sj === DEAD) continue;
        const j4 = j * 4;
        const dx = pos[j4]! - ox;
        const dy = pos[j4 + 1]! - oy;
        const dz = pos[j4 + 2]! - oz;
        const len = Math.hypot(dx, dy, dz);
        if (len === 0) continue;
        if (p.pairRange !== undefined && len > p.pairRange) continue;

        if (len < p.densityRange) {
          density += 1 / (len * len * len);
          dirdensity += Math.max(0, (dx * vx + dy * vy + dz * vz) / len) / (len * len);
        }

        let lforce = 0;
        let gforce = 0;
        let bforce = 0;

        if (csynthI && ((p.pairSpeciesMask >> sj) & 1) === 1) {
          const bd = Math.abs(i - j);
          const inRange = bd / n < p.maxBackboneDist;
          const inMatrix = i < msize && j < msize;
          const otherdensity = pos[j4 + 3]!;

          const pushapartuse = Math.min(bd, p.nonBackboneLen) * p.backboneScale + p.pushapartdelta;
          lforce += -p.pushapartlocalforce * (1 - smoothstep(0.5, 1, len / pushapartuse));
          if (p.pushapartDensityFactor !== 0) {
            gforce -= p.pushapartDensityFactor * olddensity * otherdensity * (len / p.powBaseDist) ** p.pushapartDensityPow;
          }
          if (p.fractforce !== 0 && inRange) bforce += p.fractforce * bd ** p.fractpow * len;
          if (p.xyzforce !== 0 && inRange && inMatrix && m?.dist) {
            const targlen = m.dist[i * msize + j]!;
            if (targlen < p.xyzMaxDist) gforce += ((len - targlen) * p.xyzforce) / targlen ** p.xyzpow;
          }
          if (p.backboneforce !== 0 && bd <= 1.5) bforce += (len - p.backboneScale) * p.backboneforce;

          let uwrongfade = 1;
          const rawContact = inMatrix && m?.contact ? m.contact[i * msize + j]! : 0;
          if (p.contactforcesc !== 0 && inRange && inMatrix && m?.contact) {
            let contact = rawContact;
            if (contact < -1000) {
              bforce = 0;
              gforce = 0;
              uwrongfade = 0;
            } else if (contact <= -9) {
              gforce = 0;
              uwrongfade = 0;
            } else {
              contact = Math.max(0, contact - p.contactthreshold);
              if (bd <= 1.5) contact *= p.backboneContactStrength;
              if (p.wrongfade !== 0 && p.pushapartforce !== 0) {
                // Distance at which contact attraction and global repulsion balance.
                const tlen =
                  ((p.contactforcesc * contact * p.powBaseDist) / p.pushapartforce) ** (1 / (p.pushapartpow - 1)) * p.powBaseDist;
                const cc = 1 / (p.wrongfade * p.wrongfade);
                const q = 1 - len / tlen;
                const f = cc + q * q;
                uwrongfade = (cc * cc) / (f * f);
              }
              gforce += p.contactforcesc * contact * len * uwrongfade;
            }
          }

          if (bd > 1.5 || p.ignoreBackbone !== 0) {
            gforce += -p.pushapartforce * (Math.max(len, p.backboneScale) / p.powBaseDist) ** p.pushapartpow * uwrongfade;
          }

          if (p.boidalignment !== 0 && len < p.boidrad) {
            bpx += pos[j4]!;
            bpy += pos[j4 + 1]!;
            bpz += pos[j4 + 2]!;
            bvx += (pos[j4]! - prev[j4]!) / rate;
            bvy += (pos[j4 + 1]! - prev[j4 + 1]!) / rate;
            bvz += (pos[j4 + 2]! - prev[j4 + 2]!) / rate;
            boidn++;
            gforce -= p.boidseparation * (Math.max(len, 0.01) / p.powBaseDist) ** p.boidseparationpow;
          }

          if (p.m_force !== 0 && inMatrix && m?.contact) {
            let contact = rawContact;
            if (contact <= -9) {
              gforce = 0;
              if (bd <= p.patchwidth) contact = p.patchval;
            }
            contact /= p.representativeContact;
            if (bd < 1.5) contact = (p.maxv / p.representativeContact) * p.backboneContactStrength;
            if (contact > 0) {
              const dd = p.m_k * contact ** -p.m_alpha - len;
              const dem = p.m_c * p.m_c + dd * dd;
              gforce += (p.m_force * contact * -2 * p.m_c * p.m_c * dd) / (dem * dem);
            }
          }

          gforce *= (len / p.powBaseDist) ** p.springpow;
        }

        if (outward && (sj === si || ((p.tissueMask >> sj) & 1) === 1) && len < p.protrusionRange) {
          const w = 1 - len / p.protrusionRange;
          awayWeight += w;
          awayX -= (dx * w) / len;
          awayY -= (dy * w) / len;
          awayZ -= (dz * w) / len;
        }

        const st = si * MAX_SPECIES + sj;
        lforce += adhesionForce(len, radius[i]! + radius[j]!, p.repulsion[st]!, p.adhesion[st]!, p.adhesionRange);

        const sc = (gforce + lforce + bforce) / len;
        fx += dx * sc;
        fy += dy * sc;
        fz += dz * sc;
      }
      const awayLen = Math.hypot(awayX, awayY, awayZ);
      const exposure = Math.max(0, 1 - awayWeight / p.protrusionFull);
      const outwardScale = awayLen > 1e-6 ? (protrude * exposure + thrust) / awayLen : 0;
      fx += outwardScale * awayX;
      fy += outwardScale * awayY;
      fz += outwardScale * awayZ;
      if (boidn > 0) {
        fx += (bpx / boidn - ox) * p.boidcohesion + (bvx / boidn - vx) * p.boidalignment;
        fy += (bpy / boidn - oy) * p.boidcohesion + (bvy / boidn - vy) * p.boidalignment;
        fz += (bpz / boidn - oz) * p.boidcohesion + (bvz / boidn - vz) * p.boidalignment;
      }
    }

    // ── Viscosity ──
    if (p.visc !== 0) {
      const ll = vx * vx + vy * vy + vz * vz;
      if (ll > 0) {
        const llp = ll ** (p.viscpow * 0.5 - 0.5);
        fx -= p.visc * vx * llp;
        fy -= p.visc * vy * llp;
        fz -= p.visc * vz * llp;
      }
    }

    // ── Soft spherical container ──
    if (p.modelSphereForce > 0) {
      const r = Math.hypot(ox, oy, oz);
      const xdist = r - p.modelSphereRadius;
      if (xdist > 0) {
        const sc = (p.modelSphereForce * xdist) / r;
        fx -= ox * sc;
        fy -= oy * sc;
        fz -= oz * sc;
      }
    }

    // ── Mobility, then the random terms ──
    const mob = p.mobility[si]!;
    fx *= mob;
    fy *= mob;
    fz *= mob;
    const key = (p.seed ^ Math.imul(step, 0x9e3779b1)) >>> 0;
    if (p.noiseforce !== 0 && rand01(i, key, SALT.noiseGate) < p.noiseprob) {
      fx += (rand01(i, key, SALT.noiseX) - 0.5) * p.noiseforce;
      fy += (rand01(i, key, SALT.noiseY) - 0.5) * p.noiseforce;
      if (p.dims === 3) fz += (rand01(i, key, SALT.noiseZ) - 0.5) * p.noiseforce;
    }
    const mot = p.motility[si]!;
    if (mot !== 0) {
      fx += (rand01(i, key, SALT.motileX) - 0.5) * 2 * mot;
      fy += (rand01(i, key, SALT.motileY) - 0.5) * 2 * mot;
      if (p.dims === 3) fz += (rand01(i, key, SALT.motileZ) - 0.5) * 2 * mot;
    }

    // ── Damping: the proportion of the old velocity that is kept ──
    let use = p.damp;
    const speed = Math.hypot(vx, vy, vz);
    if (p.damptype === 2) {
      const velsq = speed * speed;
      use = Math.max(0, 1 - p.damp * velsq * velsq);
    } else if (p.damptype === 3) {
      const v = Math.max(1e-14, speed);
      use = (v ** -3 + p.damp) ** (-1 / 3) / v;
    } else if (p.damptype === 4) {
      // Speed is exaggerated by the directional density, to damp collisions hard.
      const v = Math.max(1e-14, speed * (1 + dirdensity));
      use = Math.min(1, Math.max(0, ((v * p.damp) ** -3 + 1) ** (-1 / 3) / v));
    }
    if (p.damp < 0) use = -p.damp;

    // ── Integrate ──
    vx = vx * use + fx;
    vy = vy * use + fy - p.gravity;
    vz = vz * use + fz;
    const clip = soft(Math.hypot(vx, vy, vz) / p.springmaxvel);
    vx *= clip;
    vy *= clip;
    vz *= clip;
    if (!Number.isFinite(vx + vy + vz)) {
      vx = 0;
      vy = 0;
      vz = 0;
    }
    let nx = ox + vx * rate;
    let ny = oy + vy * rate;
    let nz = oz + vz * rate;
    // A particle moving against its force takes half the step: CSynth's fix for bistable pairs.
    if (vx * fx + vy * fy + vz * fz < 0) {
      nx = (ox + nx) * 0.5;
      ny = (oy + ny) * 0.5;
      nz = (oz + nz) * 0.5;
    }

    // ── Rod: `rod` ──
    const rj = t.rod[i4]!;
    const rdamp = t.rod[i4 + 3]!;
    if (rj >= 0 && rdamp >= 0) {
      const dx = pos[rj * 4]! - nx;
      const dy = pos[rj * 4 + 1]! - ny;
      const dz = pos[rj * 4 + 2]! - nz;
      const len = Math.hypot(dx, dy, dz);
      if (len >= 0.001) {
        const lo = t.rod[i4 + 1]! * p.springlen + p.springlenfix;
        const hi = t.rod[i4 + 2]! * p.springlen + p.springlenfix;
        const want = Math.min(hi, Math.max(lo, len)) * rdamp + len * (1 - rdamp);
        const sc = want / len;
        nx = pos[rj * 4]! - dx * sc;
        ny = pos[rj * 4 + 1]! - dy * sc;
        nz = pos[rj * 4 + 2]! - dz * sc;
      }
    }

    // ── Centring, fix, pull-as-fix ──
    const cd = p.springCentreDamp;
    nx = p.springCentre[0] + (nx - p.springCentre[0]) * cd;
    ny = p.springCentre[1] + (ny - p.springCentre[1]) * cd;
    nz = p.springCentre[2] + (nz - p.springCentre[2]) * cd;
    if (t.fix[i4] === 1) {
      nx = t.fix[i4 + 1]!;
      ny = t.fix[i4 + 2]!;
      nz = t.fix[i4 + 3]!;
    }
    if (p.pullfixdamp !== 0 && hasPull) {
      nx = p.pullfixdamp * t.pull[i4]! + (1 - p.pullfixdamp) * nx;
      ny = p.pullfixdamp * t.pull[i4 + 1]! + (1 - p.pullfixdamp) * ny;
      nz = p.pullfixdamp * t.pull[i4 + 2]! + (1 - p.pullfixdamp) * nz;
    }

    next[i4] = nx;
    next[i4 + 1] = ny;
    next[i4 + 2] = nz;
    next[i4 + 3] = density;
  }

  s.scratch = s.prev;
  s.prev = s.pos;
  s.pos = next;
  s.step = step + 1;
}

/** Run `steps` steps. */
export function runSprings(s: SpringState, t: SpringTopology, p: SpringParams, steps: number, m?: SpringMatrices): void {
  for (let k = 0; k < steps; k++) stepSprings(s, t, p, m);
}
