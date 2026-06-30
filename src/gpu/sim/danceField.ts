// danceField — a reinterpretation of Andy Lomas's 1992 DANCERL force-field motion
// controller (written for William Latham's SIGGRAPH film in the IBM ESME/Mutator
// language Stephen Todd & William Latham built). DANCERL drove organisms not by
// keyframes but by a *superposition of named force influences*; motion emerged from
// how those fields composed and ramped over a lifecycle. This module keeps that idea —
// the behaviour is the art — while modernising the dynamics for an interactive,
// breedable instrument.
//
// A swarm of N agents, each pos+vel in 3D. One step sums the enabled influences into a
// per-agent acceleration, then integrates semi-implicitly (Euler) with damping and a
// speed cap. The influences are reinterpreted from DANCERL's roster:
//
//   attract    — cubic containment toward the origin (DANCERL ConstraintForce: a
//                restoring "box" that keeps the swarm gathered)
//   orbit      — angular momentum about the scene centre (OrbitForce: (p×v)×p)
//   vortex     — circular stirring about the y-axis (CircleForce)
//   solenoid   — the distinctive single-coil solenoid field about the y-axis
//                (SolenoidForce, ported closely)
//   swim       — outward drift from the centre (SwimForce)
//   cohesion   — pull toward the neighbour centroid (a flocking reinterpretation of
//                DANCERL's DistanceForce bonds)
//   separation — radius repulsion between neighbours (CollisionForce)
//
// Each influence carries a strength AND an on/off enable — the "hybrid specimen" the
// Mutator breeds (see src/evo). Pure CPU here (the golden, and — at the swarm sizes a
// dancer uses — the live path too); a GPU kernel can follow the reactionDiffusion.ts
// pattern when the breeding grid needs it.

/** Resolved dance parameters — the *phenotype* of a specimen (see src/evo). Each `*On`
 *  flag is an ENABLE trait; the rest are NUMBER traits (strengths in [0,1] unless
 *  noted). */
export interface DanceParams {
  attract: number;
  orbit: number;
  vortex: number;
  solenoid: number;
  swim: number;
  cohesion: number;
  separation: number;
  /** Neighbour radius for separation, in world units. */
  sepRadius: number;
  /** Velocity retained per step (0..1) — DANCERL's LinDamp. */
  damping: number;
  /** Maximum speed (world units / step). */
  speedLimit: number;
  attractOn: boolean;
  orbitOn: boolean;
  vortexOn: boolean;
  solenoidOn: boolean;
  swimOn: boolean;
  cohesionOn: boolean;
  separationOn: boolean;
  /** Integration step (kept fixed; folded into the coefficients). */
  dt: number;
}

export interface SwarmState {
  /** Interleaved xyz per agent, length 3·n. */
  pos: Float32Array;
  /** Interleaved xyz per agent, length 3·n. */
  vel: Float32Array;
  n: number;
}

// Per-influence coefficients mapping a [0,1] strength to a sane acceleration at the
// world scale the seed uses (a shell of radius ≈ SHELL). Tuned so a neutral specimen
// (all strengths 0.5, all influences on) orbits liveliy without blowing up.
const K = {
  attract: 0.018,
  orbit: 0.05,
  vortex: 0.04,
  solenoid: 0.045,
  swim: 0.02,
  cohesion: 0.012,
  separation: 0.06,
};
const SHELL = 4.5;
const EPS = 1e-6;
/** Hard cap on radius so a degenerate specimen can't send a preview to infinity. */
const MAX_R = 60;

const DEFAULTS: DanceParams = {
  attract: 0.5,
  orbit: 0.5,
  vortex: 0.5,
  solenoid: 0.5,
  swim: 0.5,
  cohesion: 0.5,
  separation: 0.5,
  sepRadius: 1.5,
  damping: 0.94,
  speedLimit: 1.5,
  attractOn: true,
  orbitOn: true,
  vortexOn: false,
  solenoidOn: false,
  swimOn: false,
  cohesionOn: true,
  separationOn: true,
  dt: 1,
};

/** A reproducible swarm: agents on a jittered spherical shell with a small tangential
 *  kick (so orbiting has something to bite on). Deterministic in `seed`. */
export function seedSwarm(n: number, seed: number): SwarmState {
  const pos = new Float32Array(n * 3);
  const vel = new Float32Array(n * 3);
  let a = seed >>> 0;
  const rnd = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    // Uniform-ish point on a sphere via two angles, jittered radius.
    const u = rnd() * 2 - 1; // cos(theta)
    const phi = rnd() * Math.PI * 2;
    const s = Math.sqrt(Math.max(0, 1 - u * u));
    const r = SHELL * (0.7 + 0.6 * rnd());
    const x = r * s * Math.cos(phi);
    const y = r * u;
    const z = r * s * Math.sin(phi);
    pos[i * 3] = x;
    pos[i * 3 + 1] = y;
    pos[i * 3 + 2] = z;
    // Small tangential kick about the y-axis.
    const k = 0.15;
    vel[i * 3] = -z * k * 0.1 + (rnd() - 0.5) * 0.05;
    vel[i * 3 + 1] = (rnd() - 0.5) * 0.05;
    vel[i * 3 + 2] = x * k * 0.1 + (rnd() - 0.5) * 0.05;
  }
  return { pos, vel, n };
}

/** One dance step on the CPU — the golden, and the live path at dancer swarm sizes.
 *  Returns a fresh state; the input is untouched. */
export function danceStepCpu(state: SwarmState, params: Partial<DanceParams> = {}): SwarmState {
  const p = { ...DEFAULTS, ...params };
  const n = state.n;
  const src = state.pos;
  const sv = state.vel;
  const pos = new Float32Array(n * 3);
  const vel = new Float32Array(n * 3);

  // Neighbour centroid (cohesion) is the same for every agent — compute once.
  let cx = 0, cy = 0, cz = 0;
  if (p.cohesionOn && p.cohesion > 0 && n > 1) {
    for (let i = 0; i < n; i++) {
      cx += src[i * 3]!;
      cy += src[i * 3 + 1]!;
      cz += src[i * 3 + 2]!;
    }
    cx /= n;
    cy /= n;
    cz /= n;
  }

  for (let i = 0; i < n; i++) {
    const x = src[i * 3]!, y = src[i * 3 + 1]!, z = src[i * 3 + 2]!;
    const vx0 = sv[i * 3]!, vy0 = sv[i * 3 + 1]!, vz0 = sv[i * 3 + 2]!;
    const r = Math.hypot(x, y, z);
    let fx = 0, fy = 0, fz = 0;

    // attract — cubic containment toward origin (stronger the further out).
    if (p.attractOn && p.attract > 0) {
      const k = p.attract * K.attract * (1 + 0.04 * r * r);
      fx -= k * x;
      fy -= k * y;
      fz -= k * z;
    }

    // orbit — accelerate along the orbital tangent (p×v)×p (DANCERL OrbitForce).
    if (p.orbitOn && p.orbit > 0 && r > EPS) {
      // c = p × v
      const cxv = y * vz0 - z * vy0;
      const cyv = z * vx0 - x * vz0;
      const czv = x * vy0 - y * vx0;
      // t = c × p
      let tx = cyv * z - czv * y;
      let ty = czv * x - cxv * z;
      let tz = cxv * y - cyv * x;
      const tl = Math.hypot(tx, ty, tz);
      if (tl > EPS) {
        tx /= tl; ty /= tl; tz /= tl;
        const k = p.orbit * K.orbit;
        fx += k * tx; fy += k * ty; fz += k * tz;
      }
    }

    // vortex — circular stirring about the y-axis (DANCERL CircleForce).
    if (p.vortexOn && p.vortex > 0) {
      const l = Math.hypot(z, x);
      if (l > EPS) {
        const k = p.vortex * K.vortex;
        fx += k * (z / l);
        fz += k * (-x / l);
      }
    }

    // solenoid — single-coil solenoid field about the y-axis (DANCERL SolenoidForce).
    if (p.solenoidOn && p.solenoid > 0) {
      const R = 2.5;
      const lxz = Math.hypot(x, z);
      let sfx: number, sfy: number, sfz: number;
      if (lxz < EPS) {
        sfx = 0; sfy = -1; sfz = 0;
      } else {
        const kk = ((R - lxz) * (R - lxz) + y * y) / ((R + lxz) * (R + lxz) + y * y);
        const stY = (R * (1 + kk)) / (1 - kk + EPS) - lxz;
        sfx = (-y * x) / lxz;
        sfy = -stY;
        sfz = (-y * z) / lxz;
        const sl = Math.hypot(sfx, sfy, sfz);
        if (sl < EPS) { sfx = 0; sfy = -1; sfz = 0; }
        else { sfx /= sl; sfy /= sl; sfz /= sl; }
      }
      const k = p.solenoid * K.solenoid;
      fx += k * sfx; fy += k * sfy; fz += k * sfz;
    }

    // swim — outward from the centre (DANCERL SwimForce).
    if (p.swimOn && p.swim > 0 && r > EPS) {
      const k = (p.swim * K.swim) / r;
      fx += k * x; fy += k * y; fz += k * z;
    }

    // cohesion — toward the swarm centroid (reinterpreted DistanceForce bond).
    if (p.cohesionOn && p.cohesion > 0 && n > 1) {
      const k = p.cohesion * K.cohesion;
      fx += k * (cx - x); fy += k * (cy - y); fz += k * (cz - z);
    }

    // separation — radius repulsion from near neighbours (DANCERL CollisionForce).
    if (p.separationOn && p.separation > 0 && n > 1) {
      const rad = p.sepRadius;
      const rad2 = rad * rad;
      const k = p.separation * K.separation;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = x - src[j * 3]!;
        const dy = y - src[j * 3 + 1]!;
        const dz = z - src[j * 3 + 2]!;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < rad2 && d2 > EPS) {
          const w = k * (1 - d2 / rad2) / Math.sqrt(d2);
          fx += w * dx; fy += w * dy; fz += w * dz;
        }
      }
    }

    // Semi-implicit Euler with damping and a speed cap (DANCERL LinDamp + stability).
    let vx = (vx0 + fx * p.dt) * p.damping;
    let vy = (vy0 + fy * p.dt) * p.damping;
    let vz = (vz0 + fz * p.dt) * p.damping;
    const sp = Math.hypot(vx, vy, vz);
    if (sp > p.speedLimit && sp > EPS) {
      const s = p.speedLimit / sp;
      vx *= s; vy *= s; vz *= s;
    }
    let nx = x + vx * p.dt;
    let ny = y + vy * p.dt;
    let nz = z + vz * p.dt;
    // Containment backstop so a degenerate specimen can't escape to infinity.
    const nr = Math.hypot(nx, ny, nz);
    if (nr > MAX_R) {
      const s = MAX_R / nr;
      nx *= s; ny *= s; nz *= s;
      vx *= 0.5; vy *= 0.5; vz *= 0.5;
    }
    pos[i * 3] = nx; pos[i * 3 + 1] = ny; pos[i * 3 + 2] = nz;
    vel[i * 3] = vx; vel[i * 3 + 1] = vy; vel[i * 3 + 2] = vz;
  }

  return { pos, vel, n };
}

/** Advance a swarm by `steps` CPU steps. */
export function danceStepsCpu(state: SwarmState, steps: number, params: Partial<DanceParams> = {}): SwarmState {
  let s = state;
  for (let i = 0; i < steps; i++) s = danceStepCpu(s, params);
  return s;
}

// ── Field packing ────────────────────────────────────────────────────────────────
// The swarm travels the op graph as one vec3 `points` field of n=2N rows: rows [0,N)
// are positions, rows [N,2N) are velocities. So data length = 2N·3 = 6N, satisfying
// the field invariant (numCells·elementLanes), and a single feedback node carries the
// whole state across ticks.

/** Pack a swarm into a [pos(3N) ‖ vel(3N)] Float32Array (length 6N). */
export function packSwarm(state: SwarmState): Float32Array {
  const out = new Float32Array(state.n * 6);
  out.set(state.pos, 0);
  out.set(state.vel, state.n * 3);
  return out;
}

/** Unpack a [pos ‖ vel] buffer of n=2·N rows back into a swarm. */
export function unpackSwarm(data: ArrayLike<number>, rows2N: number): SwarmState {
  const n = rows2N >> 1;
  const pos = new Float32Array(n * 3);
  const vel = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) pos[i] = data[i]!;
  for (let i = 0; i < n * 3; i++) vel[i] = data[n * 3 + i]!;
  return { pos, vel, n };
}

/** Project positions to [x0,y0,x1,y1,…] (length 2N) for the top-down scatter preview. */
export function swarmXY(state: SwarmState): Float32Array {
  const out = new Float32Array(state.n * 2);
  for (let i = 0; i < state.n; i++) {
    out[i * 2] = state.pos[i * 3]!;
    out[i * 2 + 1] = state.pos[i * 3 + 1]!;
  }
  return out;
}
