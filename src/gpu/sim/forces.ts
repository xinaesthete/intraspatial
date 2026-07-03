// The force influences — the single source of truth for the dance dynamics, ported
// closely from Andy Lomas's DANCERL (docs/DANCERL.ESME). Each is a pure function of an
// agent's state (and, for the pairwise ones, the neighbour positions) returning an
// acceleration contribution (a `Vec3`). The op-graph building blocks (CPU `cpuGolden`)
// and the GPU artefact (TSL compute) both implement *these* — so the math lives once.
//
// Strength is the [0,1] breedable trait; each force multiplies it by a base coefficient
// tuned for the seed's world scale (a shell of radius ≈ SHELL). The *calm, deliberate*
// Ceilidh feel comes mostly from the integrator's DANCERL `timeFactor` and damping, not
// from shrinking these — keep the influence shapes faithful.
import { add, addScaled, cross, dot, length, normalize, readVec3, scale, sub, unpack, vec3, type Vec3, type Vec3In, ZERO3 } from "./vec3";

/** World scale the seed uses; forces are tuned around it. */
export const SHELL = 4.5;
const EPS = 1e-6;

// ── Field forces (a function of one agent's own state) ──────────────────────────────

/** Containment toward the origin — DANCERL `ConstraintForce` (a restoring "box" that
 *  keeps the swarm gathered; cubic in distance by default). F = −k·S·dᵖ⁻¹·pos. */
export function constrainForce(pos: Vec3In, strength: number, power = 3): Vec3 {
  const d = length(pos);
  if (d < EPS) return ZERO3;
  const k = 0.012 * strength * d ** (power - 1);
  return scale(pos, -k);
}

/** Outward drift from the centre — DANCERL `SwimForce`. */
export function swimForce(pos: Vec3In, strength: number): Vec3 {
  return scale(normalize(pos), 0.02 * strength);
}

/** Circular stirring about the y-axis — DANCERL `CircleForce`. A = (z, 0, −x). */
export function vortexForce(pos: Vec3In, strength: number): Vec3 {
  const [x, , z] = unpack(pos);
  const a = vec3(z, 0, -x);
  return scale(normalize(a), 0.04 * strength);
}

/** Orbital acceleration about the scene centre — DANCERL `OrbitForce`, A = (p×v)×p. */
export function orbitForce(pos: Vec3In, vel: Vec3In, strength: number): Vec3 {
  const a = cross(cross(pos, vel), pos);
  return scale(normalize(a), 0.05 * strength);
}

/** The distinctive single-coil solenoid field about the y-axis — DANCERL
 *  `SolenoidForce`, ported. Returns a unit-ish direction scaled by strength. */
export function solenoidForce(pos: Vec3In, strength: number, coilRadius = 2.5): Vec3 {
  const [x, y, z] = unpack(pos);
  const lxz = Math.hypot(x, z);
  const coeff = 0.045 * strength;
  if (lxz < EPS) return vec3(0, -coeff, 0);
  const R = coilRadius;
  const k = ((R - lxz) * (R - lxz) + y * y) / ((R + lxz) * (R + lxz) + y * y);
  const stY = (R * (1 + k)) / (1 - k + EPS) - lxz;
  const f = vec3((-y * x) / lxz, -stY, (-y * z) / lxz);
  const dir = normalize(f, EPS, [0, -1, 0]);
  return scale(dir, coeff);
}

// ── Pairwise forces (read the neighbour positions from a flat [x,y,z,…] buffer) ──────

/** Cohesion toward the centroid of neighbours within `radius` — a local reinterpretation
 *  of DANCERL's `DistanceForce` bonds (radius ≥ world ⇒ global centroid). */
export function cohereForce(i: number, pos: ArrayLike<number>, n: number, strength: number, radius: number): Vec3 {
  const p = readVec3(pos, i);
  const [px, py, pz] = unpack(p);
  const r2 = radius * radius;
  let cx = 0, cy = 0, cz = 0, count = 0;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const [qx, qy, qz] = unpack(readVec3(pos, j));
    const dx = qx - px, dy = qy - py, dz = qz - pz;
    if (dx * dx + dy * dy + dz * dz < r2) {
      cx += qx;
      cy += qy;
      cz += qz;
      count++;
    }
  }
  if (count === 0) return ZERO3;
  const centroid = vec3(cx / count, cy / count, cz / count);
  return scale(sub(centroid, p), 0.012 * strength);
}

/** Radius repulsion from near neighbours — DANCERL `CollisionForce`. Falls off toward the
 *  edge of `radius`; weighted by inverse distance so close pairs push hardest. */
export function separateForce(i: number, pos: ArrayLike<number>, n: number, strength: number, radius: number): Vec3 {
  const [px, py, pz] = unpack(readVec3(pos, i));
  const r2 = radius * radius;
  const k = 0.06 * strength;
  let acc: Vec3 = ZERO3;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const [qx, qy, qz] = unpack(readVec3(pos, j));
    const d = vec3(px - qx, py - qy, pz - qz);
    const d2 = dot(d, d);
    if (d2 < r2 && d2 > EPS) {
      const dist = Math.sqrt(d2);
      acc = addScaled(acc, d, (k * (1 - d2 / r2)) / dist);
    }
  }
  return acc;
}

/** Spring bonds to neighbours within `radius`, pulling each pair toward `restLength` —
 *  DANCERL `DistanceForce` (attract if too far, repel if too close). */
export function springForce(
  i: number,
  pos: ArrayLike<number>,
  n: number,
  strength: number,
  restLength: number,
  radius: number,
): Vec3 {
  const [px, py, pz] = unpack(readVec3(pos, i));
  const r2 = radius * radius;
  const k = 0.03 * strength;
  let acc: Vec3 = ZERO3;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const [qx, qy, qz] = unpack(readVec3(pos, j));
    const d = vec3(qx - px, qy - py, qz - pz);
    const d2 = dot(d, d);
    if (d2 < r2 && d2 > EPS) {
      const dist = Math.sqrt(d2);
      // +ve when too far (pull together), −ve when too close (push apart).
      acc = addScaled(acc, d, (k * (dist - restLength)) / dist);
    }
  }
  return acc;
}

/** Orbit a *specific* partner point — DANCERL pairwise `SetOrbit`. The couple swings
 *  around the midpoint; `partner` is the partner's position. */
export function partnerOrbitForce(pos: Vec3In, vel: Vec3In, partner: Vec3In, strength: number): Vec3 {
  const mid = scale(add(pos, partner), 0.5);
  const rel = sub(pos, mid);
  if (length(rel) < EPS) return ZERO3;
  const a = cross(cross(rel, vel), rel);
  return scale(normalize(a), 0.05 * strength);
}
