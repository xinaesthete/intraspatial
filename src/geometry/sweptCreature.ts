// Swept-creature profile — the CPU golden for the dancer's unified geometry: a superegg *nose*
// that tapers into a *tube tail* swept over the motion trail. The shader-side mirror is
// `docs-site/src/lib/creatureTsl.ts` (same CPU/TSL split as the okLab and superellipsoid pairs).
//
// The creature is a surface of revolution with a BENT axis: circular cross-sections of radius
// R(u) centred on a centreline C(u), swept from the nose tip (u = 0) to the tail (u = 1). Only
// the axial PROFILE — how the radius and the nose's forward reach vary with u — is pure and
// golden-testable here; the centreline itself is the live GPU trail (sampled in the shader) and
// the framing/normals are twist-immune analytic terms (see the TSL module).
//
//   • Nose region (surface of revolution of the superellipse front lobe): as the along-parameter
//     s goes 0 → 1 (tip → shoulder), the radius grows 0 → 1 and the forward reach shrinks 1 → 0,
//     following signed powers of sin/cos so the exponent `e` shapes it (e = 1 hemispherical,
//     e < 1 fuller/boxier egg, matching the superellipsoid convention).
//   • Body region: the radius tapers 1 → 0 from the shoulder to the tail as (1 - x)^p.
//
// Continuity at the shoulder is exact: noseRadial(1) = bodyTaper(0) = 1 and noseAxial(1) = 0.

import { signPow } from "./superellipsoid";

/** Nose radius profile: 0 at the tip (s = 0) → 1 at the shoulder (s = 1). */
export function noseRadial(s: number, e: number): number {
  return signPow(Math.sin((s * Math.PI) / 2), e);
}

/** Nose forward reach: 1 at the tip (s = 0) → 0 at the shoulder (s = 1). */
export function noseAxial(s: number, e: number): number {
  return signPow(Math.cos((s * Math.PI) / 2), e);
}

/** Body radius taper: 1 at the shoulder (x = 0) → 0 at the tail (x = 1). */
export function bodyTaper(x: number, p: number): number {
  return (1 - Math.min(Math.max(x, 0), 1)) ** p;
}
