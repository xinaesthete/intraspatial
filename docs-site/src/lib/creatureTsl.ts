// Swept-creature geometry for TSL — the shader-side mirror of `src/geometry/sweptCreature.ts`
// (golden-tested on the CPU). The dancer is a superegg NOSE that tapers into a TUBE TAIL swept
// over its motion trail: one attribute-less surface synthesised entirely from `vertexIndex`, its
// centreline read live from the GPU trail-history ring (no readback, no vertex buffers).
//
// This module holds the pure, buffer-independent pieces: the (segment × radial) grid decode and
// the axial profile functions. The renderer assembles the rest — sampling the trail ring for the
// body centreline, the nose extension along the heading, the per-ring frame, and a twist-immune
// analytic normal — because those depend on the live storage buffers and per-instance state.
//
// Thin transcription adapter — three's TSL node types are inconsistent, so nodes are loosely
// typed (same rationale as oklabTsl / supereggTsl / tslTransform).
import { abs, cos, float, Fn, int, pow, sign, sin, vec2, vertexIndex } from "three/tsl";

// biome-ignore lint/suspicious/noExplicitAny: three TSL node types are inconsistent; adapter layer.
type Tsl = any;

/** Cross-sections along the body (nose tip → tail); the tube has this many quad rings. */
export const CREATURE_SEGMENTS = 22;
/** Points around each circular cross-section (the seam closes at the last one). */
export const CREATURE_RADIAL = 9;
/** How many of the front segments form the superegg nose (the rest are the tapering tail). */
export const CREATURE_HEAD_SEGMENTS = 7;
/** Synthesised vertices (6 per quad, non-indexed). Drives the geometry's vertex count. */
export const CREATURE_VERTEX_COUNT = CREATURE_SEGMENTS * CREATURE_RADIAL * 6;

const HALF_PI = Math.PI / 2;

/** Signed power sign(x)·|x|^e (exponents here are > 0, so |x|^e is finite at 0). */
const signPow = (x: Tsl, e: Tsl): Tsl => sign(x).mul(pow(abs(x), e));

/** (ri, θ) for the current `vertexIndex`: ri ∈ [0, SEGMENTS] is the cross-section (0 = nose tip),
 *  θ ∈ [0, 2π] the angle around. Same quad-grid winding as the superegg (du = +around, dv = +along):
 *    tri A: (0,0)(1,0)(1,1)   tri B: (0,0)(1,1)(0,1). */
export const creatureCell = Fn((): Tsl => {
  const vi = int(vertexIndex);
  const quad = vi.div(int(6));
  const corner = vi.mod(int(6));
  const aroundSeg = quad.mod(int(CREATURE_RADIAL));
  const alongSeg = quad.div(int(CREATURE_RADIAL));
  const isDu = corner.equal(int(1)).or(corner.equal(int(2))).or(corner.equal(int(4)));
  const isDv = corner.equal(int(2)).or(corner.equal(int(4))).or(corner.equal(int(5)));
  const ai = float(aroundSeg).add(isDu.select(float(1), float(0))); // 0..RADIAL (around corner)
  const ri = float(alongSeg).add(isDv.select(float(1), float(0))); // 0..SEGMENTS (along corner)
  const theta = ai.div(float(CREATURE_RADIAL)).mul(2 * Math.PI);
  return vec2(ri, theta);
});

/** Nose radius profile: 0 at the tip (s = 0) → 1 at the shoulder (s = 1). */
export const noseRadial = Fn(([s, e]: [Tsl, Tsl]): Tsl => signPow(sin(s.mul(HALF_PI)), e));

/** Nose forward reach: 1 at the tip (s = 0) → 0 at the shoulder (s = 1). */
export const noseAxial = Fn(([s, e]: [Tsl, Tsl]): Tsl => signPow(cos(s.mul(HALF_PI)), e));

/** Body radius taper: 1 at the shoulder (x = 0) → 0 at the tail (x = 1). `x` is assumed clamped. */
export const bodyTaper = Fn(([x, p]: [Tsl, Tsl]): Tsl => pow(float(1).sub(x), p));
