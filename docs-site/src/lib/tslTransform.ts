// Small TSL geometry helpers for building per-instance model transforms explicitly in the vertex
// shader (reading the sim's pos/orientation state buffers), rather than relying on three's opaque
// instanced positionLocal/instanceMatrix handling. This is the level the superegg deformation will
// also work at. Thin adapter — three's TSL node types are inconsistent, so args are loosely typed
// (see oklabTsl for the same rationale).
import { abs, cos, cross, dot, Fn, length, max, mix, normalize, oneMinus, sin, step, vec3 } from "three/tsl";

// biome-ignore lint/suspicious/noExplicitAny: three TSL node types are inconsistent; adapter layer.
type Tsl = any;

/** Rotate vector `v` by the angle-axis rotation `r` (Rodrigues' formula). `|r|` is the angle, `r/|r|`
 *  the axis — the same angle-axis orientation the sim integrates (dancerGpu angPos). Numerically
 *  stable as `|r|→0`: cos→1, sin→0, (1−cos)→0 collapse the formula to `v`. */
export const rotateByAxisAngle = Fn(([r, v]: [Tsl, Tsl]) => {
  const theta = length(r);
  const k = r.div(max(theta, 1e-9));
  const c = cos(theta);
  const s = sin(theta);
  return v
    .mul(c)
    .add(cross(k, v).mul(s))
    .add(k.mul(dot(k, v).mul(oneMinus(c))));
});

/** Map a local vector `v` into a frame whose +z points along the (unit) `forward` — i.e. orient a
 *  shape modelled tip-toward-+z to face `forward`. Right = up×forward, up' = forward×right; roll
 *  about the forward axis is left free (a world-up reference, swapped to world-x when forward is
 *  near-vertical so the basis never degenerates). This replaces the sim's angle-axis facing for the
 *  render: dancers point along their velocity directly, no steering lag. */
export const orientToForward = Fn(([forward, v]: [Tsl, Tsl]) => {
  const f: Tsl = normalize(forward);
  // world-up reference, swapped to world-x when forward is near-vertical (branchless, so the basis
  // never degenerates); step(0.99,|f.y|) is 1 only near the poles.
  const upRef: Tsl = mix(vec3(0, 1, 0), vec3(1, 0, 0), step(0.99, abs(f.y)));
  const r: Tsl = normalize(cross(upRef, f));
  const u: Tsl = cross(f, r);
  return r.mul(v.x).add(u.mul(v.y)).add(f.mul(v.z));
});
