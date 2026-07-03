// Small TSL geometry helpers for building per-instance model transforms explicitly in the vertex
// shader (reading the sim's pos/orientation state buffers), rather than relying on three's opaque
// instanced positionLocal/instanceMatrix handling. This is the level the superegg deformation will
// also work at. Thin adapter — three's TSL node types are inconsistent, so args are loosely typed
// (see oklabTsl for the same rationale).
import { cos, cross, dot, Fn, length, max, oneMinus, sin } from "three/tsl";

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
  return v.mul(c).add(cross(k, v).mul(s)).add(k.mul(dot(k, v).mul(oneMinus(c))));
});
