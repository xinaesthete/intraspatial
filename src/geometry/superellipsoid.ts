// Superellipsoid (superquadric) geometry — Piet Hein's "superegg" generalised to 3D, the
// Todd–Latham *Form Synth* lineage. This is the CPU golden reference for the dancer's render
// geometry; the shader-side mirror is `docs-site/src/lib/supereggTsl.ts` (same convention as
// the okLab pair: pure CPU here, TSL there, kept honest by a shared golden test).
//
// Parametric surface over latitude η ∈ [-π/2, π/2] and longitude ω ∈ [-π, π]:
//
//   x = sx · cos^e1(η) · cos^e2(ω)
//   y = sy · cos^e1(η) · sin^e2(ω)
//   z = sz · sin^e1(η)                    (z is the pole axis)
//
// where cos^e(θ) ≡ sign(cos θ)·|cos θ|^e is the signed power. The exponents shape the form:
// e = 1 is an ellipsoid, e → 0 tends to a box, e = 2 pinches toward an octahedron. Piet Hein's
// egg is prolate and gently box-shouldered — e a little below 1 with sz > sx,sy.
//
// The surface has an ANALYTIC normal (no finite differencing), which is what lets each dancer
// carry its own exponents/scale and still light correctly:
//
//   n ∝ ( cos^(2-e1)(η)·cos^(2-e2)(ω) / sx ,
//         cos^(2-e1)(η)·sin^(2-e2)(ω) / sy ,
//         sin^(2-e1)(η)               / sz )   then normalised.
//
// Dividing by the axis scales means this is the correct object-space normal of the *scaled*
// surface, so a downstream rigid orientation (a pure rotation) transforms it correctly.

export type Vec3 = [number, number, number];

/** Signed power: sign(x)·|x|^e. The building block of superquadric coordinates. */
export function signPow(x: number, e: number): number {
  return Math.sign(x) * Math.abs(x) ** e;
}

/** Signed power guarded for non-positive exponents (the normal uses 2-e, which can be ≤ 0):
 *  clamp |x| away from 0 so |x|^e stays finite while the sign is preserved. */
export function signPowGuarded(x: number, e: number, eps = 1e-5): number {
  return Math.sign(x) * Math.max(Math.abs(x), eps) ** e;
}

/** Point on the superellipsoid surface at (η, ω). `scale` is the per-axis size (sx, sy, sz). */
export function superellipsoidPoint(eta: number, omega: number, e1: number, e2: number, scale: Vec3): Vec3 {
  const ce = signPow(Math.cos(eta), e1);
  const se = signPow(Math.sin(eta), e1);
  const co = signPow(Math.cos(omega), e2);
  const so = signPow(Math.sin(omega), e2);
  return [scale[0] * ce * co, scale[1] * ce * so, scale[2] * se];
}

/** Unit outward normal of the superellipsoid at (η, ω), accounting for the axis scale. */
export function superellipsoidNormal(eta: number, omega: number, e1: number, e2: number, scale: Vec3): Vec3 {
  const ce = signPowGuarded(Math.cos(eta), 2 - e1);
  const se = signPowGuarded(Math.sin(eta), 2 - e1);
  const co = signPowGuarded(Math.cos(omega), 2 - e2);
  const so = signPowGuarded(Math.sin(omega), 2 - e2);
  let nx = (ce * co) / scale[0];
  let ny = (ce * so) / scale[1];
  let nz = se / scale[2];
  const len = Math.hypot(nx, ny, nz) || 1;
  nx /= len;
  ny /= len;
  nz /= len;
  return [nx, ny, nz];
}

// --- Attribute-less triangle mesh -------------------------------------------------------------
// The dancer swarm renders this superegg with NO vertex buffers: the shader synthesises every
// vertex from its index (mirroring the trail geometry). These helpers define that index → (η, ω)
// mapping so the CPU golden test can validate the exact same topology the TSL walks.
//
// The parameter domain is a SLICES × STACKS grid of quads; each quad is two triangles, so
// vertex index vi ∈ [0, SLICES·STACKS·6) decodes to a grid corner. The two poles collapse to a
// point (their triangles are degenerate — standard for a UV parametrisation, harmless to draw).

/** Longitude divisions (around ω). */
export const SUPEREGG_SLICES = 24;
/** Latitude divisions (along η, pole to pole). */
export const SUPEREGG_STACKS = 16;

/** Total synthesised vertices for a SLICES × STACKS grid (6 per quad, non-indexed triangles). */
export function supereggVertexCount(slices = SUPEREGG_SLICES, stacks = SUPEREGG_STACKS): number {
  return slices * stacks * 6;
}

// Per-corner grid offsets (du = +ω step, dv = +η step) for the two triangles of a quad, wound so
// the face normal agrees with the analytic outward normal (∂ω × ∂η points outward):
//   tri A: (0,0) (1,0) (1,1)   tri B: (0,0) (1,1) (0,1)
const CORNER_DU = [0, 1, 1, 0, 1, 0];
const CORNER_DV = [0, 0, 1, 0, 1, 1];

/** Grid corner indices (gi ∈ [0, slices], gj ∈ [0, stacks]) for synthesised vertex `vi`. The
 *  stack count doesn't bound the row here — it only scales the η angle in supereggVertexAngles. */
export function supereggGridCorner(vi: number, slices = SUPEREGG_SLICES): [number, number] {
  const quad = Math.floor(vi / 6);
  const corner = vi % 6;
  const col = quad % slices;
  const row = Math.floor(quad / slices);
  return [col + (CORNER_DU[corner] ?? 0), row + (CORNER_DV[corner] ?? 0)];
}

/** (η, ω) angles for synthesised vertex `vi` — the golden of the shader's index math. */
export function supereggVertexAngles(vi: number, slices = SUPEREGG_SLICES, stacks = SUPEREGG_STACKS): [number, number] {
  const [gi, gj] = supereggGridCorner(vi, slices);
  const omega = (gi / slices) * 2 * Math.PI - Math.PI; // [-π, π]
  const eta = (gj / stacks) * Math.PI - Math.PI / 2; // [-π/2, π/2]
  return [eta, omega];
}
