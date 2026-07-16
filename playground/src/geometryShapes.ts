// Shared example shapes for the implicit (SDF/CSG) geometry-kind (ADR-0010) — consumed by both the
// mesh view (`geometryMain.ts`, dual-contoured triangles) and the raymarch view (`raymarchMain.ts`,
// exact per-pixel). All built from our ops (`box`/`sphere`/`plane`/`intersect`/`union`/…).

import { box, type Implicit, plane, sphere } from "../../src/geometry";

// ── Roofs from eaves (re-derived from aaquickhouse `CSGlib/BigRoof.cs`) ──────────────────────
// Each footprint edge (eave) extrudes up-and-inward into an inclined roof half-space; a mass capped
// by intersecting those planes is a hip roof, and unioning masses makes valleys.

/** A roof half-space through `through` with (un-normalised) normal `n`; the solid is below it. */
function roofPlane(n: [number, number, number], through: [number, number, number]): Implicit {
  const len = Math.hypot(n[0], n[1], n[2]) || 1;
  return plane(n, (n[0] * through[0] + n[1] * through[1] + n[2] * through[2]) / len);
}

/** A rectangular-footprint mass (walls) capped by four eave-derived roof planes → a hip roof. */
function roofedBox(cx: number, cz: number, hx: number, hz: number, eaveY: number, pitchRise: number): Implicit {
  const tall = 3;
  const b = 1 / pitchRise; // roof-plane normal is (out, b, 0): slope = 1/b = pitchRise
  return box(hx, tall / 2, hz)
    .translate(cx, tall / 2, cz)
    .intersect(roofPlane([1, b, 0], [cx + hx, eaveY, cz]))
    .intersect(roofPlane([-1, b, 0], [cx - hx, eaveY, cz]))
    .intersect(roofPlane([0, b, 1], [cx, eaveY, cz + hz]))
    .intersect(roofPlane([0, b, -1], [cx, eaveY, cz - hz]));
}

/** A gable roof: roof planes only on the two sides perpendicular to the ridge (along X if
 *  `ridgeAlongX`, else Z); the ends stay vertical gable walls. A wing built this way meets a main
 *  roof so its side planes form valleys landing on the main roof plane. */
function gabledBox(cx: number, cz: number, hx: number, hz: number, eaveY: number, pitchRise: number, ridgeAlongX: boolean): Implicit {
  const tall = 3;
  const b = 1 / pitchRise;
  const body = box(hx, tall / 2, hz).translate(cx, tall / 2, cz);
  return ridgeAlongX
    ? body.intersect(roofPlane([0, b, 1], [cx, eaveY, cz + hz])).intersect(roofPlane([0, b, -1], [cx, eaveY, cz - hz]))
    : body.intersect(roofPlane([1, b, 0], [cx + hx, eaveY, cz])).intersect(roofPlane([-1, b, 0], [cx - hx, eaveY, cz]));
}

/** The meshed half of the hybrid demo: a plain ridged hip-roofed house the plane BSP renders exactly
 *  (base on the ground, ridge along X). Shared so the raymarch view can mesh the same house the mesh
 *  view shows. */
export function hybridHouse(): Implicit {
  return roofedBox(0, 0, 0.6, 0.45, 0.62, 1.0); // footprint 1.2×0.9, eaves at 0.62, ridge ~1.07
}

/** The raymarched half: a lumpy, noise-displaced blob sitting on the house roof — the organic ornament
 *  that breaks the BSP (ADR-0013), so it is raymarched and composited against the meshed house by
 *  depth. The raymarch animates the noise domain over time. */
export function hybridGrowth(): Implicit {
  return sphere(0.28).displace(0.19, 3.6).translate(0.34, 1.0, 0.16);
}

export interface Shape {
  name: string;
  /** Half-extent of the mesh sampling cube (mesh view only; the raymarch view ignores it). */
  bounds: number;
  make: () => Implicit;
}

export const SHAPES: Shape[] = [
  { name: "Hip-roofed house", bounds: 1.4, make: () => roofedBox(0, 0, 1.0, 0.55, 0.5, 1.0) },
  // Cross-wing T-junction: a gabled wing meets the main wall midpoint; its side planes form valleys
  // landing on the main roof (butt end buried in the main). Subordinate ridge avoids CSG slivers.
  {
    name: "Cross-wing house (T-junction)",
    bounds: 1.4,
    make: () => roofedBox(0, 0, 1.05, 0.5, 0.5, 1.0).union(gabledBox(0, 0.7, 0.32, 0.55, 0.4, 0.9, false)),
  },
  // L-shape: two equal-height, flush arms (same half-width perpendicular to ridge → same ridge
  // height), each a COMPLETE four-plane hip mass, then unioned. The inner hips are exactly coplanar
  // with the neighbour's long-side plane (arm A's +x hip ≡ arm B's +x plane; arm B's −z hip ≡ arm A's
  // −z plane) — but in a union (SDF `min`) a coplanar cap is idempotent, so it just caps each arm at
  // the shared plane instead of letting a full-height gable poke through. The reentrant corner gets a
  // valley (arm A's +z plane meets arm B's −x plane) and the outer corner a hip, both for free. This
  // is BigRoof's "union masses → valleys"; the union absorbs the coplanarity rather than dodging it.
  {
    name: "L-shaped house",
    bounds: 1.6,
    make: () => {
      const eave = 0.5;
      const pitch = 1.0;
      const h = 0.4; // shared half-width ⇒ both ridges at eave + h
      const armA = roofedBox(-0.3, 0, 0.7, h, eave, pitch); // horizontal (long +x)
      const armB = roofedBox(0, 0.3, h, 0.7, eave, pitch); // vertical (long +z)
      return armA.union(armB);
    },
  },
  // A grid of hip-roofed houses — 16 separate masses, so the BSP takes its octree-localised path
  // (distant houses never split/classify against each other). Meshes to 16×9 clean faces.
  {
    name: "Village (4×4 houses)",
    bounds: 3.2,
    make: () => {
      const n = 4;
      const spacing = 1.4;
      const at = (k: number) => (k - (n - 1) / 2) * spacing;
      let g = roofedBox(at(0), at(0), 0.42, 0.42, 0.35, 1.0);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i || j) g = g.union(roofedBox(at(i), at(j), 0.42, 0.42, 0.35, 1.0));
      return g;
    },
  },
  // The raymarched growth on its own (the hybrid view pairs it with a meshed house). In the mesh view
  // the grid dual-contour tessellates it — a lumpy blob — since the BSP can't (noise isn't planar).
  { name: "Alien growth (noise)", bounds: 1.6, make: () => hybridGrowth() },
  { name: "Cube ∖ sphere (bite)", bounds: 1.5, make: () => box(1).subtract(sphere(1.22)) },
  { name: "Sphere ∩ cube (lens-box)", bounds: 1.2, make: () => sphere(1.1).intersect(box(0.85)) },
  {
    name: "Smooth-union blob",
    bounds: 1.7,
    make: () =>
      sphere(0.7)
        .translate(-0.45, 0, 0)
        .smoothUnion(sphere(0.7).translate(0.45, 0, 0), 0.45)
        .smoothUnion(box(0.9, 0.28, 0.28), 0.35),
  },
  {
    name: "Bolt head (cube ∩ sphere ∖ bore)",
    bounds: 1.3,
    make: () =>
      box(0.85)
        .intersect(sphere(1.05))
        .subtract(sphere(0.42).translate(0, 0.9, 0)),
  },
];
