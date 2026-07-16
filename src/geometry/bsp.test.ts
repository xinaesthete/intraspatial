// Plane-native boundary evaluation (`bsp.ts`) validated against the exact SDF golden (`evalSdf`): the
// evaluated mesh must lie ON the field's zero-set and be oriented OUTWARD. Because both the mesh and
// the raymarch consume the same `evalSdf`, these are the same red/green the render is judged by.

import { describe, expect, it } from "vitest";
import { type Brep, brepEdges, brepToMesh, evaluateBrep, mergeCoplanar } from "./bsp";
import { evalSdf, type Sdf } from "./implicit";
import { box, plane, sphere } from "./index";
import type { Vec3 } from "./superellipsoid";

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const centroid = (poly: Vec3[]): Vec3 =>
  mul(
    poly.reduce((acc, v) => add(acc, v), [0, 0, 0] as Vec3),
    1 / poly.length,
  );

/** A spread of interior sample points on a convex face — centroid, and each vertex nudged toward it. */
function facePoints(poly: Vec3[]): Vec3[] {
  const c = centroid(poly);
  return [c, ...poly.map((v) => add(v, mul(sub(c, v), 0.15)))];
}

/** No two triangles are the same (same three vertices, any order) — catches doubled coincident faces. */
function expectNoDuplicateTriangles(mesh: { positions: Float32Array; indices: Uint32Array }): void {
  const key = (i: number): string => {
    const p = mesh.positions;
    return [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]].map((x) => (x as number).toFixed(4)).join(",");
  };
  const seen = new Set<string>();
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const tri = [key(mesh.indices[t] as number), key(mesh.indices[t + 1] as number), key(mesh.indices[t + 2] as number)].sort().join("|");
    expect(seen.has(tri)).toBe(false);
    seen.add(tri);
  }
}

/** Every face lies on the surface (|SDF| ≈ 0) and its normal points outward (SDF grows along +n). A
 *  merged face is validated through its convex `parts` — the outer-loop centroid of a holed face can
 *  fall inside the hole, off the solid. */
function expectValidBoundary(node: Sdf, brep: Brep): void {
  expect(brep.faces.length).toBeGreaterThan(0);
  for (const face of brep.faces) {
    for (const piece of face.parts ?? [face.poly]) {
      for (const p of facePoints(piece)) {
        expect(Math.abs(evalSdf(node, p))).toBeLessThan(1e-5); // on the zero-set
      }
      const c = centroid(piece);
      expect(evalSdf(node, add(c, mul(face.normal, 1e-3)))).toBeGreaterThan(0); // just outside → outside
      expect(evalSdf(node, sub(c, mul(face.normal, 1e-3)))).toBeLessThan(0); // just inside → inside
    }
  }
}

// A hip roof: a box (walls + floor) capped by four inclined eave planes — the archetypal polyhedral
// CSG with no axis-aligned roof (exactly what the grid staircases and this evaluates exactly).
function roofPlane(n: Vec3, through: Vec3) {
  const l = Math.hypot(n[0], n[1], n[2]);
  return plane(n, (n[0] * through[0] + n[1] * through[1] + n[2] * through[2]) / l);
}
const hipRoof = () =>
  box(1, 1.5, 0.6)
    .translate(0, 1.5, 0)
    .intersect(roofPlane([1, 1, 0], [1, 0.5, 0]))
    .intersect(roofPlane([-1, 1, 0], [-1, 0.5, 0]))
    .intersect(roofPlane([0, 1, 1], [0, 0.5, 0.6]))
    .intersect(roofPlane([0, 1, -1], [0, 0.5, -0.6]));

describe("evaluateBrep — plane-native boundary", () => {
  it("meshes a box as six outward quads", () => {
    const brep = evaluateBrep(box(1).node, { bounds: 1.5 });
    expect(brep.faces).toHaveLength(6);
    for (const f of brep.faces) expect(f.poly).toHaveLength(4);
    expectValidBoundary(box(1).node, brep);
  });

  it("triangulates to a flat-shaded fan mesh with per-triangle provenance", () => {
    const mesh = brepToMesh(evaluateBrep(box(1).node, { bounds: 1.5 }));
    expect(mesh.indices.length / 3).toBe(12); // six quads → two tris each
    expect(mesh.vertexCount).toBe(24); // fans don't share vertices (flat shading)
    expect(mesh.facePrim.length).toBe(12); // one provenance id per triangle
    expect(new Set(mesh.facePrim)).toEqual(new Set([0])); // all from the single box primitive
  });

  it("evaluates a hip roof exactly onto the SDF surface", () => {
    const g = hipRoof();
    const brep = evaluateBrep(g.node, { bounds: 2 });
    expectValidBoundary(g.node, brep);
    // Walls + floor from the box (primId 0) and one facet per roof plane (primIds 1–4) all survive.
    const prims = new Set(brep.faces.map((f) => f.primId));
    expect(prims).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("evaluates a subtractive notch (box ∖ box) onto the surface, oriented into the cavity", () => {
    const g = box(1).subtract(box(0.4).translate(1, 0, 0)); // a square bite out of the +x face
    const brep = evaluateBrep(g.node, { bounds: 2 });
    expectValidBoundary(g.node, brep);
    // The cavity walls come from the tool primitive (primId 1), the outer shell from the main (0).
    const prims = new Set(brep.faces.map((f) => f.primId));
    expect(prims.has(0)).toBe(true);
    expect(prims.has(1)).toBe(true);
  });

  it("carries transforms (translate/scale) into world-space planes", () => {
    const g = box(0.5).scale(2).translate(3, -1, 0.5); // → a box(1) centred at (3,-1,0.5)
    const brep = evaluateBrep(g.node, { bounds: 5 });
    expect(brep.faces).toHaveLength(6);
    expectValidBoundary(g.node, brep);
    // A face centre should sit a unit from the translated centre along an axis.
    const centres = brep.faces.map((f) => centroid(f.poly));
    expect(centres.some((c) => Math.abs(c[0] - 4) < 1e-6)).toBe(true); // +x face at x = 3 + 1
  });
});

describe("evaluateBrep — coincident faces", () => {
  it("unions two abutting boxes, dropping the interior shared wall", () => {
    const shared = box(1).union(box(1).translate(2, 0, 0)); // both cap the plane x = 1, opposite normals
    const brep = evaluateBrep(shared.node, { bounds: 3 });
    expectValidBoundary(shared.node, brep);
    // The shared wall is interior to the union → no face lies on x = 1.
    const wallAt1 = brep.faces.filter((f) => f.poly.every((v) => Math.abs(v[0] - 1) < 1e-6));
    expect(wallAt1).toHaveLength(0);
    // The merged solid still spans x ∈ [−1, 3].
    const xs = brep.faces.flatMap((f) => f.poly.map((v) => v[0]));
    expect(Math.min(...xs)).toBeCloseTo(-1, 6);
    expect(Math.max(...xs)).toBeCloseTo(3, 6);
    expectNoDuplicateTriangles(brepToMesh(brep));
  });

  it("collapses same-oriented coincident caps to one face (A ∩ A = A)", () => {
    const g = box(1).intersect(box(1)); // all six planes coincide with the same orientation
    const brep = evaluateBrep(g.node, { bounds: 1.5 });
    expect(brep.faces).toHaveLength(6); // one box, not a doubled shell
    expectValidBoundary(g.node, brep);
    expectNoDuplicateTriangles(brepToMesh(brep));
  });

  it("meshes an L of two roofed arms sharing a wall+roof plane, no doubling", () => {
    // Two equal-height hip arms overlapping at the corner — the flush-arms L. Their inner walls and
    // roof planes are exactly coincident (what plain union absorbs and this must not double).
    const arm = (cx: number, cz: number, hx: number, hz: number) =>
      box(hx, 1.5, hz)
        .translate(cx, 0.75, cz)
        .intersect(roofPlane([1, 1, 0], [cx + hx, 0.5, cz]))
        .intersect(roofPlane([-1, 1, 0], [cx - hx, 0.5, cz]))
        .intersect(roofPlane([0, 1, 1], [cx, 0.5, cz + hz]))
        .intersect(roofPlane([0, 1, -1], [cx, 0.5, cz - hz]));
    const l = arm(-0.3, 0, 0.7, 0.4).union(arm(0, 0.3, 0.4, 0.7));
    const brep = evaluateBrep(l.node, { bounds: 2 });
    expectValidBoundary(l.node, brep);
    expectNoDuplicateTriangles(brepToMesh(brep));
  });
});

describe("mergeCoplanar + brepEdges", () => {
  it("leaves a box at six quads and twelve feature edges", () => {
    const merged = mergeCoplanar(evaluateBrep(box(1).node, { bounds: 1.5 }));
    expect(merged.faces).toHaveLength(6);
    for (const f of merged.faces) expect(f.poly).toHaveLength(4);
    expect(brepEdges(merged).length / 6).toBe(12); // a cube has 12 edges (6 floats per segment)
    expectValidBoundary(box(1).node, merged);
  });

  it("fuses two abutting boxes into one box — six faces, twelve edges", () => {
    const g = box(1).union(box(1).translate(2, 0, 0)); // merges to a single 4×2×2 box
    const raw = evaluateBrep(g.node, { bounds: 3 });
    const merged = mergeCoplanar(raw);
    expect(merged.faces.length).toBeLessThan(raw.faces.length); // split tops/sides fused
    expect(merged.faces).toHaveLength(6);
    expect(brepEdges(merged).length / 6).toBe(12);
    expectValidBoundary(g.node, merged);
    expectNoDuplicateTriangles(brepToMesh(merged));
  });

  it("keeps a non-convex outline as one face (corner notch)", () => {
    const g = box(1).subtract(box(0.5, 0.5, 2).translate(1, 1, 0)); // removes a corner edge, no hole
    const merged = mergeCoplanar(evaluateBrep(g.node, { bounds: 2 }));
    expect(merged.faces.some((f) => f.poly.length > 4)).toBe(true); // the L-shaped +x / +y faces
    expect(merged.faces.every((f) => !f.holes)).toBe(true);
    expectValidBoundary(g.node, merged);
  });

  it("recovers a hole when a tool pokes through a face", () => {
    const g = box(1).subtract(box(0.4, 0.4, 0.4).translate(0.8, 0, 0)); // square hole in the +x face
    const merged = mergeCoplanar(evaluateBrep(g.node, { bounds: 2 }));
    const holed = merged.faces.filter((f) => f.holes?.length);
    expect(holed).toHaveLength(1);
    expect((holed[0] as { holes?: Vec3[][] }).holes).toHaveLength(1);
    expectValidBoundary(g.node, merged); // parts still triangulate the frame correctly
  });

  it("merges the flush-arms L and yields clean line-work", () => {
    const arm = (cx: number, cz: number, hx: number, hz: number) =>
      box(hx, 1.5, hz)
        .translate(cx, 0.75, cz)
        .intersect(roofPlane([1, 1, 0], [cx + hx, 0.5, cz]))
        .intersect(roofPlane([-1, 1, 0], [cx - hx, 0.5, cz]))
        .intersect(roofPlane([0, 1, 1], [cx, 0.5, cz + hz]))
        .intersect(roofPlane([0, 1, -1], [cx, 0.5, cz - hz]));
    const l = arm(-0.3, 0, 0.7, 0.4).union(arm(0, 0.3, 0.4, 0.7));
    const raw = evaluateBrep(l.node, { bounds: 2 });
    const merged = mergeCoplanar(raw);
    expect(merged.faces.length).toBeLessThanOrEqual(raw.faces.length);
    expect(brepEdges(merged).length).toBeGreaterThan(0);
    expectValidBoundary(l.node, merged);
    expectNoDuplicateTriangles(brepToMesh(merged));
  });
});

describe("evaluateBrep — octree localisation", () => {
  // A small hip-roofed house centred at (cx, cz); each is one mass (box ∩ four roof planes).
  const house = (cx: number, cz: number) =>
    box(0.4, 0.6, 0.4)
      .translate(cx, 0.6, cz)
      .intersect(roofPlane([1, 1, 0], [cx + 0.4, 0.2, cz]))
      .intersect(roofPlane([-1, 1, 0], [cx - 0.4, 0.2, cz]))
      .intersect(roofPlane([0, 1, 1], [cx, 0.2, cz + 0.4]))
      .intersect(roofPlane([0, 1, -1], [cx, 0.2, cz - 0.4]));
  const village = (n: number, spacing: number) => {
    let g = house((0 - (n - 1) / 2) * spacing, (0 - (n - 1) / 2) * spacing);
    let first = true;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (first) {
          first = false;
          continue;
        }
        g = g.union(house((i - (n - 1) / 2) * spacing, (j - (n - 1) / 2) * spacing));
      }
    return g;
  };

  /** Order-independent geometric signature of a Brep's faces (ignoring provenance/ordering). */
  const faceKeys = (brep: Brep): string[] => {
    const r = (x: number) => Math.round(x / 1e-4);
    return brep.faces
      .map((f) => {
        const verts = f.poly
          .map((v) => `${r(v[0])},${r(v[1])},${r(v[2])}`)
          .sort()
          .join(";");
        return `${r(f.normal[0])},${r(f.normal[1])},${r(f.normal[2])}|${verts}`;
      })
      .sort();
  };

  it("gives the identical result whether localised or global", () => {
    const g = village(2, 1.6); // four masses, well separated
    const global = mergeCoplanar(evaluateBrep(g.node, { bounds: 4, octreeMaxMasses: 999 }));
    const octree = mergeCoplanar(evaluateBrep(g.node, { bounds: 4, octreeMaxMasses: 1 })); // force subdivision
    expect(faceKeys(octree)).toEqual(faceKeys(global));
  });

  it("meshes a 3×3 village validly through the octree", () => {
    const g = village(3, 1.6);
    const merged = mergeCoplanar(evaluateBrep(g.node, { bounds: 5, octreeMaxMasses: 2 }));
    expectValidBoundary(g.node, merged);
    expectNoDuplicateTriangles(brepToMesh(merged));
    expect(merged.faces.length).toBe(9 * 9); // nine separate houses, nine faces each (4 roof + 4 wall + floor)
  });

  it("stays on the global pass below the mass threshold", () => {
    // Two arms (the L) is two masses — with the default threshold it takes the simple global pass and
    // still matches the forced-octree result.
    const g = box(1).union(box(1).translate(2.5, 0, 0));
    const dflt = mergeCoplanar(evaluateBrep(g.node, { bounds: 4 }));
    const forced = mergeCoplanar(evaluateBrep(g.node, { bounds: 4, octreeMaxMasses: 1 }));
    expect(faceKeys(dflt)).toEqual(faceKeys(forced));
  });
});

describe("evaluateBrep — scope guards", () => {
  it("rejects a non-polyhedral op with a routing hint", () => {
    expect(() => evaluateBrep(box(1).subtract(sphere(0.5)).node)).toThrow(/non-polyhedral/);
  });
});
