// CPU golden tests for the Implicit (signed-distance) geometry-kind (ADR-0010): analytic distances,
// boolean semantics, exact transforms, the structure/value split (codegen slots line up with the
// param vector), and surface-nets extraction. GPU parity lives in `implicit.gpu.test.ts`.
import { describe, expect, it } from "vitest";
import { constant } from "./expr";
import { box, type Implicit, type IsoMesh, plane, sphere } from "./implicit";
import type { Vec3 } from "./superellipsoid";

const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) < tol;

describe("Implicit: analytic distances", () => {
  it("sphere is |p| − r", () => {
    const g = sphere(1);
    expect(near(g.eval([0, 0, 0]), -1)).toBe(true); // centre, inside
    expect(near(g.eval([1, 0, 0]), 0)).toBe(true); // on surface
    expect(near(g.eval([3, 0, 0]), 2)).toBe(true); // outside
  });

  it("box is exact outside and negative inside", () => {
    const g = box(1); // unit cube half-extent 1
    expect(near(g.eval([0, 0, 0]), -1)).toBe(true); // centre: distance to nearest face
    expect(near(g.eval([1, 0, 0]), 0)).toBe(true); // face
    expect(near(g.eval([2, 0, 0]), 1)).toBe(true); // outside along an axis
    expect(near(g.eval([2, 2, 0]), Math.SQRT2)).toBe(true); // outside past an edge
  });

  it("plane is a signed half-space", () => {
    const g = plane([0, 1, 0], 0); // solid where y < 0
    expect(near(g.eval([0, -2, 0]), -2)).toBe(true);
    expect(near(g.eval([5, 3, -1]), 3)).toBe(true);
    expect(g.eval([0, 0.001, 0])).toBeGreaterThan(0);
  });

  it("normalises a non-unit plane normal", () => {
    const g = plane([0, 2, 0], 1); // n̂ = (0,1,0), offset 1
    expect(near(g.eval([0, 4, 0]), 3)).toBe(true);
  });
});

describe("Implicit: displace (value-noise)", () => {
  const pts: Vec3[] = [
    [0.3, 0.1, -0.2],
    [-0.7, 0.4, 0.6],
    [1.1, -0.9, 0.2],
    [0, 0, 0],
    [-0.35, -0.15, 0.85],
  ];
  const MAX_FBM = 0.9375; // 0.5 + 0.25 + 0.125 + 0.0625 — value noise ∈ [−1,1], 4 octaves

  it("adds bounded fbm noise to the child field", () => {
    const base = sphere(1);
    const g = base.displace(0.3, 2);
    for (const p of pts) expect(Math.abs(g.eval(p) - base.eval(p))).toBeLessThanOrEqual(0.3 * MAX_FBM + 1e-6);
  });

  it("is a no-op at amp 0 and deterministic otherwise", () => {
    const base = box(0.8);
    expect(base.displace(0, 3).eval([0.3, 0.1, -0.2])).toBeCloseTo(base.eval([0.3, 0.1, -0.2]), 12);
    const g = base.displace(0.2, 3);
    expect(g.eval([0.3, 0.1, -0.2])).toBe(g.eval([0.3, 0.1, -0.2])); // pure function of position
  });

  it("actually perturbs the surface (noise is not flat)", () => {
    const base = sphere(1);
    const g = base.displace(0.4, 2.5);
    expect(Math.max(...pts.map((p) => Math.abs(g.eval(p) - base.eval(p))))).toBeGreaterThan(0.01);
  });

  it("exposes its params after the child, in canonical order", () => {
    const g = sphere(constant(1)).displace(constant(0.25), constant(2)); // f32-exact values
    expect(Array.from(g.paramVector())).toEqual([1, 0.25, 2]);
  });
});

describe("Implicit: boolean semantics", () => {
  const a = sphere(1).translate(-0.5, 0, 0);
  const b = sphere(1).translate(0.5, 0, 0);

  it("union is min of distances", () => {
    const u = a.union(b);
    for (const p of [
      [0, 0, 0],
      [1.4, 0, 0],
      [-1.4, 0, 0],
    ] as Vec3[]) {
      expect(near(u.eval(p), Math.min(a.eval(p), b.eval(p)))).toBe(true);
    }
  });

  it("intersect is max, subtract is max(a, −b)", () => {
    const p: Vec3 = [0, 0, 0];
    expect(near(a.intersect(b).eval(p), Math.max(a.eval(p), b.eval(p)))).toBe(true);
    expect(near(a.subtract(b).eval(p), Math.max(a.eval(p), -b.eval(p)))).toBe(true);
  });

  it("smoothUnion never exceeds the sharp union", () => {
    const sharp = a.union(b);
    const smooth = a.smoothUnion(b, 0.4);
    for (let t = -1.5; t <= 1.5; t += 0.25) {
      const p: Vec3 = [t, 0, 0];
      expect(smooth.eval(p)).toBeLessThanOrEqual(sharp.eval(p) + 1e-6);
    }
  });
});

describe("Implicit: exact transforms", () => {
  it("translate shifts the field", () => {
    const g = sphere(1).translate(2, 0, 0);
    expect(near(g.eval([2, 0, 0]), -1)).toBe(true);
    expect(near(g.eval([3, 0, 0]), 0)).toBe(true);
  });

  it("uniform scale scales distances", () => {
    const g = sphere(1).scale(2); // a radius-2 sphere with an exact field
    expect(near(g.eval([2, 0, 0]), 0)).toBe(true);
    expect(near(g.eval([4, 0, 0]), 2)).toBe(true);
  });
});

describe("Implicit: normals point outward", () => {
  it("sphere normal is radial", () => {
    const g = sphere(1);
    const n = g.normal([1, 0, 0]);
    expect(near(n[0], 1, 1e-3)).toBe(true);
    expect(Math.abs(n[1]) < 1e-3 && Math.abs(n[2]) < 1e-3).toBe(true);
  });
});

// The load-bearing invariant for one-pipeline-many-values: the WGSL reads exactly as many `P`
// slots as `paramVector` supplies, in the same order (a mismatch would smear values across params).
describe("Implicit: structure/value split", () => {
  const trees: Implicit[] = [
    sphere(1),
    box(0.5, 0.7, 0.9),
    plane([0, 1, 0], 0.2),
    sphere(1).translate(0.3, 0, 0).subtract(box(0.6)),
    sphere(1)
      .smoothUnion(box(0.5).translate(0.4, 0, 0), 0.3)
      .scale(1.5),
  ];

  // The distinct `P` slots referenced must be exactly `0 … paramVector.length − 1` — one slot per
  // param value, in order. (A transform's slot is *referenced* more than once because its warped
  // point expression is substituted into every leaf below it; that's the same slot reading the same
  // value, so distinct-count, not occurrence-count, is the invariant. GPU parity confirms the rest.)
  it("references exactly the slots 0…paramVector.length−1 for every tree", () => {
    for (const g of trees) {
      const indices = [...g.toWgsl().matchAll(/P\[(\d+)u\]/g)].map((m) => Number(m[1]));
      const distinct = new Set(indices);
      const n = g.paramVector().length;
      expect(distinct.size).toBe(n);
      expect(Math.max(-1, ...indices)).toBe(n - 1);
    }
  });

  it("carries breeding genes on param literals", () => {
    const g = sphere(constant(1, { name: "r", type: "number", default: 1, min: 0.2, max: 2 }));
    const specs = g.specs();
    expect(specs.length).toBe(1);
    expect(specs[0]?.name).toBe("r");
  });

  it("toWgsl emits a scene function", () => {
    expect(sphere(1).toWgsl()).toContain("fn sdScene(p: vec3<f32>)");
  });
});

describe("Implicit: surface-nets extraction", () => {
  it("meshes a sphere onto its surface with outward normals", () => {
    const g = sphere(1);
    const m = g.toMesh({ bounds: 1.6, res: 20 });
    expect(m.vertexCount).toBeGreaterThan(50);
    for (let i = 0; i < m.positions.length; i += 3) {
      const p: Vec3 = [m.positions[i] ?? 0, m.positions[i + 1] ?? 0, m.positions[i + 2] ?? 0];
      // Every vertex sits within a cell-width of the true surface (|p| ≈ 1).
      expect(Math.abs(g.eval(p))).toBeLessThan(0.12);
      // Its normal agrees with the outward radial direction.
      const r = Math.hypot(p[0], p[1], p[2]) || 1;
      const n = m.normals.slice(i, i + 3);
      const dot = ((n[0] ?? 0) * p[0] + (n[1] ?? 0) * p[1] + (n[2] ?? 0) * p[2]) / r;
      expect(dot).toBeGreaterThan(0.9);
    }
    expect(m.indices.length % 3).toBe(0);
  });

  it("produces a non-empty mesh for a CSG difference", () => {
    const g = box(1).subtract(sphere(1.25));
    const m = g.toMesh({ bounds: 1.6, res: 20 });
    expect(m.vertexCount).toBeGreaterThan(0);
    expect(m.indices.length).toBeGreaterThan(0);
  });

  // `sharpen` (dual contouring) recovers hard features that the smooth default rounds off: the QEF
  // places a vertex essentially ON a box corner (1,1,1), where the average-of-crossings sits well
  // inward. Grid step is 0.125, so nearest-vertex < 0.05 (sharpen) vs clearly rounded (smooth).
  it("sharpen recovers a box corner that the smooth default rounds", () => {
    const nearestToCorner = (m: IsoMesh): number => {
      let best = Infinity;
      for (let i = 0; i < m.positions.length; i += 3) {
        best = Math.min(best, Math.hypot((m.positions[i] ?? 0) - 1, (m.positions[i + 1] ?? 0) - 1, (m.positions[i + 2] ?? 0) - 1));
      }
      return best;
    };
    const sharp = nearestToCorner(box(1).toMesh({ bounds: 1.5, res: 24, sharpen: true }));
    const smooth = nearestToCorner(box(1).toMesh({ bounds: 1.5, res: 24 }));
    expect(sharp).toBeLessThan(0.05);
    expect(smooth).toBeGreaterThan(sharp + 0.05); // smooth clearly rounds the corner in
  });

  // Winding regression: on a centred sphere the geometric face normal (edge cross product) must
  // point outward — i.e. agree with the triangle centroid direction. Inconsistent/inverted winding
  // (which backface-culls to holes in a rendered mesh) shows up as triangles failing this.
  it("winds every triangle outward on a sphere", () => {
    const g = sphere(1);
    const m = g.toMesh({ bounds: 1.5, res: 24 });
    const pos = m.positions;
    const idx = m.indices;
    const get = (v: number): Vec3 => [pos[v * 3] ?? 0, pos[v * 3 + 1] ?? 0, pos[v * 3 + 2] ?? 0];
    let outward = 0;
    let total = 0;
    for (let t = 0; t < idx.length; t += 3) {
      const a = get(idx[t] ?? 0);
      const b = get(idx[t + 1] ?? 0);
      const c = get(idx[t + 2] ?? 0);
      const e1: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const e2: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const fn: Vec3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const cen: Vec3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
      const d = fn[0] * cen[0] + fn[1] * cen[1] + fn[2] * cen[2];
      total++;
      if (d > 0) outward++;
    }
    // Essentially all triangles agree (a handful may be near-degenerate); a winding bug fails hard.
    expect(outward / total).toBeGreaterThan(0.99);
  });
});
