import { describe, expect, it } from "vitest";
import {
  SUPEREGG_SLICES,
  SUPEREGG_STACKS,
  signPow,
  supereggVertexAngles,
  supereggVertexCount,
  superellipsoidNormal,
  superellipsoidPoint,
  type Vec3,
} from "./superellipsoid";

const UNIT: Vec3 = [1, 1, 1];

describe("signPow", () => {
  it("preserves sign and is odd", () => {
    expect(signPow(0.5, 2)).toBeCloseTo(0.25, 12);
    expect(signPow(-0.5, 2)).toBeCloseTo(-0.25, 12); // sign kept even for an even exponent
    expect(signPow(0, 3)).toBe(0);
  });
});

describe("superellipsoid at e1 = e2 = 1 (an ellipsoid)", () => {
  it("puts points on the unit sphere scaled by the axes", () => {
    const scale: Vec3 = [2, 3, 5];
    for (let i = 0; i < 20; i++) {
      const eta = -Math.PI / 2 + (Math.PI * i) / 19;
      const omega = -Math.PI + (2 * Math.PI * i) / 19;
      const [x, y, z] = superellipsoidPoint(eta, omega, 1, 1, scale);
      // point on the scaled sphere: (x/sx)² + (y/sy)² + (z/sz)² = 1
      const q = (x / scale[0]) ** 2 + (y / scale[1]) ** 2 + (z / scale[2]) ** 2;
      expect(q).toBeCloseTo(1, 10);
    }
  });

  it("normal of a UNIT sphere is the radial direction", () => {
    for (let i = 0; i < 20; i++) {
      const eta = -Math.PI / 2 + (Math.PI * (i + 0.5)) / 20; // avoid exact poles/seam
      const omega = -Math.PI + (2 * Math.PI * (i + 0.5)) / 20;
      const p = superellipsoidPoint(eta, omega, 1, 1, UNIT);
      const n = superellipsoidNormal(eta, omega, 1, 1, UNIT);
      // on a unit sphere the surface point already IS the unit normal
      expect(n[0]).toBeCloseTo(p[0], 6);
      expect(n[1]).toBeCloseTo(p[1], 6);
      expect(n[2]).toBeCloseTo(p[2], 6);
    }
  });
});

describe("analytic normal agrees with a finite-difference normal", () => {
  it("matches the surface tangent cross product", () => {
    const e1 = 0.7;
    const e2 = 0.85;
    const scale: Vec3 = [0.8, 0.8, 1.3];
    const h = 1e-4;
    for (let i = 1; i < 12; i++) {
      const eta = -Math.PI / 2 + (Math.PI * i) / 12; // interior, off the poles
      const omega = -Math.PI + (2 * Math.PI * (i + 0.3)) / 12;
      const p = superellipsoidPoint(eta, omega, e1, e2, scale);
      const pO = superellipsoidPoint(eta, omega + h, e1, e2, scale);
      const pE = superellipsoidPoint(eta + h, omega, e1, e2, scale);
      const dO: Vec3 = [pO[0] - p[0], pO[1] - p[1], pO[2] - p[2]]; // ∂/∂ω
      const dE: Vec3 = [pE[0] - p[0], pE[1] - p[1], pE[2] - p[2]]; // ∂/∂η
      // ∂ω × ∂η is the outward normal (verified analytically for the sphere case)
      const cx = dO[1] * dE[2] - dO[2] * dE[1];
      const cy = dO[2] * dE[0] - dO[0] * dE[2];
      const cz = dO[0] * dE[1] - dO[1] * dE[0];
      const cl = Math.hypot(cx, cy, cz) || 1;
      const n = superellipsoidNormal(eta, omega, e1, e2, scale);
      expect(n[0]).toBeCloseTo(cx / cl, 3);
      expect(n[1]).toBeCloseTo(cy / cl, 3);
      expect(n[2]).toBeCloseTo(cz / cl, 3);
    }
  });
});

describe("attribute-less grid topology", () => {
  it("counts 6 vertices per quad", () => {
    expect(supereggVertexCount()).toBe(SUPEREGG_SLICES * SUPEREGG_STACKS * 6);
  });

  it("keeps angles within the parameter domain and closes the seam", () => {
    const n = supereggVertexCount();
    for (let vi = 0; vi < n; vi++) {
      const [eta, omega] = supereggVertexAngles(vi);
      expect(eta).toBeGreaterThanOrEqual(-Math.PI / 2 - 1e-9);
      expect(eta).toBeLessThanOrEqual(Math.PI / 2 + 1e-9);
      expect(omega).toBeGreaterThanOrEqual(-Math.PI - 1e-9);
      expect(omega).toBeLessThanOrEqual(Math.PI + 1e-9);
    }
    // the longitude seam closes: ω = -π and ω = +π are the same meridian at any latitude
    const eta = 0.3;
    const a = superellipsoidPoint(eta, -Math.PI, 0.7, 0.7, UNIT);
    const b = superellipsoidPoint(eta, Math.PI, 0.7, 0.7, UNIT);
    expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeCloseTo(0, 10);
  });
});
