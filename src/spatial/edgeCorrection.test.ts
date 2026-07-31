import { describe, expect, it } from "vitest";
import { annulusAreasInto, annulusRectArea, cornerArea, diskRectArea } from "./edgeCorrection";

// The closed forms in edgeCorrection.ts are checked two ways: against the handful of configurations
// whose area is known exactly (disk inside the rect, rect inside the disk, halves and quarters), and
// against brute-force quadrature over a fine grid for arbitrary placements. The second is the one
// that catches a wrong quadrant sign — those cases are all interior to the formula and none of them
// has a memorable closed form.
function quadrature(cx: number, cy: number, r: number, minX: number, minY: number, maxX: number, maxY: number, n = 2000): number {
  const hx = (maxX - minX) / n;
  const hy = (maxY - minY) / n;
  const r2 = r * r;
  let inside = 0;
  for (let i = 0; i < n; i++) {
    const x = minX + (i + 0.5) * hx - cx;
    const x2 = x * x;
    for (let j = 0; j < n; j++) {
      const y = minY + (j + 0.5) * hy - cy;
      if (x2 + y * y < r2) inside++;
    }
  }
  return inside * hx * hy;
}

describe("cornerArea", () => {
  it("is the full quarter disk when the box contains it", () => {
    expect(cornerArea(3, 10, 10)).toBeCloseTo((Math.PI * 9) / 4, 10);
  });

  it("is the box when the box is inside the disk", () => {
    expect(cornerArea(10, 2, 3)).toBeCloseTo(6, 10);
  });

  it("is a half-strip when only one side binds", () => {
    // x < r, y ≥ r: ∫₀ˣ √(r²−u²) du.
    const r = 5;
    const x = 3;
    const expected = 0.5 * (x * Math.sqrt(r * r - x * x) + r * r * Math.asin(x / r));
    expect(cornerArea(r, x, 100)).toBeCloseTo(expected, 10);
  });

  it("is zero for a degenerate box", () => {
    expect(cornerArea(5, 0, 10)).toBe(0);
    expect(cornerArea(5, 10, -1)).toBe(0);
  });
});

describe("diskRectArea", () => {
  it("gives the whole disk when it sits well inside", () => {
    expect(diskRectArea(50, 50, 10, 0, 0, 100, 100)).toBeCloseTo(Math.PI * 100, 8);
  });

  it("gives the whole rectangle when the disk swallows it", () => {
    expect(diskRectArea(5, 5, 1000, 0, 0, 10, 20)).toBeCloseTo(200, 8);
  });

  it("halves at an edge and quarters at a corner", () => {
    const r = 7;
    expect(diskRectArea(0, 50, r, 0, 0, 100, 100)).toBeCloseTo((Math.PI * r * r) / 2, 8);
    expect(diskRectArea(0, 0, r, 0, 0, 100, 100)).toBeCloseTo((Math.PI * r * r) / 4, 8);
  });

  it("is zero when the disk misses the rectangle", () => {
    expect(diskRectArea(-100, -100, 5, 0, 0, 10, 10)).toBeCloseTo(0, 10);
  });

  // The real test. Each of these places the disk somewhere the formula has to combine corner terms
  // of different signs — straddling one edge, two edges, a corner, and the whole rectangle.
  it.each([
    ["interior", 50, 50, 30],
    ["straddling one edge", 10, 50, 30],
    ["straddling a corner", 5, 5, 30],
    ["straddling two opposite edges", 50, 50, 90],
    ["centre outside, overlapping", -10, 40, 40],
    ["corner outside, large", 95, 95, 60],
    ["off one end entirely but wide", 50, -20, 45],
  ])("agrees with quadrature: %s", (_label, cx, cy, r) => {
    const exact = diskRectArea(cx, cy, r, 0, 0, 100, 80);
    const approx = quadrature(cx, cy, r, 0, 0, 100, 80);
    expect(exact).toBeCloseTo(approx, 1);
    expect(Math.abs(exact - approx) / Math.max(approx, 1)).toBeLessThan(2e-3);
  });

  it("never exceeds the rectangle or the disk", () => {
    for (let cx = -20; cx <= 120; cx += 17) {
      for (let cy = -20; cy <= 100; cy += 13) {
        for (const r of [1, 15, 60, 200]) {
          const a = diskRectArea(cx, cy, r, 0, 0, 100, 80);
          expect(a).toBeGreaterThanOrEqual(-1e-9);
          expect(a).toBeLessThanOrEqual(Math.min(100 * 80, Math.PI * r * r) + 1e-6);
        }
      }
    }
  });
});

describe("annulusRectArea", () => {
  it("is the full annulus well inside", () => {
    expect(annulusRectArea(50, 50, 10, 20, 0, 0, 100, 100)).toBeCloseTo(Math.PI * (400 - 100), 8);
  });

  it("halves at an edge", () => {
    expect(annulusRectArea(0, 50, 10, 20, 0, 0, 100, 100)).toBeCloseTo((Math.PI * (400 - 100)) / 2, 8);
  });

  it("is non-negative even when both radii overhang", () => {
    expect(annulusRectArea(0, 0, 40, 50, 0, 0, 10, 10)).toBeGreaterThanOrEqual(0);
  });
});

describe("annulusAreasInto", () => {
  it("telescopes to the clipped disk", () => {
    const nBins = 12;
    const dr = 7;
    const out = new Float64Array(nBins);
    annulusAreasInto(out, 12, 9, dr, nBins, 0, 0, 100, 80);
    let sum = 0;
    for (const v of out) sum += v;
    expect(sum).toBeCloseTo(diskRectArea(12, 9, nBins * dr, 0, 0, 100, 80), 8);
  });

  it("matches per-bin annulusRectArea", () => {
    const nBins = 6;
    const dr = 11;
    const out = new Float64Array(nBins);
    annulusAreasInto(out, 3, 70, dr, nBins, 0, 0, 100, 80);
    for (let k = 0; k < nBins; k++) {
      expect(out[k]!).toBeCloseTo(annulusRectArea(3, 70, k * dr, (k + 1) * dr, 0, 0, 100, 80), 9);
    }
  });

  it("reduces to the full annulus far from every edge", () => {
    const nBins = 4;
    const dr = 5;
    const out = new Float64Array(nBins);
    annulusAreasInto(out, 500, 500, dr, nBins, 0, 0, 1000, 1000);
    for (let k = 0; k < nBins; k++) {
      const r0 = k * dr;
      const r1 = r0 + dr;
      expect(out[k]!).toBeCloseTo(Math.PI * (r1 * r1 - r0 * r0), 8);
    }
  });
});
