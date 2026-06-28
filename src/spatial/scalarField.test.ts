import { describe, it, expect } from "vitest";
import { gaussianKdeField, distanceField, cellCenter } from "./scalarField";

describe("gaussianKdeField", () => {
  it("peaks at the splatted point and decays with distance", () => {
    const f = gaussianKdeField([0], [0], { width: 41, height: 41, sigma: 1, bbox: [-5, -5, 5, 5] });
    // Find the max cell; it should sit at the centre and equal ~1 (un-normalised).
    let max = -Infinity, argc = 0, argr = 0;
    for (let r = 0; r < f.height; r++)
      for (let c = 0; c < f.width; c++)
        if (f.data[r * f.width + c]! > max) { max = f.data[r * f.width + c]!; argc = c; argr = r; }
    const [cx, cy] = cellCenter(f, argc, argr);
    expect(Math.hypot(cx, cy)).toBeLessThan(0.3); // peak ~ at the origin
    expect(max).toBeGreaterThan(0.95);
    expect(max).toBeLessThanOrEqual(1.0001);
    // One sigma out the value should be ~exp(-1/2) of the peak.
    const oned = gaussianKdeField([0], [0], { width: 3, height: 1, sigma: 1, bbox: [-1.5, -0.5, 1.5, 0.5] });
    // centre cell vs the +1 cell: ratio ~ exp(-0.5).
    const centre = oned.data[1]!, edge = oned.data[2]!;
    expect(edge / centre).toBeCloseTo(Math.exp(-0.5), 1);
  });

  it("sums overlapping kernels additively", () => {
    const sep = gaussianKdeField([-0.5, 0.5], [0, 0], { width: 1, height: 1, sigma: 2, bbox: [-0.5, -0.5, 0.5, 0.5] });
    // Single centre cell at (0,0): two kernels each exp(-0.25^2/(2*4))... compute.
    const each = Math.exp(-(0.5 * 0.5) / (2 * 4));
    expect(sep.data[0]!).toBeCloseTo(2 * each, 5);
  });
});

describe("distanceField", () => {
  it("is zero at the points and grows away from them", () => {
    const f = distanceField([0], [0], { width: 41, height: 41, bbox: [-5, -5, 5, 5] });
    let min = Infinity, argc = 0, argr = 0;
    for (let r = 0; r < f.height; r++)
      for (let c = 0; c < f.width; c++)
        if (f.data[r * f.width + c]! < min) { min = f.data[r * f.width + c]!; argc = c; argr = r; }
    const [cx, cy] = cellCenter(f, argc, argr);
    expect(Math.hypot(cx, cy)).toBeLessThan(0.3);
    // A corner is ~sqrt(2)*5 away.
    const corner = f.data[0]!;
    expect(corner).toBeCloseTo(Math.hypot(cellCenter(f, 0, 0)[0], cellCenter(f, 0, 0)[1]), 5);
  });
});
