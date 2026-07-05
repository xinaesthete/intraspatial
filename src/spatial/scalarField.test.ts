import { describe, expect, it } from "vitest";
import { cellCenter, distanceField, dtmField, gaussianKdeField } from "./scalarField";

describe("gaussianKdeField", () => {
  it("peaks at the splatted point and decays with distance", () => {
    const f = gaussianKdeField([0], [0], { width: 41, height: 41, sigma: 1, bbox: [-5, -5, 5, 5] });
    // Find the max cell; it should sit at the centre and equal ~1 (un-normalised).
    let max = -Infinity,
      argc = 0,
      argr = 0;
    for (let r = 0; r < f.height; r++)
      for (let c = 0; c < f.width; c++)
        if (f.data[r * f.width + c]! > max) {
          max = f.data[r * f.width + c]!;
          argc = c;
          argr = r;
        }
    const [cx, cy] = cellCenter(f, argc, argr);
    expect(Math.hypot(cx, cy)).toBeLessThan(0.3); // peak ~ at the origin
    expect(max).toBeGreaterThan(0.95);
    expect(max).toBeLessThanOrEqual(1.0001);
    // One sigma out the value should be ~exp(-1/2) of the peak.
    const oned = gaussianKdeField([0], [0], { width: 3, height: 1, sigma: 1, bbox: [-1.5, -0.5, 1.5, 0.5] });
    // centre cell vs the +1 cell: ratio ~ exp(-0.5).
    const centre = oned.data[1]!,
      edge = oned.data[2]!;
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
    let min = Infinity,
      argc = 0,
      argr = 0;
    for (let r = 0; r < f.height; r++)
      for (let c = 0; c < f.width; c++)
        if (f.data[r * f.width + c]! < min) {
          min = f.data[r * f.width + c]!;
          argc = c;
          argr = r;
        }
    const [cx, cy] = cellCenter(f, argc, argr);
    expect(Math.hypot(cx, cy)).toBeLessThan(0.3);
    // A corner is ~sqrt(2)*5 away.
    const corner = f.data[0]!;
    expect(corner).toBeCloseTo(Math.hypot(cellCenter(f, 0, 0)[0], cellCenter(f, 0, 0)[1]), 5);
  });
});

describe("dtmField", () => {
  it("k=1 recovers the plain distance field", () => {
    const xs = [-2, 3],
      ys = [1, -1];
    const opts = { width: 16, height: 16, bbox: [-5, -5, 5, 5] as [number, number, number, number] };
    const dtm = dtmField(xs, ys, { ...opts, k: 1 });
    const dist = distanceField(xs, ys, opts);
    for (let i = 0; i < dtm.data.length; i++) expect(dtm.data[i]!).toBeCloseTo(dist.data[i]!, 5);
  });

  it("is robust: an outlier stays high under DTM but reads as a dense site under distance", () => {
    // A tight cluster near the origin plus one lone outlier far away.
    const xs = [-0.3, 0.3, 0, 0.1, 7],
      ys = [0, 0, 0.3, -0.2, 7];
    const opts = { width: 1, height: 1, bbox: [7, 7, 7, 7] as [number, number, number, number] }; // single cell AT the outlier
    const dist = distanceField(xs, ys, opts);
    const dtm = dtmField(xs, ys, { ...opts, k: 3 });
    // At the outlier the nearest distance is ~0, but the DTM (avg of 3 nearest)
    // is large because its other neighbours are the far-away cluster.
    expect(dist.data[0]!).toBeLessThan(1);
    expect(dtm.data[0]!).toBeGreaterThan(5);
  });
});
