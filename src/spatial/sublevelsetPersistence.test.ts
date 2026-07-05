import { describe, expect, it } from "vitest";
import { distanceField, gaussianKdeField } from "./scalarField";
import { fieldBettiNumbers, sublevelsetPersistence } from "./sublevelsetPersistence";

/** Points evenly spaced on a circle of radius `R` centred at the origin. */
function ring(nPts: number, R: number): { xs: number[]; ys: number[] } {
  const xs: number[] = [],
    ys: number[] = [];
  for (let i = 0; i < nPts; i++) {
    const a = (2 * Math.PI * i) / nPts;
    xs.push(R * Math.cos(a));
    ys.push(R * Math.sin(a));
  }
  return { xs, ys };
}

describe("sublevelsetPersistence — H0 (components)", () => {
  it("a single bump has one component, no loop", () => {
    const f = gaussianKdeField([0], [0], { width: 25, height: 25, sigma: 2, bbox: [-6, -6, 6, 6] });
    const res = sublevelsetPersistence(f.data, f.width, f.height, { superlevel: true });
    const h0 = res.pairs.filter((p) => p.dim === 0);
    const h1 = res.pairs.filter((p) => p.dim === 1 && Math.abs(p.birth - p.death) > 0.05);
    expect(h0.length).toBe(1); // one essential component
    expect(h1.length).toBe(0); // no holes
  });

  it("two separated bumps: 2 components at high density, merging to 1", () => {
    const xs = [-3, 3],
      ys = [0, 0];
    const f = gaussianKdeField(xs, ys, { width: 49, height: 33, sigma: 1.5, bbox: [-6, -4, 6, 4] });
    const res = sublevelsetPersistence(f.data, f.width, f.height, { superlevel: true });
    const h0 = res.pairs.filter((p) => p.dim === 0 && Math.abs(p.birth - p.death) > 0.05);
    expect(h0.length).toBe(2); // two peaks (one essential, one dies at the saddle ~0.27)
    // Each component is born at a peak cell (high density, near its birth value).
    for (const p of h0) {
      expect(p.birthCell).toBeGreaterThanOrEqual(0);
      expect(f.data[p.birthCell]).toBeCloseTo(p.birth, 5);
    }
    // Above the saddle there are 2 components; below it they merge to 1.
    const peak = Math.max(...f.data);
    expect(fieldBettiNumbers(res, peak * 0.8)[0]).toBe(2);
    expect(fieldBettiNumbers(res, peak * 0.1)[0]).toBe(1);
  });
});

describe("sublevelsetPersistence — H1 (loops)", () => {
  it("a synthetic ring of high values encloses one hole", () => {
    // 5x5 grid: the 8 vertices at radius 1 are high, the centre and outer ring low.
    const W = 5,
      H = 5;
    const data = new Float32Array(W * H);
    const high = new Set(["1,1", "2,1", "3,1", "1,2", "3,2", "1,3", "2,3", "3,3"]);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) data[r * W + c] = high.has(`${c},${r}`) ? 2 : 0;
    const res = sublevelsetPersistence(data, W, H, { superlevel: true });
    const h1 = res.pairs.filter((p) => p.dim === 1 && Math.abs(p.birth - p.death) > 0.5);
    expect(h1.length).toBe(1);
    // Born high (value 2), dies low (the hole fills only at 0).
    expect(h1[0]!.birth).toBeCloseTo(2, 5);
    expect(h1[0]!.death).toBeCloseTo(0, 5);
    // The killing cell sits inside the hole (a low-value interior cell).
    expect(h1[0]!.deathCell).toBeDefined();
    expect(data[h1[0]!.deathCell!]).toBe(0);
    expect(fieldBettiNumbers(res, 1)[1]).toBe(1); // one loop alive mid-sweep
  });

  it("KDE superlevel set of a ring of points has one loop", () => {
    const { xs, ys } = ring(16, 5);
    const f = gaussianKdeField(xs, ys, { width: 49, height: 49, sigma: 1.1, bbox: [-8, -8, 8, 8] });
    const res = sublevelsetPersistence(f.data, f.width, f.height, { superlevel: true });
    const h1 = res.pairs.filter((p) => p.dim === 1 && Math.abs(p.birth - p.death) > 0.05);
    expect(h1.length).toBe(1); // the annulus = one persistent loop
  });

  it("union of balls (distance sublevel set) of a ring has one loop", () => {
    const { xs, ys } = ring(16, 5);
    const f = distanceField(xs, ys, { width: 49, height: 49, bbox: [-8, -8, 8, 8] });
    const res = sublevelsetPersistence(f.data, f.width, f.height); // sublevel of distance
    const h1 = res.pairs.filter((p) => p.dim === 1 && Math.abs(p.birth - p.death) > 0.1);
    expect(h1.length).toBe(1);
    // The hole is born at r=0 and dies when the balls fill the centre (~R=5).
    expect(h1[0]!.birth).toBeLessThan(1);
    expect(h1[0]!.death).toBeGreaterThan(3);
  });
});
