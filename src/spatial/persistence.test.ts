import { describe, it, expect } from "vitest";
import { vietorisRipsPersistence, bettiNumbers } from "./persistence";

function distMatrix(xs: number[], ys: number[]): { d: Float32Array; n: number } {
  const n = xs.length;
  const d = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) d[i * n + j] = Math.hypot(xs[i]! - xs[j]!, ys[i]! - ys[j]!);
  return { d, n };
}

function circle(count: number, r: number, cx = 0, cy = 0) {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < count; i++) {
    const a = (2 * Math.PI * i) / count;
    xs.push(cx + r * Math.cos(a));
    ys.push(cy + r * Math.sin(a));
  }
  return { xs, ys };
}

// most-persistent bars of a given dimension
function topBars(result: ReturnType<typeof vietorisRipsPersistence>, dim: number) {
  return result.pairs
    .filter((p) => p.dim === dim && isFinite(p.death))
    .map((p) => p.death - p.birth)
    .sort((a, b) => b - a);
}

describe("vietorisRipsPersistence", () => {
  it("a circle has exactly one persistent loop (β1 = 1)", () => {
    const { xs, ys } = circle(24, 10);
    const { d, n } = distMatrix(xs, ys);
    // maxScale above the loop's death radius (~√3·r ≈ 17.3) so it dies → a finite bar
    const res = vietorisRipsPersistence(d, n, { maxScale: 22 });
    // one dominant H1 bar, well separated from any noise
    const bars = topBars(res, 1);
    expect(bars.length).toBeGreaterThanOrEqual(1);
    expect(bars[0]!).toBeGreaterThan(8); // the loop persists over a wide band
    if (bars.length > 1) expect(bars[0]! / Math.max(bars[1]!, 1e-6)).toBeGreaterThan(3);
    // at a scale just above the edge length, β0 = 1 and β1 = 1
    const b = bettiNumbers(res, 4);
    expect(b[0]).toBe(1);
    expect(b[1]).toBe(1);
  });

  it("two well-separated blobs give β0 = 2 and no persistent loop", () => {
    const a = circle(8, 1, 0, 0);
    const b = circle(8, 1, 50, 0);
    const xs = [...a.xs, ...b.xs], ys = [...a.ys, ...b.ys];
    const { d, n } = distMatrix(xs, ys);
    const res = vietorisRipsPersistence(d, n, { maxScale: 5 });
    // two components persist (essential)
    const essential0 = res.pairs.filter((p) => p.dim === 0 && !isFinite(p.death)).length;
    expect(essential0).toBe(2);
    // each tiny blob's loop (if any) is negligible
    const bars = topBars(res, 1);
    expect(bars[0] ?? 0).toBeLessThan(2);
  });

  it("a figure-eight has two persistent loops (β1 = 2)", () => {
    const a = circle(20, 5, -5, 0);
    const b = circle(20, 5, 5, 0);
    const xs = [...a.xs, ...b.xs], ys = [...a.ys, ...b.ys];
    const { d, n } = distMatrix(xs, ys);
    // above each loop's death radius (~√3·5 ≈ 8.7)
    const res = vietorisRipsPersistence(d, n, { maxScale: 12 });
    const bars = topBars(res, 1);
    expect(bars.length).toBeGreaterThanOrEqual(2);
    expect(bars[1]!).toBeGreaterThan(3); // the two loops are both clearly persistent
  });

  it("CkNN-rescaled distance recovers a sparse ring's loop where the raw count is muddier", () => {
    // sparse outer ring (few points, large spacing)
    const { xs, ys } = circle(16, 12);
    const n = xs.length;
    // raw distances
    const raw = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) raw[i * n + j] = Math.hypot(xs[i]! - xs[j]!, ys[i]! - ys[j]!);
    // CkNN rescale: ρ_i = distance to k-th NN, d̃ = d/√(ρ_iρ_j)
    const k = 2;
    const rho = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const ds: number[] = [];
      for (let j = 0; j < n; j++) if (j !== i) ds.push(raw[i * n + j]!);
      ds.sort((p, q) => p - q);
      rho[i] = ds[k - 1]!;
    }
    const dt = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) dt[i * n + j] = i === j ? 0 : raw[i * n + j]! / Math.sqrt(rho[i]! * rho[j]!);

    const res = vietorisRipsPersistence(dt, n, { maxScale: 3 });
    // the ring's single loop is recovered on the rescaled distance
    const b = bettiNumbers(res, 1.3);
    expect(b[0]).toBe(1);
    expect(b[1]).toBe(1);
  });
});
