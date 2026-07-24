import { describe, expect, it } from "vitest";
import {
  EPANECHNIKOV,
  equivalentRadius,
  GAUSSIAN,
  KERNELS,
  kernelAt,
  kernelLabel,
  moment2,
  QUARTIC,
  roughness,
  TOPHAT,
  TRIWEIGHT,
} from "./kernels";

// Every constant in kernels.ts is a closed form. Rather than trust the algebra, each one is checked
// against numerical quadrature of the kernel it claims to describe: radial integration
// ∫f(d)·2πd·dd on a fine grid. That way a slip in a normalisation shows up here, not as a subtly
// mis-scaled mark three modules downstream.
function radialIntegral(f: (d: number) => number, r: number, steps = 200_000): number {
  const h = r / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) {
    const d = (i + 0.5) * h; // midpoint rule
    sum += f(d) * 2 * Math.PI * d * h;
  }
  return sum;
}

describe("radial kernels", () => {
  const r = 1.7;

  it("all carry unit mass — so K ⊛ β is a density and m = (K⊛β)/ρ_B is kernel-agnostic", () => {
    for (const k of KERNELS) {
      const mass = radialIntegral((dd) => kernelAt(k, dd * dd, r), r);
      expect(mass, kernelLabel(k)).toBeCloseTo(1, 4);
    }
  });

  it("μ₂ and R(K) match quadrature", () => {
    for (const k of KERNELS) {
      const m2 = radialIntegral((dd) => dd * dd * kernelAt(k, dd * dd, r), r);
      expect(m2 / moment2(k, r), `μ₂ ${kernelLabel(k)}`).toBeCloseTo(1, 3);
      const rk = radialIntegral((dd) => kernelAt(k, dd * dd, r) ** 2, r);
      expect(rk / roughness(k, r), `R(K) ${kernelLabel(k)}`).toBeCloseTo(1, 3);
    }
  });

  it("`order` is smoothness: the boundary value falls to zero and stays there", () => {
    // K(r⁻) is discontinuous only for the top-hat; every higher order vanishes at the boundary.
    expect(kernelAt(TOPHAT, (0.999 * r) ** 2, r)).toBeGreaterThan(0.1);
    for (const k of [EPANECHNIKOV, QUARTIC, TRIWEIGHT]) {
      expect(kernelAt(k, (0.999 * r) ** 2, r), kernelLabel(k)).toBeLessThan(0.01);
    }
    // ...and the approach gets flatter with order: |K(0.9r)| shrinks monotonically.
    const edge = [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT].map((k) => kernelAt(k, (0.9 * r) ** 2, r));
    for (let i = 1; i < edge.length; i++) expect(edge[i]!).toBeLessThan(edge[i - 1]!);
  });

  it("is zero outside the support and finite at the centre", () => {
    for (const k of KERNELS) {
      expect(kernelAt(k, r * r, r), kernelLabel(k)).toBe(0);
      expect(kernelAt(k, (1.5 * r) ** 2, r), kernelLabel(k)).toBe(0);
      expect(kernelAt(k, 0, r), kernelLabel(k)).toBeGreaterThan(0);
    }
  });

  it("equivalentRadius equalises scale — the only fair way to compare kernels", () => {
    const base = 2.5;
    const target = moment2(TOPHAT, base);
    for (const k of KERNELS) {
      expect(moment2(k, equivalentRadius(k, base)), kernelLabel(k)).toBeCloseTo(target, 10);
    }
    // Smoother kernels need a WIDER support to reach the same scale — the trap this guards against.
    expect(equivalentRadius(TRIWEIGHT, base)).toBeGreaterThan(equivalentRadius(EPANECHNIKOV, base));
    expect(equivalentRadius(EPANECHNIKOV, base)).toBeGreaterThan(base);
  });

  it("Epanechnikov is the least rough at matched scale (the AMISE-optimality signature)", () => {
    const base = 2.5;
    const rough = KERNELS.map((k) => ({ k, R: roughness(k, equivalentRadius(k, base)) }));
    const best = rough.reduce((a, b) => (b.R < a.R ? b : a));
    expect(kernelLabel(best.k)).toBe(kernelLabel(EPANECHNIKOV));
    // The top-hat is rougher than every smooth kernel — it pays variance for its hard edge.
    const tophat = rough.find((x) => x.k === TOPHAT)!.R;
    expect(tophat).toBeGreaterThan(rough.find((x) => x.k === QUARTIC)!.R);
    expect(tophat).toBeGreaterThan(rough.find((x) => x.k === GAUSSIAN)!.R);
  });
});
