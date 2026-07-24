import { describe, expect, it } from "vitest";
import { besselJ, kernelHankel, kernelSpectrumMin } from "./kernelSpectrum";
import { EPANECHNIKOV, GAUSSIAN, KERNELS, kernelAt, kernelLabel, QUARTIC, TOPHAT, TRIWEIGHT } from "./kernels";

// The closed forms in kernelSpectrum.ts are checked against direct numerical quadrature of the
// Hankel transform of the kernel that kernels.ts actually evaluates — the same discipline as
// kernels.test.ts. That way the two modules cannot drift apart: if someone changes the truncated
// Gaussian's normalisation, this fails.
function hankelByQuadrature(k: typeof TOPHAT, z: number, r = 1, steps = 8_000): number {
  const h = r / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) {
    const u = (i + 0.5) * h;
    sum += kernelAt(k, u * u, r) * besselJ(0, (z * u) / r) * 2 * Math.PI * u * h;
  }
  return sum;
}

/** Bessel's integral `J_m(x) = (1/π)∫₀^π cos(m t − x sin t) dt`, at a panel count far beyond what
 *  the library uses. Structurally independent of both the ascending series and the asymptotic
 *  rational forms in kernelSpectrum.ts, so agreement is real evidence rather than a tautology. */
function besselByQuadrature(m: number, x: number, panels = 400_000): number {
  const h = Math.PI / panels;
  let sum = 0;
  for (let i = 0; i <= panels; i++) {
    const t = i * h;
    const w = i === 0 || i === panels ? 1 : i % 2 ? 4 : 2;
    sum += w * Math.cos(m * t - x * Math.sin(t));
  }
  return (sum * h) / 3 / Math.PI;
}

describe("besselJ", () => {
  it("matches published values at small argument", () => {
    // The handful of constants worth hard-coding, because they are the ones every table agrees on.
    expect(besselJ(0, 0)).toBeCloseTo(1, 10);
    expect(besselJ(1, 0)).toBeCloseTo(0, 10);
    expect(besselJ(0, 1)).toBeCloseTo(0.7651976866, 8);
    expect(besselJ(1, 1)).toBeCloseTo(0.4400505857, 8);
    expect(besselJ(0, 2.404825558)).toBeCloseTo(0, 7); // the first zero of J₀
    expect(besselJ(2, 3)).toBeCloseTo(0.4860912606, 8);
  });

  it("matches independent quadrature on both sides of the split", () => {
    // Rather than hard-code large-argument constants, check against a method that shares no code
    // and no derivation with the implementation. This covers the asymptotic forms AND the upward
    // recurrence that rides on them.
    for (const [m, x] of [
      [0, 3],
      [2, 7],
      [4, 11.5], // below SERIES_MAX — the ascending series
      [0, 30],
      [1, 20],
      [3, 25],
      [4, 35], // above it — asymptotic + recurrence
    ] as [number, number][]) {
      expect(besselJ(m, x), `J${m}(${x})`).toBeCloseTo(besselByQuadrature(m, x), 7);
    }
  });

  it("is continuous across the series/asymptotic split at x = 12", () => {
    // The two regimes must agree where they meet, or the scan would see a step that is pure method.
    for (const m of [0, 1, 2, 3, 4]) {
      expect(besselJ(m, 12 - 1e-9), `J${m}`).toBeCloseTo(besselJ(m, 12 + 1e-9), 7);
    }
  });

  it("obeys the recurrence and the parity rule", () => {
    for (const x of [3, 9, 17, 31]) {
      for (const m of [1, 2, 3]) {
        expect(((2 * m) / x) * besselJ(m, x) - besselJ(m - 1, x), `m=${m} x=${x}`).toBeCloseTo(besselJ(m + 1, x), 7);
      }
    }
    expect(besselJ(0, -5)).toBeCloseTo(besselJ(0, 5), 10);
    expect(besselJ(1, -5)).toBeCloseTo(-besselJ(1, 5), 10);
  });
});

describe("kernelHankel", () => {
  it("is 1 at DC for every kernel — they all carry unit mass", () => {
    for (const k of KERNELS) expect(kernelHankel(k, 0), kernelLabel(k)).toBeCloseTo(1, 12);
  });

  it("agrees with direct quadrature of the kernel kernels.ts evaluates", () => {
    for (const k of KERNELS) {
      for (const z of [0.5, 2, 5, 9, 14]) {
        expect(kernelHankel(k, z), `${kernelLabel(k)} @ z=${z}`).toBeCloseTo(hankelByQuadrature(k, z), 5);
      }
    }
  });
});

describe("positive-definiteness of the mark kernels", () => {
  // THE FINDING. Every kernel this codebase offers has a negative-going Fourier transform in 2-D,
  // so `C = R K Rᵀ` is symmetric but NOT positive semi-definite for any of them. This is why the
  // eigen-projection in gram.ts runs on `C = M Mᵀ` instead — see that file's header.
  it("no member of the polynomial family is positive-definite", () => {
    for (const k of [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT]) {
      expect(kernelSpectrumMin(k).positiveDefinite, kernelLabel(k)).toBe(false);
    }
  });

  it("smoothness shrinks the violation monotonically but never removes it", () => {
    const mins = [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT].map((k) => kernelSpectrumMin(k).min);
    // Pinned to the measured ladder; loose enough to survive a quadrature tweak, tight enough that
    // a factor-of-two change in any entry fails.
    expect(mins[0]!).toBeCloseTo(-0.1323, 3);
    expect(mins[1]!).toBeCloseTo(-0.0586, 3);
    expect(mins[2]!).toBeCloseTo(-0.0295, 3);
    expect(mins[3]!).toBeCloseTo(-0.0159, 3);
    for (let i = 1; i < mins.length; i++) expect(mins[i]!).toBeGreaterThan(mins[i - 1]!);
    for (const m of mins) expect(m).toBeLessThan(0);
  });

  it("the paper's top-hat is the worst offender, by a factor of ~8 over the smoothest member", () => {
    expect(kernelSpectrumMin(TOPHAT).min / kernelSpectrumMin(TRIWEIGHT).min).toBeGreaterThan(7);
  });

  it("the truncated Gaussian is nearly PD, and truncation is the whole reason it is not", () => {
    // At the repo's GAUSS_TRUNC = 3σ the violation is ~1e-3 — three orders of magnitude below the
    // top-hat, but still not zero. An untruncated Gaussian would be exactly PD (its transform is a
    // Gaussian), so what is measured here is the cost of the compact support, not of the shape.
    const g = kernelSpectrumMin(GAUSSIAN);
    expect(g.positiveDefinite).toBe(false);
    expect(g.min).toBeLessThan(0);
    expect(g.min).toBeGreaterThan(-5e-3);
    expect(Math.abs(g.min)).toBeLessThan(Math.abs(kernelSpectrumMin(TRIWEIGHT).min) / 10);
  });

  it("the first negative lobe moves outward as the kernel smooths", () => {
    // Practical reading: the artefact lives at ever-higher spatial frequency, so on a raster whose
    // pixels cannot resolve it the violation is further suppressed.
    const at = [TOPHAT, EPANECHNIKOV, QUARTIC, TRIWEIGHT].map((k) => kernelSpectrumMin(k).at);
    for (let i = 1; i < at.length; i++) expect(at[i]!).toBeGreaterThan(at[i - 1]!);
    expect(at[0]!).toBeCloseTo(5.14, 1);
  });
});
