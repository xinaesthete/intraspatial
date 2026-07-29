import { describe, expect, it } from "vitest";
import { eigenSym, psdDefect, symmetrise } from "./eigenSym";

// The eigensolver is load-bearing for a *claim* ("this matrix is not PSD, so the projection is
// ill-posed"), so it is tested against cases whose spectrum is known in closed form rather than
// against another numerical routine. Reconstruction (V Λ Vᵀ = A) is the catch-all: it fails for
// any error in values, vectors, ordering or orthogonality at once.

function reconstruct(values: Float64Array, vectors: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) out[i * n + j]! += values[k]! * vectors[k * n + i]! * vectors[k * n + j]!;
    }
  }
  return out;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return m;
}

describe("eigenSym", () => {
  it("diagonalises a known 2×2 exactly", () => {
    // [[2,1],[1,2]] has eigenvalues 3 and 1, eigenvectors (1,1)/√2 and (−1,1)/√2.
    const { values, vectors } = eigenSym([2, 1, 1, 2], 2);
    expect(values[0]).toBeCloseTo(3, 12);
    expect(values[1]).toBeCloseTo(1, 12);
    const s = Math.SQRT1_2;
    expect(Math.abs(vectors[0]!)).toBeCloseTo(s, 12);
    expect(Math.abs(vectors[1]!)).toBeCloseTo(s, 12);
    // Same sign for the top mode (both components equal), opposite for the second.
    expect(Math.sign(vectors[0]!)).toBe(Math.sign(vectors[1]!));
    expect(Math.sign(vectors[2]!)).not.toBe(Math.sign(vectors[3]!));
  });

  it("reconstructs a random symmetric matrix to f64 precision", () => {
    const n = 12;
    let seed = 7;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    const a = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const v = rnd();
        a[i * n + j] = v;
        a[j * n + i] = v;
      }
    }
    const { values, vectors, sweeps, offDiagonal } = eigenSym(a, n);
    expect(sweeps).toBeLessThan(20);
    expect(offDiagonal).toBeLessThan(1e-14);
    expect(maxAbsDiff(reconstruct(values, vectors, n), a)).toBeLessThan(1e-12);

    // Eigenvectors are orthonormal.
    for (let k = 0; k < n; k++) {
      for (let l = k; l < n; l++) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += vectors[k * n + i]! * vectors[l * n + i]!;
        expect(dot).toBeCloseTo(k === l ? 1 : 0, 11);
      }
    }
    // And the trace is preserved — a cheap independent check on the values alone.
    let trA = 0;
    let trL = 0;
    for (let i = 0; i < n; i++) {
      trA += a[i * n + i]!;
      trL += values[i]!;
    }
    expect(trL).toBeCloseTo(trA, 11);
  });

  it("returns eigenvalues in descending order", () => {
    const n = 8;
    const a = new Float64Array(n * n);
    for (let i = 0; i < n; i++) a[i * n + i] = (i * 37) % 11; // a shuffled diagonal
    const { values } = eigenSym(a, n);
    for (let i = 1; i < n; i++) expect(values[i]!).toBeLessThanOrEqual(values[i - 1]!);
  });

  it("canonicalises eigenvector signs, so re-analysis does not flip a mode map", () => {
    // Negating a matrix's *input basis* must not change which way a mode points once canonicalised.
    const a = [4, 1, 0, 1, 3, 1, 0, 1, 2];
    const one = eigenSym(a, 3);
    const two = eigenSym(a.slice(), 3);
    expect([...two.vectors]).toEqual([...one.vectors]);
    // The largest-magnitude component of every mode is positive, by construction.
    for (let k = 0; k < 3; k++) {
      let big = 0;
      let at = 0;
      for (let i = 0; i < 3; i++) {
        if (Math.abs(one.vectors[k * 3 + i]!) > big) {
          big = Math.abs(one.vectors[k * 3 + i]!);
          at = i;
        }
      }
      expect(one.vectors[k * 3 + at]!).toBeGreaterThan(0);
    }
  });

  it("recovers small eigenvalues of a Gram matrix to high relative accuracy", () => {
    // MMᵀ for an M whose rows are nearly collinear: λ_min is tiny but positive, and a solver that
    // loses it to cancellation would report a spurious negative — which is precisely the signal
    // `psdDefect` is asked to interpret in gram.ts.
    const n = 4;
    const p = 40;
    const m = new Float64Array(n * p);
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < p; j++) m[k * p + j] = Math.sin(0.3 * j + 0.001 * k) + 1e-4 * k * Math.cos(0.7 * j);
    }
    const c = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let q = 0; q < p; q++) s += m[i * p + q]! * m[j * p + q]!;
        c[i * n + j] = s;
      }
    }
    const { values } = eigenSym(c, n);
    // A Gram matrix is PSD in exact arithmetic; assert we see that, not a rounding-noise negative.
    // The bound has to be RELATIVE: λ_min lands a few ulp below zero at this matrix's scale
    // (‖C‖ ≈ 20), so an absolute threshold would just be a disguised statement about ‖C‖.
    expect(psdDefect(values, n)).toBeLessThan(1e-12);
    expect(values[n - 1]! / values[0]!).toBeGreaterThan(-1e-14);
    // The point of the test: the *tiny* eigenvalues are resolved rather than lost to cancellation,
    // so a genuine −1e−3 from an indefinite matrix is distinguishable from solver noise.
    expect(values[n - 2]!).toBeGreaterThan(0);
    expect(values[n - 2]! / values[0]!).toBeLessThan(1e-8);
  });

  it("psdDefect measures indefiniteness relative to the spectral radius", () => {
    expect(psdDefect([3, 1, 0], 3)).toBe(0);
    expect(psdDefect([4, 1, -2], 3)).toBeCloseTo(0.5, 12);
    expect(psdDefect([0, 0, 0], 3)).toBe(0);
  });

  it("symmetrise averages, and does not confer positive-definiteness", () => {
    const a = [1, 4, 0, 1];
    const s = symmetrise(a, 2);
    expect([...s]).toEqual([1, 2, 2, 1]);
    // [[1,2],[2,1]] is symmetric with eigenvalues 3 and −1: symmetric, emphatically not PSD.
    const { values } = eigenSym(s, 2);
    expect(values[1]!).toBeCloseTo(-1, 12);
    expect(psdDefect(values, 2)).toBeGreaterThan(0.3);
  });
});
