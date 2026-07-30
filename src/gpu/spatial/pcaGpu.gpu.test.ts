// The device PCA against the f64 host one.
//
// Unlike `knnDescentGpu`, this kernel CANNOT be held to element-for-element agreement: the
// host accumulates the covariance in f64 and the device in f32, so the contract is a
// measured residual, not equality — the `gramMatrix.ts` situation rather than the
// `knnDescentGpu` one. What the tests below pin is that the residual is small enough to be
// invisible downstream, and, separately, the two structural properties that a tiling bug
// would break while leaving the numbers plausible: symmetry, and the tile split being
// invisible.

import { beforeAll, describe, expect, it } from "vitest";
import { columnStats, covarianceHost, pca, pcaBasis, projectScoresHost } from "../../spatial/pca";
import { expressManifold, makeManifold } from "../../spatial/syntheticManifolds";
import { covarianceGpu, pcaGpu, planTiles, projectScoresGpu } from "./pcaGpu";

const N = 4000;
const SEED = 11;

let values: Float32Array;
let dim: number;

beforeAll(() => {
  // A real-ish feature matrix: correlated columns, so the covariance has structure and an
  // off-by-one in the tiling shows up as a wrong eigenvalue rather than a shrug.
  const m = makeManifold("branching", N, SEED);
  const e = expressManifold(m, { genesPerAxis: 9, seed: SEED });
  values = e.values;
  dim = e.dim;
});

/** Largest absolute difference, relative to the largest magnitude in the oracle. */
function relResidual(got: ArrayLike<number>, want: ArrayLike<number>, count: number): number {
  let maxAbs = 0;
  let maxErr = 0;
  for (let t = 0; t < count; t++) {
    maxAbs = Math.max(maxAbs, Math.abs(want[t]!));
    maxErr = Math.max(maxErr, Math.abs(got[t]! - want[t]!));
  }
  return maxAbs > 0 ? maxErr / maxAbs : maxErr;
}

describe("covarianceGpu", () => {
  it("matches the f64 host covariance to f32 precision", async () => {
    const stats = columnStats(values, N, dim, { standardise: true });
    const host = covarianceHost(values, N, dim, stats);
    const gpu = await covarianceGpu(values, N, dim, stats);
    // 1e-5 relative is roughly what a sqrt(n)-random-walk f32 error predicts at n=4000
    // with the two-level summation, and two orders of magnitude below anything the
    // eigensolver notices. The value of measuring it rather than asserting `toBeCloseTo`
    // per element is that this number is comparable across sizes.
    expect(relResidual(gpu, host, dim * dim)).toBeLessThan(1e-5);
  });

  it("stays exactly symmetric", async () => {
    // Not a numerical property but a structural one: only the upper triangle of tiles runs
    // and the host mirrors it, so the two halves cannot drift apart. If a future change
    // computed both halves independently this would start failing in the last f32 bit,
    // which is precisely the warning worth having.
    const stats = columnStats(values, N, dim, {});
    const gpu = await covarianceGpu(values, N, dim, stats);
    for (let p = 0; p < dim; p++) {
      for (let q = 0; q < dim; q++) expect(gpu[p * dim + q]).toBe(gpu[q * dim + p]);
    }
  });

  it("centres without standardising when asked not to", async () => {
    // `standardise` changes `scale` from 1 to 1/sd, and getting that wrong on the device
    // would still produce a symmetric, plausible matrix — so both settings are checked.
    const stats = columnStats(values, N, dim, {});
    const host = covarianceHost(values, N, dim, stats);
    const gpu = await covarianceGpu(values, N, dim, stats);
    expect(relResidual(gpu, host, dim * dim)).toBeLessThan(1e-5);
    // And the un-standardised diagonal is the raw column variance, which differs from 1.
    let offOne = 0;
    for (let p = 0; p < dim; p++) if (Math.abs(gpu[p * dim + p]! - 1) > 0.01) offOne++;
    expect(offOne).toBeGreaterThan(dim / 2);
  });
});

describe("projectScoresGpu", () => {
  it("matches the host projection", async () => {
    const stats = columnStats(values, N, dim, { standardise: true });
    const basis = pcaBasis(covarianceHost(values, N, dim, stats), dim, 12);
    const host = projectScoresHost(values, N, dim, stats, basis);
    const gpu = await projectScoresGpu(values, N, dim, stats, basis);
    expect(relResidual(gpu, host, N * 12)).toBeLessThan(1e-5);
  });

  it("handles more components than the 32 lanes of a workgroup", async () => {
    // The component loop strides by 32, so `nComponents > 32` exercises a second iteration
    // — and an off-by-one there would silently leave the tail components as zeros.
    const stats = columnStats(values, N, dim, {});
    const nComp = Math.min(40, dim);
    const basis = pcaBasis(covarianceHost(values, N, dim, stats), dim, nComp);
    const gpu = await projectScoresGpu(values, N, dim, stats, basis);
    const host = projectScoresHost(values, N, dim, stats, basis);
    expect(relResidual(gpu, host, N * nComp)).toBeLessThan(1e-5);
    // Explicitly: the last component is not all zeros.
    let nonZero = 0;
    for (let i = 0; i < N; i++) if (gpu[i * nComp + nComp - 1] !== 0) nonZero++;
    expect(nonZero).toBeGreaterThan(N / 2);
  });
});

describe("pcaGpu", () => {
  it("agrees with the host PCA on variance, basis and scores", async () => {
    const host = pca(values, N, dim, { nComponents: 10, standardise: true });
    const gpu = await pcaGpu(values, N, dim, { nComponents: 10, standardise: true });

    expect(gpu.nComponents).toBe(host.nComponents);
    for (let c = 0; c < 10; c++) {
      expect(gpu.explainedVariance[c]!).toBeCloseTo(host.explainedVariance[c]!, 3);
      expect(gpu.explainedVarianceRatio[c]!).toBeCloseTo(host.explainedVarianceRatio[c]!, 5);
    }
    // The basis comes from the same f64 eigensolver on a very slightly different matrix, so
    // the components agree closely but not exactly — and the sign convention holds, which
    // is what stops a "matching" PCA from producing a mirrored embedding.
    expect(relResidual(gpu.components, host.components, 10 * dim)).toBeLessThan(1e-3);
    expect(relResidual(gpu.scores, host.scores, N * 10)).toBeLessThan(1e-3);
  });

  it("gives the same answer however the rows are split across dispatches", async () => {
    // The row tiling is the part of this module most likely to be subtly wrong, and at
    // test sizes it never triggers on its own — one dispatch covers everything. So it is
    // forced. The covariance ACCUMULATES across dispatches (a `+=` into a buffer that
    // survives between submits), which is a different and more fragile thing than the
    // descent kernel's independent row tiles, and this is the only test that runs it.
    const stats = columnStats(values, N, dim, { standardise: true });
    const whole = await covarianceGpu(values, N, dim, stats);
    for (const rowsPerTile of [64, 496, 1024]) {
      const split = await covarianceGpu(values, N, dim, stats, { rowsPerTile });
      // Not equality: a different split is a different summation order, so f32 rounding
      // differs in the last bits. The residual is what has to stay small.
      expect(relResidual(split, whole, dim * dim), `cov rowsPerTile=${rowsPerTile}`).toBeLessThan(1e-5);
    }

    const basis = pcaBasis(covarianceHost(values, N, dim, stats), dim, 8);
    const wholeScores = await projectScoresGpu(values, N, dim, stats, basis);
    for (const rowsPerTile of [64, 496]) {
      const split = await projectScoresGpu(values, N, dim, stats, basis, { rowsPerTile });
      // The projection writes each row independently, so here the split IS exactly
      // invisible — no accumulation, no reordering.
      expect(Array.from(split), `scores rowsPerTile=${rowsPerTile}`).toEqual(Array.from(wholeScores));
    }
  });

  it("spans more than one row tile when the tiling is forced small", () => {
    // `planTiles` is pure, so the interesting cases can be checked without provoking a
    // multi-second dispatch. The load-bearing part is that every ceiling actually binds:
    // the workgroup one is the bug that cost a day in `umapLayoutGpu`.
    const big = { maxBufferFloats: 1e9, maxWorkgroups: 65535 };
    expect(planTiles(1e6, 300, (300 * 300) / 2, big)).toBe(44432); // work-bound
    expect(planTiles(1e6, 300, 1, { ...big, maxBufferFloats: 300 * 1000 })).toBe(992); // buffer-bound
    expect(planTiles(1e6, 4, 1, { ...big, maxWorkgroups: 64 })).toBe(128); // workgroup-bound
    expect(planTiles(100, 4, 1, big)).toBe(96); // n-bound, rounded to whole row blocks
    expect(planTiles(8, 4, 1, big)).toBe(16); // never zero, however tight
  });
});
