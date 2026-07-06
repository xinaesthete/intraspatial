// Getis-Ord Gi* hotspot statistic on a grid — a Local Indicator of Spatial
// Association (LISA). For each cell it asks: is the *neighbourhood* sum of values
// significantly higher (hot) or lower (cold) than you'd expect if the values were
// shuffled at random? The output is a z-score grid: large positive = hot spot,
// large negative = cold spot, ~0 = unremarkable. Directly interpretable.
//
// This is a *composition* primitive: the expensive windowed sum is the separable
// convolution (GPU); the global mean/variance and the closed-form standardisation
// are cheap CPU. `pointHotspotsGpu` chains it onto the KDE splat, so a raw point
// cloud becomes a hotspot map in one call:  points -> splat -> window -> z.
//
// With a box window this is the classic binary-weights Gi*. Because the windowed
// sum uses clamp-to-edge, every cell carries the full window weight W=(2r+1)^2, so
// the Gi* denominator simplifies to a constant. (A Gaussian window gives a softer,
// "fuzzier" LISA — see the windowing discussion in the toolbox doc.)
import { boxKernel, convolveSeparableGpu } from "./convolveSeparable";
import { type SplatOptions, splatDensityGpu } from "./splatDensity";

export interface HotspotField {
  /** Row-major width*height Gi* z-scores. */
  z: Float32Array;
  width: number;
  height: number;
  /** Global mean and population std of the input grid (for reference). */
  mean: number;
  std: number;
}

/** Getis-Ord Gi* z-scores for a grid, using a square box neighbourhood of the
 *  given radius (in cells). */
export async function getisOrdGpu(
  grid: ArrayLike<number>,
  width: number,
  height: number,
  opts: { radius?: number } = {},
): Promise<HotspotField> {
  const radius = opts.radius ?? 2;
  const n = width * height;
  if (grid.length !== n) throw new Error("getisOrd: grid length != width*height");

  // Windowed neighbourhood sum, Σ_j w_ij x_j  (GPU).
  const localSum = await convolveSeparableGpu(grid, width, height, boxKernel(radius));

  // Global mean and population std (CPU; cheap).
  let sum = 0,
    sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = grid[i]!;
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const std = Math.sqrt(variance);

  // Box window with clamp-to-edge → every cell has W weights, all 1.
  const win = 2 * radius + 1;
  const W = win * win;
  // Gi* = (Σ w_ij x_j − X̄·ΣW) / (S · sqrt( (n·ΣW² − (ΣW)²)/(n−1) ))
  const denom = std * Math.sqrt((W * (n - W)) / (n - 1));
  const z = new Float32Array(n);
  if (denom > 0) {
    for (let i = 0; i < n; i++) z[i] = (localSum[i]! - mean * W) / denom;
  }
  return { z, width, height, mean, std };
}

export interface PointHotspotOptions extends SplatOptions {
  /** Box-neighbourhood radius (in grid cells) for the Gi* window. Default 2. */
  radius?: number;
}

/** End-to-end: a point cloud → KDE density → Getis-Ord hotspot z-scores.
 *  Composes `splatDensityGpu` and `getisOrdGpu`. */
export async function pointHotspotsGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: PointHotspotOptions,
): Promise<HotspotField & { bbox: [number, number, number, number] }> {
  const density = await splatDensityGpu(xs, ys, opts);
  const hot = await getisOrdGpu(density.data, density.width, density.height, { radius: opts.radius ?? 2 });
  return { ...hot, bbox: density.bbox };
}
