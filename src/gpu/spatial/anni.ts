// Average Nearest Neighbour Index (ANNI) — a classic, interpretable test for
// whether a point pattern is clustered, random, or dispersed. It *composes* the
// GPU `nearestNeighborDistancesGpu` primitive: the GPU does the O(N^2) nearest-
// neighbour search, this adds the (cheap, CPU) summary statistic on top.
//
// R = observed mean NN distance / expected mean NN distance under complete spatial
// randomness (CSR). R < 1 clustered, R ~ 1 random, R > 1 dispersed. A z-score gives
// significance vs the CSR null (see Clark & Evans 1954).
import { nearestNeighborDistancesGpu } from "./nnDistance";

export interface AnniResult {
  /** Mean observed nearest-neighbour distance. */
  meanObserved: number;
  /** Expected mean NN distance under CSR for this density. */
  meanExpected: number;
  /** R = meanObserved / meanExpected. <1 clustered, ~1 random, >1 dispersed. */
  index: number;
  /** Standard-normal z-score of the deviation from CSR. */
  zScore: number;
  /** Plain-language reading of the index + significance. */
  interpretation: "clustered" | "random" | "dispersed";
  n: number;
  area: number;
}

export interface AnniOptions {
  /** Study-region area. Default: area of the points' axis-aligned bounding box. */
  area?: number;
}

/** Average Nearest Neighbour Index for a point cloud. */
export async function anniGpu(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  opts: AnniOptions = {},
): Promise<AnniResult> {
  const n = xs.length;
  if (n < 2) throw new Error("anni: need at least 2 points");

  let area = opts.area;
  if (area === undefined) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xs[i]!, y = ys[i]!;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    area = (maxX - minX) * (maxY - minY);
  }
  if (!(area > 0)) throw new Error("anni: study area must be > 0");

  const nn = await nearestNeighborDistancesGpu(xs, ys);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += nn[i]!;
  const meanObserved = sum / n;

  // CSR expectations (Clark & Evans). density λ = n / area.
  const density = n / area;
  const meanExpected = 0.5 / Math.sqrt(density);
  const index = meanObserved / meanExpected;
  const standardError = 0.26136 / Math.sqrt(n * density);
  const zScore = (meanObserved - meanExpected) / standardError;

  // |z| > 1.96 is significant at p<0.05; otherwise read as random.
  let interpretation: AnniResult["interpretation"] = "random";
  if (zScore <= -1.96) interpretation = "clustered";
  else if (zScore >= 1.96) interpretation = "dispersed";

  return { meanObserved, meanExpected, index, zScore, interpretation, n, area };
}
