// The mosquito-bite neighbourhood: the set of sampled offsets each cell gathers from every
// step. Distances follow Beta(1, concentration) — biting is mostly local, larger concentration
// ⇒ tighter neighbourhood ⇒ less geographic smoothing. Generated CPU-side and uploaded as a
// buffer the kernel merely reads (so there is no RNG in the kernel), and *seeded* so runs are
// reproducible — required for the future data-fit fitness and for the invariant tests
// (ADR-0011, decision 4).
//
// Beta(1, c) has a closed-form inverse-CDF: F(x) = 1 − (1−x)^c ⇒ x = 1 − (1−U)^(1/c) for
// U ~ Uniform(0,1). That removes the @stdlib/random dependency entirely.
import { mulberry32 } from "../../../evo/rng";

/** A neighbourhood packed for upload as an array of `{dx, dy, weight}` (matching the kernel's
 *  `NbhdPoint` struct: three f32 per sample). */
export interface Neighbourhood {
  /** `count` × 3 f32: [dx, dy, weight] per sample, row-major. */
  data: Float32Array;
  count: number;
}

export interface NeighbourhoodParams {
  /** Physical width the grid spans, in km (with `gridWidth` gives cell size). */
  mapWidthInKm: number;
  /** Maximum bite distance, in km — the scale the Beta(0,1) draw maps onto. */
  maxDistanceInKm: number;
  /** Beta shape parameter: higher ⇒ more local biting, less smoothing. */
  concentration: number;
  /** Number of samples ("mosquitos"). */
  count: number;
  /** Grid width in cells, to convert km ⇒ cells. */
  gridWidth: number;
  /** Seed for reproducibility. */
  seed: number;
}

/** Draw `x ~ Beta(1, c)` by inverse-CDF from a uniform `u ∈ [0,1)`. */
export function betaOneC(u: number, c: number): number {
  return 1 - (1 - u) ** (1 / c);
}

/** Generate a seeded neighbourhood: `count` offsets with uniform random direction and a
 *  Beta(1, concentration) distance (in cells). Deterministic in `seed`. */
export function makeNeighbourhood(p: NeighbourhoodParams): Neighbourhood {
  const rng = mulberry32(p.seed);
  const cellWidthInKm = p.mapWidthInKm / p.gridWidth;
  const data = new Float32Array(p.count * 3);
  for (let i = 0; i < p.count; i++) {
    const direction = rng() * 2 * Math.PI;
    const distance01 = betaOneC(rng(), p.concentration);
    const distanceInCells = (distance01 * p.maxDistanceInKm) / cellWidthInKm;
    data[i * 3 + 0] = Math.round(Math.cos(direction) * distanceInCells);
    data[i * 3 + 1] = Math.round(Math.sin(direction) * distanceInCells);
    data[i * 3 + 2] = 1;
  }
  return { data, count: p.count };
}
