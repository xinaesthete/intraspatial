// The wand's patch as a subgraph (docs/wand-metric-and-regions.md): from whitened channels u, the
// region around a seed where the smoothed distance to the seed's mixture still curves up, opened by a
// disc so branches narrower than it drop, then the piece holding the seed. Every node is a general op;
// only the seed and the scales are the wand's.
//
// Needs its ops registered (`registerWandOps`): the region pack, the Gaussian, the Laplacian, the
// threshold and the mask arithmetic. A page that never loads the built-in registrar has none of them.
import type { Graph } from "./graph";
import type { GpuField } from "./handle";
import { convolveSeparableOp } from "./ops/convolveSeparable";
import { invertFieldOp, minFieldsOp, scaleFieldOp } from "./ops/fieldArithmetic";
import { laplacianOp } from "./ops/fieldCalculus";
import { REGION_OPS } from "./ops/regionOps";
import { thresholdOp } from "./ops/threshold";
import { hasOp, registerOp } from "./registry";

/** Register the ops the wand's subgraph uses. Idempotent, and safe beside `registerBuiltinOps`. */
export function registerWandOps(): void {
  const ops = [...REGION_OPS, convolveSeparableOp, thresholdOp, laplacianOp, scaleFieldOp, minFieldsOp, invertFieldOp];
  for (const op of ops) if (!hasOp(op.name)) registerOp(op);
}

export interface WandPatchParams {
  /** The seed, as a fractional pixel (pixel centres at integers). */
  readonly col: number;
  readonly row: number;
  /** σ of the smoothing on d before its Laplacian, in pixels; 0 = none. */
  readonly sigma: number;
  /** Also stop where d exceeds this; 0 = off. */
  readonly tol: number;
  /** `false`: a plain flood on d ≤ tol, without the Laplacian test. */
  readonly laplacian: boolean;
  /** Opening radius in pixels; below 0.5 = no opening. */
  readonly rho: number;
}

/** Strictly above `t` for distances, which are square roots of integers: ≥ t + this. */
const ABOVE = 1e-4;

/** One op with one output. */
const op1 = (g: Graph, name: string, inputs: Record<string, GpuField>, params?: Record<string, unknown>): GpuField =>
  g.op(name, inputs, params)[0]!;

/**
 * Build the patch for `u` (a grid with a channel axis) into `g`. `cell` is the unopened flood (the
 * seed always in), `patch` the opened piece holding the seed: 0/1 grids, `patch` empty when the
 * opening removed the seed's own pixel.
 */
export function wandPatch(g: Graph, u: GpuField, p: WandPatchParams): { d: GpuField; cell: GpuField; patch: GpuField } {
  const seed = { col: p.col, row: p.row };
  const d = op1(g, "mixtureDistance", { in: u }, seed);
  let inside: GpuField | undefined;
  if (p.laplacian) {
    const smooth =
      p.sigma > 0
        ? op1(
            g,
            "convolveSeparable",
            { grid: d },
            { kernel: "gaussian", sigma: p.sigma, radius: Math.min(64, Math.max(1, Math.ceil(3 * p.sigma))) },
          )
        : d;
    inside = op1(g, "threshold", { in: op1(g, "laplacian", { in: smooth }) }, { thresh: 0 });
  }
  if (p.tol > 0) {
    // d ≤ tol, as −d ≥ −tol.
    const near = op1(g, "threshold", { in: op1(g, "scaleField", { in: d }, { s: -1 }) }, { thresh: -p.tol });
    inside = inside ? op1(g, "minFields", { a: inside, b: near }) : near;
  }
  // Neither test: everything is in, and the flood is the whole raster.
  inside ??= op1(g, "threshold", { in: d }, { thresh: Number.NEGATIVE_INFINITY });
  const cell = op1(g, "floodFrom", { in: inside }, { ...seed, seedIn: true });
  if (p.rho < 0.5) return { d, cell, patch: cell };
  // Opening by a disc: erode (no outside pixel within ρ, the raster's outside counting), dilate (within
  // ρ of what is left), then the piece holding the seed.
  const eroded = op1(g, "threshold", { in: op1(g, "distanceTransform", { in: cell }, { border: true }) }, { thresh: p.rho + ABOVE });
  const toEroded = op1(g, "distanceTransform", { in: op1(g, "invertField", { in: eroded }) });
  const opened = op1(g, "threshold", { in: op1(g, "scaleField", { in: toEroded }, { s: -1 }) }, { thresh: -p.rho });
  return { d, cell, patch: op1(g, "floodFrom", { in: opened }, seed) };
}
