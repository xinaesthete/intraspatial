// Registers the built-in ops. Importing this module (or the graph entry point)
// populates the registry; `listOps()` then drives the React Flow palette.
import { hasOp, registerOp } from "../registry";
import type { OpType } from "../op";

// Register-if-absent. `registerOp` throws on a duplicate name (a useful guard against
// real collisions), but the registry module outlives a Vite HMR re-evaluation of this
// module, so a blind re-register on hot reload would throw. Guarding on `hasOp` keeps
// every registrar idempotent against the registry's actual state.
const reg = (op: OpType): void => {
  if (!hasOp(op.name)) registerOp(op);
};
import { splatDensityOp } from "./splatDensity";
import { convolveSeparableOp } from "./convolveSeparable";
import { getisOrdOp } from "./getisOrd";
import { thresholdOp } from "./threshold";
import { addGridsOp } from "./addGrids";
import { kthNeighborDistanceOp } from "./kthNeighborDistance";
import { fuzzyAdjacencyOp } from "./fuzzyAdjacency";
import { fuzzyAdjacencyAdaptiveOp } from "./fuzzyAdjacencyAdaptive";
import { membershipToDistanceOp } from "./membershipToDistance";
import { vietorisRipsOp } from "./vietorisRips";
import { reactionDiffusionStepOp } from "./reactionDiffusion";
import { feedbackOp } from "./feedback";

let registered = false;

/** Idempotently register the built-in ops. */
export function registerBuiltinOps(): void {
  if (registered) return;
  registered = true;
  // image / spatial-grid front
  reg(splatDensityOp);
  reg(convolveSeparableOp);
  reg(getisOrdOp);
  reg(thresholdOp);
  reg(addGridsOp);
  // fuzzy TDA front
  reg(kthNeighborDistanceOp);
  reg(fuzzyAdjacencyOp);
  reg(fuzzyAdjacencyAdaptiveOp);
  reg(membershipToDistanceOp);
  reg(vietorisRipsOp);
  // simulation front
  reg(reactionDiffusionStepOp);
  reg(feedbackOp);
}

registerBuiltinOps();

// --- Element-algebra op pack (ADR-0004) ------------------------------------------
// Registered SEPARATELY and opt-in, NOT from registerBuiltinOps(). Eagerly importing
// these modules into the always-loaded registry added enough module-graph weight to
// tip Dawn-on-Node's fragile collection/teardown over the edge in unrelated GPU test
// forks (the splat-render `graph_pipeline` fork segfaulted at collection; ADR-0002/0003).
// Callers that need complex/vec/quaternion ops — the element tests, and the playground
// composer — invoke `registerElementOps()` explicitly. It is idempotent.
let elementRegistered = false;

/** Register the complex / vec / quaternion algebra ops + the complex RD step. Async
 *  and dynamic-import based on purpose: the modules enter the graph only when a caller
 *  actually asks, so a fork that never calls this (e.g. the base graph GPU tests) never
 *  pays their load cost. Idempotent; safe to call repeatedly. */
export async function registerElementOps(): Promise<void> {
  if (elementRegistered) return;
  elementRegistered = true;
  const [complex, arith, rd, calc] = await Promise.all([
    import("./complexOps"),
    import("./fieldArithmetic"),
    import("./reactionDiffusionComplex"),
    import("./fieldCalculus"),
  ]);
  reg(complex.complexOp);
  reg(complex.realPartOp);
  reg(complex.imagPartOp);
  reg(complex.conjugateOp);
  reg(complex.magnitudeOp);
  reg(arith.addFieldsOp);
  reg(arith.subFieldsOp);
  reg(arith.mulFieldsOp);
  reg(arith.scaleFieldOp);
  reg(arith.dotFieldsOp);
  reg(arith.crossFieldsOp);
  reg(arith.normalizeFieldOp);
  reg(rd.reactionDiffusionComplexOp);
  reg(calc.gradientOp);
  reg(calc.gradientMagnitudeOp);
  reg(calc.laplacianOp);
  reg(calc.divergenceOp);
  reg(calc.structureOrientationOp);
}

let waveletRegistered = false;

/** Register the wavelet-domain op pack (forward/inverse DWT + detail thresholding,
 *  ADR-0006). Opt-in and dynamic-import based, for the same reason as
 *  `registerElementOps` — keep these modules off the always-loaded registry path so
 *  unrelated GPU test forks stay at baseline weight (ADR-0002/0003 Dawn fragility).
 *  Idempotent. */
export async function registerWaveletOps(): Promise<void> {
  if (waveletRegistered) return;
  waveletRegistered = true;
  const wav = await import("./waveletOps");
  reg(wav.fdwtOp);
  reg(wav.idwtOp);
  reg(wav.thresholdDetailOp);
}

export {
  splatDensityOp,
  convolveSeparableOp,
  getisOrdOp,
  thresholdOp,
  addGridsOp,
  kthNeighborDistanceOp,
  fuzzyAdjacencyOp,
  fuzzyAdjacencyAdaptiveOp,
  membershipToDistanceOp,
  vietorisRipsOp,
  reactionDiffusionStepOp,
  feedbackOp,
};
