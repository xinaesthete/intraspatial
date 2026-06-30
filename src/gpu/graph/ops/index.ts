// Registers the built-in ops. Importing this module (or the graph entry point)
// populates the registry; `listOps()` then drives the React Flow palette.
import { registerOp } from "../registry";
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
  registerOp(splatDensityOp);
  registerOp(convolveSeparableOp);
  registerOp(getisOrdOp);
  registerOp(thresholdOp);
  registerOp(addGridsOp);
  // fuzzy TDA front
  registerOp(kthNeighborDistanceOp);
  registerOp(fuzzyAdjacencyOp);
  registerOp(fuzzyAdjacencyAdaptiveOp);
  registerOp(membershipToDistanceOp);
  registerOp(vietorisRipsOp);
  // simulation front
  registerOp(reactionDiffusionStepOp);
  registerOp(feedbackOp);
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
  const [complex, arith, rd] = await Promise.all([
    import("./complexOps"),
    import("./fieldArithmetic"),
    import("./reactionDiffusionComplex"),
  ]);
  registerOp(complex.complexOp);
  registerOp(complex.realPartOp);
  registerOp(complex.imagPartOp);
  registerOp(complex.conjugateOp);
  registerOp(complex.magnitudeOp);
  registerOp(arith.addFieldsOp);
  registerOp(arith.subFieldsOp);
  registerOp(arith.mulFieldsOp);
  registerOp(arith.scaleFieldOp);
  registerOp(arith.dotFieldsOp);
  registerOp(arith.crossFieldsOp);
  registerOp(arith.normalizeFieldOp);
  registerOp(rd.reactionDiffusionComplexOp);
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
  registerOp(wav.fdwtOp);
  registerOp(wav.idwtOp);
  registerOp(wav.thresholdDetailOp);
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
