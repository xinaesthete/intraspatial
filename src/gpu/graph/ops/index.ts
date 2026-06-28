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
}

registerBuiltinOps();

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
};
