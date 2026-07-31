// Register the opt-in op packs (element algebra + wavelet + manifold, ADR-0004/0006)
// into the
// composer's op registry. In `src/` these are registered via async, dynamic-import
// registrars (`registerElementOps`/`registerWaveletOps`) kept off the Node test-fork
// module graph for Dawn-on-Node teardown reasons. In the browser there is no such
// concern, and a dynamic import resolves to a *separate* module instance under Vite
// (the same duplication that forces `dedupe: ["typegpu"]`), so registering through it
// would populate a different registry than the app reads. Here we import the op
// objects statically — the SAME graph-module instance the app uses — and register
// them synchronously, so they are present before the first palette render.
import { hasOp, registerOp } from "../../src/gpu/graph";
import { complexOp, conjugateOp, imagPartOp, magnitudeOp, realPartOp } from "../../src/gpu/graph/ops/complexOps";
import { FORCE_OPS } from "../../src/gpu/graph/ops/danceForces";
import {
  addFieldsOp,
  crossFieldsOp,
  dotFieldsOp,
  mulFieldsOp,
  normalizeFieldOp,
  scaleFieldOp,
  subFieldsOp,
} from "../../src/gpu/graph/ops/fieldArithmetic";
import { divergenceOp, gradientMagnitudeOp, gradientOp, laplacianOp, structureOrientationOp } from "../../src/gpu/graph/ops/fieldCalculus";
import { reactionDiffusionComplexOp } from "../../src/gpu/graph/ops/reactionDiffusionComplex";
import { UMAP_OPS } from "../../src/gpu/graph/ops/umapOps";
import { fdwtOp, idwtOp, thresholdDetailOp } from "../../src/gpu/graph/ops/waveletOps";

/** Idempotently register the element-algebra, wavelet and manifold op packs. Guards on the
 *  registry's actual state (`hasOp`) rather than a module-local flag, so it survives
 *  Vite HMR re-evaluation of this module (the registry module itself persists, so a
 *  blind re-register would throw "duplicate op"). */
export function registerExtraOps(): void {
  const ops = [
    complexOp,
    realPartOp,
    imagPartOp,
    conjugateOp,
    magnitudeOp,
    addFieldsOp,
    subFieldsOp,
    mulFieldsOp,
    scaleFieldOp,
    dotFieldsOp,
    crossFieldsOp,
    normalizeFieldOp,
    reactionDiffusionComplexOp,
    fdwtOp,
    idwtOp,
    thresholdDetailOp,
    gradientOp,
    gradientMagnitudeOp,
    laplacianOp,
    divergenceOp,
    structureOrientationOp,
    ...FORCE_OPS,
    // Manifold pack: knn / fuzzyGraph / umapLayout primitives plus the composed `umap`
    // node. Registered here rather than through `registerUmapOps` for the module-instance
    // reason above.
    ...UMAP_OPS,
  ];
  for (const op of ops) if (!hasOp(op.name)) registerOp(op);
}
