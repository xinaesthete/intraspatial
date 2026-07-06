// The bridge from the dancer's params to the Mutator (src/evo). The breeding strip is
// aesthetic selection — Todd & Latham's method — applied to the dance: you breed by eye,
// and because each cell is a *live* mini-simulation you are selecting on *behaviour*, not
// a frozen frame. That is the whole point (the thing surface-mimicry misses).
//
// We evolve a curated subset of DancerParams (the ones whose variation reads as a
// different dance); the rest hold at their defaults. A trait near 0 effectively switches
// its influence off, so the continuous genes already give the hybrid on/off character.

import { type Specimen, specimenToParams, type TraitSpace, traitSpaceFromParams } from "../../../../src/evo";
import type { ParamSpec } from "../../../../src/gpu/graph/op";
import { type DancerParams, DEFAULT_DANCER_PARAMS } from "./sim";

/** The evolvable traits (mirror of DancerParams fields, with breeding bounds). */
export const DANCER_TRAIT_SPECS: ParamSpec[] = [
  { name: "constrain", type: "number", default: 0.5, min: 0, max: 1 },
  { name: "cohere", type: "number", default: 0.4, min: 0, max: 1 },
  { name: "separate", type: "number", default: 0.6, min: 0, max: 1 },
  { name: "orbit", type: "number", default: 0.35, min: 0, max: 1 },
  { name: "swim", type: "number", default: 0, min: 0, max: 0.8 },
  { name: "vortex", type: "number", default: 0, min: 0, max: 1 },
  { name: "solenoid", type: "number", default: 0, min: 0, max: 1 },
  { name: "partner", type: "number", default: 0.5, min: 0, max: 1 },
  { name: "caller", type: "number", default: 1, min: 0, max: 2 },
  { name: "callerSpeed", type: "number", default: 0.6, min: 0.1, max: 1.5 },
  { name: "period", type: "int", default: 480, min: 180, max: 900 },
  { name: "speedLimit", type: "number", default: 1.2, min: 0.4, max: 2.5 },
  { name: "face", type: "number", default: 0.5, min: 0, max: 1.5 },
  // render traits — appearance, bred alongside behaviour (okLCH colour ramp + cone size)
  { name: "hueSlow", type: "number", default: 4.6, min: 0, max: 6.283 },
  { name: "hueFast", type: "number", default: 3.5, min: 0, max: 6.283 },
  { name: "chroma", type: "number", default: 0.14, min: 0, max: 0.3 },
  { name: "lightSlow", type: "number", default: 0.52, min: 0.2, max: 0.9 },
  { name: "lightFast", type: "number", default: 0.86, min: 0.4, max: 1 },
  { name: "speedRef", type: "number", default: 1.2, min: 0.4, max: 2.5 },
  { name: "sizeBase", type: "number", default: 0.8, min: 0.3, max: 1.6 },
  { name: "sizeSpin", type: "number", default: 0.9, min: 0, max: 3 },
  // creature shape — swept superegg-nose + tapering tube-tail
  { name: "noseRound", type: "number", default: 0.78, min: 0.2, max: 1.8 },
  { name: "tubeRadius", type: "number", default: 0.14, min: 0.05, max: 0.4 },
  { name: "tubeTaper", type: "number", default: 1.4, min: 0.4, max: 3 },
  { name: "thinSpeed", type: "number", default: 0.5, min: 0, max: 1.5 },
  { name: "noseAspect", type: "number", default: 1.6, min: 0.5, max: 3.5 },
];

export const DANCER_TRAIT_SPACE: TraitSpace = traitSpaceFromParams(DANCER_TRAIT_SPECS);

/** Decode a specimen to full DancerParams (evolved traits over the fixed defaults). */
export function specimenToDancerParams(specimen: Specimen): DancerParams {
  const evolved = specimenToParams(DANCER_TRAIT_SPACE, specimen) as Partial<DancerParams>;
  return { ...DEFAULT_DANCER_PARAMS, ...evolved };
}
