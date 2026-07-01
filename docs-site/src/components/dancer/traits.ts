// The bridge from the dancer's params to the Mutator (src/evo). The breeding strip is
// aesthetic selection — Todd & Latham's method — applied to the dance: you breed by eye,
// and because each cell is a *live* mini-simulation you are selecting on *behaviour*, not
// a frozen frame. That is the whole point (the thing surface-mimicry misses).
//
// We evolve a curated subset of DancerParams (the ones whose variation reads as a
// different dance); the rest hold at their defaults. A trait near 0 effectively switches
// its influence off, so the continuous genes already give the hybrid on/off character.
import type { ParamSpec } from "../../../../src/gpu/graph/op";
import { specimenToParams, traitSpaceFromParams, type Specimen, type TraitSpace } from "../../../../src/evo";
import { DEFAULT_DANCER_PARAMS, type DancerParams } from "./sim";

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
  { name: "period", type: "int", default: 200, min: 60, max: 400 },
  { name: "speedLimit", type: "number", default: 1.2, min: 0.4, max: 2.5 },
  { name: "face", type: "number", default: 0.5, min: 0, max: 1.5 },
];

export const DANCER_TRAIT_SPACE: TraitSpace = traitSpaceFromParams(DANCER_TRAIT_SPECS);

/** Decode a specimen to full DancerParams (evolved traits over the fixed defaults). */
export function specimenToDancerParams(specimen: Specimen): DancerParams {
  const evolved = specimenToParams(DANCER_TRAIT_SPACE, specimen) as Partial<DancerParams>;
  return { ...DEFAULT_DANCER_PARAMS, ...evolved };
}
