// The evolutionary layer — Todd & Latham "Mutator" aesthetic-selection primitives,
// generic over any op/subgraph's `ParamSpec[]`. Pure + deterministic (seeded RNG), no
// GPU, so it unit-tests like the repo's CPU goldens. The dancer field is the first
// consumer; anything that declares bounded params gets a trait-space for free.
//
// Naming note: we use "specimen" (one bred individual) and "trait" (one evolvable
// parameter axis) rather than genome/gene — this repo also handles real
// biological/genomic data, and the collision would be confusing.
//
// See the plan: trait-space (traitSpace) ← derived from ParamSpec[]; specimen = a
// playhead in that space carrying position + velocity; mutator = the operators that
// move it (mutate/marry/steer/advance); pedigree = the serialisable lineage.

export type { BreedOptions } from "./mutator";
export { advance, breed, marry, mutate, steer, toward } from "./mutator";
export type { BirthOp, Pedigree, PedigreeNode } from "./pedigree";
export { ancestry, emptyPedigree, fromJSON, recordBirth, select, specimenId, toJSON } from "./pedigree";
export type { Rng } from "./rng";
export { gauss, hashSeed, mulberry32, uniform } from "./rng";
export type { SerializedSpecimen, Specimen } from "./specimen";
export { cloneSpecimen, deserializeSpecimen, neutralSpecimen, randomSpecimen, serializeSpecimen } from "./specimen";
export type { Trait, TraitKind, TraitSpace } from "./traitSpace";
export { paramsToSpecimen, specimenToParams, traitSpaceFromParams, withLocked } from "./traitSpace";
