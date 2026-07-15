// The HsPf parameter seam (ADR-0011, decision 7): the one thing this artefact must get right,
// because it is cheap now and expensive to retrofit. The sim is driven by a plain `Params`
// object built from this declared `ParamSpec[]`, which is also the bridge to the whole Mutator
// spectrum in `src/evo` (`traitSpaceFromParams` → `specimenToParams` feeds a live genome
// straight back in; `withLocked` is freeze; `steer`/`mutate` are exploration).
//
// Names are dotted paths (`fitness.A.pp`) — a real hierarchical namespace over flat string keys,
// filterable by prefix — and each spec carries orthogonal `tags`. The interaction model is
// filter → apply: `filterSpecs` selects an *ordered* set; an operation is applied to it by
// locking the complement (`complementNames` → `withLocked`).
//
// Genotype leaf labels use the minus/plus-per-locus encoding mm/mp/pm/pp = [--, -+, +-, ++],
// matching the original's data keys. See src/gpu/sim/hspf/CONTEXT.md.
import type { ParamSpec, Params } from "../../graph/op";
import type { HspfParams } from "./kernel";
import { DEFAULT_FITNESS, type Vec4 } from "./math";

export type { Params };

const A = DEFAULT_FITNESS.A;
const S = DEFAULT_FITNESS.S;
const GENO = ["mm", "mp", "pm", "pp"] as const;

function fitnessSpecs(row: "A" | "S", values: Vec4, label: string): ParamSpec[] {
  return GENO.map((g, i) => ({
    name: `fitness.${row}.${g}`,
    type: "number",
    default: values[i] ?? 0,
    min: 0,
    max: 1.5,
    step: 0.01,
    describe: `${label} fitness of genotype ${g}`,
    tags: ["fitness", row === "A" ? "background" : "sickle"],
  }));
}

/** The full HsPf parameter schema. Ordering here defines the manual-UI layout and the default
 *  MIDI-CC binding order within a filtered set. */
export const HSPF_PARAM_SPECS: ParamSpec[] = [
  ...fitnessSpecs("A", A, "Background"),
  ...fitnessSpecs("S", S, "Sickle"),
  {
    name: "spread.concentration",
    type: "number",
    default: 6,
    min: 1,
    max: 50,
    step: 0.5,
    describe: "Beta shape: higher ⇒ more local biting, less smoothing",
    tags: ["spread"],
  },
  {
    name: "spread.maxDistanceKm",
    type: "number",
    default: 2000,
    min: 100,
    max: 5000,
    step: 50,
    describe: "Maximum bite distance (km)",
    tags: ["spread"],
  },
  {
    name: "spread.mapWidthKm",
    type: "number",
    default: 10000,
    min: 1000,
    max: 20000,
    step: 100,
    describe: "Physical width the grid spans (km)",
    tags: ["spread"],
  },
  {
    name: "spread.mosquitoCount",
    type: "int",
    default: 1000,
    min: 100,
    max: 25000,
    step: 100,
    describe: "Number of sampled bites per cell (Monte-Carlo gather; higher = smoother but slower)",
    tags: ["spread"],
  },
  {
    name: "dynamics.twoBiteRate",
    type: "number",
    default: 0,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Fraction of transmission that is two-bite recombination",
    tags: ["dynamics"],
  },
  {
    name: "init.mm",
    type: "number",
    default: 0.9,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Initial frequency of genotype -- (mm)",
    tags: ["init"],
  },
  {
    name: "init.mp",
    type: "number",
    default: 0,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Initial frequency of genotype -+ (mp)",
    tags: ["init"],
  },
  {
    name: "init.pm",
    type: "number",
    default: 0,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Initial frequency of genotype +- (pm)",
    tags: ["init"],
  },
  {
    name: "init.pp",
    type: "number",
    default: 0.1,
    min: 0,
    max: 1,
    step: 0.01,
    describe: "Initial frequency of genotype ++ (pp)",
    tags: ["init"],
  },
];

/** All declared defaults as a flat `Params` object — the starting genome. */
export function defaultHspfParams(): Params {
  const out: Params = {};
  for (const s of HSPF_PARAM_SPECS) out[s.name] = s.default;
  return out;
}

function num(params: Params, name: string): number {
  const v = params[name];
  return typeof v === "number" ? v : 0;
}

/** The typed sim configuration a `Params` genome resolves to: kernel params, neighbourhood
 *  generation settings (grid width + seed supplied by the driver), and the initial genotype
 *  vector. This is the boundary between the flat exploration surface and the sim's typed inputs. */
export interface HspfConfig {
  hspf: HspfParams;
  neighbourhood: { mapWidthInKm: number; maxDistanceInKm: number; concentration: number; count: number };
  init: Vec4;
}

/** Resolve a flat `Params` genome into the sim's typed configuration. */
export function toHspfConfig(params: Params): HspfConfig {
  const fitRow = (row: "A" | "S"): Vec4 => [
    num(params, `fitness.${row}.mm`),
    num(params, `fitness.${row}.mp`),
    num(params, `fitness.${row}.pm`),
    num(params, `fitness.${row}.pp`),
  ];
  return {
    hspf: {
      twoBiteRate: num(params, "dynamics.twoBiteRate"),
      fitness: { A: fitRow("A"), S: fitRow("S") },
    },
    neighbourhood: {
      mapWidthInKm: num(params, "spread.mapWidthKm"),
      maxDistanceInKm: num(params, "spread.maxDistanceKm"),
      concentration: num(params, "spread.concentration"),
      count: Math.round(num(params, "spread.mosquitoCount")),
    },
    init: [num(params, "init.mm"), num(params, "init.mp"), num(params, "init.pm"), num(params, "init.pp")],
  };
}

export interface ParamFilter {
  /** Match specs whose dotted `name` starts with this prefix (e.g. "fitness", "fitness.A"). */
  pathPrefix?: string;
  /** Match specs carrying this tag. */
  tag?: string;
  /** Match specs whose `name` contains this substring. */
  nameIncludes?: string;
}

/** Select an *ordered* subset of specs (declaration order preserved) by any combination of
 *  path-prefix, tag, and name-substring. The empty filter matches everything. This ordered set
 *  is what an operation is applied to, and the future MIDI-CC binding target. */
export function filterSpecs(specs: ReadonlyArray<ParamSpec>, filter: ParamFilter = {}): ParamSpec[] {
  const prefix = filter.pathPrefix;
  return specs.filter((s) => {
    if (prefix !== undefined && !(s.name === prefix || s.name.startsWith(`${prefix}.`))) return false;
    if (filter.tag !== undefined && !(s.tags ?? []).includes(filter.tag)) return false;
    if (filter.nameIncludes !== undefined && !s.name.includes(filter.nameIncludes)) return false;
    return true;
  });
}

/** The names *not* in `selected` — i.e. the set to lock (`withLocked`) so that an operation
 *  (mutate/steer/randomize/…) affects only the filtered set. */
export function complementNames(specs: ReadonlyArray<ParamSpec>, selected: ReadonlyArray<ParamSpec>): string[] {
  const keep = new Set(selected.map((s) => s.name));
  return specs.filter((s) => !keep.has(s.name)).map((s) => s.name);
}
