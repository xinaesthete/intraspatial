// The trait-space: the bridge that turns any op/subgraph's `ParamSpec[]` into an
// evolvable specimen, and back. This is what makes the Mutator *generic* — the dancer
// field is just the first consumer; anything that declares typed params (with
// `min`/`max`) gets a trait-space for free (the op author controls the trait-space by
// how they declare params).
//
// (Naming: we deliberately avoid "gene"/"genome" — this repo also handles real
// biological/genomic data, so a "specimen" is one bred individual and a "trait" is one
// evolvable parameter axis.)
//
// Classification of each ParamSpec:
//   number / int  with finite [min,max]  → a NUMBER trait, the specimen carries a
//                                           normalised coord in [0,1] over [min,max]
//                                           (int re-quantised on read-out)
//   enum (options)                        → a NUMBER trait over [0, options.len-1],
//                                           rounded to an index on read-out
//   bool                                  → an ENABLE trait (one bit) — the on/off of
//                                           a force influence (the hybrid specimen)
//   number/int without bounds             → FIXED (non-evolvable) — passed through
//                                           untouched, never mutated
//
// Pure + deterministic; no GPU. `specimenToParams` feeds straight into a graph node's
// `params`, so a live specimen can drive a simulation with no graph rebuild.
import type { ParamSpec, Params } from "../gpu/graph/op";
import type { Specimen } from "./specimen";

export type TraitKind = "number" | "enable" | "fixed";

export interface Trait {
  paramName: string;
  kind: TraitKind;
  /** Whether the artist has locked this trait (excluded from mutation/marriage). */
  locked: boolean;
  // NUMBER traits:
  min: number;
  max: number;
  /** Re-quantise on read-out (int params, enum index). */
  isInt: boolean;
  enumOptions?: string[];
  /** Index into a specimen's `pos`/`vel` for NUMBER traits, into `enable` for ENABLE
   *  traits; `-1` for FIXED traits. */
  slot: number;
  /** Read-out value for FIXED traits (the param's default). */
  fixedValue?: unknown;
}

export interface TraitSpace {
  traits: Trait[];
  /** Length of a specimen's `pos`/`vel` arrays. */
  numCount: number;
  /** Length of a specimen's `enable` array. */
  enableCount: number;
  byParam: Record<string, number>; // paramName → index into `traits`
}

function isFinNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

/** Derive a trait-space from a param schema. The order of `traits` follows `specs`. */
export function traitSpaceFromParams(specs: ParamSpec[]): TraitSpace {
  const traits: Trait[] = [];
  const byParam: Record<string, number> = {};
  let numCount = 0;
  let enableCount = 0;

  for (const s of specs) {
    let trait: Trait;
    if (s.type === "bool") {
      trait = { paramName: s.name, kind: "enable", locked: false, min: 0, max: 1, isInt: true, slot: enableCount++ };
    } else if (s.type === "enum" && s.options && s.options.length > 1) {
      trait = {
        paramName: s.name,
        kind: "number",
        locked: false,
        min: 0,
        max: s.options.length - 1,
        isInt: true,
        enumOptions: s.options,
        slot: numCount++,
      };
    } else if ((s.type === "number" || s.type === "int") && isFinNum(s.min) && isFinNum(s.max) && s.max > s.min) {
      trait = {
        paramName: s.name,
        kind: "number",
        locked: false,
        min: s.min,
        max: s.max,
        isInt: s.type === "int",
        slot: numCount++,
      };
    } else {
      // Unbounded number, single-option enum, or anything we can't bound: pass the
      // declared default through untouched.
      trait = { paramName: s.name, kind: "fixed", locked: true, min: 0, max: 0, isInt: false, slot: -1, fixedValue: s.default };
    }
    byParam[s.name] = traits.length;
    traits.push(trait);
  }

  return { traits, numCount, enableCount, byParam };
}

/** Decode a specimen to concrete params for a graph node. */
export function specimenToParams(space: TraitSpace, sp: Specimen): Params {
  const out: Params = {};
  for (const trait of space.traits) {
    if (trait.kind === "fixed") {
      out[trait.paramName] = trait.fixedValue;
    } else if (trait.kind === "enable") {
      out[trait.paramName] = sp.enable[trait.slot] !== 0;
    } else {
      const t = clamp01(sp.pos[trait.slot] ?? 0);
      let v = trait.min + t * (trait.max - trait.min);
      if (trait.enumOptions) {
        const idx = Math.min(trait.enumOptions.length - 1, Math.max(0, Math.round(v)));
        // biome-ignore lint/style/noNonNullAssertion: bound checked above
        out[trait.paramName] = trait.enumOptions[idx]!;
        continue;
      }
      if (trait.isInt) v = Math.round(v);
      out[trait.paramName] = v;
    }
  }
  return out;
}

/** Encode concrete params into a specimen (positions only; velocity starts at zero).
 *  Values outside a trait's range clamp into it. */
export function paramsToSpecimen(space: TraitSpace, params: Params, seed: number): Specimen {
  const pos = new Float64Array(space.numCount);
  const vel = new Float64Array(space.numCount);
  const enable = new Uint8Array(space.enableCount);
  for (const trait of space.traits) {
    if (trait.kind === "fixed") continue;
    const raw = params[trait.paramName];
    if (trait.kind === "enable") {
      enable[trait.slot] = raw ? 1 : 0;
      continue;
    }
    let v: number;
    if (trait.enumOptions) {
      const idx = trait.enumOptions.indexOf(String(raw));
      v = idx >= 0 ? idx : 0;
    } else {
      v = isFinNum(raw) ? raw : trait.min;
    }
    pos[trait.slot] = clamp01((v - trait.min) / (trait.max - trait.min));
  }
  return { pos, vel, enable, seed };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Return a copy of `space` with the named traits locked (excluded from mutation and
 *  marriage). This is the "trait-fix" the Studio UI exposes — the artist freezes the
 *  parts they like and breeds the rest. */
export function withLocked(space: TraitSpace, lockedParamNames: Iterable<string>): TraitSpace {
  const locked = new Set(lockedParamNames);
  return {
    ...space,
    traits: space.traits.map((t) => ({ ...t, locked: t.kind === "fixed" ? true : locked.has(t.paramName) })),
  };
}
