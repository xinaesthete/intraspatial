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
//                                           normalised coord in [0,1] over [min,max],
//                                           through the param's `warp` (int
//                                           re-quantised on read-out)
//   enum (options)                        → a NUMBER trait over [0, options.len-1],
//                                           rounded to an index on read-out
//   bool                                  → an ENABLE trait (one bit) — the on/off of
//                                           a force influence (the hybrid specimen)
//   number/int without bounds             → FIXED (non-evolvable) — passed through
//                                           untouched, never mutated
//
// Warps. The normalised coord is a position in the warp's unit space, as a SuperCollider
// ControlSpec's is: read-out is `warpMap(coord, {min, max, warp})`, encoding is
// `warpUnmap`. Absent (or "lin") warp ⇒ `min + coord * (max - min)`, as always. So a
// `freq` of 50–3000 Hz with `warp: "exp"` reads 387 Hz (the geometric mean) at 0.5, and
// mutation, marriage and steering, all of which work in coord space, move evenly in pitch.
// An exp warp over a range touching or crossing zero can't be traversed; such a param
// falls back to linear, with a console warning (rather than becoming FIXED, which would
// silently stop it evolving).
//
// Delta. A param's `delta` (its own units) becomes the trait's `deltaNorm`, the size of
// one mutation step in coord units, which `mutate` multiplies into its Gaussian step
// alongside the global `rate` (σ = rate · deltaNorm; absent ⇒ deltaNorm 1, as before).
// The rule:
//   linear (and enum, over [0, options-1])  deltaNorm = delta / (max - min)
//   warped  the coord distance from the default to `default + delta`, i.e.
//           |unmap(default + delta) - unmap(default)|; if `default + delta` passes max,
//           from `default - delta` instead; if both leave the range, the larger clipped
//           distance. (A missing/non-numeric default is taken as the range's warped
//           midpoint.) So the step is `delta` at the default, and the same *ratio*
//           (for exp) everywhere else.
//
// Pure + deterministic; no GPU. `specimenToParams` feeds straight into a graph node's
// `params`, so a live specimen can drive a simulation with no graph rebuild.
import type { ParamSpec, Params } from "../gpu/graph/op";
import { type Warp, warpIsLinear, warpMap, warpUnmap, warpValid } from "../gpu/graph/warp";
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
  /** How the coord maps onto [min, max] (NUMBER traits). Absent ⇒ linear; only set for a
   *  non-linear warp valid over the range. */
  warp?: Warp;
  /** One mutation step in coord units, from the param's `delta` (see the module comment).
   *  Absent ⇒ 1, i.e. the step is the mutation `rate` alone. */
  deltaNorm?: number;
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
    } else if (isEvolvableNumber(s)) {
      trait = numberTrait(s, numCount++);
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

/** Whether a non-bool param becomes a NUMBER trait: an enum with a choice to make, or a
 *  number/int with a finite, non-empty [min, max]. */
function isEvolvableNumber(s: ParamSpec): boolean {
  if (s.type === "enum") return !!s.options && s.options.length > 1;
  return (s.type === "number" || s.type === "int") && isFinNum(s.min) && isFinNum(s.max) && s.max > s.min;
}

/** The NUMBER trait for an enum or bounded number/int param (see `isEvolvableNumber`). */
function numberTrait(s: ParamSpec, slot: number): Trait {
  const enumOptions = s.type === "enum" ? s.options : undefined;
  const trait: Trait = enumOptions
    ? { paramName: s.name, kind: "number", locked: false, min: 0, max: enumOptions.length - 1, isInt: true, enumOptions, slot }
    : { paramName: s.name, kind: "number", locked: false, min: s.min ?? 0, max: s.max ?? 0, isInt: s.type === "int", slot };
  const warp = trait.enumOptions ? undefined : traitWarp(s, trait.min, trait.max);
  if (warp !== undefined) trait.warp = warp;
  const deltaNorm = normalisedDelta(s, trait.min, trait.max, warp);
  if (deltaNorm !== undefined) trait.deltaNorm = deltaNorm;
  return trait;
}

/** The warp a NUMBER trait keeps: undefined for linear, or for a warp that can't traverse
 *  [min, max] (warned, then treated as linear). */
function traitWarp(s: ParamSpec, min: number, max: number): Warp | undefined {
  if (s.warp === undefined) return undefined;
  if (!warpValid(s.warp, min, max)) {
    console.warn(
      `intraspatial/evo: param "${s.name}" has warp ${JSON.stringify(s.warp)}, which can't traverse [${min}, ${max}] ` +
        `(an exp warp needs min and max non-zero and of the same sign); treating it as linear`,
    );
    return undefined;
  }
  return warpIsLinear(s.warp) ? undefined : s.warp;
}

/** A param's `delta` as one mutation step in coord units (the rule is in the module
 *  comment); undefined when it has none, or an unusable one (warned). */
function normalisedDelta(s: ParamSpec, min: number, max: number, warp: Warp | undefined): number | undefined {
  const delta = s.delta;
  if (delta === undefined) return undefined;
  if (!isFinNum(delta) || delta < 0) {
    console.warn(`intraspatial/evo: param "${s.name}" has delta ${delta}, not a finite number ≥ 0; ignoring it`);
    return undefined;
  }
  if (warp === undefined) return delta / (max - min);
  const spec = { min, max, warp };
  const d = isFinNum(s.default) ? Math.min(max, Math.max(min, s.default)) : warpMap(0.5, spec);
  const u = warpUnmap(d, spec);
  if (d + delta <= max) return warpUnmap(d + delta, spec) - u;
  if (d - delta >= min) return u - warpUnmap(d - delta, spec);
  return Math.max(1 - u, u);
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
      let v = trait.warp === undefined ? trait.min + t * (trait.max - trait.min) : warpMap(t, trait);
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
    pos[trait.slot] = trait.warp === undefined ? clamp01((v - trait.min) / (trait.max - trait.min)) : warpUnmap(v, trait);
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
