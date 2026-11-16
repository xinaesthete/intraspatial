// Param units (ADR-0018 decision 5, as amended 2026-11-16). A `ParamSpec.units` is a name, and
// what it means depends on whether something can resolve it:
//
// - A FIXED unit ("Hz", "s", "dB", "rad", "%") means the same everywhere. It is a label, and it
//   converts only into itself.
// - A RELATIVE unit means something only against a context the graph or the host supplies:
//   "cells" and "world" against a field's placement; samples against a sample rate; beats against
//   a tempo; scale degrees against a scale. Nobody can list these in advance, so they are not
//   enumerated here. A `UnitResolver` claims some unit names and resolves them against its
//   context into a `ResolvedUnit`: a label plus a map into a BASE unit. Two resolved units
//   convert into each other exactly when they share a base. The map is a function, not a factor,
//   because some relative units are not linear: scale degrees reach semitones through a table.
//
// This module carries the spatial resolver (`spatialUnits`) because the spatial units are this
// repo's own. Resolvers for other domains (audio, music) belong with whoever owns their context.

import type { ResolvedPlacement } from "./handle";

/** A param's unit, resolved in some context. */
export interface ResolvedUnit {
  /** The unit's own name, shown beside a value in it ("cells", "world units", "Hz"). */
  readonly label: string;
  /** The unit `toBase` lands in ("µm", "semitones"). Two resolved units convert into each other
   *  exactly when their bases are equal. */
  readonly base: string;
  /** A value in this unit, in the base unit. */
  toBase(v: number): number;
  /** The inverse of `toBase`. Absent where there is none (chromatic semitones have no scale
   *  degree), so nothing can be converted *into* this unit. */
  fromBase?(v: number): number;
}

/** Resolves one family of relative units against a context of type `C`. */
export interface UnitResolver<C> {
  /** The unit names this resolver claims. */
  readonly units: readonly string[];
  /** The unit in context, or undefined when the context lacks what it needs. */
  resolve(unit: string, ctx: C): ResolvedUnit | undefined;
}

const identity = (v: number) => v;

/** A fixed unit: labelled by its name, its own base, converting only into itself. */
export function fixedUnit(unit: string): ResolvedUnit {
  return { label: unit, base: unit, toBase: identity, fromBase: identity };
}

/** Resolve `units` with the first resolver that claims it; a name no resolver claims is a fixed
 *  unit. Undefined when `units` is absent (undeclared), or when the claiming resolver can't
 *  resolve it in `ctx`. One context serves several resolvers: pass an object satisfying each
 *  of their context types. */
export function resolveUnit<C>(
  units: string | undefined,
  ctx: C,
  resolvers: readonly UnitResolver<NoInfer<C>>[],
): ResolvedUnit | undefined {
  if (units === undefined) return undefined;
  const resolver = resolvers.find((r) => r.units.includes(units));
  return resolver ? resolver.resolve(units, ctx) : fixedUnit(units);
}

/** `v` in unit `from`, in unit `to`; undefined when they don't share a base, or `to` can't be
 *  converted into. */
export function convertUnit(v: number, from: ResolvedUnit, to: ResolvedUnit): number | undefined {
  if (from.base !== to.base || !to.fromBase) return undefined;
  return to.fromBase(from.toBase(v));
}

/** What the spatial units resolve against: the placement of the field a param applies to, and
 *  the store's physical scale. */
export interface SpatialUnitContext {
  /** The field's placement (ADR-0018). Absent ⇒ array space: "cells" are just cells, and a
   *  "world" length has nothing to convert against. */
  placement?: ResolvedPlacement;
  /** Micrometres per world unit, from the store's metadata (e.g. `micrometresPerUnit` for an MDV
   *  table, NGFF axis units for OME-Zarr). Absent ⇒ unknown, which is not the same as 1. */
  micrometresPerUnit?: number;
}

/** The world size of one cell, when the placement's x and y steps are the same length
 *  (ADR-0018 decision 2: `|worldFromArray · ê|`, derived, never stored); undefined when they
 *  differ, since one length then has no single size in cells. */
function isotropicCellSize(p: ResolvedPlacement): number | undefined {
  const [ax, ay] = p.worldFromArray.axes;
  const sx = Math.hypot(...ax);
  const sy = Math.hypot(...ay);
  return sx > 0 && Math.abs(sx - sy) <= 1e-9 * Math.max(sx, sy) ? sx : undefined;
}

/** "cells" and "world" (ADR-0018 decision 5). A world length's base is µm when the store says how
 *  long a world unit is, else the placement's system ("world:<system>"), else "world". A cell
 *  length reaches the same base through the placement's cell size, so cells and world convert
 *  into each other whenever the field is placed with square cells, and both show in µm whenever
 *  the scale is known ("2 cells = 0.8 µm"). Unplaced, "cells" is its own base. */
export const spatialUnits: UnitResolver<SpatialUnitContext> = {
  units: ["cells", "world"],
  resolve(unit, ctx) {
    const world = worldUnit(ctx);
    if (unit === "world") return world;
    return unit === "cells" ? cellUnit(ctx.placement, world) : undefined;
  },
};

/** A world length, in µm when the scale is known, else in the placement's own system. */
function worldUnit({ placement, micrometresPerUnit: um }: SpatialUnitContext): ResolvedUnit {
  if (um !== undefined && Number.isFinite(um) && um > 0) {
    return { label: "world units", base: "µm", toBase: (v) => v * um, fromBase: (v) => v / um };
  }
  return { label: "world units", base: placement ? `world:${placement.system}` : "world", toBase: identity, fromBase: identity };
}

/** A cell length, reaching `world`'s base through the cell size when the cells are square. */
function cellUnit(placement: ResolvedPlacement | undefined, world: ResolvedUnit): ResolvedUnit {
  const cell = placement ? isotropicCellSize(placement) : undefined;
  if (cell === undefined) return fixedUnit("cells");
  return { label: "cells", base: world.base, toBase: (v) => world.toBase(v * cell), fromBase: (v) => world.fromBase!(v) / cell };
}
