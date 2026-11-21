// Shared by the spring model's ops: value conventions and the small checks every op repeats.
//
//   pos, prev, force   points, vec4   xyz and one spare lane (density for positions)
//   species            points, u32    `DEAD` marks an unused slot
//   radius             points, f32
//   density            points, vec2   (density, directional density)
import { DEAD } from "#src/gpu/sim/springs/state";
import type { ElementType, FieldValue, Shape } from "../../handle";
import type { ParamSpec, Params } from "../../op";

export const VEC2: ElementType = { kind: "vec", n: 2 };
/** The species value of an unused particle slot. */
export const DEAD_SPECIES = DEAD;
/** What "no cutoff" is sent to a kernel as: Tint rejects a constant infinity. */
export const NO_CUTOFF = 3.0e38;

export function pointsN(shape: Shape, who: string): number {
  if (shape.kind !== "points") throw new Error(`${who}: expected a points field`);
  return shape.n;
}

/** All the given shapes are points fields for the same particles; returns how many. */
export function sameParticles(shapes: readonly Shape[], who: string): number {
  const n = pointsN(shapes[0]!, who);
  for (const s of shapes) if (pointsN(s, who) !== n) throw new Error(`${who}: inputs are for different numbers of particles`);
  return n;
}

export function requireVec(el: ElementType | undefined, lanes: 2 | 4, who: string): void {
  if (!el || el.kind !== "vec" || el.n !== lanes) throw new Error(`${who}: expected a vec${lanes} points field`);
}

export function requireScalar(el: ElementType | undefined, who: string): void {
  if (el && el.kind !== "scalar") throw new Error(`${who}: expected a scalar points field`);
}

/** A reader of numeric params that falls back to the declared default. */
export function numbers(specs: readonly ParamSpec[], who: string): (p: Params, name: string) => number {
  return (p, name) => {
    const v = p[name] ?? specs.find((s) => s.name === name)?.default;
    if (typeof v !== "number" || Number.isNaN(v)) throw new Error(`${who}: param "${name}" must be a number`);
    return v;
  };
}

/** A cutoff as the kernel takes it: finite, with "none" as a very large number. */
export function cutoff(v: number): number {
  return Number.isFinite(v) ? v : NO_CUTOFF;
}

export function hostF32(v: FieldValue | undefined, who: string): ArrayLike<number> {
  if (!v?.data) throw new Error(`${who}: cpu golden needs host data`);
  return v.data;
}

export function residentBuffer(v: FieldValue | undefined, who: string): GPUBuffer {
  if (!v?.buffer) throw new Error(`${who}: resident op received a non-resident input`);
  return v.buffer.buffer;
}

/** The `pairRange` param every pair op takes. */
export const PAIR_RANGE: ParamSpec = {
  name: "pairRange",
  type: "number",
  default: NO_CUTOFF,
  min: 0,
  describe: "Pairs further apart than this are ignored. Leave very large for no cutoff.",
};
