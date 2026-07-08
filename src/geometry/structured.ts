// Structured geometry — the structural / instancing layer of the grammar (ADR-0010). Where a
// Swept node warps one profile pointwise, a **structural op** (`stack`, `branch`) *instances* a
// child: it places N copies of it, each under a rigid Placement. Chaining structural ops multiplies
// the instances (a `stack` of a `branch`), and the whole structural tree over one leaf **flattens**
// to a flat Placement list — so a Structured is just `{ base: Swept, instances: Mat4[] }`.
//
// That flattening is the load-bearing decision: because every instance is the *same* base horn
// under a different rigid transform, lowering is one base mesh + N placements — on the CPU a
// concatenation (the golden), on the GPU a single **instanced draw** (one base buffer, N transforms;
// the demo). Heterogeneous trees (different leaf structures per branch) are the next step: they
// bucket by structure rather than flatten, keeping "structure = a coarse key, not per-instance data".
//
// **Counts are continuous.** `count` may be fractional: `n + f` emits n full instances plus one
// *emergent* instance whose fold weight `smoothstep(f)` rides its uniform scale, so a new member
// grows in from a point instead of popping (see `splitCount`). The control stays discrete — a UI
// snaps to integers — while the *transition* between integers goes fractional, letting animation fold
// members in and out smoothly. Crucially the fold lives entirely inside the placement Mat4, so the
// pipeline is unchanged: still one base mesh + N transforms, still one instanced draw, no recompile.
//
// Deferred (designed-for): true recursion (`sub` — each branch is itself a sub-tree), per-instance
// progressions as expressions over the instance index, and structure that varies along the sweep.

import { type AngleUnit, isAngle, unitToRadians } from "./angle";
import type { AngleLike } from "./angle-like";
import { applyNormal, applyPoint, compose, IDENTITY, type Mat4, mul, rotX, rotZ, scaleUniform, translate } from "./placement";
import type { Mesh, Swept, TessellateOptions } from "./swept";

/** Options for `stack`: repeat the child `count` times up the sweep axis, each lifted by `step`
 *  (default = the base horn's length, so copies stack contiguously), optionally spun by `twist`
 *  per step and scaled by `scale` per step (compounding — a tapering tower). */
export interface StackOptions {
  step?: number;
  twist?: AngleLike;
  scale?: number;
}

/** Options for `branch`: `count` copies splayed evenly around the axis, each tilted `angle` out
 *  from it, uniformly `scale`d, and optionally spun `twist` about its own axis (a whorl). */
export interface BranchOptions {
  angle?: AngleLike;
  scale?: number;
  twist?: AngleLike;
}

/** Structural angles are scalar constants (unlike the pointwise Param-expressions of `bend`/etc.):
 *  a bare number is in the catalogue unit, an `Angle` is canonical. Expressions over the instance
 *  index are designed-for, not built. */
function toRad(a: AngleLike | undefined, unit: AngleUnit): number {
  if (a === undefined) return 0;
  if (isAngle(a)) return a.radians;
  if (typeof a === "number") return a * unitToRadians(unit);
  throw new Error("structured: angles must be a number or Angle (expressions over the instance index are not supported yet)");
}

const COUNT_EPS = 1e-6;

/** C1 emergence weight: zero slope at both ends, so a folding instance has no velocity kink as it
 *  completes (frac→1) and the next one begins (frac→0). */
const smoothstep = (f: number): number => {
  const t = f < 0 ? 0 : f > 1 ? 1 : f;
  return t * t * (3 - 2 * t);
};

/** Split a possibly-fractional `count` into whole *full* instances (fold weight 1) plus a residual
 *  fraction. The count is at least 1 — the leaf always exists. When `frac > 0`, one extra *emergent*
 *  instance is placed with fold weight `smoothstep(frac)` folded into its **uniform scale**, so the
 *  assembly grows a new member continuously (from a point) rather than popping it in at integers.
 *  The whole fold lives inside the placement matrix: downstream is still one base mesh + N Mat4s. */
function splitCount(count: number): { full: number; frac: number } {
  const c = Math.max(1, count);
  const full = Math.floor(c + COUNT_EPS);
  const frac = c - full;
  return { full, frac: frac < COUNT_EPS ? 0 : frac };
}

/** The Placements for a `stack` of `count` along +z. Fractional counts grow the top segment in from
 *  a point at its true anchor (`i·step`) — the tower's existing rungs never move. */
export function stackPlacements(baseLength: number, count: number, opts: StackOptions | undefined, unit: AngleUnit): Mat4[] {
  const { full, frac } = splitCount(count);
  const step = opts?.step ?? baseLength;
  const twist = toRad(opts?.twist, unit);
  const scale = opts?.scale ?? 1;
  const total = full + (frac > 0 ? 1 : 0);
  const out: Mat4[] = [];
  for (let i = 0; i < total; i++) {
    const w = i < full ? 1 : smoothstep(frac); // the emergent top rung folds in by weight
    out.push(compose([translate(0, 0, i * step), rotZ(i * twist), scaleUniform(scale ** i * w)]));
  }
  return out;
}

/** The Placements for a `branch` whorl of `count` around +z. The angular denominator is the
 *  *continuous* count, so as `count` grows the existing arms **re-space** while the emergent arm —
 *  which starts coincident with arm 0 and migrates to its slot — grows in from a point. Integer
 *  counts reduce to even `2π·j/n` spacing, identical to the discrete path. */
export function branchPlacements(count: number, opts: BranchOptions | undefined, unit: AngleUnit): Mat4[] {
  const { full, frac } = splitCount(count);
  const angle = toRad(opts?.angle, unit);
  const scale = opts?.scale ?? 1;
  const twist = toRad(opts?.twist, unit);
  const c = Math.max(1, count); // continuous denominator ⇒ arms re-space as the whorl grows
  const total = full + (frac > 0 ? 1 : 0);
  const out: Mat4[] = [];
  for (let j = 0; j < total; j++) {
    const w = j < full ? 1 : smoothstep(frac);
    const phi = (2 * Math.PI * j) / c;
    out.push(compose([rotZ(phi), rotX(angle), rotZ(twist), scaleUniform(scale * w)]));
  }
  return out;
}

/** A Structured Geometry: a shared base Swept horn instanced by a flat list of Placements. */
export class Structured {
  readonly kind = "structured" as const;
  readonly base: Swept;
  readonly instances: readonly Mat4[];
  readonly angleUnit: AngleUnit;

  constructor(base: Swept, instances: readonly Mat4[], angleUnit: AngleUnit) {
    this.base = base;
    this.instances = instances;
    this.angleUnit = angleUnit;
  }

  /** Number of leaf instances. */
  get count(): number {
    return this.instances.length;
  }

  /** Compose a new structural op's Placements onto the existing assembly — the new op is the
   *  *outer* transform (each existing instance is placed by each new Placement). Instance count
   *  multiplies. */
  private composeWith(newPlacements: Mat4[]): Structured {
    const instances = newPlacements.flatMap((n) => this.instances.map((e) => mul(n, e)));
    return new Structured(this.base, instances, this.angleUnit);
  }

  /** Stack the whole assembly `count` times up the axis. */
  stack(count: number, opts?: StackOptions): Structured {
    return this.composeWith(stackPlacements(this.base.length, count, opts, this.angleUnit));
  }

  /** Branch the whole assembly into a `count`-fold whorl. */
  branch(count: number, opts?: BranchOptions): Structured {
    return this.composeWith(branchPlacements(count, opts, this.angleUnit));
  }

  /** Tessellate on the CPU — the golden. The base mesh is built once, then each instance's
   *  Placement transforms a copy of it, with indices offset per instance. */
  tessellate(opts?: TessellateOptions): Mesh {
    const base = this.base.tessellate(opts);
    const n = this.instances.length;
    const bv = base.vertexCount;
    const bi = base.indices.length;
    const positions = new Float32Array(n * bv * 3);
    const normals = new Float32Array(n * bv * 3);
    const indices = new Uint32Array(n * bi);
    for (let k = 0; k < n; k++) {
      const m = this.instances[k] ?? IDENTITY;
      for (let v = 0; v < bv; v++) {
        const o = v * 3;
        const p = applyPoint(m, [base.positions[o] ?? 0, base.positions[o + 1] ?? 0, base.positions[o + 2] ?? 0]);
        const nrm = applyNormal(m, [base.normals[o] ?? 0, base.normals[o + 1] ?? 0, base.normals[o + 2] ?? 0]);
        const d = (k * bv + v) * 3;
        positions[d] = p[0];
        positions[d + 1] = p[1];
        positions[d + 2] = p[2];
        normals[d] = nrm[0];
        normals[d + 1] = nrm[1];
        normals[d + 2] = nrm[2];
      }
      for (let e = 0; e < bi; e++) indices[k * bi + e] = (base.indices[e] ?? 0) + k * bv;
    }
    return { positions, normals, indices, vertexCount: n * bv, slices: base.slices, stacks: base.stacks };
  }

  /** Pull to a combined Mesh on the CPU (the single async boundary). GPU rendering instances the
   *  base geometry — build it with `createSweptGpu(base)` and draw `instanceMatrices()`. */
  async toMesh(opts?: TessellateOptions): Promise<Mesh> {
    return this.tessellate(opts);
  }

  /** The instance Placements packed as a flat `array<mat4x4<f32>>` (16 f32 each) — the per-instance
   *  transform buffer a GPU instanced draw binds. Same matrices as the CPU golden, so they agree. */
  instanceMatrices(): Float32Array {
    const out = new Float32Array(this.instances.length * 16);
    for (let k = 0; k < this.instances.length; k++) {
      const m = this.instances[k] ?? IDENTITY;
      for (let i = 0; i < 16; i++) out[k * 16 + i] = m[i] ?? 0;
    }
    return out;
  }
}
