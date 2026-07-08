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

const clampCount = (n: number): number => Math.max(1, Math.floor(n));

/** The Placements for a `stack` of `count`, along +z. */
export function stackPlacements(baseLength: number, count: number, opts: StackOptions | undefined, unit: AngleUnit): Mat4[] {
  const n = clampCount(count);
  const step = opts?.step ?? baseLength;
  const twist = toRad(opts?.twist, unit);
  const scale = opts?.scale ?? 1;
  const out: Mat4[] = [];
  for (let i = 0; i < n; i++) out.push(compose([translate(0, 0, i * step), rotZ(i * twist), scaleUniform(scale ** i)]));
  return out;
}

/** The Placements for a `branch` whorl of `count` around +z. */
export function branchPlacements(count: number, opts: BranchOptions | undefined, unit: AngleUnit): Mat4[] {
  const n = clampCount(count);
  const angle = toRad(opts?.angle, unit);
  const scale = opts?.scale ?? 1;
  const twist = toRad(opts?.twist, unit);
  const out: Mat4[] = [];
  for (let j = 0; j < n; j++) {
    const phi = (2 * Math.PI * j) / n;
    out.push(compose([rotZ(phi), rotX(angle), rotZ(twist), scaleUniform(scale)]));
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
