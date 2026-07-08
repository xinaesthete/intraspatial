// Swept geometry — the horn-lineage Geometry-kind (ADR-0010), re-derived from Stephen Todd's
// FormGrow / horn grammar (the Todd–Latham lineage). A Swept geometry is a
// **Profile** (a cross-section that is a function of `(s, θ)`) swept along the Sweep coordinate
// `s ∈ [0, 1]` under an ordered **Transform-stack** of `s`(/`θ`)-parameterised, closed-form
// coordinate transforms. It is **pointwise closed-form**:
//
//     eval(s, θ) → (position, normal)
//
// with NO recurrence along `s` — the stack is a composition of pointwise maps applied to a base
// placement, not an integrated marching frame (a marching frame, if ever needed, becomes an
// explicit `integrate` op; ADR-0010 open questions). Because it is pointwise closed-form, the CPU
// golden is a nested `(s, θ)` loop and the GPU kernel is one invocation per vertex — the same
// function twice, agreeing by construction (ADR-0003).
//
// The fluent chain (`horn().radius(…).bend(…).twist(…).scale(…)`) is **pure and synchronous**: it
// builds this IR and evaluates nothing. `await` appears exactly once, at `toMesh` (the pull
// boundary). A built chain is a first-class value — inspectable, serialisable, breedable.
//
// Framing strategy (the Swept surface's normal along `s`): this slice uses **central finite
// differences** in `(s, θ)` — `normalize(cross(∂p/∂θ, ∂p/∂s))`. Framing a cross-section along a
// curved sweep has no fully-general closed-form normal (ADR-0010); the strategy is deliberately
// swappable, and finite-difference is the honest slice-one choice (identical on CPU and GPU).

import type { ParamSpec } from "../gpu/graph/op";
import { type AngleUnit, isAngle, unitToRadians } from "./angle";
import type { AngleLike } from "./angle-like";
import { collectSpecs, type Expr, type ExprLike, evalExpr, toExpr, wgslExpr, wgslFloat } from "./expr";
import { signPow, type Vec3 } from "./superellipsoid";

// ── The Transform-stack ───────────────────────────────────────────────────────────────

/** One closed-form, `s`(/`θ`)-parameterised coordinate transform in a Swept node's
 *  Transform-stack. `bend`/`twist` carry an angle expression (radians); `scale` a factor. */
export type Transform = { kind: "bend"; angle: Expr } | { kind: "twist"; angle: Expr } | { kind: "scale"; factor: Expr };

/** Apply one transform to a point at `(s, θ)` — the CPU image of its WGSL codegen. */
function applyTransform(t: Transform, s: number, theta: number, p: Vec3): Vec3 {
  const [x, y, z] = p;
  switch (t.kind) {
    case "bend": {
      // Rotate about +X: curves the sweep axis in the y–z plane as the angle grows along s.
      const a = evalExpr(t.angle, s, theta);
      const c = Math.cos(a);
      const sn = Math.sin(a);
      return [x, y * c - z * sn, y * sn + z * c];
    }
    case "twist": {
      // Rotate about +Z (the travel axis): spins the profile around the sweep.
      const a = evalExpr(t.angle, s, theta);
      const c = Math.cos(a);
      const sn = Math.sin(a);
      return [x * c - y * sn, x * sn + y * c, z];
    }
    case "scale": {
      // Scale the cross-section (a taper when the factor varies along s).
      const f = evalExpr(t.factor, s, theta);
      return [x * f, y * f, z];
    }
  }
}

/** The WGSL statement(s) that apply one transform to the local `var p: vec3<f32>`. */
function transformWgsl(t: Transform): string {
  switch (t.kind) {
    case "bend":
      return `{ let a = ${wgslExpr(t.angle)}; let c = cos(a); let sn = sin(a); p = vec3<f32>(p.x, p.y * c - p.z * sn, p.y * sn + p.z * c); }`;
    case "twist":
      return `{ let a = ${wgslExpr(t.angle)}; let c = cos(a); let sn = sin(a); p = vec3<f32>(p.x * c - p.y * sn, p.x * sn + p.y * c, p.z); }`;
    case "scale":
      return `{ let f = ${wgslExpr(t.factor)}; p = vec3<f32>(p.x * f, p.y * f, p.z); }`;
  }
}

// ── Materialised output ───────────────────────────────────────────────────────────────

/** A tessellated Swept surface as plain typed arrays — the **interop / portability escape
 *  hatch** materialisation form (off the render path; ADR-0010). Positions and normals are
 *  flat `[x,y,z, …]` (3 f32 per vertex; a flat `array<f32>` dodges the vec3-stride-16 trap).
 *  The vertex grid is `(slices + 1) × (stacks + 1)` corners; `indices` wind two triangles per
 *  quad. */
export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  slices: number;
  stacks: number;
}

export interface TessellateOptions {
  /** Profile divisions around θ. */
  slices?: number;
  /** Sweep divisions along s. */
  stacks?: number;
}

const DEFAULT_SLICES = 24;
const DEFAULT_STACKS = 16;

/** Finite-difference step for the framing-strategy normal. Chosen large enough that the f32
 *  cancellation in `∂p` stays well under the parity tolerance (a smaller step trades truncation
 *  error for catastrophic cancellation on the GPU). */
export const FRAME_H = 1e-2;

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** `(s, θ)` for grid corner `(col, row)` of a `slices × stacks` tessellation. */
export function gridSampleAngles(col: number, row: number, slices: number, stacks: number): { s: number; theta: number } {
  return { s: row / stacks, theta: (col / slices) * 2 * Math.PI };
}

// ── The Swept Geometry value ──────────────────────────────────────────────────────────

/** Config for a `horn()` / Swept generator. `angleUnit` governs bare-number angles (the
 *  catalogue default; never a hidden global). */
export interface HornConfig {
  radius?: ExprLike;
  /** Superellipse exponent of the profile (1 = circle, <1 boxier, >1 pinched). Constant in
   *  this slice. */
  exponent?: number;
  /** Sweep length L (the base straight-axis extent before transforms). */
  length?: number;
  angleUnit?: AngleUnit;
}

/** A lazy, typed Swept Geometry — the value the horn grammar builds. Immutable: every op
 *  returns a new `Swept`, so a built chain is a stable first-class value. */
export class Swept {
  readonly kind = "swept" as const;
  readonly profileRadius: Expr;
  readonly exponent: number;
  readonly length: number;
  readonly stack: readonly Transform[];
  readonly angleUnit: AngleUnit;

  constructor(init: {
    profileRadius: Expr;
    exponent: number;
    length: number;
    stack: readonly Transform[];
    angleUnit: AngleUnit;
  }) {
    this.profileRadius = init.profileRadius;
    this.exponent = init.exponent;
    this.length = init.length;
    this.stack = init.stack;
    this.angleUnit = init.angleUnit;
  }

  private with(patch: Partial<{ profileRadius: Expr; stack: readonly Transform[] }>): Swept {
    return new Swept({
      profileRadius: patch.profileRadius ?? this.profileRadius,
      exponent: this.exponent,
      length: this.length,
      stack: patch.stack ?? this.stack,
      angleUnit: this.angleUnit,
    });
  }

  /** Convert an angle argument to a canonical-radians `Expr`. A typed `Angle` is already
   *  canonical; a bare number / expression is scaled by the catalogue's default unit. */
  private angleExpr(a: AngleLike): Expr {
    if (isAngle(a)) return toExpr(a.radians);
    const factor = unitToRadians(this.angleUnit);
    if (typeof a === "number") return toExpr(a * factor);
    return factor === 1 ? a : { kind: "mul", a: { kind: "const", value: factor }, b: a };
  }

  private push(t: Transform): Swept {
    return this.with({ stack: [...this.stack, t] });
  }

  /** Set the profile radius as an `{s, θ}`-expression (a bare number is a constant). */
  radius(r: ExprLike): Swept {
    return this.with({ profileRadius: toExpr(r) });
  }

  /** Bend the sweep axis: rotate about +X by an angle that varies along `s`. */
  bend(a: AngleLike): Swept {
    return this.push({ kind: "bend", angle: this.angleExpr(a) });
  }

  /** Twist the profile about the travel axis: rotate about +Z by an angle along `s`. */
  twist(a: AngleLike): Swept {
    return this.push({ kind: "twist", angle: this.angleExpr(a) });
  }

  /** Scale (taper) the cross-section by a factor along `s`. */
  scale(f: ExprLike): Swept {
    return this.push({ kind: "scale", factor: toExpr(f) });
  }

  /** The breeding surface: every `ParamSpec` gene carried by this geometry's expressions. */
  specs(): ParamSpec[] {
    const out: ParamSpec[] = [];
    collectSpecs(this.profileRadius, out);
    for (const t of this.stack) collectSpecs(t.kind === "scale" ? t.factor : t.angle, out);
    return out;
  }

  /** The base swept placement before the Transform-stack: the profile point at `(s, θ)` on the
   *  straight axis. */
  private base(s: number, theta: number): Vec3 {
    const r = evalExpr(this.profileRadius, s, theta);
    return [r * signPow(Math.cos(theta), this.exponent), r * signPow(Math.sin(theta), this.exponent), s * this.length];
  }

  /** The pointwise closed-form position at `(s, θ)` — profile placement then the stack in order. */
  position(s: number, theta: number): Vec3 {
    let p = this.base(s, theta);
    for (const t of this.stack) p = applyTransform(t, s, theta, p);
    return p;
  }

  /** The pointwise `(position, normal)` at `(s, θ)`. The normal is the finite-difference
   *  framing strategy — `normalize(cross(∂p/∂θ, ∂p/∂s))`, outward for a plain sweep. */
  eval(s: number, theta: number): { position: Vec3; normal: Vec3 } {
    const position = this.position(s, theta);
    const dTheta = sub3(this.position(s, theta + FRAME_H), this.position(s, theta - FRAME_H));
    const dS = sub3(this.position(s + FRAME_H, theta), this.position(s - FRAME_H, theta));
    const n = cross(dTheta, dS);
    const len = Math.hypot(n[0], n[1], n[2]);
    const normal: Vec3 = len > 1e-8 ? [n[0] / len, n[1] / len, n[2] / len] : [0, 0, 1];
    return { position, normal };
  }

  /** Tessellate on the CPU — the golden `(s, θ)` loop. */
  tessellate(opts: TessellateOptions = {}): Mesh {
    const slices = opts.slices ?? DEFAULT_SLICES;
    const stacks = opts.stacks ?? DEFAULT_STACKS;
    const cols = slices + 1;
    const rows = stacks + 1;
    const vertexCount = cols * rows;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const { s, theta } = gridSampleAngles(col, row, slices, stacks);
        const { position, normal } = this.eval(s, theta);
        const i = (row * cols + col) * 3;
        positions[i] = position[0];
        positions[i + 1] = position[1];
        positions[i + 2] = position[2];
        normals[i] = normal[0];
        normals[i + 1] = normal[1];
        normals[i + 2] = normal[2];
      }
    }
    return { positions, normals, indices: gridIndices(slices, stacks), vertexCount, slices, stacks };
  }

  /** Pull this geometry to a Mesh (the single async boundary). CPU by default; pass
   *  `{ mode: "gpu", device, root }` to lower to the `"use gpu"` TGSL per-vertex kernel — the
   *  same closed-form function, run once per vertex. Positions/normals differ only by backend
   *  arithmetic; the index winding is identical. */
  async toMesh(opts: ToMeshOptions = {}): Promise<Mesh> {
    if (opts.mode === "gpu") {
      const { device, root } = opts;
      if (!device || !root) throw new Error("toMesh(gpu): a device and root are required");
      const slices = opts.slices ?? DEFAULT_SLICES;
      const stacks = opts.stacks ?? DEFAULT_STACKS;
      // Dynamic import keeps the `"use gpu"` / device module out of the CPU module graph
      // (Dawn-on-Node teardown is sensitive to what pulls the kernel in; element.gpu notes).
      const { sweptMeshGpu } = await import("./sweptGpu");
      return sweptMeshGpu(device, root, this, slices, stacks);
    }
    return this.tessellate(opts);
  }

  /** Lower this geometry's per-vertex position+normal evaluation to a WGSL compute shader —
   *  the codegen mirror of {@link position}/{@link eval}. Pure string assembly (no device
   *  dependency); `sweptGpu` binds and dispatches it. */
  toWgsl(): string {
    return sweptShaderWgsl(this);
  }
}

/** An angle argument to a transform-op. */
export type { AngleLike } from "./angle-like";

export interface ToMeshOptions extends TessellateOptions {
  mode?: "cpu" | "gpu";
  device?: GPUDevice;
  root?: import("../gpu/graph/backend").Root;
}

/** Two triangles per quad over the `(slices + 1) × (stacks + 1)` vertex grid, wound so the
 *  face normal agrees with the analytic outward normal. */
export function gridIndices(slices: number, stacks: number): Uint32Array {
  const cols = slices + 1;
  const indices = new Uint32Array(slices * stacks * 6);
  let k = 0;
  for (let row = 0; row < stacks; row++) {
    for (let col = 0; col < slices; col++) {
      const a = row * cols + col;
      const b = a + 1;
      const c = a + cols;
      const dd = c + 1;
      indices[k++] = a;
      indices[k++] = c;
      indices[k++] = dd;
      indices[k++] = a;
      indices[k++] = dd;
      indices[k++] = b;
    }
  }
  return indices;
}

// ── Generator-op + catalogue ──────────────────────────────────────────────────────────

/** Default profile radius, sweep length, and profile exponent for a bare `horn()`. */
const DEFAULT_RADIUS = 0.5;
const DEFAULT_LENGTH = 2;
const DEFAULT_EXPONENT = 1;

/** A Catalogue: the registered geometry-ops bound to a config (the default Angle unit). The
 *  surface a user composes from. Slice one exposes the `horn()` generator. */
export interface Catalogue {
  angleUnit: AngleUnit;
  horn(config?: HornConfig): Swept;
}

/** Build a Catalogue with an explicit default Angle unit (default `deg`, matching the FormGrow
 *  horn grammar's `twist(360)`). */
export function catalogue(opts: { angleUnit?: AngleUnit } = {}): Catalogue {
  const angleUnit = opts.angleUnit ?? "deg";
  return {
    angleUnit,
    horn(config: HornConfig = {}): Swept {
      return new Swept({
        profileRadius: toExpr(config.radius ?? DEFAULT_RADIUS),
        exponent: config.exponent ?? DEFAULT_EXPONENT,
        length: config.length ?? DEFAULT_LENGTH,
        stack: [],
        angleUnit: config.angleUnit ?? angleUnit,
      });
    },
  };
}

const defaultCatalogue = catalogue();

/** The horn Generator-op — a Swept geometry from nothing, ready to chain transforms onto.
 *  Uses the default catalogue (degrees); construct your own with {@link catalogue} for a
 *  different default Angle unit. */
export function horn(config?: HornConfig): Swept {
  return defaultCatalogue.horn(config);
}

// ── WGSL codegen ──────────────────────────────────────────────────────────────────────

/** Assemble the full WGSL compute shader for a Swept geometry: decode each grid corner to
 *  `(s, θ)`, evaluate the closed-form position, and derive the finite-difference normal — the
 *  GPU image of {@link Swept.tessellate}. Buffers are flat `array<f32>` (3 lanes/vertex). */
export function sweptShaderWgsl(g: Swept): string {
  const stack = g.stack.map(transformWgsl).join("\n    ");
  return /* wgsl */ `
struct Params { slices: u32, stacks: u32, vertexCount: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> outPos: array<f32>;
@group(0) @binding(2) var<storage, read_write> outNor: array<f32>;

const TWO_PI: f32 = 6.283185307179586;
const FRAME_H: f32 = ${wgslFloat(FRAME_H)};
const E: f32 = ${wgslFloat(g.exponent)};
const L: f32 = ${wgslFloat(g.length)};

fn signpow(x: f32, e: f32) -> f32 { return sign(x) * pow(abs(x), e); }

fn evalPos(s: f32, th: f32) -> vec3<f32> {
  let r = ${wgslExpr(g.profileRadius)};
  var p = vec3<f32>(r * signpow(cos(th), E), r * signpow(sin(th), E), s * L);
    ${stack}
  return p;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.vertexCount) { return; }
  let cols = params.slices + 1u;
  let col = i % cols;
  let row = i / cols;
  let s = f32(row) / f32(params.stacks);
  let th = f32(col) / f32(params.slices) * TWO_PI;
  let p = evalPos(s, th);
  let dTheta = evalPos(s, th + FRAME_H) - evalPos(s, th - FRAME_H);
  let dS = evalPos(s + FRAME_H, th) - evalPos(s - FRAME_H, th);
  var n = cross(dTheta, dS);
  let len = length(n);
  if (len > 1e-8) { n = n / len; } else { n = vec3<f32>(0.0, 0.0, 1.0); }
  let o = i * 3u;
  outPos[o] = p.x; outPos[o + 1u] = p.y; outPos[o + 2u] = p.z;
  outNor[o] = n.x; outNor[o + 1u] = n.y; outNor[o + 2u] = n.z;
}
`;
}
