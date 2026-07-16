// Implicit geometry — the signed-distance Geometry-kind (ADR-0010), the natural home for CSG.
// An Implicit geometry is a scalar field `f(p) → distance`: negative inside the solid, positive
// outside, zero on the surface. Booleans are min/max combinators over these fields, so CSG is
// *closed-form and pointwise* the same way the Swept kind is — which is exactly what lets one IR
// lower two ways that agree by construction (the repo's CPU-golden == GPU-kernel discipline,
// ADR-0003):
//
//   • `evalSdf(node, p)`   — the CPU evaluator (the golden).
//   • `wgslSdf(node, …)`   — a WGSL distance-function fragment over a shader point `p`.
//
// This is deliberately NOT the `@typegpu/sdf` model (compose `"use gpu"` functions in *source*).
// A CSG tree is built at *runtime* and must stay inspectable / serialisable / breedable /
// round-trippable (ADR-0010, ADR-0012), so — exactly as `expr.ts` does for param-expressions — we
// keep a typed IR and **codegen WGSL from it**, never author a static kernel. `@typegpu/sdf` is a
// *formula vocabulary* (its `smin`/primitive closed-forms are the WGSL we emit) and a *raymarch
// render backend*, not our composition model.
//
// Structure / value split (mirrors `expr.ts`): a primitive's numeric params flow through a uniform
// buffer `P`, so the emitted WGSL depends only on the tree's *structure*. Two same-structure trees
// (a sphere-minus-box with different radii) share one pipeline; you vary them by uploading a new
// `paramVector()`. Param literals may carry a `ParamSpec` gene, so a CSG tree breeds like a horn.
//
// Scope (first slice, ADR-0010's deferred implicit kind): generators `sphere`/`box`/`plane`;
// booleans `union`/`intersect`/`subtract`/`smoothUnion`; domain transforms `translate`/`scale`
// (both **exact** distance-preserving maps). CPU extraction (`toMesh`) gives a renderable golden —
// smooth surface nets by default, opt-in `sharpen` (dual contouring) for hard CSG edges. Designed-for,
// not built: the GPU extraction pass (classify
// → compact → connect; see `docs/explainers/surface-nets-dual-contouring.html`), rotation and
// non-uniform warps (which only *bound* distance, so they need a Lipschitz note), and the
// argmin-primitive-id provenance channel (ADR-0012) — the immutable IR + structural pre-order already
// give each primitive a stable address for it.

import type { ParamSpec } from "../gpu/graph/op";
import {
  collectConsts,
  collectSpecs,
  type Expr,
  type ExprLike,
  evalExpr,
  toExpr,
  type UniformCtx,
  wgslExpr,
  wgslExprUniform,
} from "./expr";
import type { Vec3 } from "./superellipsoid";

// ── The SDF IR ────────────────────────────────────────────────────────────────────────
//
// A small typed DAG (ADR-0007), the vector-domain sibling of `expr.ts`'s scalar `Expr`. Scalar
// params are `Expr` (reused wholesale — a bare number is a constant, and a constant carries an
// optional gene), so the breeding surface / structure-value machinery is shared, not re-derived.
// Params are constants over the domain point in this slice; the free variable is `p`, not `{s,θ}`.

/** A signed-distance-field node. Generators are leaves; booleans and transforms are interior. */
export type Sdf =
  | { kind: "sphere"; radius: Expr }
  | { kind: "box"; half: [Expr, Expr, Expr] }
  /** Half-space `dot(p, n̂) − d ≤ 0`. `n` need not be unit; it is normalised on evaluation. */
  | { kind: "plane"; n: [Expr, Expr, Expr]; d: Expr }
  | { kind: "union"; a: Sdf; b: Sdf }
  | { kind: "intersect"; a: Sdf; b: Sdf }
  | { kind: "subtract"; a: Sdf; b: Sdf }
  | { kind: "smoothUnion"; a: Sdf; b: Sdf; k: Expr }
  | { kind: "translate"; t: [Expr, Expr, Expr]; child: Sdf }
  | { kind: "scale"; factor: Expr; child: Sdf }
  /** Add `amp · fbm(freq · p)` to a child field — a value-noise displacement that makes an organic,
   *  lumpy surface. Breaks the exact distance property (the raymarch under-relaxes for it), so it is a
   *  *display* op: the raymarch animates it via a time drift; the grid mesher can still tessellate it. */
  | { kind: "displace"; child: Sdf; amp: Expr; freq: Expr };

/** Visit every scalar `Expr` param of the tree in the **canonical pre-order** — own params before
 *  children, in field order. `paramVector`, `specs`, and `wgslSdf` all traverse in this exact order,
 *  so the GPU `P` buffer, the breeding surface, and the emitted `P[k]` slots line up lane-for-lane
 *  (the GPU parity test is the backstop against drift). */
function eachParam(node: Sdf, visit: (e: Expr) => void): void {
  switch (node.kind) {
    case "sphere":
      visit(node.radius);
      break;
    case "box":
      visit(node.half[0]);
      visit(node.half[1]);
      visit(node.half[2]);
      break;
    case "plane":
      visit(node.n[0]);
      visit(node.n[1]);
      visit(node.n[2]);
      visit(node.d);
      break;
    case "union":
    case "intersect":
    case "subtract":
      eachParam(node.a, visit);
      eachParam(node.b, visit);
      break;
    case "smoothUnion":
      eachParam(node.a, visit);
      eachParam(node.b, visit);
      visit(node.k);
      break;
    case "translate":
      visit(node.t[0]);
      visit(node.t[1]);
      visit(node.t[2]);
      eachParam(node.child, visit);
      break;
    case "scale":
      visit(node.factor);
      eachParam(node.child, visit);
      break;
    case "displace":
      eachParam(node.child, visit);
      visit(node.amp);
      visit(node.freq);
      break;
  }
}

// ── CPU golden ────────────────────────────────────────────────────────────────────────

/** Evaluate a constant scalar param on the CPU (`{s,θ}` are unused — implicit params are
 *  constants over the domain point). */
const val = (e: Expr): number => evalExpr(e, 0, 0);

function sdBox(p: Vec3, b: Vec3): number {
  const qx = Math.abs(p[0]) - b[0];
  const qy = Math.abs(p[1]) - b[1];
  const qz = Math.abs(p[2]) - b[2];
  const ox = Math.max(qx, 0);
  const oy = Math.max(qy, 0);
  const oz = Math.max(qz, 0);
  return Math.hypot(ox, oy, oz) + Math.min(Math.max(qx, qy, qz), 0);
}

/** The exact smooth-union closed-form (matches `@typegpu/sdf` `opSmoothUnion` and the WGSL
 *  `smin` below): a polynomial blend that rounds the union seam over width `k`. */
function smin(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

// ── value noise (the `displace` op) ─────────────────────────────────────────────────────
// A hash-lattice value noise + 4-octave fbm. The hash is pure u32 integer arithmetic (emulated with
// `Math.imul`/`>>>0`), *identical* to the WGSL in `SDF_PREAMBLE`, so the only CPU/GPU divergence is
// the final f32-vs-f64 interpolation — small enough for the parity test's tolerance.

/** iq's integer hash, u32 → u32 (bit-exact with the WGSL `hashU`). */
function hashU(n: number): number {
  let m = (n ^ (n << 13)) >>> 0;
  const sq = Math.imul(m, m) >>> 0;
  m = (Math.imul(m, (Math.imul(sq, 15731) + 789221) >>> 0) + 1376312589) >>> 0;
  return m >>> 0;
}
/** Value at an integer lattice corner, in `[0, 1]`. */
function hashLattice(ix: number, iy: number, iz: number): number {
  const n = (Math.imul(ix, 1619) + Math.imul(iy, 31337) + Math.imul(iz, 6971)) >>> 0;
  return (hashU(n) & 0x7fffffff) / 0x7fffffff;
}
const quintic = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10);
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/** Trilinearly-interpolated, quintic-smoothed value noise in `[-1, 1]`. */
function valueNoise(px: number, py: number, pz: number): number {
  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const iz = Math.floor(pz);
  const ux = quintic(px - ix);
  const uy = quintic(py - iy);
  const uz = quintic(pz - iz);
  const x00 = lerp(hashLattice(ix, iy, iz), hashLattice(ix + 1, iy, iz), ux);
  const x10 = lerp(hashLattice(ix, iy + 1, iz), hashLattice(ix + 1, iy + 1, iz), ux);
  const x01 = lerp(hashLattice(ix, iy, iz + 1), hashLattice(ix + 1, iy, iz + 1), ux);
  const x11 = lerp(hashLattice(ix, iy + 1, iz + 1), hashLattice(ix + 1, iy + 1, iz + 1), ux);
  return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz) * 2 - 1;
}

/** Four-octave fractal Brownian motion of {@link valueNoise}. */
function fbm3(px: number, py: number, pz: number): number {
  let s = 0;
  let a = 0.5;
  let qx = px;
  let qy = py;
  let qz = pz;
  for (let o = 0; o < 4; o++) {
    s += a * valueNoise(qx, qy, qz);
    qx *= 2;
    qy *= 2;
    qz *= 2;
    a *= 0.5;
  }
  return s;
}

/** Signed distance from `p` to the surface described by `node` — the golden reference. Negative
 *  inside, positive outside, zero on the surface. */
export function evalSdf(node: Sdf, p: Vec3): number {
  switch (node.kind) {
    case "sphere":
      return Math.hypot(p[0], p[1], p[2]) - val(node.radius);
    case "box":
      return sdBox(p, [val(node.half[0]), val(node.half[1]), val(node.half[2])]);
    case "plane": {
      const nx = val(node.n[0]);
      const ny = val(node.n[1]);
      const nz = val(node.n[2]);
      const len = Math.hypot(nx, ny, nz) || 1;
      return (p[0] * nx + p[1] * ny + p[2] * nz) / len - val(node.d);
    }
    case "union":
      return Math.min(evalSdf(node.a, p), evalSdf(node.b, p));
    case "intersect":
      return Math.max(evalSdf(node.a, p), evalSdf(node.b, p));
    case "subtract":
      return Math.max(evalSdf(node.a, p), -evalSdf(node.b, p));
    case "smoothUnion":
      return smin(evalSdf(node.a, p), evalSdf(node.b, p), val(node.k));
    case "translate":
      return evalSdf(node.child, [p[0] - val(node.t[0]), p[1] - val(node.t[1]), p[2] - val(node.t[2])]);
    case "scale": {
      const f = val(node.factor);
      return evalSdf(node.child, [p[0] / f, p[1] / f, p[2] / f]) * f;
    }
    case "displace": {
      const freq = val(node.freq);
      return evalSdf(node.child, p) + val(node.amp) * fbm3(p[0] * freq, p[1] * freq, p[2] * freq);
    }
  }
}

/** Central-difference gradient of the field, normalised — the surface normal. Exact-enough for the
 *  golden; the analytic gradient is a later optimisation. */
export function normalSdf(node: Sdf, p: Vec3, eps = 1e-4): Vec3 {
  const dx = evalSdf(node, [p[0] + eps, p[1], p[2]]) - evalSdf(node, [p[0] - eps, p[1], p[2]]);
  const dy = evalSdf(node, [p[0], p[1] + eps, p[2]]) - evalSdf(node, [p[0], p[1] - eps, p[2]]);
  const dz = evalSdf(node, [p[0], p[1], p[2] + eps]) - evalSdf(node, [p[0], p[1], p[2] - eps]);
  const len = Math.hypot(dx, dy, dz) || 1;
  return [dx / len, dy / len, dz / len];
}

// ── WGSL codegen ──────────────────────────────────────────────────────────────────────
//
// The mirror of `evalSdf`: the same tree, so CPU and GPU distances agree (the GPU parity test pins
// this). Each node emits a WGSL `f32` expression for the distance at a **point expression** (a WGSL
// `vec3<f32>` string); domain transforms wrap that point expression, so no matrices or temporaries
// are needed for `translate`/`scale`. Param **values** are read from the storage buffer `P` via
// `wgslExprUniform` — own params are read *before* recursing into children, matching `eachParam`.

/** A WGSL `f32` distance expression for `node` at the point expression `pt`. `emit` renders each
 *  scalar param — either `P[k]` reads (the uniform path, one pipeline many values) or baked literals
 *  (self-contained WGSL for embedding, e.g. a `wgslFn` raymarch material). Own params are emitted
 *  *before* recursing into children, matching `eachParam`. */
export function wgslSdf(node: Sdf, pt: string, emit: (e: Expr) => string): string {
  switch (node.kind) {
    case "sphere":
      return `(length(${pt}) - ${emit(node.radius)})`;
    case "box": {
      const bx = emit(node.half[0]);
      const by = emit(node.half[1]);
      const bz = emit(node.half[2]);
      return `sdBox(${pt}, vec3<f32>(${bx}, ${by}, ${bz}))`;
    }
    case "plane": {
      const nx = emit(node.n[0]);
      const ny = emit(node.n[1]);
      const nz = emit(node.n[2]);
      const dd = emit(node.d);
      return `(dot(${pt}, normalize(vec3<f32>(${nx}, ${ny}, ${nz}))) - ${dd})`;
    }
    case "union":
      return `min(${wgslSdf(node.a, pt, emit)}, ${wgslSdf(node.b, pt, emit)})`;
    case "intersect":
      return `max(${wgslSdf(node.a, pt, emit)}, ${wgslSdf(node.b, pt, emit)})`;
    case "subtract":
      return `max(${wgslSdf(node.a, pt, emit)}, -(${wgslSdf(node.b, pt, emit)}))`;
    case "smoothUnion": {
      // a, then b, then k — matches eachParam. Bind children to locals so `smin`'s args don't
      // re-expand the subtrees.
      const a = wgslSdf(node.a, pt, emit);
      const b = wgslSdf(node.b, pt, emit);
      const k = emit(node.k);
      return `smin(${a}, ${b}, ${k})`;
    }
    case "translate": {
      const tx = emit(node.t[0]);
      const ty = emit(node.t[1]);
      const tz = emit(node.t[2]);
      return wgslSdf(node.child, `(${pt} - vec3<f32>(${tx}, ${ty}, ${tz}))`, emit);
    }
    case "scale": {
      const f = emit(node.factor);
      return `(${wgslSdf(node.child, `(${pt} / ${f})`, emit)} * ${f})`;
    }
    case "displace": {
      // Child first (emits its params), then amp, then freq — matching eachParam. The noise domain
      // carries `uNoiseTime` (a private global the raymarch sets per frame; 0 for the static golden),
      // so the lumps drift over time without disturbing the base shape.
      const child = wgslSdf(node.child, pt, emit);
      const amp = emit(node.amp);
      const freq = emit(node.freq);
      return `(${child} + ${amp} * fbm3((${pt}) * ${freq} + vec3<f32>(0.0, uNoiseTime, uNoiseTime * 0.5)))`;
    }
  }
}

/** WGSL preamble: the primitive/combinator helpers `wgslSdf` references. */
export const SDF_PREAMBLE = /* wgsl */ `
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}
// Value noise for the displace op — bit-exact hash with the CPU golden (u32 integer arithmetic).
// A render pass sets uNoiseTime per frame to drift the noise domain (animation); it defaults to 0, so
// the static field the CPU golden and parity harness see is at uNoiseTime = 0.
var<private> uNoiseTime: f32 = 0.0;
fn hashU(n0: u32) -> u32 {
  let n = (n0 << 13u) ^ n0;
  return n * (n * n * 15731u + 789221u) + 1376312589u;
}
fn hashLattice(c: vec3<i32>) -> f32 {
  let n = u32(c.x) * 1619u + u32(c.y) * 31337u + u32(c.z) * 6971u;
  return f32(hashU(n) & 0x7fffffffu) / f32(0x7fffffff);
}
fn quintic3(t: vec3<f32>) -> vec3<f32> { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
fn valueNoise(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let ci = vec3<i32>(i);
  let u = quintic3(p - i);
  let x00 = mix(hashLattice(ci + vec3<i32>(0, 0, 0)), hashLattice(ci + vec3<i32>(1, 0, 0)), u.x);
  let x10 = mix(hashLattice(ci + vec3<i32>(0, 1, 0)), hashLattice(ci + vec3<i32>(1, 1, 0)), u.x);
  let x01 = mix(hashLattice(ci + vec3<i32>(0, 0, 1)), hashLattice(ci + vec3<i32>(1, 0, 1)), u.x);
  let x11 = mix(hashLattice(ci + vec3<i32>(0, 1, 1)), hashLattice(ci + vec3<i32>(1, 1, 1)), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z) * 2.0 - 1.0;
}
fn fbm3(p0: vec3<f32>) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p0;
  for (var o = 0; o < 4; o = o + 1) {
    s = s + a * valueNoise(q);
    q = q * 2.0;
    a = a * 0.5;
  }
  return s;
}
`;

/** The scene distance function `sdScene(p) -> f32` for a tree — the reusable GPU building block a
 *  raymarch pass or (future) extraction pass calls. By default reads params from the storage buffer
 *  `P` (see {@link Implicit.paramVector}); with `bakeConstants` the params are baked as literals so
 *  the WGSL is self-contained (for embedding in a `wgslFn` / another shader, rebuilt per param set). */
export function sdSceneWgsl(node: Sdf, opts: { bakeConstants?: boolean } = {}): string {
  const ctx: UniformCtx = { next: 0 };
  const emit = opts.bakeConstants ? (e: Expr) => wgslExpr(e) : (e: Expr) => wgslExprUniform(e, ctx);
  const body = wgslSdf(node, "p", emit);
  return `${SDF_PREAMBLE}\nfn sdScene(p: vec3<f32>) -> f32 {\n  return ${body};\n}\n`;
}

// ── CPU surface-nets extraction ─────────────────────────────────────────────────────────
//
// The `implicit → mesh` bridge (ADR-0010), CPU golden. One vertex per straddling cell: **smooth by
// default** — the average of its edge crossings (surface nets), which traces smooth isosurfaces (e.g.
// volume-derived) cleanly; **`sharpen`** opts into a per-cell QEF (dual contouring) that recovers
// sharp CSG edges/corners, at the cost of some jitter on *curved* creases (one vertex per cell can't
// trace a curved edge). Quads are dual to the grid's bipolar edges; per-vertex normals come from the
// exact field gradient. The GPU compute-pass version is the documented next step.

/** A tessellated Implicit surface as plain typed arrays — the interop / portability form (ADR-0010).
 *  Positions/normals are flat `[x,y,z, …]`; `indices` triangulate the dual quads. */
export interface IsoMesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
}

export interface TessellateOptions {
  /** Half-extent of the sampling cube `[-bounds, bounds]³`. Must enclose the surface. */
  bounds?: number;
  /** Cells per axis. */
  res?: number;
  /** Recover sharp CSG edges/corners via a per-cell QEF (dual contouring) instead of the smooth
   *  average-of-crossings (surface nets). Default `false` — smooth is right for volume-derived
   *  isosurfaces; `sharpen` trades some curved-crease jitter for hard edges on CSG. */
  sharpen?: boolean;
}

const DEFAULT_BOUNDS = 2;
const DEFAULT_RES = 24;

// The 12 edges of a cell as corner-index pairs (corner bit = x | y<<1 | z<<2).
const CELL_EDGES: [number, number][] = [
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7], // x-dir
  [0, 2],
  [1, 3],
  [4, 6],
  [5, 7], // y-dir
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7], // z-dir
];
const CORNER_OFFSET: Vec3[] = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
];

/** Solve the symmetric-positive-definite 3×3 system `A x = b` (A given by its upper triangle) by its
 *  adjugate. Returns null if `A` is effectively singular. The dual-contouring QEF's normal equations. */
function solveSym3(
  a00: number,
  a01: number,
  a02: number,
  a11: number,
  a12: number,
  a22: number,
  b0: number,
  b1: number,
  b2: number,
): Vec3 | null {
  const c00 = a11 * a22 - a12 * a12;
  const c01 = a02 * a12 - a01 * a22;
  const c02 = a01 * a12 - a02 * a11;
  const det = a00 * c00 + a01 * c01 + a02 * c02;
  if (Math.abs(det) < 1e-12) return null;
  const c11 = a00 * a22 - a02 * a02;
  const c12 = a02 * a01 - a00 * a12;
  const c22 = a00 * a11 - a01 * a01;
  const id = 1 / det;
  return [(c00 * b0 + c01 * b1 + c02 * b2) * id, (c01 * b0 + c11 * b1 + c12 * b2) * id, (c02 * b0 + c12 * b1 + c22 * b2) * id];
}

/** Extract the zero-isosurface on the CPU over a `res³` grid — the golden for the `implicit → mesh`
 *  bridge. One vertex per straddling cell: by default the **smooth** average of its edge crossings
 *  (surface nets); with `sharpen`, a per-cell **QEF** (dual contouring) that recovers sharp CSG
 *  edges. Quads are dual to the grid's bipolar edges. */
export function tessellateSdf(node: Sdf, opts: TessellateOptions = {}): IsoMesh {
  const bound = opts.bounds ?? DEFAULT_BOUNDS;
  const N = opts.res ?? DEFAULT_RES;
  const sharpen = opts.sharpen ?? false;
  const step = (2 * bound) / N;
  const at = (i: number) => -bound + i * step;

  // Corner field values on the (N+1)³ lattice.
  const S = N + 1;
  const vals = new Float32Array(S * S * S);
  const vIdx = (i: number, j: number, k: number) => (k * S + j) * S + i;
  for (let k = 0; k < S; k++)
    for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) vals[vIdx(i, j, k)] = evalSdf(node, [at(i), at(j), at(k)]);

  // One vertex per active cell; `cellVert[cellIndex] = vertex slot` (or -1).
  const cellVert = new Int32Array(N * N * N).fill(-1);
  const cIdx = (i: number, j: number, k: number) => (k * N + j) * N + i;
  const positions: number[] = [];
  const normals: number[] = [];

  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        // Gather the 8 corner signs.
        const corner: number[] = [];
        for (let c = 0; c < 8; c++) {
          const o = CORNER_OFFSET[c] ?? [0, 0, 0];
          corner.push(vals[vIdx(i + o[0], j + o[1], k + o[2])] ?? 0);
        }
        // Place the cell vertex. Smooth (default): the mass point — the average of the edge
        // crossings (surface nets), which traces smooth isosurfaces cleanly. `sharpen`: additionally
        // accumulate a QEF `min Σ (nᵢ·(x−pᵢ))²` over the crossings' Hermite data (point `pᵢ` +
        // gradient `nᵢ`) and place the vertex on the intersection of their tangent planes — the sharp
        // edge/corner — falling back to the smooth mass point when it overshoots the cell.
        let a00 = 0;
        let a01 = 0;
        let a02 = 0;
        let a11 = 0;
        let a12 = 0;
        let a22 = 0;
        let b0 = 0;
        let b1 = 0;
        let b2 = 0;
        let mx = 0; // mass point (crossing centroid)
        let my = 0;
        let mz = 0;
        let n = 0;
        for (const [ca, cb] of CELL_EDGES) {
          const fa = corner[ca] ?? 0;
          const fb = corner[cb] ?? 0;
          if (fa < 0 === fb < 0) continue;
          const t = fa / (fa - fb);
          const oa = CORNER_OFFSET[ca] ?? [0, 0, 0];
          const ob = CORNER_OFFSET[cb] ?? [0, 0, 0];
          const px = at(i + oa[0] + (ob[0] - oa[0]) * t);
          const py = at(j + oa[1] + (ob[1] - oa[1]) * t);
          const pz = at(k + oa[2] + (ob[2] - oa[2]) * t);
          mx += px;
          my += py;
          mz += pz;
          n++;
          if (!sharpen) continue;
          const [nx, ny, nz] = normalSdf(node, [px, py, pz]);
          const dpl = nx * px + ny * py + nz * pz;
          a00 += nx * nx;
          a01 += nx * ny;
          a02 += nx * nz;
          a11 += ny * ny;
          a12 += ny * nz;
          a22 += nz * nz;
          b0 += dpl * nx;
          b1 += dpl * ny;
          b2 += dpl * nz;
        }
        if (n === 0) continue;
        mx /= n;
        my /= n;
        mz /= n;
        let p: Vec3 = [mx, my, mz];
        if (sharpen) {
          // Tikhonov term toward the mass point keeps flat / near-parallel cells well-conditioned.
          const lam = 0.08;
          a00 += lam;
          a11 += lam;
          a22 += lam;
          b0 += lam * mx;
          b1 += lam * my;
          b2 += lam * mz;
          const sol = solveSym3(a00, a01, a02, a11, a12, a22, b0, b1, b2) ?? [mx, my, mz];
          // If the QEF overshoots the cell (typical on a *curved* crease) fall back to the smooth
          // mass point rather than clamping to a cell face — clamping makes adjacent overshoot cells
          // snap to different faces, which reads as a sawtooth.
          const tol = 0.1 * step;
          const inCell =
            sol[0] >= at(i) - tol &&
            sol[0] <= at(i + 1) + tol &&
            sol[1] >= at(j) - tol &&
            sol[1] <= at(j + 1) + tol &&
            sol[2] >= at(k) - tol &&
            sol[2] <= at(k + 1) + tol;
          if (inCell) p = [sol[0], sol[1], sol[2]];
        }
        const nrm = normalSdf(node, p);
        cellVert[cIdx(i, j, k)] = positions.length / 3;
        positions.push(p[0], p[1], p[2]);
        normals.push(nrm[0], nrm[1], nrm[2]);
      }
    }
  }

  // Quads dual to bipolar grid edges. For a bipolar grid edge along axis A (near corner `here` at
  // (i,j,k), far corner at +A), the 4 cells sharing it are ordered **CCW as seen from +A** — this
  // winds a front face whose right-hand normal points +A. The surface normal must point inside→out
  // (increasing SDF); the SDF increases +A exactly when the near corner is inside (`here < 0`), so
  // we FLIP when the near corner is outside (`here > 0`). Same rule on all three axes — the earlier
  // per-axis guesses disagreed, which backface-culled the mis-wound faces into holes.
  const indices: number[] = [];
  const quad = (a: number, b: number, c: number, dd: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || dd < 0) return;
    if (flip) indices.push(a, c, b, a, dd, c);
    else indices.push(a, b, c, a, c, dd);
  };
  for (let k = 0; k < S; k++) {
    for (let j = 0; j < S; j++) {
      for (let i = 0; i < S; i++) {
        const here = vals[vIdx(i, j, k)] ?? 0;
        const flip = here > 0;
        // +x edge → CCW-from-+x cells in the (y,z) plane.
        if (i < N && j >= 1 && k >= 1 && here < 0 !== (vals[vIdx(i + 1, j, k)] ?? 0) < 0) {
          quad(
            cellVert[cIdx(i, j - 1, k - 1)] ?? -1,
            cellVert[cIdx(i, j, k - 1)] ?? -1,
            cellVert[cIdx(i, j, k)] ?? -1,
            cellVert[cIdx(i, j - 1, k)] ?? -1,
            flip,
          );
        }
        // +y edge → CCW-from-+y cells in the (z,x) plane.
        if (j < N && i >= 1 && k >= 1 && here < 0 !== (vals[vIdx(i, j + 1, k)] ?? 0) < 0) {
          quad(
            cellVert[cIdx(i - 1, j, k - 1)] ?? -1,
            cellVert[cIdx(i - 1, j, k)] ?? -1,
            cellVert[cIdx(i, j, k)] ?? -1,
            cellVert[cIdx(i, j, k - 1)] ?? -1,
            flip,
          );
        }
        // +z edge → CCW-from-+z cells in the (x,y) plane.
        if (k < N && i >= 1 && j >= 1 && here < 0 !== (vals[vIdx(i, j, k + 1)] ?? 0) < 0) {
          quad(
            cellVert[cIdx(i - 1, j - 1, k)] ?? -1,
            cellVert[cIdx(i, j - 1, k)] ?? -1,
            cellVert[cIdx(i, j, k)] ?? -1,
            cellVert[cIdx(i - 1, j, k)] ?? -1,
            flip,
          );
        }
      }
    }
  }

  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    indices: Uint32Array.from(indices),
    vertexCount: positions.length / 3,
  };
}

// ── The Implicit Geometry value ─────────────────────────────────────────────────────────

/** A lazy, typed Implicit Geometry — a signed-distance field built by CSG ops. Immutable: every op
 *  returns a new `Implicit`, so a built tree is a stable first-class value (inspectable,
 *  serialisable, breedable — the provenance address of each primitive is its pre-order position). */
export class Implicit {
  readonly kind = "implicit" as const;
  readonly node: Sdf;

  constructor(node: Sdf) {
    this.node = node;
  }

  // Boolean-ops (defined only on Implicit; to boolean a Swept you bridge swept→implicit first).
  /** Union: the space in either solid (`min` of distances). */
  union(other: Implicit): Implicit {
    return new Implicit({ kind: "union", a: this.node, b: other.node });
  }
  /** Intersection: the space in both solids (`max`). */
  intersect(other: Implicit): Implicit {
    return new Implicit({ kind: "intersect", a: this.node, b: other.node });
  }
  /** Difference: this solid with `other` carved out (`max(a, −b)`). */
  subtract(other: Implicit): Implicit {
    return new Implicit({ kind: "subtract", a: this.node, b: other.node });
  }
  /** Smooth union: union with the seam rounded over width `k` (a bare number is a constant). */
  smoothUnion(other: Implicit, k: ExprLike): Implicit {
    return new Implicit({ kind: "smoothUnion", a: this.node, b: other.node, k: toExpr(k) });
  }

  // Transform-ops (on Implicit they compose into the domain warp; both are exact distance maps).
  /** Translate the field by `(x, y, z)`. */
  translate(x: ExprLike, y: ExprLike, z: ExprLike): Implicit {
    return new Implicit({ kind: "translate", t: [toExpr(x), toExpr(y), toExpr(z)], child: this.node });
  }
  /** Uniformly scale the field by `factor` (exact: distances scale with it). */
  scale(factor: ExprLike): Implicit {
    return new Implicit({ kind: "scale", factor: toExpr(factor), child: this.node });
  }

  /** Add `amp · fbm(freq · p)` noise to the field — a lumpy, organic displacement. A *display* op
   *  (breaks the exact distance property); the raymarch under-relaxes and animates it. */
  displace(amp: ExprLike, freq: ExprLike): Implicit {
    return new Implicit({ kind: "displace", child: this.node, amp: toExpr(amp), freq: toExpr(freq) });
  }

  /** The breeding surface: every `ParamSpec` gene carried by this tree's param literals, in
   *  canonical order. */
  specs(): ParamSpec[] {
    const out: ParamSpec[] = [];
    eachParam(this.node, (e) => collectSpecs(e, out));
    return out;
  }

  /** This tree's numeric params in the exact order the GPU kernel's `P` buffer reads them — so a
   *  same-structure tree renders through one pipeline with only a new `paramVector` uploaded. The
   *  CPU golden reads the same literals directly, so parity holds. */
  paramVector(): Float32Array {
    const out: number[] = [];
    eachParam(this.node, (e) => collectConsts(e, out));
    return Float32Array.from(out);
  }

  /** Signed distance at `p` — the golden. */
  eval(p: Vec3): number {
    return evalSdf(this.node, p);
  }

  /** Surface normal at `p` (field gradient). */
  normal(p: Vec3): Vec3 {
    return normalSdf(this.node, p);
  }

  /** Extract the surface as a Mesh on the CPU (naive surface nets) — the single async boundary is
   *  elsewhere; extraction itself is pure. GPU extraction (compaction pass) is designed-for. */
  toMesh(opts?: TessellateOptions): IsoMesh {
    return tessellateSdf(this.node, opts);
  }

  /** The `sdScene(p)` WGSL — the codegen'd distance function, the GPU-facing building block (a
   *  raymarch pass binds it; the parity harness samples it). Pure string assembly. Pass
   *  `{ bakeConstants: true }` for a self-contained `sdScene` (literals, no `P` buffer). */
  toWgsl(opts: { bakeConstants?: boolean } = {}): string {
    return sdSceneWgsl(this.node, opts);
  }
}

// ── Generator-ops ───────────────────────────────────────────────────────────────────────

/** A sphere of the given radius, centred at the origin. */
export function sphere(radius: ExprLike): Implicit {
  return new Implicit({ kind: "sphere", radius: toExpr(radius) });
}

/** An axis-aligned box with the given half-extents, centred at the origin. A single number is a
 *  cube. */
export function box(hx: ExprLike, hy?: ExprLike, hz?: ExprLike): Implicit {
  const x = toExpr(hx);
  return new Implicit({ kind: "box", half: [x, hy === undefined ? x : toExpr(hy), hz === undefined ? x : toExpr(hz)] });
}

/** A half-space bounded by the plane `dot(p, n̂) = d` (solid on the `< d` side). `n` need not be
 *  unit. The plane-based primitive for classic CSG solids. */
export function plane(n: Vec3, d: ExprLike = 0): Implicit {
  return new Implicit({ kind: "plane", n: [toExpr(n[0]), toExpr(n[1]), toExpr(n[2])], d: toExpr(d) });
}
