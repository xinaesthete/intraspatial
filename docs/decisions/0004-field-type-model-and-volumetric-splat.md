# ADR-0004 — Field type model (element algebra ⊥ tensor axes ⊥ domain) and volumetric splat

Status: **proposed** (2026-06-29)

## Decision

Replace the implicit "scalar field on a 2D grid" model in the op-graph runtime with
a field described by **three orthogonal facets**, and make the **element type** a
closed set of small algebras distinct from **open tensor axes**:

```ts
Field = { domain; element; axes?; dtype }

Domain =
  | { kind: "grid"; size: [w, h] | [w, h, d] }   // 2D or 3D, one shape kind
  | { kind: "points"; n; dim: 2 | 3 }
  | { kind: "matrix"; rows; cols }
  | { kind: "scalar" }
  | { kind: "opaque"; name }

ElementType =                                     // CLOSED set; each carries an algebra
  | { kind: "scalar" }
  | { kind: "complex" }                           // 2 lanes · mul / conj / exp / FFT
  | { kind: "quaternion" }                        // 4 lanes · Hamilton product, normalize, slerp
  | { kind: "vec"; n: 2 | 3 | 4 }                 // dot / cross / normalize
  | { kind: "mat"; n: 2 | 3 }                     // later: structure / diffusion tensors

TensorAxis = { name: string; length: number }     // OPEN, runtime length; NO per-element algebra
```

- **Element algebra and tensor axes are different kinds, deliberately.** The element
  type is a closed, compile-time-known small algebra with typed arithmetic
  (`complexMul`, `conjugate`, `quatMul`, `slerp`, `dot`, `cross`). A tensor axis is
  an open, runtime-length bulk dimension (e.g. genes) with only *axis-parametric*
  ops (`reduceAxis`, `contract`, `selectAxis(g)`, `softmax`) and **no** intrinsic
  arithmetic. The op registry type-checks against `element`/`axes` at graph-build
  time, so `complexMul` is rejected on a gene field and `reduceAxis("gene", …)` is
  rejected on a complex field. The two op families never bleed together.
- **Reaction–diffusion `(u, v)` becomes one `complex` grid behind a single feedback
  node**, removing the two-feedback-node workaround. `u`/`v` are genuinely one
  signal with two components under a channel-wise linear update; complex is the
  honest representation and unlocks complex arithmetic (FFT-domain diffusion later).
- **Gene-expression density is `element: scalar, axes: [{ name: "gene", length: G }]`**
  — explicitly *not* a wide `vecN`. We do **not** target huge G: typical MDV use is
  ~10 genes viewed interactively. The axis exists so the same op *graph* is correct
  at any G, not so we vectorise hundreds of lanes.
- **Scale equivariance is a design goal, first-class.** The op set is authored so the
  *same graph* is the unit of work at every scale: a handful of genes / a small
  volume runs interactively in-browser on WebGPU; the identical operations, unchanged,
  run as a batch/cluster job at large scale. Ops must therefore stay free of
  interactive-only assumptions (no "small enough to materialise" baked in) and the
  lazy-pull executor stays the seam where a small pull and a large fused/streamed
  pull are the same description with a different backend.

### Volumetric splat (points → voxel grid)

The 2D additive-render splat ([`splatDensity`](../../src/gpu/spatial/splatDensity.ts))
does **not** generalise to 3D — there is no render-blend into a 3D texture. The
element-vs-axis split selects the strategy:

- **Fixed small element (scalar / complex / vec / quaternion) → slice-stack additive
  render.** A separable kernel factors: isotropic 3D Gaussian splat = the existing 2D
  Gaussian splat × a 1D z-profile. Loop the z-support; one additive instanced draw of
  all points per z-slice (2D view into the 3D/array texture), weight =
  `inplane_gaussian × exp(-z²/2σ²)`. Cost ~`z_support` draws total, full f32
  precision, no atomics. Reuses the existing render primitive almost verbatim.
- **Tensor axis (gene volume) → gather off a 3D uniform-grid index.** Bin points into
  a 3D cell grid (scan + counting-sort — the spatial index already on the roadmap),
  then one thread per voxel pulls points from neighbouring cells and accumulates the
  **whole G-vector in registers / workgroup memory, written once, coalesced.** No
  atomics, deterministic, exact for *any* radial kernel. This is the decisive reason
  gather beats scatter here: scatter needs an atomic *per channel per voxel*, so a
  G-gene volume is G atomics per footprint-voxel — a non-starter.
- **Scatter + fixed-point atomics** is kept only as a niche fallback for *scalar*
  splat where points ≫ voxels and the volume is dense (the "no f32 atomics →
  fixed-point accumulate" constraint from the spatial front).

The **3D uniform-grid spatial index is the load-bearing primitive** under both the
gather splat and the discrete-cell front; build it first.

### Dense vs lazy volumes

A dense gene volume need not be materialised, and at large scale must not be
(128³ × 200 genes × f32 ≈ 1.7 GB). The gather/index path yields the *occupied*
voxels naturally; combined with the lazy-pull executor, a "gene-density volume" is
preferably a **lazy field that is only ever sampled / fused into its consumer**, not
a stored dense grid. This is the same seam that makes scale equivariance work: small
interactive pulls materialise; large pulls fuse and stream.

## Context & provenance

The user's direction over three exchanges:

1. *"represent things like `u, v` … as a single complex signal (so we shouldn't need
   all the fiddling around with two feedback nodes)"* — and gene-expression-density
   tensors *"where the number of genes varies at runtime"*, plus *"volumetric
   domains, which implies our strategy for `KDESplat` might not apply."*
2. *"We should have different models in the type system for things like `vec2` `vec4`
   vs N-tensor. We should support appropriate complex arithmetic operations on more
   specific 2d/3d/quaternion fields and treat gene tensors as something different."*
   → element algebra is a closed set distinct from tensor axes; this is the crux.
   Also: *"I'm not so much concerned about 'multi-channel splat' … as 'splat into a
   volume'"* (the former is trivial; the volume is the real problem).
3. *"I don't expect this to necessarily scale to huge numbers of genes, frequently in
   MDV we may look at ~10 at a time… although it's appealing to represent a set of
   operations that at small scales are suitable for interactive exploration but at
   larger scales can be run equivalently as a cluster job."* → bounds G (no wide-lane
   ambition) and elevates scale equivariance to a design goal.

Grounding in the current runtime (mapped this session):
- [`Shape`](../../src/gpu/graph/handle.ts) is a discriminated union with **no channel
  or component concept**; `grid` is `{width, height}`, `Dtype` is `f32|i32|u32`,
  `FieldValue.data` is one flat typed array. "Scalar field on a 2D grid" is baked in
  by *omission*, so the change is additive metadata, not a type-system rewrite.
- [`feedback()`](../../src/gpu/graph/graph.ts) already wraps an arbitrary `GpuField`
  of any shape/dtype; the two-node RD pattern is forced only by
  [`reactionDiffusionStepOp`](../../src/gpu/graph/ops/reactionDiffusion.ts) declaring
  separate `u`/`v` ports, not by the delay machinery.
- 2D is assumed only at the centralised `gridShape()` chokepoints
  (`convolveSeparable.ts`, `reactionDiffusion.ts`, `splatDensity.ts`), which keeps the
  3D blast radius small.

Caveat to verify before building: render-to-3D-slice in WebGPU goes through a per-slice
2D render-pass attachment (`depthSlice` / array-layer view); confirm that path is stable
on Dawn-on-Node (see ADR-0002 / ADR-0003 teardown notes) before relying on slice-stack.
Slice-stack is exact only for **separable** kernels — a radial Epanechnikov (compact
support, the spatial front's preferred kernel) is not separable, so gather is the
general-correctness path and slice-stack is the Gaussian fast path.

## Consequences

- [`Shape`](../../src/gpu/graph/handle.ts) becomes `Domain` (grow `grid` to a 2-or-3
  tuple, give `points` a `dim`); `element` (default `scalar`) and `axes` (default
  none) are new, so **every existing op is untouched**. `FieldValue.data` stays a flat
  typed array; strides derive from `domain × element-lanes × axes`.
- Op definitions gain element/axis preconditions checked at build time; a new
  algebraic op family (complex/quaternion/vec) and a new axis-parametric family
  (reduce/contract/select over named axes) join the registry.
- RD collapses to a single `complex` feedback node as a free side effect (good first
  pilot: exercises the whole element path end to end through shape, kernel
  lane-indexing, feedback, and composer).
- New primitive to build first: the **3D uniform-grid spatial index**
  (scan + counting-sort), which then carries both the gather splat and the
  discrete-cell front.
- Scale equivariance constrains future op authoring: no interactive-only assumptions;
  the lazy executor is the small↔large seam; large gene volumes are fused/lazy, not
  dense.
- Revisit if a use case actually needs large G (then reconsider planar `vec4`-tile
  packing for vectorised lanes), or if WebGPU render-to-3D-slice proves Dawn-unstable
  (then gather becomes the only volumetric path).

## Implementation status

- **Element-type model: landed (2026-06-29).** `ElementType` (scalar/complex/vec/quaternion)
  + `elementLanes`/`elementsEqual`/`elementLabel`/`elementOf` added to
  [`handle.ts`](../../src/gpu/graph/handle.ts) as an optional facet on `GpuField`/`FieldValue`
  (absent ⇒ scalar, so every existing op is untouched). `OpType.inferElements` (optional,
  parallel to `inferShapes`) threads through the builder ([`graph.ts`](../../src/gpu/graph/graph.ts));
  rejection of a wrong element happens there, at graph-build time. Pure algebra in
  [`elementMath.ts`](../../src/gpu/graph/elementMath.ts) (CPU reference the WGSL will mirror).
  Ops: `complex`/`realPart`/`imagPart`/`conjugate`/`magnitude`
  ([`complexOps.ts`](../../src/gpu/graph/ops/complexOps.ts)) and
  `add`/`sub`/`mul`/`scale`/`dot`/`cross`/`normalize`
  ([`fieldArithmetic.ts`](../../src/gpu/graph/ops/fieldArithmetic.ts)) — CPU Tier-1.
  Validated by [`element.test.ts`](../../src/gpu/graph/element.test.ts) (CPU, 9 tests).
- **RD collapse to one `complex` node: landed.** [`reactionDiffusionComplex.ts`](../../src/gpu/graph/ops/reactionDiffusionComplex.ts)
  carries the Gray–Scott `(U,V)` state as a single `complex` field (re=U, im=V) behind one
  feedback node, the identical kernel underneath. [`reactionDiffusionComplex.test.ts`](../../src/gpu/graph/reactionDiffusionComplex.test.ts)
  proves it matches the legacy two-feedback-node loop bit-for-bit (CPU, 3 tests) and
  [`element.gpu.test.ts`](../../src/gpu/graph/element.gpu.test.ts) runs it on Dawn.
- **GPU complex-multiply kernel: exists and validated, not yet wired into the op.**
  [`complexMulGpu.ts`](../../src/gpu/graph/ops/complexMulGpu.ts) is a TGSL `"use gpu"`
  compute pass (ADR-0003 layout-bound pipeline) validated directly against the CPU
  reference in [`element.gpu.test.ts`](../../src/gpu/graph/element.gpu.test.ts) (real Dawn,
  ≤1e-5). `mulFields` stays CPU Tier-1 for now — see the registration note below.
- **Element ops are an OPT-IN pack (`registerElementOps()`), not eagerly registered.**
  Finding worth recording: eagerly importing the element op modules (and especially a second
  `"use gpu"` TGSL kernel) into the always-loaded op registry added enough module-graph
  weight to tip **Dawn-on-Node's collection/teardown** over the edge in *unrelated* GPU test
  forks — the splat-**render** `graph_pipeline` fork segfaulted at collection (`tests 0ms`),
  stochastically (5–6/6). Bisected by reverting subsets; baseline 0/6, full 6/6. Fix:
  [`registerElementOps()`](../../src/gpu/graph/ops/index.ts) loads the element modules via
  **dynamic `import()`**, so a fork that never asks for them never pays the cost; the base
  RD op module is kept byte-identical to baseline (complex variant lives in its own file).
  This is the same Dawn fragility ADR-0002/0003 already isolate per-file; the lesson is that
  even pure-JS module-graph growth on the *eager* registry path can tip the render fork.
  Consequence/TODO: the playground composer must call `registerElementOps()` to surface the
  new ops; wiring `complexMulGpu` back into `mulFields` waits on understanding this interaction.
- **Still proposed:** the tensor `axes` facet, the 3D `Domain`, volumetric splat, and
  dense-vs-lazy. The element work was scoped first deliberately (smallest blast radius,
  everything builds on it).
