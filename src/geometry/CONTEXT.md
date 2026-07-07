# Context — Procedural geometry-ops

The ubiquitous language for the composable geometry-ops catalogue (ADR-0010): a horn-grammar-first
procedural-geometry system, re-derived from Stephen Todd's FormGrow, built to also host
CSG/implicit geometry. This glossary covers **only** that design; the datasource context has its
own glossary (see the root [`CONTEXT-MAP.md`](../../CONTEXT-MAP.md)).

## Language

### The value and its kinds

**Geometry**:
The lazy, typed value a geometry expression produces — an IR node, not samples. Built purely and
synchronously by the fluent chain; carries a Geometry-kind and an accumulated description.
Inspectable, serialisable, breedable, and lowerable; evaluated only when pulled.
_Avoid_: shape, mesh (a mesh is one *kind* / one materialised form, not the value), model.

**Geometry-kind**:
The tag on a Geometry node — `swept`, `implicit`, or `mesh` — that decides how it evaluates and
which ops apply. "Different geometry-types" in the design means different kinds coexisting in one
IR, bridged explicitly.
_Avoid_: type (overloaded), representation (fine informally).

**Swept**:
The kind for horn-lineage geometry: a Profile swept along the sweep coordinate under a
Transform-stack. Pointwise closed-form — `eval(s, θ) → (position, normal)`.
_Avoid_: generalized cylinder (fine informally), revolution (a swept form need not be a surface of
revolution).

**Implicit**:
The kind for signed-distance geometry: `p → distance`, with Boolean-ops as min/max combinators.
The natural home for CSG.
_Avoid_: SDF (fine informally for the field itself; the *kind* is Implicit), field (means a
datasource/op-graph field elsewhere).

**Profile**:
The cross-section of a Swept geometry — a function of **both** `(s, θ)`, so it may be anisotropic
and morph along the sweep (a superellipse today; ribbons/arrow-heads later). `superellipsoid.ts`
is the profile primitive; `sweptCreature.ts`'s taper functions are special-case profiles.
_Avoid_: cross-section (fine informally), outline, section.

### The ops

**Geometry-op**:
A node-constructor in the catalogue. Pure and synchronous: it takes a Geometry (and params) and
returns a new Geometry, building IR. Never does I/O; never awaits.
_Avoid_: operation (fine in prose), command, step.

**Generator-op**:
A Geometry-op that makes a Geometry from nothing — kind-specific. `horn()`/`superegg()` → Swept;
`sphere()`/`box()` → Implicit.
_Avoid_: primitive (means the spatial-analysis primitives elsewhere), source (datasource term).

**Transform-op**:
A Geometry-op that warps an existing Geometry — **kind-polymorphic**. `bend`/`twist`/`curl`/
`sweep`/`scale`/`translate`. On a Swept node it appends to the Transform-stack; on an Implicit
node it composes into the domain warp.
_Avoid_: modifier, deformer, filter.

**Boolean-op**:
`union` / `subtract` / `intersect` — min/max of signed distances. Defined **only on Implicit**;
booleaning a Swept geometry requires a Bridge first.
_Avoid_: CSG-op (CSG is the broader idea), merge, combine.

**Bridge**:
An **explicit** op that converts one Geometry-kind to another — `swept→mesh` (tessellate),
`implicit→mesh` (surface-net / marching), `swept→implicit`. Never an implicit coercion; kind
changes are always visible in the IR.
_Avoid_: convert, cast, adapter.

**Horn**:
The archetypal Swept Generator-op and the grammar's first citizen — a Profile swept under a
Transform-stack of `s`-parameterised rotations (bend/twist/curl/…). The re-derived essence of
Stephen Todd's FormGrow horn.
_Avoid_: tube, cylinder, form.

### The swept model

**Sweep coordinate** (`s`):
The along-horn parameter, `s ∈ [0, 1]` (base → tip). The first free variable of every Swept param
expression; the axis the Transform-stack and Profile vary over.
_Avoid_: t, u, length, rib index (a rib is a *sampled* `s`).

**Spoke coordinate** (`θ`):
The around-profile parameter, `θ ∈ [0, 2π)`. The second free variable of a Swept param expression;
what makes anisotropic profiles (ribbons, arrow-heads) expressible.
_Avoid_: v, phi, angle (reserve "angle" for a rotation *magnitude*), spoke index.

**Transform-stack**:
The ordered list of `s`(/`θ`)-parameterised, closed-form coordinate Transform-ops a Swept node
carries. Evaluating `(s, θ)` places the Profile point, then applies the stack in order. Closed-form
(no recurrence), so the CPU golden and the per-vertex TGSL kernel are the same function twice.
_Avoid_: transform chain, matrix stack (it is not restricted to affine matrices), pipeline.

**Framing strategy**:
The swappable policy that supplies a Swept surface's normal/orientation frame along `s`
(analytic-through-the-Jacobian / finite-difference / rotation-minimizing). Named because framing a
cross-section along a curved sweep has **no fully-general closed-form** solution — twisting normals
and degeneracies are expected, and the reference system mitigates by keeping pathological regions
off-camera. Authority: Stephen Todd.
_Avoid_: normal mode, frame (fine for the per-`s` frame *value* the strategy yields).

### Parameters and breeding

**Param expression**:
A Geometry-op parameter that is itself an expression over the free-variable environment — `{s, θ}`
now, `{globals, sim-channels, data}` designed-for. Pure: the `s`/`θ`-dependence is always explicit.
A bare scalar is a **constant**; any progression is a named expression.
_Avoid_: dynamic param, curve, animation.

**Ramp**:
A named Param-expression constructor for a progression along `s` — `ramp(360)`, `linear(0, 360)`,
`ease(…)`. Where the ergonomics of "twist harder toward the tip" live — not baked into the op.
_Avoid_: gradient, envelope, sweep (that is the coordinate).

**Angle**:
The typed rotation-magnitude value, with explicit constructors `deg` / `rad` / `turns`,
canonicalised to radians internally. A catalogue-level default unit (passed in at construction)
governs bare numbers — never a hidden global.
_Avoid_: degrees/radians as bare numbers, rotation.

**ParamSpec** (a.k.a. **gene**):
The breeding metadata a Param-expression's numeric literals carry — `[name, default, min, max,
step, …]`. The Mutator (`src/evo`) breeds these. Called a *gene* in the FormGrow lineage; ParamSpec
is the type here.
_Avoid_: knob, uniform (that is one lowered form of a param), hyperparameter.

### Evaluation and output

**Lower**:
Compile a Geometry IR to a runnable form for a backend — a **CPU golden** `(s, θ)` loop and a
`"use gpu"` **TGSL** per-vertex kernel (ADR-0003). Never TSL.
_Avoid_: compile (fine in prose), codegen, emit.

**Pull**:
Evaluate a Geometry through the op-graph executor — the single async boundary (`await`). Produces
one **materialisation form**: the **lazy** value (unevaluated), a resident **GPU buffer** (stays
on-device), or a **plain typed array** (positions/normals/indices with an explicit
shape/element/basis schema). Rendering uses the on-device forms only — a Mesh-kind Geometry pulled
to a GPU buffer, or the lazy value lowered straight into a render op — **never** a round-trip
through a CPU mesh. The typed array is a portability / interop escape hatch, off the render path.
_Avoid_: evaluate/run (fine in prose), execute, render (render is a downstream consumer).

**Catalogue**:
The registered set of Geometry-ops, constructed with its config (e.g. default Angle unit). The
surface a user composes from; reuses the `src/gpu/graph` registry.
_Avoid_: library, registry (that is the shared op-graph mechanism), toolbox (means the spatial
primitives elsewhere).
