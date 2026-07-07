# ADR-0010 — Procedural geometry as composable geometry-ops

Status: **proposed / exploratory** (2026-07-07)

## Context

We want to bring the **procedural geometry** of Stephen Todd's *FormGrow* / horn grammar
(prior art in `~/code/organicart`, chiefly `TS/horn.ts`) into this repo — **not** as a code
lift or a wrapper, but as a clean re-derivation in this repo's idiom. The goal is a
**standalone, composable geometry-ops catalogue**: horn-grammar-first, but built to also host
CSG / implicit-surface / boolean geometry as other *geometry-types*. The dancer (DANCERL) may
consume it; a future CSynth/FoldSynth is another consumer (which will later want to reference
the Java UI, and other codebases for CSG). It does not need to be comprehensive now — the point
is to get the architecture right so adding ops later is straightforward.

**Provenance / credit.** FormGrow and the horn grammar come from the long-running **Todd–Latham**
collaboration, and the credit here leads with **Stephen Todd**: he wrote the bulk of the code and
drove much of the creative direction of the horn system (Peter Todd contributed some parts). We
name the lineage *Todd–Latham*, leading with Todd — not "Latham FormGrow".

The organicart horn grammar is already the target ergonomic shape:

```
horn(name).ribs(10).radius(50).stack(1000).bend(0).curl(0)
          .twist(360, {k}).sweep(4000).branch(...).scale(...).spiral(...)
```

Each verb (`$bend`/`$twist`/`$curl`/`$scale`/`$sweep`/…) is implemented by one `processTran(…)`
that does two things at once: it **appends a coordinate transform** — a rotation in a chosen
plane (`twr(x,y,…)` for bend, `twr(y,z,…)` for curl, `twr(x,z,…)` for twist) whose angle is a
function of the position along the horn — and it **declares a gene** (`[name, default, min, max,
step, …]`, i.e. a `ParamSpec`) so the Mutator can breed that knob. A horn is a **swept
generalized cylinder**: a profile cross-section swept along a sweep parameter, under an
accumulated stack of transforms, compiled (in organicart) to a per-vertex shader.

This repo already has the substrate to host that cleanly: the demand-pulled **op-graph**
(`src/gpu/graph`, ADR-0003 `"use gpu"` TGSL kernels, memoised executor, backend seam), the
**expression-IR ⇄ DSL ⇄ graph** duality (ADR-0007), the CPU-golden + GPU-kernel discipline that
keeps `forces.ts`/`dancerGpu` honest, the Mutator/`ParamSpec` breeding surface (`src/evo`), and
existing geometry goldens `src/geometry/superellipsoid.ts` (the "superegg" profile) and
`src/geometry/sweptCreature.ts` (a bent-axis surface of revolution — already a special-case
horn).

## Decision

Build a **geometry-ops catalogue** as a lazy, typed **geometry expression-IR** that reuses/extends
the ADR-0007 substrate and lowers to **CPU golden + TGSL (TypeGPU)** — never TSL. The fluent
chain is sugar that *builds* this IR; nothing evaluates until pulled.

1. **Concept re-derivation, not a lift.** `horn.ts` is a *specification/reference*, never copied
   or wrapped. A mechanical port would drag in genes/`COL`/three.js/DOM and poison the clean core.

2. **The chain builds a lazy, typed geometry expression-IR.** A horn transform is fundamentally a
   *per-point* coordinate → (position, normal) map — Level-2 expression-IR (ADR-0007), assembled
   into a generator that surfaces as a Level-1 node in `src/gpu/graph`. Reuse the ADR-0007 IR
   structure (typed DAG, `ParamSpec`-carrying literals). Op parameters can themselves be
   expressions (`twist(ramp(360))`), in the core from day one.

3. **GPU backend is TGSL / TypeGPU, not TSL.** Two lowering backends: **CPU JS** (golden
   reference) and a `"use gpu"` **TGSL** kernel (ADR-0003). This aligns the geometry IR with the
   `src/` core (~all TGSL) and ADR-0009's demotion of three.js/TSL. See the **amendment to
   ADR-0007** (its backend named "TSL (GPU)"; corrected to TGSL).

4. **Minimal core first.** In the core now: **expression-valued params**. Deferred (designed-for,
   not built): the two ADR-0007 *surfaces* — the playground **graph ⇄ IR round-trip** (nearer
   term; the IR is a graph and the playground already edits graphs) and the **DSL-text grammar**
   (furthest / still speculative — nobody has designed that grammar concretely). Sequence: core IR
   → graph round-trip → DSL-text (only if it earns its place).

5. **Build synchronously, evaluate once.** The fluent chain is **pure and synchronous** and
   contains **no promises** — building an IR needs no I/O. `await` appears exactly once, at the
   terminal evaluation boundary (`await g.toMesh(device)` / `pull`). This deliberately diverges
   from the eager `*Gpu` primitives (`splatDensityGpu`'s await-per-step materialises to CPU and
   round-trips every step — the anti-pattern the op-graph exists to remove). A built chain is a
   first-class value: inspectable, serialisable, breedable, round-trippable.

6. **Geometry is tagged multi-representation.** A `Geometry` node carries a `kind ∈ {swept,
   implicit, mesh}`:
   - **transform-ops** (`bend`/`twist`/`scale`/`translate`) are **kind-polymorphic** — on a
     `swept` node they append to the along-`s` transform stack; on an `implicit` node they compose
     into the domain warp;
   - **generator-ops** are kind-specific (`horn`/`superegg` → `swept`; `sphere`/`box` → `implicit`);
   - **boolean-ops** (`union`/`subtract`/`intersect`) live on `implicit` (min/max of SDFs); to
     boolean a horn you first apply an explicit **bridge** `swept → implicit`;
   - **bridges are explicit ops**, never implicit coercions (`swept→mesh` tessellate,
     `implicit→mesh` surface-net/marching, `swept→implicit`). This is what "different
     geometry-types" means concretely.

7. **The swept-kind is pointwise closed-form.** `eval(s ∈ [0,1], θ ∈ [0,2π)) → (position,
   normal)`, built from:
   - a **profile** — a cross-section that is a function of `(s, θ)` (superellipse today; radius /
     exponents / anisotropy vary along `s`). `superellipsoid.ts` becomes the profile primitive;
     `sweptCreature.ts`'s `noseRadial`/`bodyTaper` become special-case profiles the grammar
     subsumes. A general `(s, θ)` profile is what makes protein-cartoon cases (flat ribbons,
     arrow-heads that flare then collapse) expressible.
   - an ordered **transform stack** — `s`(/`θ`)-parameterised, closed-form coordinate transforms
     (`bend`/`twist`/`curl`/`sweep`/`tilt`/`flap`/`scale`), composed in order.
   Because it is closed-form and pointwise: **CPU golden = a nested `(s, θ)` loop; GPU = one TGSL
   invocation per vertex; they agree by construction.**

8. **Param expressions live over `{s, θ}` now, `{globals, sim-channels, data}` next.** The
   free-variable environment of a param expression is `{s, θ}` in the core (option 2), structured
   so simulation channels / global inputs / data fields (option 3, the fuller ADR-0007/0004
   vision) slot in **additively** and within close reach. Params are **pure expressions** — the
   `s`/`θ`-dependence is always explicit; a bare scalar is a **constant**, and any progression is
   a **named expression** (`ramp(360)`, `linear(0,360)`, `ease(…)`). No hidden op-baked ramp; the
   ergonomics live in a small library of ramp/profile constructors, not in the op.

9. **Angles are a typed value, not an ambient mode.** A lightweight `Angle` with explicit
   constructors (`deg`, `rad`, `turns`), canonicalised to radians internally, plus conversions.
   Bare-number ergonomics take a **catalogue-level default unit** passed in explicitly at
   construction — never a hidden global that makes the same expression mean two things.

### Scope line (first slice)

**In:** `horn()` → one `swept` node; a `superellipse` profile with `radius` an `{s}`-expression;
three transforms — `bend`, `twist`, `scale` — as pure `s`(/`θ`)-expression transforms carrying
`ParamSpec`; lazy sync build → `await g.toMesh(device)` lowering to **both** a CPU golden `(s,θ)`
loop **and** a `"use gpu"` TGSL per-vertex kernel, with a golden **parity test** (ADR-0003
pattern).

**Out (designed-for, not built):** CSG / implicit / boolean; the **structural / instancing /
recursion** ops (`branch`/`spoke`/`spiral`/`stack`/`sub` — the recursive sub-horn tree); the
DSL-text grammar; the playground graph round-trip; a Profile sub-grammar (`.ribbon().arrowhead()`).
The IR must let a `swept` node later sit as a *leaf under a structural node*.

## Why

- **One compute abstraction.** Geometry joins the existing op-graph / expression-IR rather than
  becoming a third parallel system; it inherits device, memo, backends, TGSL, and breeding.
- **Fidelity to the grammar without the entanglement.** The swept transform-stack *is* the horn
  grammar; re-deriving it as pointwise closed-form keeps the character while shedding
  genes/`COL`/three.js/DOM coupling.
- **Testable by construction.** Pointwise closed-form means the CPU golden and the per-vertex TGSL
  kernel are the same function twice — the discipline the repo already trusts.
- **Room for CSG.** Tagged kinds + explicit bridges host implicit/boolean geometry natively
  without forcing the horn into an SDF (which would throw away its rib/profile/parametric
  structure) or into meshes (fragile booleans, no GPU lowering).

## Consequences / open questions

- **Normal framing along `s` is an unsolved sub-problem.** Framing a cross-section along a curved
  sweep has no fully-general closed-form normal — you get twisting/degeneracies. There is no
  claimed cure here: the reference system mitigates by **steering pathological regions out of
  camera view**. The swept-kind therefore gets an explicit, **swappable framing strategy**
  (analytic profile-normal-through-the-transform-Jacobian / finite-difference in `(s,θ)` /
  rotation-minimizing frame). **Authority to consult: Stephen Todd.**
- **Closed-form vs `s`-integration.** We assert closed-form-in-`s` (no recurrence); the `twr`
  primitive looks purely closed-form. If some horn behaviour genuinely needs a *marching frame*
  (step `n` depends on step `n−1`), it becomes an explicit `integrate` op rather than bending the
  pointwise contract. Some pathological cases are expected. Confirm with Stephen Todd.
- **Materialisation forms — rendering stays entirely on-GPU.** A pulled geometry resolves to one
  of three forms — the **lazy** IR / handle, a resident **GPU buffer** (stays on-device), or a
  **plain typed array** (positions/normals/indices with an explicit shape/element/basis schema).
  **Rendering never requires the CPU typed-array form, and never a round-trip through a CPU mesh.**
  A geometry's GPU-resident representation is consumed on-device: either as a resident buffer handed
  straight to a raster/raymarch pass, or — per ADR-0009 (rendering-as-ops) — consumed *within the
  op-graph itself*, so a whole `generate → transform → render` chain runs GPU-resident with **no
  standalone download at any step**. The plain typed array is a **portability / interop escape
  hatch only** (FAIR export, deck.gl / SpatialData.js handoff), off the render path. `toMesh` yields
  GPU buffers by default; a CPU array is an explicit opt-in, never implied by drawing. The
  `composable-interpretable` doc is revised to teach this build-lazy / pull-once model (and that
  the typed-array form is interop-only) instead of the eager await-per-step read.
  **Current substrate (honest state):** this GPU-resident render path is the *target* this design
  commits to, **not a capability available today**. The op-graph is **Tier-1** now (`src/gpu/graph/
  handle.ts`) — every op is GPU-native, but intermediate `FieldValue`s round-trip to the CPU
  between ops — and **no render op exists yet** (ADR-0009 is exploratory). No-download rendering
  depends on **Tier-2 resident-buffer edges** plus a render op landing first; until then a geometry
  pulled for display still materialises through a buffer/array. We build toward the target and do
  not claim it prematurely.
- **Profile wants its own sub-grammar eventually.** The arrow-head/cartoon need signals that
  profiles (`circle`/`superellipse`/`ribbon`/`arrow`, morphing along `s`) will grow a small
  composable grammar. Slice one treats the profile as a `(s,θ)` expression; the sub-grammar is
  designed-for.
- **Module home.** Extend `src/geometry/` and reuse the `src/gpu/graph` registry rather than
  starting a new top-level tree.

## References

- ADR-0003 (`"use gpu"` TGSL kernels), ADR-0004 (field/element type model), ADR-0007
  (expression-IR ⇄ DSL ⇄ graph duality — **amended here**: backend TSL → TGSL), ADR-0009
  (rendering as ops; three.js/TSL demoted to presentation).
- `src/gpu/graph` (executor, memo, registry, backend seam), `src/geometry/{superellipsoid,
  sweptCreature}.ts` (profile primitives the grammar subsumes), `src/evo` (Mutator / `ParamSpec`).
- Prior art: Stephen Todd's FormGrow / horn grammar — `~/code/organicart` `TS/horn.ts`
  (`processTran`, `twr`, the `$bend/$twist/$curl/$scale/$sweep` verbs).
- `src/geometry/CONTEXT.md` (the geometry ubiquitous language) and root `CONTEXT-MAP.md`.
