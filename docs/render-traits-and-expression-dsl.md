# Render traits & the expression IR (DSL ⇄ graph ⇄ compiled)

Status: **planning** (2026-07-03). Design doc; not yet implemented. Companion ADR:
[`decisions/0007-expression-ir-dsl-graph-duality.md`](decisions/0007-expression-ir-dsl-graph-duality.md).

## What this is for

Today the dancer's look is hard-coded: cones whose per-instance colour is a CPU-computed
`THREE.Color` uploaded every frame, one fixed material, no per-instance shape variation. We want the
**visual traits** — mesh geometry and PBR material — to be **mappable from simulation state** (a
dancer's speed, spin, the forces on it, its figure, its age…) or from **somewhat arbitrary
expressions** over those channels, both **globally** (constant traits) and **per instance**.

Concretely the near-term wants are:

- **Procedural superegg geometry** — an instanced [superellipsoid](https://en.wikipedia.org/wiki/Superellipsoid)
  (Piet Hein's superegg; the Todd–Latham *Form Synth* lineage) whose exponents / axis scales /
  stretch change per instance from `(angular) velocity`, forces, etc.
- **Procedural texturing → PBR** — noise / pattern fields mapped to `baseColor`, `roughness`,
  `metalness`, `emissive`, clearcoat…
- **okLab / okLCH colour** — perceptually-uniform colour so scalar fields map to colour that reads
  correctly and interpolates without muddy midpoints. Useful far beyond the dancer.

The **long-term** want (this doc's real subject) is that the mapping mechanism is not one-off shader
code but a small **expression language** that is **bidirectional with the playground's visual
dataflow graph** — edit a mapping as text *or* as nodes, and drop a **DSL node inside the graph** as
a textual escape hatch — compiling to **both TSL (GPU) and CPU JS**, and breedable by the existing
Mutator.

## The core idea: one IR, three surfaces, N backends

The mistake would be to treat "a DSL", "the node graph", and "the shader" as three separate things to
keep in sync. They are three *views* of **one canonical typed expression IR**:

```
            parse ▲ print                 layout ▲ extract
   DSL text ──────┼──────►  Expression IR  ◄──────┼────── playground graph (nodes/edges)
                  │        (typed AST/DAG)         │
                  └──────────────┬─────────────────┘
                                 │ emit
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
                  TSL         CPU JS      (future: WGSL,
                (GPU node)  (golden ref)   serialised form)
```

- **The IR is the source of truth.** A typed DAG of nodes: channel references, literals/uniforms,
  operators, function calls, colour-space conversions, swizzles. Deterministic, hashable, serialisable.
- **DSL text ⇄ IR** is `parse` / `print`. Round-tripping text→IR→text is normalising, not lossy.
- **Graph ⇄ IR** is `extract` / `layout`. A subgraph of the playground *is* an IR DAG with 2-D
  positions; a **DSL node** is a graph node whose body is DSL text that parses to a sub-IR compiled
  inline. "Expand DSL node to nodes" = `print`-to-graph; "collapse nodes to a DSL node" = `extract`
  then `print`-to-text.
- **IR → TSL** and **IR → CPU JS** are two backends emitting from the same DAG — the same
  dual-target discipline `src/gpu/sim/forces.ts` already follows by hand (CPU golden + TSL kernel),
  but generated from one description instead of written twice.

**Precedent.** This is Houdini's **VEX ⇄ VOP** duality (write a shader as code or as a node network,
convert either way), and the model behind **MaterialX** and Blender's shader nodes. We are not
inventing the pattern; we are applying it to *sim-state → render-trait* mappings and grounding it in
this repo's existing typed-field IR (see ADR-0004) and Mutator.

## Layers

### 1. Channels — the inputs

The sim is GPU-resident: `pos, vel, accel, angPos, angVel` live in storage buffers on the device the
renderer draws from. **Channels** expose the ones the render wants as per-instance storage buffers
read in TSL by `instanceIndex` — the same GPU-buffer bridge already used for `instanceMatrix` and the
trail history ring (no readback).

A **channel registry** names them and records type + how to obtain them:

| Channel | Type | Source |
|---|---|---|
| `position` | vec3 | `pos` buffer |
| `velocity` | vec3 | `vel` buffer |
| `speed` | f32 | `\|velocity\|` (derived) |
| `acceleration` | vec3 | `accel` buffer |
| `forceMag` | f32 | `\|accel\|` proxy, or a dedicated force accumulator |
| `spin` | f32 | `\|angVel\|` |
| `angVel` | vec3 | `angVel` buffer |
| `figure` | u32 | caller figure code (uniform) |
| `age` | f32 | frames since spawn / since last figure onset |
| `radius` | f32 | `\|position\|` |
| `neighbours` | f32 | count within a radius (already computed in the step kernel) |

Derived channels are themselves trivial IR expressions (`speed = length(velocity)`), so the registry
is just a set of *named root expressions* — no special-casing.

### 2. Traits — the mappings

A **trait** binds an IR expression (of the channels) to a **render target**. Targets are a fixed,
typed set; the *expression* feeding each is arbitrary.

- **Geometry targets** (consumed in the vertex stage): superellipsoid `e1`, `e2`; per-axis scale
  `sx, sy, sz`; velocity-aligned stretch; angular twist.
- **Material targets** (fragment stage, PBR): `baseColor` (colour-typed), `roughness`, `metalness`,
  `emissive` (colour), `emissiveIntensity`, `clearcoat`, `clearcoatRoughness`, plus procedural
  **texture fields** (noise/patterns in object/world space or driven by channels).

**Global vs per-instance** is not a distinction in the model — a global trait is simply an expression
with no channel dependence (a constant / uniform). The compiler hoists channel-free sub-expressions
to uniforms automatically.

**Breeding.** Numeric literals in the IR carry `ParamSpec` metadata (range, scale, label), so the
Mutator breeds the *constants* immediately (its existing `ParamSpec[]` surface). Because the IR is
serialisable, breeding the *structure* (crossover/mutation over the DAG — grow/prune nodes) is a
later, natural extension, and the honest end of the Todd–Latham "mutate the form, not just the
dials" lineage.

### 3. Colour algebra — okLab / okLCH

Colour-typed values are tagged with their **space** (`srgb`, `linear`, `oklab`, `oklch`). Conversions
are IR nodes; the compiler inserts them so that, e.g., a `mix` between two colours happens in okLab
(perceptual) even if the endpoints were authored in sRGB, and the final material output is converted
back to the renderer's working space. This is the first concrete piece (see below) and is independent
of the DSL — it lands as plain CPU + TSL functions with a golden test, then becomes IR nodes.

## Type system

Expressions are typed: `f32` (and eventually `f16`, per the numerical-precision thread), `vec2/3/4`,
`bool`, `u32`, and `colour<space>`. Types drive:

- **checking** — a scalar can't feed a vec3 target without an explicit broadcast; colour spaces are
  tracked so conversions are inserted, not guessed;
- **backend emission** — the same node emits `mix()` in TSL and a CPU `mix` helper;
- **the graph UI** — port colours/shapes by type, connection validity.

This reuses, rather than reinvents, the repo's field/element type model (ADR-0004): a render
expression is a field computation whose element types include colour.

## Compilation & correctness

- **IR → TSL**: emit a TSL node graph (`Fn`, channel storage `.element(instanceIndex)`, `mix`,
  `pow`, `mx_noise_*`, the okLab helpers). Fed into `material.positionNode` / `colorNode` /
  `roughnessNode` / etc. Zero readback — consistent with the trails/matrix work.
- **IR → CPU JS**: emit a plain function over `Float32Array` channels — the **golden reference**, and
  the path for CPU-side consumers (the distance-matrix heatmap palette, breeding-strip thumbnails).
- **Cross-check**: the same repo discipline as `forces.ts` / `dancerGpu` — a golden test asserts
  CPU and GPU evaluate a set of expressions to the same values within tolerance. Keeping the two
  backends honest is *itself* a demonstration of the tool's value for maths/data-science (the
  numerical-accuracy identity this project deliberately foregrounds).

## Near-term, incremental adoption (the "less generic" changes that anticipate this)

We build the demo's look *now*, hand-authored in TSL, but structured so each piece is obviously the
manual version of an IR expression — so adopting the IR later is refactoring, not rewriting.

1. **okLab / okLCH module** (first — chosen). `src/color/oklab.ts`: sRGB ⇄ linear ⇄ okLab ⇄ okLCH,
   pure + golden-tested against Ottosson's reference values; a matching TSL module. Immediately
   upgrade the trail colour, the instance colour, and the matrix heatmap to perceptual ramps.
2. **Channel bridge**. Expose `vel` / `accel` / `angVel` (and derived `speed`/`spin`/`forceMag`) as
   render-readable storage buffers, mirroring `setMatrixTarget` / `setTrailTarget`. Move instance
   **colour** off the per-frame CPU `setColorAt` upload (the last stale-readback path) into a TSL
   `colorNode` reading the bridge — an okLCH ramp on `speed`/`figure`.
3. **Superegg geometry**. Instanced superellipsoid base mesh; TSL `positionNode`/`normalNode`
   (analytic superquadric normals) with per-instance `e1,e2`/scale/stretch read from the bridge.
   Hand-authored mapping today = one IR expression tomorrow.
4. **Procedural PBR**. `mx_noise_*`-driven `roughness`/`emissive` fields, colour via okLab.
5. **Extract the registry + trait table** from steps 2–4 into the channel registry and a typed
   trait→target table — the seam the IR/DSL plugs into.

Each of 1–4 names its channels and targets explicitly, so step 5 is a lift into data, and the DSL/IR
(this doc's subject) then generates what we will have hand-written.

## Open questions

- **IR granularity vs the existing op-graph.** The playground graph is coarse (ops = fields). The
  expression IR is fine (per-scalar math). Are these one graph at two zoom levels, or two graphs with
  a boundary (an "expression op" whose interior is the IR)? Leaning: the latter first (an expression
  node type whose body is IR/DSL), unify later.
- **How much of TSL to expose.** The IR needn't cover all of TSL. Start with the operators/functions
  the traits need (arith, `mix/clamp/smoothstep`, `pow`, length/normalize, noise, colour conv) and
  grow. A raw-TSL escape node covers the rest without blocking.
- **f16 in expressions.** Where does reduced precision belong (texture-ish material fields yes;
  geometry positions probably not)? Ties into the wider precision thread (task #19).
- **Serialised form.** The IR's on-disk/JSON form is also the breeding genotype and the
  copy-paste/interchange format. Define it once; DSL text and graph both derive from it.
