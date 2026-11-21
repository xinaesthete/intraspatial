# Spring model as graph ops

Design note, 2026-11-21. Not an ADR. Supersedes the `SpringsGpu` object as the way the spring
model runs. PR #62 is held until this reshape is in.

## Decision

- **One op per term, each with its own GPU kernel** and CPU golden. A simulation is a graph of
  them with its state on feedback edges.
- **Fusion is a later spike.** Nothing here is built for it yet. The ops are written so that it
  has something to measure against (see "For the fusion spike").
- **The fused step stays as the yardstick.** `stepSprings` (CPU) is the reference the summed ops
  must equal. `createSpringsGpu` (hand-fused WGSL) is the speed a compiler would have to match.
- **State never leaves the device inside the loop.** The host supplies settings and reads
  through explicit pulls. Views bind the resident state.
- **Kernels are TGSL** (`"use gpu"`, ADR-0003), like the other graph ops. The fused kernel is
  raw WGSL; that is one more way it is the legacy path.

This follows the dance forces (`src/gpu/graph/ops/danceForces.ts`): a force op outputs a
per-particle field, the fields are summed, an integrator closes one feedback loop. Those ops run
on the CPU; these have kernels.

## Built (slices 1 to 4): the whole model

All in `src/gpu/graph/ops/springs/`, each with a TGSL kernel and a CPU golden.

| group | ops | file |
| --- | --- | --- |
| value | `bondTable` bundle: `link` (u32), `spring` (vec4f), `n × K` each | `bondTable.ts` |
| bonded | `springForce`, `torsionForce` | `springForce.ts`, `torsionForce.ts` |
| one particle | `viscosity`, `container`, `pullForce`, `mobility` | `bodyForces.ts` |
| random | `tick` (the step counter), `noiseForce`, `motilityForce` | `tick.ts`, `randomForces.ts` |
| pairs, new terms | `pushApart`, `adhesion`, `outward`, `pairDensity` | one file each |
| pairs, CSynth | `pushApartGlobal`, `pushApartDensity`, `fractalForce`, `backboneForce`, `distanceForce`, `contactForce`, `lorentzianContact`, `boids` | `csynthPairs.ts` |
| sum | `addForces` (`addFields` runs on the host) | `addForces.ts` |
| integrate | `springIntegrate` | `springIntegrate.ts` |
| constraints | `rodConstraint`, `centreConstraint`, `fixConstraint`, `pullFix` | `constraints.ts` |
| bonds | `bondBreak`, `bondForm` | `bondRules.ts` |
| division | `divisionDraw`, `divisionPlan`, `divideMove`, `restAfterDivision`, `inheritField` | `division.ts` |
| leaders | `leaderStatus`, `leaderApply` | `leaders.ts` |

- **`springGraph(state, topology, params, matrices?)`** (`src/gpu/sim/springs/graph.ts`) builds the
  graph from the same arguments `stepSprings` takes, with only the terms that are on, in the
  fused step's order: forces, sum, mobility, random terms, integrate, constraints.
- **`spheroidGraph(scenario)`** adds the scenario's rules. What they rewrite (the bond table, species,
  radius, the step) is then state on feedback edges, so the whole tick runs on the device and
  the host is not in the loop. `readSpringState` is the explicit download, for a readout.
- **Division is a plan and then copies.** `divisionPlan` matches the k-th particle that wants to
  divide with the k-th free slot by two prefix sums (`src/gpu/scan`). What "wants to" means is an
  input (`divisionDraw` is the model's own answer), and each per-particle field goes through its
  own `inheritField`, so division does not need to know what the particles carry
  ([`agent-programs.md`](agent-programs.md)).
- **`kernelOp`** (`kernelOp.ts`) is the part every op repeats: pipeline cache, leases, bind group,
  shape checks. An op file is its layout, its kernel, its uniforms and its CPU maths.
- **Random terms** draw from the same hash on the device as on the host (`rngTgsl.ts`), keyed on
  a step counter that is itself a value on a feedback edge.
- **CSynth's gate.** A pair whose contact is missing or excluded switches other terms off for
  that pair. Each term works its own gate out from the contact map; none reaches into another.

Changes to the runtime (`src/gpu/graph/executor.ts`):

- **`advance(..., { materialise })`.** It downloaded its sink every tick. `"last"` downloads once
  at the end of a batch; `"none"` returns the buffer for a view to bind.
- **A resident bundle can sit on a feedback edge.** The commit adopted and released a value's own
  buffer only; a bundle's are in its parts. The bond table needs this.
- **`simStateValue` and `downloadValue`**: the value a feedback node holds (what a view binds),
  and the explicit way to bring one to the host.
- **A feedback value that only feeds itself now advances.** A tick computed only what the sink
  depends on, so a counter (whose next value nothing else reads) stayed at its seed.

Checked (`springOps.gpu.test.ts`): every kernel against its golden; the graph against
`stepSprings`, on the CPU (2×10⁻⁵) and on the device (5×10⁻⁴), with every term of the model on
at once over a three-species state with dead slots, rods, pins, pulls, torsion springs and a
contact map that gates pairs; the chromatin scenarios (distances, contacts) and the spheroid
(2D, 3D) built from their own params; a whole run with one download, and one with none.

Rules (`springRules.gpu.test.ts`): each of break, form, division and leader selection gives
exactly the host function's result on the same state, on the device and in its golden. The
spheroid's whole tick as a graph, run on the CPU, makes exactly the host scenario's decisions
over 30 steps in 2D and 3D (same species, same bond table). On the device it runs 60 steps with
no download and ends within a tenth of the host's cell count; single precision can tip a bond
over its yield a step early, after which the runs are different runs of the same model.

One host rule changed so the device could match it. `selectLeaders` walked the cells in order
and let each new leader inhibit later candidates, which is sequential. Now a candidate is
blocked by a leader that stays or by a candidate with a lower index. They differ only when two
candidates fall near each other in one round. The seed sweep in `spring-model.md` predates this.

Measured under Node on Dawn/Metal, one download at the end of the run, single runs.

| case | particles | terms | fused kernel | graph | ratio |
| --- | --- | --- | --- | --- | --- |
| bonded chain | 64 to 8,192 | 1 | 0.04 ms | 0.12 to 0.15 ms | 3 to 4 |
| synthetic, 4 pair terms | 256 | 5 | 0.28 | 0.48 | 1.7 |
| | 1,024 | 5 | 0.75 | 1.24 | 1.6 |
| | 8,192 | 5 | 4.2 | 8.8 | 2.1 |
| spheroid, 3D | 1,790 | 5 | 0.81 | 0.91 | 1.1 |
| | 4,774 | 5 | 2.02 | 2.46 | 1.2 |
| chromatin, distances | 400 | 2 | 0.47 | 0.40 | 0.9 |

The spheroid's whole tick (forces, turnover, division, leaders), 3D, against the legacy runner,
which downloads every ten steps and runs division and leader selection on the host:

| particles | legacy runner | graph, no download | ratio |
| --- | --- | --- | --- |
| 4,190 | 3.3 ms a step | 3.9 ms a tick | 1.16 |
| 7,174 | 6.0 | 7.1 | 1.18 |
| 10,716 | 9.4 | 11.2 | 1.19 |

The buffer pool did not grow over 100 ticks at any size.

- **A fixed cost of about 0.1 ms a tick** (submits, the executor's bookkeeping, sources uploaded
  again each tick) is all that separates the two when the kernels are cheap.
- **Pair passes are cheaper apart than expected.** Each does less per pair than the fused loop,
  and the fused kernel pays for every term's branches whether or not it is on. For the two real
  scenarios the graph is within a fifth of the fused kernel, or ahead.
- **So fusion has less to win than the pass count suggests.** The synthetic four-term case, at
  about twice, is where it would show.

## For the fusion spike

Ops fall into four groups by the loop their kernel runs. Terms in one group could share a loop.

| loop | over | terms |
| --- | --- | --- |
| particle | nothing: one particle | gravity, container, viscosity, noise, motility, pull |
| bond | the particle's own bond slots | spring |
| quad | the four-particle windows the particle is in | torsion |
| pair | every other particle within `pairRange` | push-apart (local, global, density), density, adhesion, outward, fractal, backbone, distance target, contact, Lorentzian contact, boids |

The pair group is where fusion should pay, though the measurements above say by less than the
pass count suggests. The three simplest CSynth pair terms already share one kernel that a
uniform selects a term from: fusion in miniature, and the shape an interpreter would have
([`agent-programs.md`](agent-programs.md)).

## Values

| value | shape | notes |
| --- | --- | --- |
| `pos`, `prev` | points, vec4 | xyz and density; velocity is `pos − prev`, as in CSynth. vec4 because a `vec3` array is 16 bytes a sample on the device and 12 on the host |
| `species`, `radius` | points, u32 / f32 | changed only by division and leader selection. `species` 255 is a dead slot |
| `density` | points, vec2 | density and directional density, from `pairDensity`; `springIntegrate` copies the first into `pos.w` |
| `repulsion`, `adhesion` | matrix, species × species | inputs to `adhesion` |
| `bondTable` | bundle: `link` (n·K, u32), `spring` (n·K, vec4f) | one writer per tick: break, then form. Two parts, so no kernel bitcasts and the empty marker is never an f32 |
| `rod`, `fix`, `pull`, `torsion` | points | inputs; host-owned, sent down when edited |
| `force` | points, vec4 | what every force op outputs (w unused) |

## Ops

| group | ops | in → out |
| --- | --- | --- |
| forces | one per term above | state (+ `bondTable`) → `force` |
| sum | `addFields` (exists) | forces → `force` |
| integrate | `springIntegrate`: mobility, damping types 1–4, speed cap, half-step fix | `pos`, `prev`, `force` → `pos`, `prev` |
| constraints | `rodConstraint`, `centre`, `fixConstraint` | `pos` → `pos` |
| bonds | `bondBreak`, `bondForm` | `pos`, `bondTable` → `bondTable` |
| growth | `divide`, `selectLeaders` | state → state |
| readouts | counts, centroid, bounds, bond stress | state → small values |

A scenario becomes a function that builds a graph and returns its state handles. `advance` runs
it. The playground and the views take the state from the sim store.

## What the views need from it

The deck layers take a plain `SimBuffers` (positions, attributes, bond table, `n`, `K`, layout as
parameters), not the legacy object. The graph version must offer:

- the **final** positions of a tick, never an intermediate, in a buffer with `COPY_SRC`;
- a defined "tick complete" point (the history ring copies positions then);
- a bond table edited on the host sent down without touching positions (drawn lines);
- optionally, more resident values to bind: bond strain per slot, particle birth step.

## What is not straightforward

| issue | handling |
| --- | --- |
| **Static sources are uploaded every tick** (matrices, rods, the species tables; the bond table and species too when no rule rewrites them). | Small, but it is host work inside the loop. Wants a resident source that uploads once. |
| **Leaders are not cleared when leader selection is switched off**, as the host scenario does. | A graph is built for one set of rules. Live settings are slice 5. |
| **Totals and events** (bonds broken, divisions) are not counted on the device. | Reductions, with the readouts. |
| **`addFields` is host-only.** | `addForces` stands in. Making `addFields` resident would replace it. |

## Slices

1. ~~Bond table, `springForce`, `springIntegrate`, a chain as a graph on resident feedback
   edges, parity and timing.~~ Built.
2. ~~Pair ops: push-apart (local), adhesion, outward, density.~~ Built.
3. ~~The remaining terms: mobility, noise, motility, viscosity, container, centring, rods, fix,
   pull, torsion, and CSynth's other pair terms.~~ Built.
4. ~~`bondBreak`, `bondForm`, `divide`, `selectLeaders` as ops, all on the device, with the
   bundle-feedback fix.~~ Built.
5. Scenarios as graph builders; the page and the views read the sim state; the runner goes.
6. The fusion spike: one kernel per loop group, then across groups. Compare with per-op and with
   the hand-fused kernel.

Explainer candidate: a simulation as a graph of force ops around one feedback loop, and what fusing does to the pass count.
