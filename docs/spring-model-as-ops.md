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

## Built (slices 1 and 2)

| piece | where (`src/gpu/graph/ops/springs/` unless said) |
| --- | --- |
| `bondTable` bundle: `link` (u32) and `spring` (vec4f), `n × K` each | `bondTable.ts` |
| `springForce`: bonded springs | `springForce.ts` |
| `pushApart`: CSynth's local push-apart | `pushApart.ts` |
| `adhesion`: contact repulsion and adhesion by species pair | `adhesion.ts` |
| `outward`: protrusion (fades when buried) and thrust | `outward.ts` |
| `pairDensity`: density and directional density | `pairDensity.ts` |
| `addForces`: resident sum of two forces (`addFields` runs on the host) | `addForces.ts` |
| `springIntegrate`: damping types 1 to 4, speed cap, half-step fix; dead particles stay put | `springIntegrate.ts` |
| `springGraph(state, topology, params)`: builds the graph from the terms that are on; refuses params that need a term with no op yet (`termsWithoutOps`) | `src/gpu/sim/springs/graph.ts` |
| `advance(..., { materialise: "last" \| "none" })` | `src/gpu/graph/executor.ts` |

Checked (`springOps.gpu.test.ts`): every kernel against its golden; the graph against
`stepSprings` on the CPU (2×10⁻⁵) and on the device (5×10⁻⁴), bonded-only for all four damping
types and with every pair term on over a three-species state with dead slots; a whole run with
one download, and one with none.

`advance` used to download its sink every tick. `materialise: "last"` downloads once at the end
of a batch; `"none"` returns the buffer for a view to bind.

Measured under Node on Dawn/Metal, one download at the end of the run.

Bonded springs only (two ops):

| particles | fused kernel | graph |
| --- | --- | --- |
| 64 | 0.04 ms a step | 0.14 ms a tick |
| 1,024 | 0.04 | 0.12 |
| 8,192 | 0.04 | 0.15 |

All four pair terms on (five force ops, three sums, density, integrate):

| particles | fused kernel | graph | ratio |
| --- | --- | --- | --- |
| 256 | 0.28 ms a step | 0.48 ms a tick | 1.7 |
| 1,024 | 0.75 | 1.24 | 1.6 |
| 4,096 | 2.4 | 4.5 | 1.9 |
| 8,192 | 4.2 | 8.8 | 2.1 |

- **Bonded only:** the gap does not grow with size, so it is per-tick overhead: submits, the
  executor's bookkeeping, and sources uploaded again each tick.
- **With pair terms:** four O(n²) passes against the fused kernel's one cost about twice as
  much, not four times, because each pass does less per pair. The ratio rises slowly with size.
  This is the number the fusion spike has to beat.

## For the fusion spike

Ops fall into four groups by the loop their kernel runs. Terms in one group could share a loop.

| loop | over | terms |
| --- | --- | --- |
| particle | nothing: one particle | gravity, container, viscosity, noise, motility, pull |
| bond | the particle's own bond slots | spring (built) |
| quad | the four-particle windows the particle is in | torsion |
| pair | every other particle within `pairRange` | push-apart local (built), density (built), adhesion (built), outward (built), push-apart global, density push-apart, fractal, backbone, distance target, contact, Lorentzian contact, boids |

The pair group is where fusion should pay: fifteen pair ops are fifteen O(n²) passes, one fused
kernel is one.

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
| A resident **bundle on a feedback edge**. The executor's commit adopts and releases `value.buffer` only; a bundle's buffers are in `parts`. Read, not yet tested. | Fix in the executor when `bondBreak` first puts the table on a feedback edge (slice 4). Until then the table is a source and is uploaded each tick. |
| **Coupling between pair terms.** A contact marked missing zeroes the global terms for that pair. | Model the gate as a pair weight both terms read, not as one term reaching into another. |
| **Order inside the fused step.** Force cap per bond, mobility before noise, constraints after integration. | The op graph fixes the same order by its edges. Parity with `stepSprings` is the test. |
| **Division needs free slots.** | k-th divider to k-th free slot by prefix sums (`src/gpu/scan`). The CPU rule changes to match if it must. |
| **Sources are uploaded every tick** (bond table, species, radius, the two tables). | Small, but it is host work inside the loop. They become state on feedback edges as the ops that change them arrive (slice 4); static inputs want a resident source. |
| **`addFields` is host-only.** | `addForces` stands in. Making `addFields` resident would replace it. |

## Slices

1. ~~Bond table, `springForce`, `springIntegrate`, a chain as a graph on resident feedback
   edges, parity and timing.~~ Built.
2. ~~Pair ops: push-apart (local), adhesion, outward, density.~~ Built.
3. The remaining terms: mobility, noise, motility, viscosity, container, centring, rods, fix, pull,
   torsion, and CSynth's other pair terms (global and density push-apart, fractal, backbone,
   distance targets, contacts, boids). After this the spheroid's forces are all ops.
4. `bondBreak`, `bondForm`, `divide`, `selectLeaders` as ops, all on the device. Includes the
   bundle-feedback fix.
5. Scenarios as graph builders; the page and the views read the sim state; the runner goes.
6. The fusion spike: one kernel per loop group, then across groups. Compare with per-op and with
   the hand-fused kernel.

Explainer candidate: a simulation as a graph of force ops around one feedback loop, and what fusing does to the pass count.
