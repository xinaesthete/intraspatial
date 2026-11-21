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

## Built (slices 1 to 5): the whole model, and a runner for it

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
- **`createSpringSim(scenario, { backend })`** (`src/gpu/sim/springs/sim.ts`) is what a page or a view
  drives. `advance(steps)` runs on the device and downloads nothing; `buffers()` hands a view the
  positions, species, radius and bond table to bind; `sync()` is the explicit download. A change
  of settings is noticed, and the graph is rebuilt from the running state. The playground's GPU
  setups run on it.
- **Every input is state.** What a rule rewrites is fed its new value; everything else (matrices,
  rods, the species tables, and the bond table and species when nothing rewrites them) is fed
  itself. A self-fed value is uploaded for the first tick and stays on the device, where a plain
  source is uploaded every tick. After the first tick the loop makes no uploads and no downloads.
- **`scenarioGraph(scenario)`** builds any scenario's whole tick: a scenario says which rules it
  applies through `graphRules()`. It adds the scenario's rules. What they rewrite (the bond table, species,
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

## Fusion: packed bundles and an interpreted pair kernel (slice 6, a spike)

**Result.** One interpreted pass over the pairs does the work of the model's four pair ops at the
speed of the hand-fused kernel, with the terms chosen by a program that is data. It needs the
state and inputs packed, each in one buffer.

### Packed bundles

A kernel may bind eight storage buffers. A kernel doing the work of many ops needs more values
than that, however it is produced. So a bundle can be **packed**: its parts are ranges of one
buffer (`FieldValue.packed`, `ResidentBuffer.offset` and `.sliceOf`, `packOp`).

| who binds | how |
| --- | --- |
| an op that wants one part | that range, as its own binding: its kernel is unchanged |
| a fused kernel | the whole buffer, once, as raw words, with part offsets passed as data |
| an op that binds whole buffers | the executor hands it a copy (`OpType.slices` says an op takes ranges) |

- Offsets are multiples of 256 bytes, the alignment a binding may start on (`packedLayout`).
- Ownership is the root buffer's. A slice is a view; `payloadsOf` reports roots.
- On the host a packed bundle is an ordinary bundle of separate arrays.
- `springGraph(..., { packed: true })` keeps the state in one bundle on one feedback edge and
  the fixed inputs in another. The ops are the same and the results are the same bits (tested
  over the spheroid's whole tick, 2D and 3D).
- Read and write never share a buffer: a tick reads the state bundle and `pack` writes a new one,
  and the feedback edge swaps them.

### The interpreter

`pairInterpreter` (`pairInterpreter.ts`) takes the packed state and inputs and outputs a force
and a density. Its program is two parts of the inputs bundle: `[opcode, paramBase]` pairs and a
block of parameters. `springGraph(..., { fusePairs: true })` uses it in place of `pushApart`,
`adhesion`, `outward` and `pairDensity`, which remain the definition.

- **Each particle decodes the program once, into registers.** The loop over pairs then reads
  nothing but positions. The first version read opcodes and parameters per pair per
  instruction, and was slower than the separate ops by half again.
- **Terms that need the whole neighbourhood** (the outward push) gather into registers and are
  finished after the loop.
- **A term may appear once in a program.** Registers are named per term. A program with repeats
  would need registers in an array.
- Four storage buffers: state, inputs, and the two outputs.

Checked: against its own golden; against the three force ops it replaces, summed (10⁻⁴); the
density against `pairDensity`; a graph with it against `stepSprings`; the spheroid's whole tick.

### Measured

Node on Dawn/Metal. Best of four rounds, the variants taken in turn within each round. Another
session was using the same GPU, and rounds differed by up to a half at one size, hence best-of.

Synthetic tissue, four pair terms on, ms a tick:

| particles | hand-fused | per-op | per-op, packed | interpreted |
| --- | --- | --- | --- | --- |
| 1,024 | 0.82 | 1.36 | 1.38 | 0.81 |
| 4,096 | 2.73 | 4.90 | 4.92 | 2.74 |
| 8,192 | 4.64 | 9.52 | 9.42 | 5.34 |

Spheroid, 3D, whole tick, ms a tick:

| particles | legacy runner | per-op | per-op, packed | interpreted pairs |
| --- | --- | --- | --- | --- |
| 7,174 | 6.78 | 9.42 | 8.73 | 7.36 |
| 10,716 | 9.28 | 11.32 | 10.75 | 8.70 |

- **Packing costs nothing measurable** for the per-op graph.
- **The interpreter matches the hand-fused kernel** to within a sixth, and is 1.7 to 1.8 times
  faster than the separate ops where four pair terms are on.
- **At the larger spheroid the graph with interpreted pairs is ahead of the legacy runner**,
  which also downloads every ten steps.

### Not done

- CSynth's other pair terms are still separate ops. They need the contact gate as a shared
  per-pair value, and the two matrices in the inputs bundle.
- The bond, particle and quad loops are not fused. They are cheap; the gain would be fewer
  dispatches, about 0.1 ms a tick in all.
- Compiled fusion was not tried. The interpreter already matches the hand-fused kernel, so it
  would have to win on something other than speed.
- Software Vulkan (CI) has not run these kernels yet.

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
| **A change of settings is a round trip.** The graph reads settings when it is built, so `createSpringSim` downloads the state and rebuilds. | Fine for a slider; settings as graph inputs would avoid it. |
| **The playground still downloads every batch**, because it draws on a 2D canvas from the host's copy. | The deck views bind `buffers()` and leave the download out. |
| **The legacy `SpringsGpu` and `createSpheroidGpuRunner` are still there**, as the fused baseline and because the views branch uses them. | Remove once nothing calls them. |
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
5. ~~Scenarios as graph builders; a runner the page drives; every input held on the device.~~
   Built. The page's views still read the host's copy.
6. ~~The fusion spike.~~ Done as packed bundles and an interpreted pair kernel; see above.

Explainer candidate: a simulation as a graph of force ops around one feedback loop, and what fusing does to the pass count.
